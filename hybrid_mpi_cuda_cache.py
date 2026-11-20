import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI
from numba import cuda, float64, njit, types
from numba.extending import intrinsic
from llvmlite import ir

TILE_DIM = 16
SHARED_DIM = 18
GLOBAL_NX = GLOBAL_NY = 512
LOCAL_N = 256
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
DX = 1.0 / 513.0


def gpu_info():
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,uuid,compute_cap',
        '--format=csv,noheader'], text=True).strip().splitlines()[0]
    name, uuid, capability = [v.strip() for v in raw.split(',')]
    return {'name': name, 'uuid': uuid, 'compute_capability': capability}


def cpu_info():
    info = subprocess.check_output(['lscpu', '-J'], text=True)
    topology = json.loads(info)
    stable = [row for row in topology['lscpu'] if row['field'].rstrip(':') in
              ('Architecture', 'CPU(s)', 'On-line CPU(s) list', 'Vendor ID', 'Model name',
               'Model', 'CPU family', 'Stepping', 'Thread(s) per core', 'Core(s) per socket',
               'Socket(s)', 'L1d cache', 'L1i cache', 'L2 cache', 'L3 cache', 'NUMA node(s)')]
    signature = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    line_path = Path('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size')
    line = int(line_path.read_text()) if line_path.exists() else 64
    return {'signature': signature, 'topology': topology, 'cache_line_bytes': line,
            'logical_cpus': os.cpu_count(), 'architecture': platform.machine()}


def shared_context(comm):
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node.size != 4:
        raise RuntimeError('All four ranks must share one compute node.')
    win = MPI.Win.Allocate_shared(64 * 8 + 63 if node.rank == 0 else 0, 1, comm=node)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint8, shape=(64 * 8 + 63,))
    offset = node.bcast((-raw.ctypes.data) % 64 if node.rank == 0 else None, root=0)
    view = np.ndarray(buffer=buf, dtype=np.uint64, offset=offset, shape=(64,))
    if view.ctypes.data % 64:
        raise RuntimeError('Shared window is not locally aligned to 64 bytes.')
    win.Lock_all()
    if node.rank == 0:
        view[:] = 0
    win.Sync()
    node.Barrier()
    win.Sync()
    return node, win, view


def init_hybrid_context(github_user):
    comm = MPI.COMM_WORLD
    if comm.size != 4:
        raise RuntimeError(f'Requires exactly 4 MPI ranks; got {comm.size}.')
    if os.environ.get('NUMBA_ENABLE_CUDASIM') == '1':
        raise RuntimeError('Simulator is prohibited for official telemetry.')
    cuda.select_device(0)
    gpu = comm.bcast(gpu_info() if comm.rank == 0 else None, root=0)
    own_uuid = gpu_info()['uuid']
    if not comm.allreduce(own_uuid == gpu['uuid'], op=MPI.LAND):
        raise RuntimeError('Every rank must use the same physical GPU 0.')
    cpu = comm.bcast(cpu_info() if comm.rank == 0 else None, root=0)
    if cpu['cache_line_bytes'] != 64:
        raise RuntimeError('This experiment requires 64-byte CPU cache lines.')
    cart = comm.Create_cart((2, 2), periods=(False, False), reorder=False)
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    node, win, view = shared_context(comm)
    material = f'{github_user.strip()}::{cpu["signature"]}::{gpu["uuid"]}'
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big')
    return dict(comm=comm, cart_comm=cart, rank=comm.rank, coords=cart.Get_coords(comm.rank),
                neighbors=(up, down, left, right), node_comm=node, win=win,
                shm_view=view, cpu=cpu, gpu=gpu, INSTANCE_SEED=seed,
                github_username=github_user.strip())


def initialize_fields(ctx):
    y0, x0 = ctx['coords']
    iy = y0 * LOCAL_N + np.arange(258)
    ix = x0 * LOCAL_N + np.arange(258)
    y, x = np.meshgrid(iy * DX, ix * DX, indexing='ij')
    phase = (ctx['INSTANCE_SEED'] % 1_000_003) / 1_000_003 * 2 * np.pi
    u = np.sin(np.pi * x) * np.sin(np.pi * y)
    u *= 1 + 0.05 * np.cos(2 * np.pi * x + phase) * np.cos(2 * np.pi * y)
    physical = (iy[:, None] == 0) | (iy[:, None] == 513) | (ix[None, :] == 0) | (ix[None, :] == 513)
    u[physical] = 0
    kappa = 0.1 + 0.025 * np.sin(2 * np.pi * x + phase) * np.sin(2 * np.pi * y)
    source = 0.01 * (1 + 0.2 * np.cos(2 * np.pi * x + phase) * np.cos(2 * np.pi * y))
    return tuple(np.ascontiguousarray(v, dtype=np.float64) for v in (u, kappa, source))


@cuda.jit
def pack_halos_kernel(u, packed):
    t = cuda.grid(1)
    if t < LOCAL_N:
        j = t + 1
        packed[0, t] = u[1, j]
        packed[1, t] = u[LOCAL_N, j]
        packed[2, t] = u[j, 1]
        packed[3, t] = u[j, LOCAL_N]


@cuda.jit
def unpack_halos_kernel(u, packed, up, down, left, right):
    t = cuda.grid(1)
    if t < LOCAL_N:
        j = t + 1
        if up:
            u[0, j] = packed[0, t]
        if down:
            u[257, j] = packed[1, t]
        if left:
            u[j, 0] = packed[2, t]
        if right:
            u[j, 257] = packed[3, t]


@cuda.jit
def tiled_stencil_kernel(u, out, kappa, source, dt, inv_dx2, region):
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), float64)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockIdx.x * TILE_DIM, cuda.blockIdx.y * TILE_DIM
    tid = ty * TILE_DIM + tx
    for q in range(tid, SHARED_DIM * SHARED_DIM, TILE_DIM * TILE_DIM):
        sy, sx = q // SHARED_DIM, q % SHARED_DIM
        yi, xi = by + sy, bx + sx
        if region == 1 and (yi == 0 or yi == 257 or xi == 0 or xi == 257):
            tile[sy, sx] = 0.0
        else:
            tile[sy, sx] = u[yi, xi]
    cuda.syncthreads()
    i, j = by + ty + 1, bx + tx + 1
    border = i == 1 or i == LOCAL_N or j == 1 or j == LOCAL_N
    if region == 0 or (region == 1 and not border) or (region == 2 and border):
        sy, sx = ty + 1, tx + 1
        v = tile[sy, sx]
        k = kappa[i, j]
        flux = (0.5 * (k + kappa[i-1, j]) * (tile[sy-1, sx] - v)
                + 0.5 * (k + kappa[i+1, j]) * (tile[sy+1, sx] - v)
                + 0.5 * (k + kappa[i, j-1]) * (tile[sy, sx-1] - v)
                + 0.5 * (k + kappa[i, j+1]) * (tile[sy, sx+1] - v)) * inv_dx2
        out[i, j] = v + dt * (flux + source[i, j] - LAMBDA_DAMP * v)


class Pipeline:
    def __init__(self, ctx, host_u, host_k, host_s, dt):
        self.ctx, self.dt = ctx, dt
        self.interior, self.boundary = cuda.stream(), cuda.stream()
        self.u = cuda.to_device(host_u)
        self.out = cuda.to_device(host_u)
        self.k = cuda.to_device(host_k)
        self.source = cuda.to_device(host_s)
        self.send_d = cuda.device_array((4, LOCAL_N), np.float64)
        self.recv_d = cuda.device_array((4, LOCAL_N), np.float64)
        self.send_h = cuda.pinned_array((4, LOCAL_N), np.float64)
        self.recv_h = cuda.pinned_array((4, LOCAL_N), np.float64)
        self.recv_h[:] = 0
        self.present = tuple(int(n != MPI.PROC_NULL) for n in ctx['neighbors'])

    def exchange_start(self):
        c = self.ctx['cart_comm']
        requests = []
        opposite = (1, 0, 3, 2)
        for d, neighbor in enumerate(self.ctx['neighbors']):
            if neighbor != MPI.PROC_NULL:
                requests.append(c.Irecv(self.recv_h[d], source=neighbor, tag=100+opposite[d]))
        pack_halos_kernel[2, 128, self.boundary](self.u, self.send_d)
        self.send_d.copy_to_host(self.send_h, stream=self.boundary)
        self.boundary.synchronize()
        for d, neighbor in enumerate(self.ctx['neighbors']):
            if neighbor != MPI.PROC_NULL:
                requests.append(c.Isend(self.send_h[d], dest=neighbor, tag=100+d))
        return requests

    def exchange_finish(self, requests):
        MPI.Request.Waitall(requests)
        self.recv_d.copy_to_device(self.recv_h, stream=self.boundary)
        unpack_halos_kernel[2,128,self.boundary](self.u, self.recv_d, *self.present)

    def step_serialized(self):
        self.exchange_finish(self.exchange_start())
        tiled_stencil_kernel[(16,16),(16,16),self.boundary](
            self.u, self.out, self.k, self.source, self.dt, 1/DX**2, 0)
        self.boundary.synchronize()
        self.u, self.out = self.out, self.u


def step_overlapped(p):
    # Deep-only launch avoids even speculative tile reads of mutable ghosts.
    tiled_stencil_kernel[(16,16),(16,16),p.interior](
        p.u, p.out, p.k, p.source, p.dt, 1/DX**2, 1)
    requests = p.exchange_start()
    p.exchange_finish(requests)
    tiled_stencil_kernel[(16,16),(16,16),p.boundary](
        p.u, p.out, p.k, p.source, p.dt, 1/DX**2, 2)
    p.interior.synchronize()
    p.boundary.synchronize()
    p.u, p.out = p.out, p.u


class VolatileLoad(ir.instructions.LoadInstr):
    def descr(self, buf):
        out = []
        super().descr(out)
        buf.append(''.join(out).replace('load ', 'load volatile ', 1))


class VolatileStore(ir.instructions.StoreInstr):
    def descr(self, buf):
        out = []
        super().descr(out)
        buf.append(''.join(out).replace('store ', 'store volatile ', 1))


@intrinsic
def volatile_increment(typingctx, array, index):
    signature = types.void(array, index)
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = builder.gep(ary.data, [args[1]])
        load = VolatileLoad(builder.block, ptr, name='counter_read')
        load.align = 8
        builder._insert(load)
        value = builder.add(load, ir.Constant(ir.IntType(64), 1))
        store = VolatileStore(builder.block, value, ptr)
        store.align = 8
        builder._insert(store)
        return context.get_dummy_value()
    return signature, codegen


@njit(nogil=True)
def host_shm_accumulate(shm_view, index, iterations):
    for _ in range(iterations):
        volatile_increment(shm_view, index)


def benchmark_shm_strides(ctx, iterations=SHM_ITERS, repeats=3):
    node, win, view = ctx['node_comm'], ctx['win'], ctx['shm_view']
    rank = node.rank
    host_shm_accumulate(view, rank, 1)
    llvm = host_shm_accumulate.inspect_llvm(host_shm_accumulate.signatures[0])
    audit = {'volatile_load_i64': len(re.findall(r'load volatile i64', llvm)),
             'volatile_store_i64': len(re.findall(r'store volatile i64', llvm)),
             'sha256': hashlib.sha256(llvm.encode()).hexdigest()}
    if not audit['volatile_load_i64'] or not audit['volatile_store_i64']:
        raise AssertionError('Volatile RMW absent from generated LLVM.')
    if ctx['rank'] == 0:
        Path('host_shm_accumulate.ll').write_text(llvm)
    node.Barrier()
    result = {}
    for trial in range(repeats):
        for stride in ((1,8,16), (16,1,8), (8,16,1))[trial % 3]:
            if rank == 0:
                view[:] = 0
            win.Sync()
            node.Barrier()
            win.Sync()
            start = MPI.Wtime()
            host_shm_accumulate(view, rank * stride, iterations)
            elapsed = MPI.Wtime() - start
            win.Sync()
            node.Barrier()
            win.Sync()
            maximum = node.allreduce(elapsed, op=MPI.MAX)
            passed = all(int(view[r*stride]) == iterations for r in range(4))
            if not node.allreduce(passed, op=MPI.LAND):
                raise AssertionError('Shared counter verification failed.')
            result.setdefault(str(stride), []).append(maximum * 1000)
    sweep = {key: {'spacing_bytes': int(key)*8, 'samples_ms': values,
                    'median_ms': float(np.median(values))} for key, values in result.items()}
    sweep['false_sharing_ratio_1_over_8'] = sweep['1']['median_ms']/sweep['8']['median_ms']
    sweep['padding_ratio_8_over_16'] = sweep['8']['median_ms']/sweep['16']['median_ms']
    return sweep, audit


def diagnostics(ctx, p, host_k, host_s, step):
    current = p.u.copy_to_host()
    interior = current[1:257,1:257]
    mass = float(interior.sum() * DX**2)
    energy = float(0.5 * np.square(interior).sum() * DX**2)
    values = np.array([mass, energy], dtype=np.float64)
    total = np.empty_like(values)
    ctx['cart_comm'].Allreduce(values, total, op=MPI.SUM)
    entry = {'step': step, 'mass': float(total[0]), 'energy_l2': float(total[1])}
    if step:
        previous = p.out.copy_to_host()
        old_mass = float(previous[1:257,1:257].sum() * DX**2)
        boundary_flux = 0.0
        # Internal interface flux cancels exactly between neighbor ranks.
        for d, neighbor in enumerate(ctx['neighbors']):
            if neighbor != MPI.PROC_NULL:
                continue
            if d == 0:
                a,b,ka,kb = previous[1,1:257],previous[0,1:257],host_k[1,1:257],host_k[0,1:257]
            elif d == 1:
                a,b,ka,kb = previous[256,1:257],previous[257,1:257],host_k[256,1:257],host_k[257,1:257]
            elif d == 2:
                a,b,ka,kb = previous[1:257,1],previous[1:257,0],host_k[1:257,1],host_k[1:257,0]
            else:
                a,b,ka,kb = previous[1:257,256],previous[1:257,257],host_k[1:257,256],host_k[1:257,257]
            boundary_flux += float((0.5*(ka+kb)*(b-a)).sum())
        expected = p.dt * (boundary_flux + float(host_s[1:257,1:257].sum()*DX**2)
                           - LAMBDA_DAMP*old_mass)
        residual = ctx['cart_comm'].allreduce(mass-old_mass-expected, op=MPI.SUM)
        entry['global_heat_balance_abs_residual'] = abs(residual)
        if abs(residual) >= 1e-12:
            raise AssertionError(f'Distributed heat balance failed: {residual}')
    if ctx['rank'] == 0:
        print('ENERGY', json.dumps(entry), flush=True)
    return entry


def run_mode(ctx, fields, dt, mode):
    p = Pipeline(ctx, *fields, dt)
    action = p.step_serialized if mode == 'serialized' else lambda: step_overlapped(p)
    action()
    p.u.copy_to_device(fields[0])
    p.out.copy_to_device(fields[0])
    cuda.synchronize()
    records = [diagnostics(ctx, p, fields[1], fields[2], 0)]
    times = []
    for endpoint in (100,200,300):
        ctx['cart_comm'].Barrier()
        start = MPI.Wtime()
        for _ in range(100):
            action()
        cuda.synchronize()
        times.append(ctx['cart_comm'].allreduce(MPI.Wtime()-start, op=MPI.MAX))
        records.append(diagnostics(ctx, p, fields[1], fields[2], endpoint))
    return p.u.copy_to_host()[1:257,1:257].copy(), sum(times)*1000, records


def ptx_audit(ctx):
    ptx = tiled_stencil_kernel.inspect_asm(tiled_stencil_kernel.signatures[0])
    counts = {op: len(re.findall(r'\b'+re.escape(op)+r'\b', ptx))
              for op in ('st.shared.f64','ld.shared.f64','bar.sync')}
    # Some LLVM backends express identical 64-bit shared loads/stores as b64/u64.
    counts['shared_stores_any_64bit'] = len(re.findall(r'\bst\.shared\.(?:f64|b64|u64)\b',ptx))
    counts['shared_loads_any_64bit'] = len(re.findall(r'\bld\.shared\.(?:f64|b64|u64)\b',ptx))
    if not counts['bar.sync'] or not counts['shared_stores_any_64bit'] or not counts['shared_loads_any_64bit']:
        raise AssertionError(f'Shared-memory PTX audit failed: {counts}')
    if ctx['rank'] == 0:
        Path('tiled_stencil_kernel.ptx').write_text(ptx)
    counts['sha256'] = hashlib.sha256(ptx.encode()).hexdigest()
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('github_username')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?',args.github_username):
        raise ValueError('Set a valid real GitHub username.')
    ctx = init_hybrid_context(args.github_username)
    try:
        fields = initialize_fields(ctx)
        kmax = ctx['cart_comm'].allreduce(float(fields[1].max()),op=MPI.MAX)
        dtmax = DX**2/(4*kmax)
        # Includes damping in the positivity bound, and a 10% safety margin.
        dt = 0.9/(4*kmax/DX**2 + LAMBDA_DAMP)
        serial, serial_ms, serial_energy = run_mode(ctx, fields, dt, 'serialized')
        overlap, overlap_ms, overlap_energy = run_mode(ctx, fields, dt, 'overlapped')
        diff = ctx['cart_comm'].allreduce(float(np.max(np.abs(serial-overlap))), op=MPI.MAX)
        if diff >= 1e-12:
            raise AssertionError(f'Serialized/overlapped mismatch: {diff}')
        audit = ptx_audit(ctx)
        sweep, llvm = benchmark_shm_strides(ctx)
        metadata = {'rank': ctx['rank'], 'coords': ctx['coords'],
                    'cpu_affinity': sorted(os.sched_getaffinity(0)),
                    'shared_base_mod_64': int(ctx['shm_view'].ctypes.data % 64)}
        ranks = ctx['cart_comm'].gather(metadata, root=0)
        if ctx['rank'] == 0:
            import numba, mpi4py
            manifest = dict(schema_version=1, status='executed_on_real_gpu',
                github_username=ctx['github_username'], INSTANCE_SEED=ctx['INSTANCE_SEED'],
                cpu=ctx['cpu'], gpu=ctx['gpu'], ranks=ranks,
                python=sys.version, numpy=np.__version__, numba=numba.__version__,
                mpi4py=mpi4py.__version__, mpi_library=MPI.Get_library_version(),
                global_shape=[512,512], local_shape=[258,258], steps=300,
                boundary_condition='homogeneous Dirichlet', diffusivity='conservative face average',
                source='0.01*(1+0.2*cos(2*pi*x+phase)*cos(2*pi*y))',
                dx=DX, kappa_max=kmax, dt=dt, dt_max=dtmax, courant_ratio=dt/dtmax,
                lambda_damp=LAMBDA_DAMP, max_diff=diff,
                gpu_serialized_ms=serial_ms, gpu_overlapped_ms=overlap_ms,
                overlap_speedup=serial_ms/overlap_ms,
                timing_scope='max-rank synchronized wall time; warmup and diagnostics excluded',
                serialized_energy=serial_energy, overlapped_energy=overlap_energy,
                ptx_audit=audit, llvm_audit=llvm, shm_iterations=SHM_ITERS,
                shm_repeats=3, shm_benchmark=sweep,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            Path('hybrid_lab_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(manifest,indent=2),flush=True)
    finally:
        ctx['win'].Unlock_all()
        ctx['win'].Free()
        ctx['node_comm'].Free()
        ctx['cart_comm'].Free()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Rank {MPI.COMM_WORLD.rank}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        MPI.COMM_WORLD.Abort(1)
