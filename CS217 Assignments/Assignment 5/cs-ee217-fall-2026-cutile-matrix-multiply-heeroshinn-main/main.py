#!/usr/bin/env python3
"""Driver for the cuTile tiled matrix multiplication. Do not modify this file.

Usage:
    python main.py                  # A, B and C are 1000 x 1000
    python main.py <m>              # m x m
    python main.py <m> <k> <n>      # A: m x k, B: k x n, C: m x n
Options:
    --block-m/--block-n/--block-k   block sizes (powers of 2, at least 16; default 64 x 64 x 32)
    --dump DIR                      save the compiled kernel (Tile IR and cubin) in DIR

Exit codes: 0 passed, 1 wrong results, 2 bad arguments, 3 exception in tilemm.py, 4 no GPU.
"""
import argparse
import contextlib
import glob
import os
import sys
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser(
        usage="python main.py [m | m k n] [--block-m BM] [--block-n BN] [--block-k BK] "
              "[--dump DIR]")
    p.add_argument("dims", nargs="*", type=int, help="m, or m k n (default 1000)")
    p.add_argument("--block-m", type=int, default=64)
    p.add_argument("--block-n", type=int, default=64)
    p.add_argument("--block-k", type=int, default=32)
    p.add_argument("--dump", metavar="DIR", help="save the compiled kernel in DIR")
    args = p.parse_args()

    if len(args.dims) == 0:
        args.m = args.k = args.n = 1000
    elif len(args.dims) == 1:
        args.m = args.k = args.n = args.dims[0]
    elif len(args.dims) == 3:
        args.m, args.k, args.n = args.dims
    else:
        p.error("give no sizes, one size (m), or three sizes (m k n)")
    if min(args.m, args.k, args.n) <= 0:
        p.error("matrix dimensions must be positive")
    for name in ("block_m", "block_n", "block_k"):
        b = getattr(args, name)
        if b < 16 or b & (b - 1):
            p.error(f"--{name.replace('_', '-')} must be a power of 2 and at least 16 (got {b})")
    return args


args = parse_args()

# These must be set before cuTile is imported
if args.dump:
    dump_dir = os.path.abspath(args.dump)
    os.makedirs(dump_dir, exist_ok=True)
    os.environ["CUDA_TILE_DUMP_TILEIR"] = dump_dir
    # A private compile cache, so the kernel is compiled (not reused) and its cubin can be found
    os.environ["CUDA_TILE_CACHE_DIR"] = os.path.join(dump_dir, "cache")

import cupy as cp              # noqa: E402
import cupyx.profiler          # noqa: E402

try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError
except Exception:
    print("No CUDA GPU is visible. Run on Bender.", file=sys.stderr)
    sys.exit(4)

from tilemm import tiled_matmul  # noqa: E402


@contextlib.contextmanager
def no_cupy_matmul():
    """Make CuPy's matrix products raise while tilemm.py runs: the kernel must do the work.
    (The @ operator and ndarray.dot can't be patched; the CI source check catches those.)"""
    def blocked(*a, **kw):
        raise RuntimeError("tilemm.py must compute C with your cuTile kernel, "
                           "not with CuPy's matrix multiplication")
    targets = [(cp, n) for n in ("matmul", "dot", "tensordot", "einsum", "inner", "vdot", "outer")]
    targets += [(cp.linalg, n) for n in ("multi_dot", "matrix_power")]
    saved = [(obj, n, getattr(obj, n)) for obj, n in targets if hasattr(obj, n)]
    for obj, n, _ in saved:
        setattr(obj, n, blocked)
    try:
        yield
    finally:
        for obj, n, f in saved:
            setattr(obj, n, f)


def sync():
    cp.cuda.get_current_stream().synchronize()


def guarded(rows, cols, body, guard_value, guard):
    """A rows x cols matrix inside a larger buffer whose ends hold guard_value.

    Reads past the edge of A or B pick up NaN and make C wrong; writes past the
    edge of C change its guard and are reported."""
    buf = cp.full((rows * cols + 2 * guard,), guard_value, dtype=cp.float32)
    mat = buf[guard:guard + rows * cols].reshape(rows, cols)
    mat[...] = body
    return buf, mat


def run_student(A, B, C):
    with no_cupy_matmul():
        tiled_matmul(A, B, C, BLOCK_M=args.block_m, BLOCK_N=args.block_n, BLOCK_K=args.block_k)


m, k, n = args.m, args.k, args.n
# Loose enough to accept a kernel that multiplies in TF32 (10-bit mantissa) on the tensor cores
tolerance = 5e-3
flops = 2.0 * m * n * k

# Set up the problem ---------------------------------------------------------

print("\nSetting up the problem...", end="", flush=True)
t0 = time.perf_counter()
rng = cp.random.default_rng(217)
A_h = rng.random((m, k), dtype=cp.float32)
B_h = rng.random((k, n), dtype=cp.float32)
big = max(args.block_m, args.block_n, args.block_k)
# TILEMM_NO_GUARD=1 allocates the matrices exactly, without CuPy's memory pool, so
# compute-sanitizer sees out-of-bounds accesses
guard = big * (max(m, k, n) + big)
if os.environ.get("TILEMM_NO_GUARD") == "1":
    guard = 0
    cp.cuda.set_allocator(None)
SENTINEL = -12345.0
A_buf, A = guarded(m, k, A_h, cp.nan, guard)
B_buf, B = guarded(k, n, B_h, cp.nan, guard)
C_buf, C = guarded(m, n, cp.nan, SENTINEL, guard)
del A_h, B_h
sync()
print(f"{time.perf_counter() - t0:f} s")
print(f"    A: {m} x {k}\n    B: {k} x {n}\n    C: {m} x {n}")
props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
print(f"    Blocks: BLOCK_M={args.block_m}, BLOCK_N={args.block_n}, BLOCK_K={args.block_k}, "
      f"device={props['name'].decode()}")

# Launch the kernel (the first call also compiles it) ------------------------

print("Launching kernel (first call, includes compiling)...", end="", flush=True)
t0 = time.perf_counter()
try:
    run_student(A, B, C)
    sync()
except Exception:
    print("\n")
    traceback.print_exc()
    print("\nYour code in tilemm.py raised an exception (see the traceback above). For a cuTile "
          "compile error, the message points at the line of the kernel it could not compile.")
    sys.exit(3)
print(f"{time.perf_counter() - t0:f} s")

if args.dump:
    import sqlite3
    n_cubins = 0
    for db in glob.glob(os.path.join(dump_dir, "cache", "*.db")):
        con = sqlite3.connect(db)
        for table, in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in con.execute(f"SELECT * FROM {table}"):
                for value in row:
                    if isinstance(value, bytes) and value[:4] == b"\x7fELF":
                        n_cubins += 1
                        with open(os.path.join(dump_dir, f"matmul_kernel_{n_cubins}.cubin"), "wb") as f:
                            f.write(value)
        con.close()
    print(f"    Saved the Tile IR and {n_cubins} cubin file(s) in {dump_dir}; "
          f"view the machine code with: cuobjdump -sass {args.dump}/*.cubin")

# Verify correctness ---------------------------------------------------------

print("Verifying results...", flush=True)
ref = A.astype(cp.float64) @ B.astype(cp.float64)
got = C.astype(cp.float64)
scale = cp.maximum(cp.abs(ref), 1.0)
rel = cp.abs(got - ref) / scale
bad = ~(rel <= tolerance)          # NaN counts as wrong
wrong = int(bad.sum())
total = m * n
guard_ok = bool((C_buf[:guard] == SENTINEL).all() and (C_buf[guard + total:] == SENTINEL).all())
max_err = float(rel.max()) if wrong == 0 else float("nan")

if wrong > 0:
    rows, cols = cp.nonzero(bad)
    for row, col in zip(rows[:5].tolist(), cols[:5].tolist()):
        print(f"    Mismatch at row {row}, col {col}: expected {float(ref[row, col]):f}, "
              f"got {float(got[row, col]):f}")
    print(f"\nTEST FAILED: {wrong} of {total} elements are wrong")
    nans = int(cp.isnan(got).sum())
    if nans == total:
        print("    C is all NaN, which is how main.py fills it: the kernel never stored to C. "
              "Check that tiled_matmul launches it over the whole grid")
    elif nans > 0:
        print(f"    {nans} elements are NaN: either they were never stored (is the grid big "
              "enough?), or a load read past the edge of A or B, where main.py puts NaN "
              "(use padding_mode=ct.PaddingMode.ZERO)")
if not guard_ok:
    print("\nTEST FAILED: the kernel wrote outside C")
if wrong > 0 or not guard_ok:
    print()
    sys.exit(1)
print(f"TEST PASSED {total}")
print(f"    Max relative error: {max_err:.2e}")

# Time the kernel ------------------------------------------------------------


def median_ms(f):
    r = cupyx.profiler.benchmark(f, n_warmup=10, n_repeat=50)
    return float(cp.median(cp.asarray(r.gpu_times[0]))) * 1e3


ms = median_ms(lambda: run_student(A, B, C))
print(f"\nKernel time: {ms:.4f} ms (median), {flops / ms / 1e6:.1f} GFLOP/s")
print()
