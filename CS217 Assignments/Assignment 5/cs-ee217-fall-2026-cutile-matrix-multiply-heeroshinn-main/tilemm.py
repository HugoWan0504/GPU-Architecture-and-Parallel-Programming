"""Tiled matrix multiplication in cuTile: C = A x B

main.py calls tiled_matmul().
See the cuTile documentation: https://docs.nvidia.com/cuda/cutile-python/

CUDA                                    cuTile
----                                    ------
blockIdx.y, blockIdx.x                  ct.bid(0), ct.bid(1)
threadIdx, one element per thread       none: the block works on whole tiles
__shared__ tiles, __syncthreads()       none: ct.load() a tile, the compiler stages it
bounds checks, zero padding             ct.load(..., padding_mode=ct.PaddingMode.ZERO);
                                        ct.store() drops elements past the edge
sum += As[ty][i] * Bs[i][tx]            acc = ct.mma(a_tile, b_tile, acc)
kernel<<<grid, block>>>(...)            ct.launch(stream, grid, kernel, (args...))
"""
import cupy as cp
import cuda.tile as ct


@ct.kernel
def matmul_kernel(A, B, C,
                  BLOCK_M: ct.Constant[int], BLOCK_N: ct.Constant[int], BLOCK_K: ct.Constant[int]):

    #####################################################################
    #
    # Compute one BLOCK_M x BLOCK_N tile of C = A x B
    #   where A is a (M x K) matrix
    #   where B is a (K x N) matrix
    #   where C is a (M x N) matrix
    #
    # M, N and K need not be multiples of the block sizes.
    #
    #####################################################################

    # Which tile of C this block computes
    bm = ct.bid(0)   # tile row of C: rows bm*BLOCK_M ... bm*BLOCK_M + BLOCK_M-1
    bn = ct.bid(1)   # tile column of C

    # This block's BLOCK_M x BLOCK_N tile of C, accumulated in fp32
    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

    # Loop over the tiles along K, like the loop over phases in your CUDA kernel.
    # ct.num_tiles() counts tiles of the given shape along an axis, including a partial one.
    num_k_tiles = ct.num_tiles(A, axis=1, shape=(BLOCK_M, BLOCK_K))
    for kt in range(num_k_tiles):
        a_tile = ct.load(A, index=(bm, kt), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(kt, bn), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, index=(bm, bn), tile=acc)


def tiled_matmul(A, B, C, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    """Launch matmul_kernel to compute C = A x B.

    A (M x K), B (K x N) and C (M x N) are float32 CuPy arrays on the GPU.
    Pass the block sizes through to the kernel; don't hard-code them.
    """
    M, _ = A.shape
    _, N = B.shape

    # One block per output tile, including partial tiles along the edges.
    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)

    # Invoke the cuTile kernel -----------------------------------------------
    ct.launch(cp.cuda.get_current_stream(), grid, matmul_kernel,
              (A, B, C, BLOCK_M, BLOCK_N, BLOCK_K))
