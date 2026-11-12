#include <stdio.h>
#include "support.h"

// Block dimensions: make TILE=<n> sets both, make TILE_X=<x> TILE_Y=<y> sets each one
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif
#ifndef TILE_X
#define TILE_X TILE_SIZE
#endif
#ifndef TILE_Y
#define TILE_Y TILE_SIZE
#endif

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    // Choose a tile depth that supports square and rectangular blocks.
    const int TILE_K = (TILE_X < TILE_Y) ? TILE_X : TILE_Y;

    __shared__ float tileA[TILE_Y][TILE_K];
    __shared__ float tileB[TILE_K][TILE_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    for (int start = 0; start < k; start += TILE_K) {

        // Load an A tile: TILE_Y rows by TILE_K columns.
        if (tx < TILE_K) {
            const int aCol = start + tx;

            tileA[ty][tx] = (row < m && aCol < k)
                        ? A[row * k + aCol]
                        : 0.0f;
        }

        // Load a B tile: TILE_K rows by TILE_X columns.
        if (ty < TILE_K) {
            const int bRow = start + ty;

            tileB[ty][tx] = (bRow < k && col < n)
                        ? B[bRow * n + col]
                        : 0.0f;
        }

        // Wait until both shared-memory tiles are ready.
        __syncthreads();

        for (int p = 0; p < TILE_K; p++) {
            sum += tileA[ty][p] * tileB[p][tx];
        }

        // Wait until everyone finishes reading before replacing the tiles.
        __syncthreads();
    }

    // Only threads within the output matrix write a result.
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
        
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    // Threads per block in x and y; use these, not hard-coded numbers
    const unsigned int BLOCK_X = TILE_X;
    const unsigned int BLOCK_Y = TILE_Y;
	
    /*************************************************************************/
    //INSERT CODE HERE
    dim3 block(BLOCK_X, BLOCK_Y);

    dim3 grid(
        (n + BLOCK_X - 1) / BLOCK_X,
        (m + BLOCK_Y - 1) / BLOCK_Y
    );

    /*************************************************************************/

    // Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE
    mysgemm<<<grid, block>>>(m, n, k, A, B, C);

    /*************************************************************************/

    // Catch invalid launch configurations
    CHECK(cudaGetLastError());
}


