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

__global__ void matAdd(int rows, int cols, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (rows x cols) matrix
     *   where B is a (rows x cols) matrix
     *   where C is a (rows x cols) matrix
     *
     * All matrices are stored in row-major order.
     *
     ********************************************************************/

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
        
    /*************************************************************************/

}

void basicMatAdd(int rows, int cols, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------

    // Threads per block in x and y; use these, not hard-coded numbers
    const unsigned int BLOCK_X = TILE_X;
    const unsigned int BLOCK_Y = TILE_Y;
	
    /*************************************************************************/
    //INSERT CODE HERE

    /*************************************************************************/
	
	// Invoke CUDA kernel -----------------------------------------------------

    /*************************************************************************/
    //INSERT CODE HERE

    /*************************************************************************/

    // Catch invalid launch configurations
    CHECK(cudaGetLastError());

}

