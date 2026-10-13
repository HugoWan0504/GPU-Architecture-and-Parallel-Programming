/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    dim3 dim_grid, dim_block;

    // Allocate and initialize host memory
    if(argc == 1) {
        in_elements = 1000000;
    } else if(argc == 2) {
        in_elements = atoi(argv[1]);
    }
    if(argc > 2 || (int)in_elements < 1) {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./reduction          # Input of size 1,000,000 is used"
           "\n    Usage: ./reduction <m>      # Input of size m is used"
           "\n");
        exit(0);
    }
    initVector(&in_h, in_elements);

    out_elements = in_elements / (BLOCK_SIZE<<1);
    if(in_elements % (BLOCK_SIZE<<1)) out_elements++;

    out_h = (float*)malloc(out_elements * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n", in_elements);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMalloc((void**)&in_d, in_elements * sizeof(float)));
    CHECK(cudaMalloc((void**)&out_d, out_elements * sizeof(float)));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMemcpy(in_d, in_h, in_elements * sizeof(float),
        cudaMemcpyHostToDevice));

    CHECK(cudaMemset(out_d, 0, out_elements * sizeof(float)));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x = out_elements; dim_grid.y = dim_grid.z = 1;
    #ifndef OPTIMIZED
    printf("Running Naive Reduction..."); fflush(stdout);
    naiveReduction<<<dim_grid, dim_block>>>(out_d, in_d, in_elements);
    #else
    printf("Running Optimized Reduction..."); fflush(stdout);
    optimizedReduction<<<dim_grid, dim_block>>>(out_d, in_d, in_elements);
    #endif
    CHECK(cudaGetLastError());       // bad launch configuration
    CHECK(cudaDeviceSynchronize());  // errors inside the kernel

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMemcpy(out_h, out_d, out_elements * sizeof(float),
        cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    /* Check each block's partial sum, then the total */
    verify(in_h, in_elements, out_h, out_elements, BLOCK_SIZE<<1);

    // Free memory ------------------------------------------------------------

    cudaFree(in_d); cudaFree(out_d);
    free(in_h); free(out_h);

    return 0;
}

