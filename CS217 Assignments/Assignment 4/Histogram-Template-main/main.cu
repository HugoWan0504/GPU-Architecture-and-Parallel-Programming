#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "support.h"
#include "kernel.cu"

// make builds this file twice: histogram-basic, and histogram-privatized with -DPRIVATIZED
#ifdef PRIVATIZED
#define PROGRAM "./histogram-privatized"
#else
#define PROGRAM "./histogram-basic"
#endif

int main(int argc, char* argv[])
{
    Timer timer;

    // Load kernels when the program starts rather than at their first launch, so the kernel time below
    // doesn't include loading the kernel. Must be set before the first CUDA call
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int *in_h;
    unsigned int* bins_h;
    unsigned int *in_d;
    unsigned int* bins_d;
    unsigned int num_elements, num_bins;

    if(argc == 1) {
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    }
    if(argc > 3 || (int)num_elements < 1 || (int)num_bins < 1) {
        printf("\n    Invalid input parameters!"
           "\n    Usage: " PROGRAM "            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: " PROGRAM " <m>        # Input: m, Bins: 4,096"
           "\n    Usage: " PROGRAM " <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }
    initVector(&in_h, num_elements, num_bins);
    bins_h = (unsigned int*) malloc(num_bins*sizeof(unsigned int));
    if(bins_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int)));
    CHECK(cudaMalloc((void**)&bins_d, num_bins * sizeof(unsigned int)));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMemcpy(in_d, in_h, num_elements * sizeof(unsigned int),
        cudaMemcpyHostToDevice));

    CHECK(cudaMemset(bins_d, 0, num_bins * sizeof(unsigned int)));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Leave nonzero values in shared memory, so a private histogram that is not zeroed gives wrong counts
    dirtySharedMemory();

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    // CUDA events time the kernel on the GPU; the host timer above also counts launch overhead
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_ms;
    CHECK(cudaEventCreate(&kernel_start));
    CHECK(cudaEventCreate(&kernel_stop));
    CHECK(cudaEventRecord(kernel_start));

    #ifndef PRIVATIZED
    printf("Running Basic Histogram..."); fflush(stdout);
    basicHistogram(in_d, bins_d, num_elements, num_bins);
    #else
    printf("Running Privatized Histogram..."); fflush(stdout);
    privatizedHistogram(in_d, bins_d, num_elements, num_bins);
    #endif
    CHECK(cudaGetLastError());       // bad launch configuration
    CHECK(cudaEventRecord(kernel_stop));
    CHECK(cudaDeviceSynchronize());  // errors inside the kernel

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
#if CUDART_VERSION >= 5000
    printf("    Kernel time: %f ms\n", kernel_ms);
#else
    // GPGPU-Sim (CUDA 4.0) doesn't simulate time: its events report a meaningless value
    printf("    Kernel time: not measured in GPGPU-Sim; use gpu_sim_cycle in the simulator output\n");
#endif
    cudaEventDestroy(kernel_start); cudaEventDestroy(kernel_stop);

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    CHECK(cudaMemcpy(bins_h, bins_d, num_bins * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    cudaFree(in_d); cudaFree(bins_d);
    free(in_h); free(bins_h);

    return 0;
}

