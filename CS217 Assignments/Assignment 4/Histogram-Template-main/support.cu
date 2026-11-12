#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(unsigned int **vec_h, unsigned int size, unsigned int num_bins)
{
    *vec_h = (unsigned int*)malloc(size*sizeof(unsigned int));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }
    srand(217);
    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%num_bins);
    }

}

// Checks every bin against a histogram computed on the CPU, and the total count against num_elements
void verify(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

  const unsigned maxShown = 5;
  unsigned wrong = 0, zeros = 0;
  unsigned long long gpuTotal = 0;

  // Compute reference bins
  unsigned int* bins_ref = (unsigned int*) calloc(num_bins, sizeof(unsigned int));
  if(bins_ref == NULL) FATAL("Unable to allocate host");
  for(unsigned int i = 0; i < num_elements; ++i) {
      ++bins_ref[input[i]];
  }

  printf("\n");

  // Compare to reference bins
  for(unsigned int binIdx = 0; binIdx < num_bins; ++binIdx) {
      gpuTotal += bins[binIdx];
      if(bins[binIdx] == 0) ++zeros;
      if(bins[binIdx] != bins_ref[binIdx]) {
        if(wrong < maxShown) {
          printf("    Mismatch in bin %u: expected %u, got %u\n", binIdx, bins_ref[binIdx], bins[binIdx]);
        }
        ++wrong;
      }
  }

  printf("    Total count: %u (CPU), %llu (GPU)\n", num_elements, gpuTotal);
  free(bins_ref);

  if(wrong > 0) {
    printf("TEST FAILED: %u of %u bins are wrong\n", wrong, num_bins);
    if(zeros == num_bins) {
      printf("    All bins are zero: check that the kernel ran and added its counts to bins, and (privatized)"
             " that threads call __syncthreads() before reading the private histogram to add it to bins\n");
    } else if(gpuTotal > num_elements) {
      printf("    The GPU counted more elements than the input has: check that each element is counted once,"
             " and (privatized) that each block's private histogram is set to zero before use\n");
    } else if(gpuTotal < num_elements) {
      printf("    The GPU counted fewer elements than the input has: check that every element is counted"
             " (bounds and strides), that updates use atomics, that all counts are added to bins, and"
             " (privatized) that __syncthreads() separates zeroing, counting and adding to bins\n");
    }
    printf("\n");
    exit(1);
  }
  printf("TEST PASSED\n\n");

}

// Fills the shared memory of every SM with nonzero values. Shared memory is not cleared between kernels,
// so a histogram kernel that forgets to zero its private histogram starts from these values and fails.
__global__ void dirtySharedKernel(unsigned int words)
{
    extern __shared__ unsigned int junk[];
    for(unsigned int i = threadIdx.x; i < words; i += blockDim.x) {
        junk[i] = 0xDEADBEEF + i;
    }
}

void dirtySharedMemory() {
#if CUDART_VERSION >= 5000
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t bytes = prop.sharedMemPerBlock;
    dirtySharedKernel<<<prop.multiProcessorCount * 8, 256, bytes>>>(bytes / sizeof(unsigned int));
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
#endif
    // Skipped under GPGPU-Sim (CUDA 4.0): the extra kernel would add to the simulation time and to the
    // statistics the report asks for
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

