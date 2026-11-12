#include <stdio.h>

#define BLOCK_SIZE 512

// Basic histogram: every thread adds its elements directly to the global histogram bins
__global__ void basic_histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    // BASIC HISTOGRAM IMPLEMENTATION
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elements) {
        atomicAdd(&bins[input[i]], 1u);
    }

    /*************************************************************************/
}

// Privatized histogram: each block counts into its own histogram in shared memory,
// then adds it to the global histogram bins
__global__ void privatized_histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    // PRIVATIZED HISTOGRAM IMPLEMENTATION
    extern __shared__ unsigned int hist_s[];

    // Phase 1: initialize every bin in this block's private histogram
    for (unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        hist_s[bin] = 0;
    }

    __syncthreads();

    // Phase 2: count this block's input elements into shared memory
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elements) {
        atomicAdd(&hist_s[input[i]], 1u);
    }

    __syncthreads();

    // Phase 3: merge every private bin into the global histogram
    for (unsigned int bin = threadIdx.x; bin < num_bins; bin += blockDim.x) {
        atomicAdd(&bins[bin], hist_s[bin]);
    }

    /*************************************************************************/
}

// Launches basic_histo_kernel
void basicHistogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    /*************************************************************************/
    //INSERT CODE HERE
    unsigned int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    basic_histo_kernel<<<num_blocks, BLOCK_SIZE>>>(input, bins, num_elements, num_bins);

    /*************************************************************************/

}

// Launches privatized_histo_kernel
void privatizedHistogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    /*************************************************************************/
    //INSERT CODE HERE
    unsigned int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    privatized_histo_kernel<<<num_blocks, BLOCK_SIZE, num_bins * sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);

    /*************************************************************************/

}
