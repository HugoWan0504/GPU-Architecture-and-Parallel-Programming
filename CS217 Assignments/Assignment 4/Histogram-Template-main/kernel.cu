#include <stdio.h>

#define BLOCK_SIZE 512

// Basic histogram: every thread adds its elements directly to the global histogram bins
__global__ void basic_histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    // BASIC HISTOGRAM IMPLEMENTATION


    /*************************************************************************/
}

// Privatized histogram: each block counts into its own histogram in shared memory,
// then adds it to the global histogram bins
__global__ void privatized_histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    // PRIVATIZED HISTOGRAM IMPLEMENTATION


    /*************************************************************************/
}

// Launches basic_histo_kernel
void basicHistogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    /*************************************************************************/
    //INSERT CODE HERE


    /*************************************************************************/

}

// Launches privatized_histo_kernel
void privatizedHistogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    /*************************************************************************/
    //INSERT CODE HERE


    /*************************************************************************/

}
