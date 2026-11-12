/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512


__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION
    __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    unsigned int i = start + tid;
    unsigned int j = start + BLOCK_SIZE + tid;

    partialSum[tid] = (i < size) ? in[i] : 0.0f;
    partialSum[tid + BLOCK_SIZE] = (j < size) ? in[j] : 0.0f;

    __syncthreads();

    // contiguous active threads
    // for (unsigned int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2) {
    //   unsigned int index = 2 * stride * tid;

    //   if (index + stride < 2 * BLOCK_SIZE) {
    //       partialSum[index] += partialSum[index + stride];
    //   }

    //   __syncthreads();
    // }

    // interleaved active threads
    for (unsigned int stride = 1; stride < 2 * BLOCK_SIZE; stride *= 2) {
      if (tid % stride == 0) {
          unsigned int index = 2 * tid;
          partialSum[index] += partialSum[index + stride];
      }

      __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = partialSum[0];
    }

}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
    __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    unsigned int i = start + tid;
    unsigned int j = start + BLOCK_SIZE + tid;

    partialSum[tid] = (i < size) ? in[i] : 0.0f;
    partialSum[tid + BLOCK_SIZE] = (j < size) ? in[j] : 0.0f;

    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = partialSum[0];
    }


}