/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "support.h"

void initVector(float **vec_h, unsigned size)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }
    srand(217);
    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%100)/100.00;
    }

}


// Checks each block's partial sum against the CPU, then the total.
// Block b should hold the sum of input[b*elements_per_block] up to the next block's start (or the end).
void verify(float* input, unsigned num_elements, float* partial, unsigned num_partial,
    unsigned elements_per_block) {

  // Float sums of up to elements_per_block values in [0, 1) differ from the exact sum by
  // far less than this; a missed or duplicated element is off by at least 0.01
  const double relativeTolerance = 1e-5, absoluteTolerance = 1e-3;
  const unsigned maxShown = 5;
  unsigned wrong = 0, zeros = 0;
  double cpuSum = 0.0, gpuSum = 0.0;

  printf("\n");

  for(unsigned b = 0; b < num_partial; ++b) {
    double expected = 0.0;
    for(unsigned i = b*elements_per_block; i < num_elements && i < (b+1)*elements_per_block; ++i) {
      expected += input[i];
    }
    cpuSum += expected;
    gpuSum += partial[b];
    if (partial[b] == 0.0f) ++zeros;
    // Written as !(... <= ...) so that NaN counts as wrong
    if (!(fabs(partial[b] - expected) <= relativeTolerance*fabs(expected) + absoluteTolerance)) {
      if (wrong < maxShown) {
        printf("    Mismatch in block %u (elements %u to %u): expected %f, got %f\n", b,
          b*elements_per_block, (b+1)*elements_per_block < num_elements ?
          (b+1)*elements_per_block - 1 : num_elements - 1, expected, partial[b]);
      }
      ++wrong;
    }
  }

  printf(" Sum: %f (CPU), %f (GPU)\n", cpuSum, gpuSum);
  if (wrong > 0) {
    printf("TEST FAILED: %u of %u block partial sums are wrong\n", wrong, num_partial);
    if (zeros == num_partial) {
      printf("    All partial sums are zero: check that the kernel ran and wrote out[blockIdx.x]\n");
    }
    printf("\n");
    exit(1);
  }
  printf("TEST PASSED\n\n");

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

