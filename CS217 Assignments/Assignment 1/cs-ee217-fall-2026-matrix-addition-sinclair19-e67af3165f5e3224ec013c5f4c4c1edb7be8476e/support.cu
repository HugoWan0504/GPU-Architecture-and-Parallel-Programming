#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, unsigned int rows, unsigned int cols) {

  const size_t maxShown = 5;
  size_t total = (size_t)rows*cols, wrong = 0, zeros = 0;

  for(size_t row = 0; row < rows; ++row) {
    for(size_t col = 0; col < cols; ++col) {
      size_t i = row*cols + col;
      float expected = A[i] + B[i];
      if (C[i] == 0.0f) ++zeros;
      // A single float add is exact on CPU and GPU; != also rejects NaN
      if (C[i] != expected) {
        if (wrong < maxShown) {
          printf("    Mismatch at row %zu, col %zu: expected %f (= %f + %f), got %f\n",
            row, col, expected, A[i], B[i], C[i]);
        }
        ++wrong;
      }
    }
  }

  if (wrong > 0) {
    printf("\nTEST FAILED: %zu of %zu elements are wrong\n", wrong, total);
    if (zeros == total) {
      printf("    C is all zeros: check that the kernel ran and C was copied back to the host\n");
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

