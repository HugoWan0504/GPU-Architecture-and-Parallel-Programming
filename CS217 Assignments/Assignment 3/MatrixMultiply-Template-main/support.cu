#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(float *A, float *B, float *C, unsigned int m, unsigned int k,
  unsigned int n) {

  const size_t maxShown = 5;
  const float tolerance = 1e-5;
  size_t total = (size_t)m*n, wrong = 0, zeros = 0;

  for(size_t row = 0; row < m; ++row) {
    for(size_t col = 0; col < n; ++col) {
      float sum = 0;
      for(size_t i = 0; i < k; ++i) {
        sum += A[row*k + i]*B[i*n + col];
      }
      float got = C[row*n + col];
      if (got == 0.0f) ++zeros;
      // Relative error, but absolute for tiny sums; written so NaN counts as wrong
      float scale = sum < 0 ? -sum : sum;
      if (scale < 1.0f) scale = 1.0f;
      float diff = got - sum;
      if (!(diff <= tolerance*scale && diff >= -tolerance*scale)) {
        if (wrong < maxShown) {
          printf("    Mismatch at row %zu, col %zu: expected %f, got %f\n",
            row, col, sum, got);
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
  printf("TEST PASSED %zu\n\n", total);

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

