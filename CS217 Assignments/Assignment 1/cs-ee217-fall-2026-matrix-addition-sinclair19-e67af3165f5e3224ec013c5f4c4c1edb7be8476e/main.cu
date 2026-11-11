#include <stdio.h>
#include <stdlib.h>
#include "support.h"
#include "kernel.cu"

int main (int argc, char *argv[])
{

    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t mat_sz;
    int rows, cols;

    if (argc == 1) {
        rows = cols = 1000;
    } else if (argc == 2) {
        rows = cols = atoi(argv[1]);
    } else if (argc == 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    } else {
        rows = cols = 0;
    }

    if (rows <= 0 || cols <= 0) {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./mat-add                # All matrices are 1000 x 1000"
      "\n    Usage: ./mat-add <m>            # All matrices are m x m"
      "\n    Usage: ./mat-add <m> <n>        # All matrices are m x n (m rows, n columns)"
      "\n");
        exit(0);
    }

    mat_sz = (size_t)rows*cols;

    A_h = (float*) malloc( sizeof(float)*mat_sz );
    for (size_t i=0; i < mat_sz; i++) { A_h[i] = (rand()%100)/100.00; }

    B_h = (float*) malloc( sizeof(float)*mat_sz );
    for (size_t i=0; i < mat_sz; i++) { B_h[i] = (rand()%100)/100.00; }

    C_h = (float*) malloc( sizeof(float)*mat_sz );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    A: %d x %d\n    B: %d x %d\n    C: %d x %d\n", rows, cols,
        rows, cols, rows, cols);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    // Wrap CUDA API calls in CHECK(...) here and below, e.g. CHECK(cudaMalloc(...));
    CHECK(cudaMalloc((void**)&A_d, sizeof(float) * mat_sz));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float) * mat_sz));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float) * mat_sz));

    /*************************************************************************/
    CHECK(cudaDeviceSynchronize());
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);
	
    /*************************************************************************/
    //INSERT CODE HERE
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float) * mat_sz, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float) * mat_sz, cudaMemcpyHostToDevice));

    /*************************************************************************/
    CHECK(cudaDeviceSynchronize());
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel using standard mat-add interface -------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    basicMatAdd(rows, cols, A_d, B_d, C_d);

    CHECK(cudaDeviceSynchronize());
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables to host ------------------------------------------
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    /*************************************************************************/
    //INSERT CODE HERE
    CHECK(cudaMemcpy(C_h, C_d, sizeof(float) * mat_sz, cudaMemcpyDeviceToHost));

    /*************************************************************************/
    CHECK(cudaDeviceSynchronize());
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...\n"); fflush(stdout);

    verify(A_h, B_h, C_h, rows, cols);


    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    /*************************************************************************/
    //INSERT CODE HERE
    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
    
    /*************************************************************************/
    return 0;
}

