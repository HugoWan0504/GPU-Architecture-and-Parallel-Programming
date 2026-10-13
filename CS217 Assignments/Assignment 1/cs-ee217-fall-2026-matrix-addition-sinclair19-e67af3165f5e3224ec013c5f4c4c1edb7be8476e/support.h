#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void verify(float *A, float *B, float *C, unsigned int rows, unsigned int cols);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        printf("\n"); fflush(stdout); /* finish any partial stdout line first */\
        fprintf(stderr, "[%s:%d] " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

// Wrap CUDA API calls: CHECK(cudaMalloc(...));
#define CHECK(call) \
    do {\
        cudaError_t err_ = (call);\
        if (err_ != cudaSuccess) FATAL("CUDA error: %s", cudaGetErrorString(err_));\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
