#include <chrono>
#include <sys/time.h>
#include "common.cuh"
#include <cmath>
#include <cassert>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define DEBUG

#ifdef DEBUG
#define CUDA_CALL(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
#define cudaCheckError(ans) ans
#endif


__device__ float mul(const float a, const float b)
{
    return a * b;
}


__global__ void norm2(const int n, const float *x, float *z) {
    // cg::grid_group grid = cg::this_grid();
    int warpSize = 32;
    __shared__ float sdata[32];

    int tid = threadIdx.x;
    int id = blockDim.x * blockIdx.x + tid;
    float val = 0;

    if (id < n) {
        val = mul(x[id], x[id]);
    }

    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0) {
        sdata[warpID] = val;
    }
    __syncthreads();

    if (warpID == 0) {
        val = (lane < blockDim.x / warpSize) ? sdata[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (tid == 0) {
            atomicAdd(z, val);
        }
    }

    // grid.sync();

    // if (grid.thread_rank() == 0) {
        // *z = sqrtf(*z);
    // }
}


double my_clock()
{
    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);

}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <problem_size> <run_times>\n", argv[0]);
        return 1;
    }

    const size_t n = std::atoi(argv[1]);
    const int runs = std::atoi(argv[2]);
    const size_t byteSize = n * sizeof(float);
    const size_t bytez = sizeof(float);

    if (runs <= 0 || n <= 0) {
        fprintf(stderr, "Please enter positive integers for the problem size, run times.\n");
        return 1;
    }

    setGPU();

    // float *x = (float *)malloc(byteSize);
    // float *z = (float *)malloc(bytez);

    // if (x == NULL) {
    //     printf("Failed to allocate memory in CPU.\n");
    //     exit(-1);
    // }

    float *xGPU, *zGPU;
    CUDA_CALL(cudaMallocManaged((float **)&xGPU, byteSize));
    CUDA_CALL(cudaMallocManaged((float **)&zGPU, bytez));

    for (size_t i = 0; i < n; ++i) {
        xGPU[i] = static_cast<float>(1 + i);
    }

    // CUDA_CALL(cudaMemcpy(xGPU, x, byteSize, cudaMemcpyHostToDevice));

    int minGridSize;
    int bestBlockSize;

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &bestBlockSize,
        norm2,
        0,
        0
    );
    // bestBlockSize = 128;

    int grid = (n + bestBlockSize - 1) / bestBlockSize;

    double start_time = my_clock();

    for (int i = 0; i < runs; ++i) {

        // void* kernelArgs[] = { (void*)&n, (void*)&xGPU, (void*)&zGPU };
        norm2<<<grid, bestBlockSize>>>(n, xGPU, zGPU);
        // CUDA_CALL(cudaLaunchCooperativeKernel((void*)norm2, grid, bestBlockSize, kernelArgs));

        cudaDeviceSynchronize();
        *zGPU = sqrtf(*zGPU);
        // CUDA_CALL(cudaMemcpy(&z, zGPU, bytez, cudaMemcpyDeviceToHost));
    }


    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;

    printf("\nAverage execution time over %d runs: \t%f s.\n", runs, averageDuration);
    printf("Result:\t%.9f.\n", *zGPU);

    // free(x);
    CUDA_CALL(cudaFree(xGPU));
    CUDA_CALL(cudaFree(zGPU));
    return 0;
}
