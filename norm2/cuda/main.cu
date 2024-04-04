#include <chrono>
#include <sys/time.h>
#include "common.cuh"
#include <cmath>

__device__ float mul(const float a, const float b)
{
    return a * b;
}

__global__ void norm2(const int n, const float *x, float *z)
{
    int warpSize = 32;
    __shared__ float sdata[32];

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = blockDim.x * bid + tid;
    float val = 0;

    if (id < n)
    {
        val = mul(x[id], x[id]);
    }

    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    if (lane == 0)
        sdata[warpID] = val;
    __syncthreads();

    if (warpID == 0)
    {
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (tid == 0)
        {
            atomicAdd(z, val);
        }
    }
}

double my_clock()
{
    /* struct timeval { long        tv_sec;
                long        tv_usec;        };

    struct timezone { int   tz_minuteswest;
                 int        tz_dsttime;      };     */

    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <problem_size> <run_times> <blockSize>\n", argv[0]); // 1D grid and block
        return 1;
    }

    const size_t n = std::atoi(argv[1]);
    const int runs = std::atoi(argv[2]);
    const int blockSize = std::atoi(argv[3]);
    const size_t byteSize = n * sizeof(float);
    const size_t bytez = sizeof(float);

    if (runs <= 0 || n <= 0 || blockSize <= 0 || blockSize > 1024)
    {
        fprintf(stderr, "Please enter positive integers for the problem size, run times and blockSize, blockSize should be\
        lower than 1024.\n");
        return 1;
    }

    setGPU();

    float *x = (float *)malloc(byteSize);
    float z = 0;

    if (x != NULL)
    {
        memset(x, 0, byteSize);
    }
    else
    {
        printf("fail to allocate memory in CPU.\n");
        free(x);
        exit(-1);
    }

    cudaError_t errorX, errorZ;
    float *xGPU, *zGPU;
    errorX = cudaMalloc((float **)&xGPU, byteSize);
    errorZ = cudaMalloc((float **)&zGPU, bytez);
    if (errorX == cudaSuccess && errorZ == cudaSuccess)
    {
        cudaMemset(xGPU, 0, byteSize);
        cudaMemset(zGPU, 0, bytez);
    }
    else
    {
        printf("fail to allocate memory in GPU.\n");
        if (errorX == cudaSuccess)
            cudaFree(xGPU);
        if (errorZ == cudaSuccess)
            cudaFree(zGPU);
        free(x);
        exit(-1);
    }

    for (size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<float>(i + 1);
    }

    cudaMemcpy(xGPU, x, byteSize, cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((n + block.x - 1) / block.x);

    double start_time = my_clock();

    for (int i = 0; i < runs; ++i)
    {
        norm2<<<grid, block>>>(n, xGPU, zGPU);
        cudaDeviceSynchronize();
        cudaMemcpy(&z, zGPU, bytez, cudaMemcpyDeviceToHost);
        z = sqrtf(z);
        cudaFree(zGPU);
        cudaMalloc((float **)&zGPU, bytez);
        cudaMemset(zGPU, 0, bytez);
    }

    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;

    printf("Average execution time over %d runs: \t%.9f ms.\n", runs, averageDuration);
    printf("result:\t%.9f.\n", z);

    free(x);
    cudaFree(xGPU);
    cudaFree(zGPU);
    return 0;
}