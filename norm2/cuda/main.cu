#include <chrono>
#include <sys/time.h>
#include "common.cuh"
#include <cmath>
#include <cassert>
#include <cooperative_groups.h>


__device__ float mul(const float a, const float b)
{
    return a * b;
}

__global__ void norm2(const int n, const float *x, float *z)
{
    int warpSize = 32;
    __shared__ float sdata[32];        // share memory is shared by different threads within 1 block 

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = blockDim.x * bid + tid;
    float val = 0;

    if (id < n)
    {
        val = mul(x[id], x[id]);
    }

    unsigned mask = 0xFFFFFFFFU;  //used in shuffle, to determine which threads will be used for shuffle operation
    int lane = threadIdx.x % warpSize;   // there are 32 threads within a warp, so each f represents 1111 such 4 threads
    int warpID = threadIdx.x / warpSize;

    //
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);    //reduce within each warp

    if (lane == 0) 
        sdata[warpID] = val;
    __syncthreads();               //used to synchronize different threads to write data in shared memory

    if (warpID == 0)               //now all needed data has already be written in sdata, and we just need to use it in warp 0
    {
        assert (tid == lane);
        val = (lane < blockDim.x / warpSize) ? sdata[lane] : 0;   // in case blockDim != 1024(this case sdata doesnt need 32 value)
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (tid == 0)
        {
            atomicAdd(z, val);                  //add data from different block, as all values needed have already been reduced
        }                                       //in tid 0 in each block, so we just sum them up, and get the square root outside
    }                                           //the kernal function

    
    //todo: nvidia collective communication grid-level 
}

double my_clock()
{
    /* struct timeval { long        tv_sec;
                long        tv_usec;        };

    struct timezone { int   tz_minuteswest;
                 int        tz_dsttime;      };     */

    struct timeval tp;
    struct timezone tzp;

    gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char *argv[])
{
    // if (argc < 4)
    if (argc < 3)
    {
        // fprintf(stderr, "Usage: %s <problem_size> <run_times> <blockSize>\n", argv[0]); // 1D grid and block
        fprintf(stderr, "Usage: %s <problem_size> <run_times>\n", argv[0]); 
        return 1;
    }

    const size_t n = std::atoi(argv[1]);
    const int runs = std::atoi(argv[2]);
    // const int blockSize = std::atoi(argv[3]);
    const size_t byteSize = n * sizeof(float);
    const size_t bytez = sizeof(float);


    // if (runs <= 0 || n <= 0 || blockSize <= 0 || blockSize > 1024)
    if (runs <= 0 || n <= 0 )
    {
        // fprintf(stderr, "Please enter positive integers for the problem size, run times and blockSize, blockSize should be\
        // lower than 1024.\n");
        fprintf(stderr, "Please enter positive integers for the problem size, run times.\n");

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
    errorX = cudaMalloc((float **)&xGPU, byteSize);     //allocate global memory in GPU
    errorZ = cudaMalloc((float **)&zGPU, bytez);
    if (errorX == cudaSuccess && errorZ == cudaSuccess)
    {
        cudaMemset(xGPU, 0, byteSize);
        // cudaMemset(zGPU, 0, bytez);
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
        x[i] = static_cast<float>(1+i);
    }

    cudaMemcpy(xGPU, x, byteSize, cudaMemcpyHostToDevice);

    // dim3 block(blockSize);
    // dim3 grid((n + block.x - 1) / block.x);

    // test cudaOccupancyMaxPotentialBlockSize
    int minGridSize;
    int bestBlockSize;

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &bestBlockSize,
        norm2,
        0,
        0
    );

    int grid = (n + bestBlockSize - 1) / bestBlockSize;

    double start_time = my_clock();

    for (int i = 0; i < runs; ++i)
    {
        cudaMemset(zGPU, 0, bytez);
        // norm2<<<grid, block>>>(n, xGPU, zGPU);
        norm2<<<grid, bestBlockSize>>>(n, xGPU, zGPU);
        cudaDeviceSynchronize();
        cudaMemcpy(&z, zGPU, bytez, cudaMemcpyDeviceToHost);
        z = sqrtf(z);
    }

    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;

    printf("\nAverage execution time over %d runs: \t%f s.\n", runs, averageDuration);
    printf("result:\t%.9f.\n", z);

    free(x);
    cudaFree(xGPU);
    cudaFree(zGPU);
    return 0;
}
