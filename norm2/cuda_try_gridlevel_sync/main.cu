#include <chrono>
#include <sys/time.h>
#include "common.cuh"
#include <cmath>
#include <cassert>
#include <cooperative_groups.h>
using namespace cooperative_groups;

__device__ float mul(const float a, const float b)
{
    return a * b;
}

// __global__ void norm2(const int n, const float *x, float *z)
// {
//     int warpSize = 32;
//     __shared__ float sdata[32]; // share memory is shared by different threads within 1 block

//     int bid = blockIdx.x;
//     int tid = threadIdx.x;
//     int id = blockDim.x * bid + tid;
//     float val = 0;

//     if (id < n)
//     {
//         val = mul(x[id], x[id]);
//     }

//     unsigned mask = 0xFFFFFFFFU;       // used in shuffle, to determine which threads will be used for shuffle operation
//     int lane = threadIdx.x % warpSize; // there are 32 threads within a warp, so each f represents 1111 such 4 threads
//     int warpID = threadIdx.x / warpSize;

//     //
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1)
//         val += __shfl_down_sync(mask, val, offset); // reduce within each warp

//     if (lane == 0)
//         sdata[warpID] = val;
//     __syncthreads(); // used to synchronize different threads to write data in shared memory

//     if (warpID == 0) // now all needed data has already be written in sdata, and we just need to use it in warp 0
//     {
//         assert(tid == lane);
//         val = (lane < blockDim.x / warpSize) ? sdata[lane] : 0; // in case blockDim != 1024(this case sdata doesnt need 32 value)
//         for (int offset = warpSize / 2; offset > 0; offset >>= 1)
//             val += __shfl_down_sync(mask, val, offset);
//         if (tid == 0)
//         {
//             atomicAdd(z, val); // add data from different block, as all values needed have already been reduced
//         } // in tid 0 in each block, so we just sum them up, and get the square root outside
//     } // the kernal function

//     // todo: nvidia collective communication grid-level
// }

// __device__ float thread_sum(const int n, const float *x) {
//     int id = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0.0f;

//     if (id < n) {
//         sum = x[id] * x[id];
//     }

//     return sum;
// }

// __device__ float reduce_sum(thread_group g, float val, float *sdata) {
//     // int lane = threadIdx.x % warpSize;
//     // int warpID = threadIdx.x / warpSize;
//     int lane = g.thread_rank();

//     // unsigned mask = 0xFFFFFFFFU;
//     // for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//     //     val += __shfl_down_sync(mask, val, offset);
//     // }

//     // if (lane == 0) {
//     //     sdata[warpID] = val;
//     // }
//     // __syncthreads();

//     // val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
//     // if (warpID == 0) {
//     //     for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//     //         val += __shfl_down_sync(mask, val, offset);
//     //     }
//     // }

//     unsigned mask = 0xFFFFFFFFU;
//     for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//         val += __shfl_down_sync(mask, val, offset);
//     }

//     if (lane == 0) {
//         sdata[warpID] = val;
//     }
//     block.sync();

//     val = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
//     if (warpID == 0) {
//         for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
//             val += __shfl_down_sync(mask, val, offset);
//         }
//     }

//     return val;
// }

// __global__ void norm2(const int n, const float *x, float *z) {
//     extern __shared__ float sdata[];

//     float my_sum = thread_sum(n, x);

//     auto g = this_thread_block();   // get all threads within a block
    
//     auto tileIdx = g.thread_rank() / 32; // similar to get the warpID
//     float* t = &sdata[32 * tileIdx];

//     auto tile32 = tiled_partition(g, 32);
//     float tile_sum = reduce_sum(tile32, my_sum, t);

//     if (tile32.thread_rank() == 0) {
//         atomicAdd(z, tile_sum);
//     }
// }


__device__ float reduce_sum_tile_shfl(cooperative_groups::thread_block_tile<32> g, float val) {
    for (int i = g.size() / 2; i > 0; i /= 2) {
        val += g.shfl_down(val, i);
    }
    return val;
}



__global__ void norm2(const int n, const float *x, float *z) {
    extern __shared__ float sdata[];
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (globalId < n) {
        sum = x[globalId] * x[globalId];
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile32 = cooperative_groups::tiled_partition<32>(block);

    float tile_sum = reduce_sum_tile_shfl(tile32, sum);

    if (tile32.thread_rank() == 0) {
        sdata[tile32.meta_group_rank()] = tile_sum;
    }

    block.sync();  // Synchronize all threads within the block

    if (tile32.meta_group_rank() == 0) {
        float block_sum = sdata[threadIdx.x];
        block_sum = reduce_sum_tile_shfl(tile32, block_sum);

        if (threadIdx.x == 0) {
            atomicAdd(z, block_sum);
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

    gettimeofday(&tp, &tzp);
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
    errorX = cudaMalloc((float **)&xGPU, byteSize); // allocate global memory in GPU
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
        x[i] = static_cast<float>(1);
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
        0);

    printf("system-opimized bestBlockSize = %d\n", bestBlockSize);
    printf("system-opimized minGridSize = %d\n", minGridSize);

    int grid = max(minGridSize, static_cast<int>((n + bestBlockSize - 1) / bestBlockSize));
    // change to n/2 to vectorize it
    // int sharedSize = bestBlockSize * sizeof(float); 

    printf("actual GridSize = %d\n", grid);

    double start_time = my_clock();

    for (int i = 0; i < runs; ++i)
    {
        // cudaMemset(zGPU, 0, bytez);
        // norm2<<<grid, block>>>(n, xGPU, zGPU);
        norm2<<<grid, bestBlockSize>>>(n, xGPU, zGPU);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        }
        cudaDeviceSynchronize();
        cudaMemcpy(&z, zGPU, bytez, cudaMemcpyDeviceToHost);
        z = sqrtf(z);
    }
    cudaDeviceSynchronize();

    double end_time = my_clock();
    double averageDuration = (end_time - start_time) / runs;

    printf("Average execution time over %d runs: \t%f ms.\n", runs, averageDuration);
    printf("result:\t%.9f.\n\n", z);

    free(x);
    cudaFree(xGPU);
    cudaFree(zGPU);
    return 0;
}