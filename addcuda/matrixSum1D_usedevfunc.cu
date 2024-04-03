#include <stdio.h>
#include "common.cuh"

__device__ float add(const float x, const float y){  //device func can return other types besides void
    return x + y;
}


__global__ void addfromGPU(float *A, float *B, float *C, const int N){  //kernal func can only return void
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + blockDim.x*bid;

    if (id >= N) return;
    C[id] = add(A[id], B[id]);
}


void initializeData(float *p, int nElement){
    for(int i = 0; i < nElement; i++){
        p[i] = (float)(rand() & 0xFF)/10.f;
    }
    // return;
}

int main(void){
    
    //1.set GPU device
    setGPU();

    //2.allocate memory for host and device
    int iElemCount = 513;                           //number of elements
    size_t iByteCount = iElemCount * sizeof(float); //bytes of assigned array
    

    float *pA, *pB, *pC;                            //create pointers for arrays in host
    pA = (float*)malloc(iByteCount);
    pB = (float*)malloc(iByteCount);
    pC = (float*)malloc(iByteCount);
    if (pA!=NULL&&pB!=NULL&&pC!=NULL){     //for robust
        memset(pA, 0, iByteCount);
        memset(pB, 0, iByteCount);
        memset(pC, 0, iByteCount);
    }
    else
    {
        printf("fail to allocate memory in CPU.\n");
        free(pA);
        free(pC);
        free(pB);
        exit(-1);
    }

    cudaError_t errorA, errorB, errorC;
    float *pDA, *pDB, *pDC;
    errorA = cudaMalloc((float**)&pDA, iByteCount);
    errorB = cudaMalloc((float**)&pDB, iByteCount);
    errorC = cudaMalloc((float**)&pDC, iByteCount);
    if (errorA == cudaSuccess && errorB == cudaSuccess && errorC == cudaSuccess){
        cudaMemset(pDA, 0, iByteCount);
        cudaMemset(pDB, 0, iByteCount);
        cudaMemset(pDC, 0, iByteCount);
    }
    else
    {
        printf("fail to allocate memory in GPU.\n");
        if (errorA == cudaSuccess) cudaFree(pDA);     //be cautious of these cudaFree, pDA is not allocated for memory at first
        if (errorB == cudaSuccess) cudaFree(pDB);     //so dont try to use cudaFree if it is not allocated by cudaMalloc
        if (errorC == cudaSuccess) cudaFree(pDC);
        free(pA);
        free(pC);
        free(pB);
        exit(-1);
    }

    
    //3.initialize data in host
    srand(666);  //set random seed
    initializeData(pA, iElemCount);
    initializeData(pB, iElemCount);

    //4.copy data in host to device
    cudaMemcpy(pDA, pA, iByteCount, cudaMemcpyHostToDevice);
    cudaMemcpy(pDB, pB, iByteCount, cudaMemcpyHostToDevice);

    //5.call kernal function to compute
    dim3 block(32); 
    // dim3 grid(iElemCount/32);
    dim3 grid((iElemCount+block.x-1)/block.x); //common way for rounding up(ceiling)

    addfromGPU<<<grid, block>>>(pDA, pDB, pDC, iElemCount);
    cudaDeviceSynchronize();

    //6.copy computed data from device to host
    cudaMemcpy(pC, pDC, iByteCount, cudaMemcpyDeviceToHost);  //implicitly synchronize

    for (int i = 0; i < 10; i++){
        printf("idx:%2d\tmatrix_A:%.2f\tmatrix_B:%.2f\tmatrix_C:%.2f\n",i+1, pA[i],pB[i],pC[i]);
    }

    //7.free memory in both host and device
    free(pA);
    free(pB);
    free(pC);
    cudaFree(pDA);
    cudaFree(pDB);
    cudaFree(pDC);
    
    cudaDeviceReset();


    return 0;
}