#pragma once
#include <stdio.h>
#include <stdlib.h>


void setGPU(){
	
	int iDeviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&iDeviceCount);

	if (error != cudaSuccess || iDeviceCount == 0)
	{
		printf("\nno cuda capable GPU found\n");
		exit(-1);
	}
	else{
		printf("\nthe count of gpus is %d\n", iDeviceCount);
	}

	int iDev = 0;
	error = cudaSetDevice(iDev);
	if (error != cudaSuccess)
	{
		printf("\nfail to set gpu 0 for computing\n");
		exit(-1);
	}
	else
	{
		printf("set gpu 0 for compuiting\n");
	}
}
