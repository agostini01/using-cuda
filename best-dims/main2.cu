#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void MyKernel(int *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] *= array[idx]; 
  } 
} 

void launchMyKernel(int *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size

  // The minimum grid size needed to achieve the
  // maximum occupancy for a full device launch
  int minGridSize; 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  // Round up according to array size 
  gridSize = (arrayCount + blockSize - 1) / blockSize; 

  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
  // printf("%d,%d,%d,%d\n", minGridSize, gridSize, blockSize, maxActiveBlocks );
  printf("minGridSize: %d\n", minGridSize);
  printf("gridSize: %d\n", gridSize);
  printf("blockSize: %d\n", blockSize);
  printf("maxActiveBlocks: %d\n", maxActiveBlocks);
  printf("props.warpSize: %d\n", props.warpSize);
  printf("props.maxThreadsPerMultiProcessor: %d\n", props.maxThreadsPerMultiProcessor);
}

int main(int argc, char const *argv[])
{
    const int N = 1000000;
    int* array = new int[N];
    launchMyKernel(array, N);
    delete[] (array);
    return 0;
}
