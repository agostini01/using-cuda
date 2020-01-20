#ifndef FINDCLOSESTGPU_H
#define FINDCLOSESTGPU_H

#include "cuda_runtime.h"

int FindClosestGPU (float3* points, int* indices, int count);

__global__ void FindClosest (float3* points, int* indices, int count);
__global__ void FindClosestOpt (float3* points, int* indices, int count);

#ifdef SMALL_ARRAY
int FindClosestGPUcte (float3* points, int* indices, int count);
__global__ void FindClosestOptCte (int* indices, int count);
#endif // SMALL_ARRAY

#endif //FINDCLOSESTGPU_H