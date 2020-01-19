#ifndef FINDCLOSESTGPU_H
#define FINDCLOSESTGPU_H

#include "cuda_runtime.h"

int FindClosestGPU (float3* points, int* indices, int count);

__global__ void FindClosest (float3* points, int* indices, int count);

#endif //FINDCLOSESTGPU_H