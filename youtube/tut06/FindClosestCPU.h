#ifndef FINDCLOSESTCPU_H
#define FINDCLOSESTCPU_H

#include "cuda_runtime.h"

void FindClosestCPU (float3* points, int* indices, int count);

#endif //FINDCLOSESTCPU_H