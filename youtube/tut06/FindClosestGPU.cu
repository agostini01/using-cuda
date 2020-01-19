#include <iostream>
#include "FindClosestGPU.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

int FindClosestGPU (float3* points, int* indices, int count) {

    // Allocate and copy to device
    float3 * dev_points;
    int * dev_indices;

    if (cudaMalloc(&dev_points, sizeof(float3)*count) != cudaSuccess)
    {
        cout<< "Could not allocate dev_points" << endl;
        return 1;
    }
    if (cudaMalloc(&dev_indices, sizeof(int)*count) != cudaSuccess)
    {
        cout<< "Could not allocate dev_indices" << endl;
        cudaFree(dev_points);
        return 1;
    }
    if (cudaMemcpy(dev_points, points, sizeof(float3) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<< "Could not copy d_a" << endl;
        cudaFree(dev_points);
        cudaFree(dev_indices);
        return 1;
    }

    // Invoke kernel
    FindClosest<<<(count/32)+1, 32>>>(dev_points, dev_indices, count);

    // Wait for kernel

    // Copy data back

    return 0;
}

__global__ void FindClosest (float3* points, int* indices, int count) {
    if (count <= 1) return;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < count)
    {

        float3 thisPoint = points[idx];
        float distToClosest = 3.40282e38f; // float.max value for initial dist

        for (int i = 0; i < count; i++)
        {
            if (i == idx) continue;
            float dist = 
                (thisPoint.x - points[i].x) *
                (thisPoint.x - points[i].x) +
                (thisPoint.y - points[i].y) *
                (thisPoint.y - points[i].y) +
                (thisPoint.z - points[i].z) *
                (thisPoint.z - points[i].z)
            ;
            if(dist < distToClosest) {
                distToClosest = dist;
                indices[idx] = i;
            }
        }
    }
}