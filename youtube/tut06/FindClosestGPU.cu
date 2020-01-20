#include <iostream>
#include "config.h"
#include "FindClosestGPU.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#ifdef SMALL_ARRAY
__constant__ float3 points_cte [N];
#endif // SMALL_ARRAY

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
        cout<< "Could not copy points" << endl;
        cudaFree(dev_points);
        cudaFree(dev_indices);
        return 1;
    }

    // Invoke kernel
    FindClosestOpt<<<(count/512)+1, 512>>>(dev_points, dev_indices, count);

    // Copy data back
    if (cudaMemcpy(indices, dev_indices, sizeof(int) * count, cudaMemcpyDeviceToHost)!=cudaSuccess)
    {
        cout<< "Could not copy back from device" << endl;
        cudaFree(dev_points);
        cudaFree(dev_indices);
        delete[] dev_points;
        delete[] dev_indices;
        return 1;
    }

    // Cleanup
    cudaFree(dev_points);
    cudaFree(dev_indices);

    return 0;
}


/// Non-optimized version of find closest point
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

/// Optimized to do one write instead of updating the indices array for every
/// best point found
__global__ void FindClosestOpt (float3* points, int* indices, int count) {
    if (count <= 1) return;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < count)
    {

        float3 thisPoint = points[idx];
        float distToClosest = 3.40282e38f; // float.max value for initial dist
        int tmp; // local variable for best index

        //#pragma unroll 10
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
                tmp = i;
            }
        }
        indices[idx] = tmp;
    }
}

#ifdef SMALL_ARRAY
int FindClosestGPUcte (float3* points, int* indices, int count) {

    // Allocate and copy to device
    int * dev_indices;

    if (cudaMalloc(&dev_indices, sizeof(int)*count) != cudaSuccess)
    {
        cout<< "Could not allocate dev_indices" << endl;
        return 1;
    }
    if (cudaMemcpyToSymbol(points_cte, points, sizeof(float3) * count)!=cudaSuccess)
    {
        cout<< "Could not copy points" << endl;
        cudaFree(dev_indices);
        printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }

    // Invoke kernel
    FindClosestOptCte<<<(count/512)+1, 512>>>(dev_indices, count);

    // Copy data back
    if (cudaMemcpy(indices, dev_indices, sizeof(int) * count, cudaMemcpyDeviceToHost)!=cudaSuccess)
    {
        cout<< "Could not copy back from device" << endl;
        cudaFree(dev_indices);
        delete[] dev_indices;
        return 1;
    }

    // Cleanup
    cudaFree(dev_indices);

    return 0;
}
/// Optimized to:
/// do one write instead of updating the indices array for every
///    best point found
/// constant memory for points
__global__ void FindClosestOptCte (int* indices, int count) {
    if (count <= 1) return;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < count)
    {

        float3 thisPoint = points_cte[idx];
        float distToClosest = 3.40282e38f; // float.max value for initial dist
        int tmp; // local variable for best index

        for (int i = 0; i < count; i++)
        {
            if (i == idx) continue;
            float dist = 
                (thisPoint.x - points_cte[i].x) *
                (thisPoint.x - points_cte[i].x) +
                (thisPoint.y - points_cte[i].y) *
                (thisPoint.y - points_cte[i].y) +
                (thisPoint.z - points_cte[i].z) *
                (thisPoint.z - points_cte[i].z)
            ;
            if(dist < distToClosest) {
                distToClosest = dist;
                tmp = i;
            }
        }
        indices[idx] = tmp;
    }
}
#endif // SMALL_ARRAY