#include "FindClosestCPU.h"
#include "cuda_runtime.h"
#include "math.h"

void FindClosestCPU (float3* points, int* indices, int count) {
    if (count <= 1) return;

    for (int curPoint = 0; curPoint < count; curPoint++)
    {
        float distToClosest = 3.40282e38f; // float.max value for initial dist

        for (int i = 0; i < count; i++)
        {
            if (i == curPoint) continue;
            float dist = sqrt(
                (points [curPoint].x - points[i].x) *
                (points [curPoint].x - points[i].x) +
                (points [curPoint].y - points[i].y) *
                (points [curPoint].y - points[i].y) +
                (points [curPoint].z - points[i].z) *
                (points [curPoint].z - points[i].z)
            );
            if(dist < distToClosest) {
                distToClosest = dist;
                indices[curPoint] = i;
            }
        }
    }
}

void FindClosestCPUopt (float3* points, int* indices, int count) {
    if (count <= 1) return;

    for (int curPoint = 0; curPoint < count; curPoint++)
    {
        float distToClosest = 3.40282e38f; // float.max value for initial dist

        for (int i = 0; i < count; i++)
        {
            if (i == curPoint) continue;
            float dist = 
                (points [curPoint].x - points[i].x) *
                (points [curPoint].x - points[i].x) +
                (points [curPoint].y - points[i].y) *
                (points [curPoint].y - points[i].y) +
                (points [curPoint].z - points[i].z) *
                (points [curPoint].z - points[i].z)
            ;
            if(dist < distToClosest) {
                distToClosest = dist;
                indices[curPoint] = i;
            }
        }
    }
}