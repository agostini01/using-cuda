#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <ctime>

#include "FindClosestCPU.h"
#include "FindClosestGPU.h"

using namespace std;


int main(int argc, char const *argv[])
{
    srand (time(NULL));

    const int count = 10000;
    int * indexOfClosest = new int[count];
    float3 * points = new float3[count];

    for (int i = 0; i < count; i++)
    {
        points[i].x = (float) ((rand() % 10000) - 5000);
        points[i].y = (float) ((rand() % 10000) - 5000);
        points[i].z = (float) ((rand() % 10000) - 5000);
    }

    // Run throught the algorithm several times
    cout << "Collecting fastest CPU runtime..." << endl;
    long fastestTime = 100000000;

    for (int q = 0; q < 20; q++)
    {
        // Time the run
        long startTime = clock();
        FindClosestCPU(points, indexOfClosest, count);
        long finishTime = clock();

        long currentTime = finishTime - startTime;
        cout << "Run "<< q << "\ttook " << currentTime << "ms" << endl;

        if (currentTime < fastestTime)
            fastestTime = currentTime;
    }
    cout << "!!! Fastest CPU time: " << fastestTime << "ms" << endl<<endl;

    for (int i = 0; i < 5; i++)
    {
        cout << "Point: " << i
             << "\tClosest to point: " << indexOfClosest[i] << endl;
    }

    // if (cudaMalloc(&d_a, sizeof(int)*count) != cudaSuccess)
    // {
    //     cout<< "Could not allocate d_a" << endl;
    //     return 1;
    // }
    // if (cudaMalloc(&d_b, sizeof(int)*count) != cudaSuccess)
    // {
    //     cout<< "Could not allocate d_b" << endl;
    //     cudaFree(d_a);
    //     return 1;
    // }

    // if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    // {
    //     cout<< "Could not copy d_a" << endl;
    //     cudaFree(d_a);
    //     cudaFree(d_b);
    //     return 1;
    // }
    
    // if (cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    // {
    //     cout<< "Could not copy d_b" << endl;
    //     cudaFree(d_a);
    //     cudaFree(d_b);
    //     return 1;
    // }

    // AddInts <<<count/256+1, 256>>>(d_a,d_b,count);

    // if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost)!=cudaSuccess)
    // {
    //     cout<< "Could not copy back from device" << endl;
    //     cudaFree(d_a);
    //     cudaFree(d_b);
    //     delete[] h_a;
    //     delete[] h_b;
    //     return 1;
    // }

    // for (int i = 0; i < 5; i++)
    // {
    //     cout <<"It is: " << h_a[i] << endl;
    // }
    

    delete[] indexOfClosest;
    delete[] points;

    return 0;
}

