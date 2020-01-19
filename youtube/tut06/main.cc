#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <ctime>

#include "FindClosestCPU.h"
#include "FindClosestGPU.h"

#define NSAMPLES 5

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

    long fastestTime;
    // Run throught the algorithm several times
    cout << "Collecting fastest CPU runtime..." << endl;
    fastestTime = 100000000;

    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        long startTime = clock();
        FindClosestCPU(points, indexOfClosest, count);
        long finishTime = clock();

        long currentTime = finishTime - startTime;
        cout << "Run "<< q << "\ttook " << currentTime/CLOCKS_PER_SEC*1000 << "ms" << endl;

        if (currentTime < fastestTime)
            fastestTime = currentTime;
    }
    cout << "!!! Fastest CPU time: " << fastestTime << "ms" << endl<<endl;
    
    cout << "Collecting fastest CPU opt runtime..." << endl;
    fastestTime = 100000000;
    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        long startTime = clock();
        FindClosestCPUopt(points, indexOfClosest, count);
        long finishTime = clock();

        long currentTime = finishTime - startTime;
        cout << "Run "<< q << "\ttook " << currentTime/CLOCKS_PER_SEC*1000 << "ms" << endl;

        if (currentTime < fastestTime)
            fastestTime = currentTime;
    }
    cout << "!!! Fastest CPU time: " << fastestTime << "ms" << endl<<endl;

    for (int i = 0; i < 5; i++)
    {
        cout << "Point: " << i
             << "\tClosest to point: " << indexOfClosest[i] << endl;
    }

    delete[] indexOfClosest;
    delete[] points;

    return 0;
}

