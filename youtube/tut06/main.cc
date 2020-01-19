#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

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

    auto fastestTime=chrono::high_resolution_clock::now();

    // CPU RUN =================================================================
    // Run throught the algorithm several times
    cout << "Collecting CPU runtime..." << endl;
    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        auto startTime = chrono::high_resolution_clock::now();
        FindClosestCPU(points, indexOfClosest, count);
        auto finishTime = chrono::high_resolution_clock::now();

        auto currentTime = chrono::duration_cast<chrono::microseconds>(finishTime - startTime);
        cout << "Run " << q << "\ttook " << currentTime.count() << "us" << endl;
    }
    cout << endl;
    
    // CPU opt RUN =============================================================
    // Run throught the algorithm several times
    cout << "Collecting CPU opt runtime..." << endl;
    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        auto startTime = chrono::high_resolution_clock::now();
        FindClosestCPUopt(points, indexOfClosest, count);
        auto finishTime = chrono::high_resolution_clock::now();

        auto currentTime = chrono::duration_cast<chrono::microseconds>(finishTime - startTime);
        cout << "Run " << q << "\ttook " << currentTime.count() << "us" << endl;
    }
    cout << endl;


    for (int i = 0; i < 5; i++)
    {
        cout << "Point: " << i
             << "\tClosest to point: " << indexOfClosest[i] << endl;
    }
    cout<<endl;
    
    // GPU RUN =================================================================
    // Run throught the algorithm several times
    cout << "Collecting GPU runtime..." << endl;
    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        auto startTime = chrono::high_resolution_clock::now();
        FindClosestGPU(points, indexOfClosest, count);
        auto finishTime = chrono::high_resolution_clock::now();

        auto currentTime = chrono::duration_cast<chrono::microseconds>(finishTime - startTime);
        cout << "Run " << q << "\ttook " << currentTime.count() << "us" << endl;
    }
    cout << endl;
    
    for (int i = 0; i < 5; i++)
    {
        cout << "Point: " << i
             << "\tClosest to point: " << indexOfClosest[i] << endl;
    }
    cout<<endl;

    delete[] indexOfClosest;
    delete[] points;

    return 0;
}

