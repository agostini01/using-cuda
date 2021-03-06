#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

#include "config.h"
#include "FindClosestCPU.h"
#include "FindClosestGPU.h"

using namespace std;

int main(int argc, char const *argv[])
{
    srand (time(NULL));

    const int count = N;
    int * indexOfClosest = new int[count];
    float3 * points = new float3[count];

    for (int i = 0; i < count; i++)
    {
        points[i].x = (float) ((rand() % 10000) - 5000);
        points[i].y = (float) ((rand() % 10000) - 5000);
        points[i].z = (float) ((rand() % 10000) - 5000);
    }


    if(false){
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
    }
    
    
    // GPU RUN =================================================================
    // Run throught the algorithm several times
    cudaDeviceReset();
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

    #ifdef SMALL_ARRAY 
    // GPU cte RUN =============================================================
    // Run throught the algorithm several times
    cudaDeviceReset();
    cout << "Collecting GPU cte runtime..." << endl;
    for (int q = 0; q < NSAMPLES; q++)
    {
        // Time the run
        auto startTime = chrono::high_resolution_clock::now();
        FindClosestGPUcte(points, indexOfClosest, count);
        auto finishTime = chrono::high_resolution_clock::now();

        auto currentTime = chrono::duration_cast<chrono::microseconds>(finishTime - startTime);
        cout << "Run " << q << "\ttook " << currentTime.count() << "us" << endl;
    }
    cout << endl;
    #endif // SMALL_ARRAY
    
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

