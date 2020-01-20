#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

__global__ void MyKernel(unsigned long long * time) {
    __shared__ float shared[32];

    //const int idx = 0; //perform a broadcast
    //const int idx = blockIdx.x; // perform a broadcast
    const int idx = threadIdx.x; // no bank conflict

    // time the access
    unsigned long long startTime = clock();
    shared[idx]++;
    unsigned long long finishTime = clock();

    *time = (finishTime - startTime);
}

int main(int argc, char const *argv[])
{

    unsigned long long time;
    unsigned long long * d_time;

    cudaMalloc(&d_time, sizeof(unsigned long long));

    for (int i = 0; i < 10; i++)
    {
        MyKernel<<< 1,32 >>>(d_time);
        cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        const unsigned long long overhead = 0;
        cout<<"Time: "<<(time-overhead)/32<<endl<<endl;
    }

    cudaFree(d_time);
    cudaDeviceReset();
    return 0;
}

