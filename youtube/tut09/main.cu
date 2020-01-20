#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;


/// With k20m and k40m GPUs banks are organized in sets of 8 bytes,
/// for this reason, conflicts happen when accesses to doubles fall on the
/// same bank
__global__ void MyKernelHomogeneos(unsigned long long * time) {
    const unsigned sharedSize = 4096;
    __shared__ double shared[sharedSize];
    unsigned long long startTime;
    unsigned long long finishTime;

    // const int idx = 0; //perform a broadcast
    // const int idx = blockIdx.x; // perform a broadcast
    // const int idx = threadIdx.x; // no bank conflict - each therad access different bank
    // const int idx = threadIdx.x*2; // bank conflict starts
    // const int idx = threadIdx.x*32; // worst bank conflict - all threads access same bank
    // const int idx = threadIdx.x*128; // same worst bank conflict

    const int idx = threadIdx.x*4; // current test
    if (idx < sharedSize) {

    // time the access an homogeneous array
    startTime = clock();
    shared[idx]++;
    finishTime = clock();

    time[threadIdx.x] = (finishTime - startTime);
    }
}

struct test{
    double x,y,w,z;

    // Adding padding here offsets the 4-way bank conflict
    // at the cost of wasting shared memory
    // comment it out to test difference
    // double padding;
};

struct test2{
    // This will present a 32-way bank conflict if no padding is added
    //    with:
    //              __shared__ test2 shared[32];
    //              shared[threadIdx.x].x[0]++;
    double x[32];

    // Adding padding here offsets the 32-way bank conflict
    // at the cost of wasting shared memory
    // comment it out to test difference
    // double padding;
};

__global__ void MyKernelDS(unsigned long long * time) {
    const unsigned sharedSize = 1024;
    __shared__ test shared[sharedSize];
    unsigned long long startTime;
    unsigned long long finishTime;

    const int idx = threadIdx.x*1; // current test

    // time the access an homogeneous array
    startTime = clock();
    shared[idx].x++;
    finishTime = clock();

    time[threadIdx.x] = (finishTime - startTime);
}

int main(int argc, char const *argv[])
{
    const unsigned nThreads = 32;

    unsigned long long time[nThreads];
    unsigned long long * d_time;

    cudaMalloc(&d_time, sizeof(unsigned long long)*nThreads);

    const unsigned long long overhead = 0;
    for (int r = 0; r < 10; r++)
    {
        cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
        // MyKernelHomogeneos<<< 1,nThreads >>>(d_time);
        MyKernelDS<<< 1,nThreads >>>(d_time);
        cudaMemcpy(&time, d_time, sizeof(unsigned long long)*nThreads, cudaMemcpyDeviceToHost);

        cout << "Time:\t";
        for (int i = 0; i < nThreads; i++)
        {
         cout<<(time[i]-overhead)/32<<"\t";
        }
        cout << endl<<endl;
    }
    
    cudaFree(d_time);
    cudaDeviceReset();
    return 0;
}