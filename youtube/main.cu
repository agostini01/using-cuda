#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

__global__ void AddInts(int * a, int* b, int count)
{
    int id = blockIdx.x * blockDim.x * threadIdx.x;
    if (id < count)
    {
        a[id]+b[id];
    }

}

int main(int argc, char const *argv[])
{
    srand (time(NULL));
    int count = 100;
    int * h_a = new int[count];
    int * h_b = new int[count];

    for (int i = 0; i < count; i++)
    {
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }

    cout << "Prior to addition:" << endl;
    for (int i = 0; i < 5; i++)
    {
        cout << h_a[i] << " " << h_b[i] << endl;
    }

    int *d_a, *d_b;

    if (cudaMalloc(&d_a, sizeof(int)*count) != cudaSuccess)
    {
        cout<< "Could not allocate d_a" << endl;
        return 1;
    }
    if (cudaMalloc(&d_b, sizeof(int)*count) != cudaSuccess)
    {
        cout<< "Could not allocate d_b" << endl;
        cudaFree(d_a);
        return 1;
    }

    if (cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<< "Could not copy d_a" << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }
    
    if (cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice)!=cudaSuccess)
    {
        cout<< "Could not copy d_b" << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    AddInts <<<count/256+1, 256>>>(d_a,d_b,count);

    if (cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost)!=cudaSuccess)
    {
        cout<< "Could not copy back from device" << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
        delete[] h_a;
        delete[] h_b;
    }

    for (int i = 0; i < 5; i++)
    {
        cout <<"It is: " h_a[i] << endl;
    }
    

    delete[] h_a;
    delete[] h_b;

    return 0;
}

