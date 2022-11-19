#include <stdio.h>
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel_01()
{
	printf("Hello, world!\n");
}

__host__ void launch_01()
{
    printf("launch kernel 01!\n");

    kernel_01<<<2, 2>>>();
    cudaDeviceSynchronize();

    printf("\n\n");
}

////////////////////////////////

__global__ void kernel_02()
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    printf("I am from %i block, %i thread (global index: %i)\n", blockIdx.x, threadIdx.x, tid);
}

__host__ void launch_02()
{
    printf("launch kernel 02!\n");

    kernel_02<<<2, 2>>>();
    cudaDeviceSynchronize();

    printf("\n\n");
}

////////////////////////////////

__global__ void kernel_03(int* array, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) 
    {
        array[tid] += tid;
    }
}

__host__ void launch_03()
{
    printf("launch kernel 03!\n");

    constexpr int ARRAY_SIZE = 4;
    constexpr int BLOCK_SIZE = 4;

    int* host_array;
    host_array = (int*)std::malloc(sizeof(int) * ARRAY_SIZE);

    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        host_array[i] = 1;
    }

    int* device_array;
    cudaMalloc(&device_array, sizeof(int) * ARRAY_SIZE);

    printf("Print array before\n");
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        printf("a[%zi] = %i\n", i, host_array[i]);
    }

    cudaMemcpy(device_array, host_array, sizeof(int) * ARRAY_SIZE, cudaMemcpyKind::cudaMemcpyHostToDevice);

    constexpr auto GRID_SIZE = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_03<<<GRID_SIZE, BLOCK_SIZE>>>(device_array, ARRAY_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(host_array, device_array, sizeof(int) * ARRAY_SIZE, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    printf("Print array after\n");
    for (size_t i = 0; i < ARRAY_SIZE; i++)
    {
        printf("a[%zi] = %i\n", i, host_array[i]);
    }

    cudaFree(device_array);
    std::free(host_array);

    printf("\n\n");
}

////////////////////////////////

int main()
{
    launch_01();

    launch_02();

    launch_03();

	return 0;
}