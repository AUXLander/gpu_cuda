#pragma once
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void kernel_axpy_float(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy);
__global__ void kernel_axpy_double(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy);

template<class T, size_t BLOCK_SIZE>
struct axpy_cuda
{   
    T* device_x;
    T* device_y;

    std::string name() const
    {
        return std::string("advanced GPU CUDA multi thread implementation (block size ") + std::to_string(BLOCK_SIZE) + ")";
    }

    inline void prepare(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {
        cudaMalloc(&device_x, sizeof(T) * n);
        cudaMalloc(&device_y, sizeof(T) * n);

        cudaMemcpy(device_x, x, sizeof(T) * n, cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(device_y, y, sizeof(T) * n, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }

    inline void main(size_t n, T a, T*, size_t incx, T*, size_t incy)
    {
        const size_t grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        if constexpr (std::is_same<T, float>::value)
        {
            kernel_axpy_float<<<grid_size, BLOCK_SIZE >>>(n, a, device_x, incx, device_y, incy);
        }

        if constexpr (std::is_same<T, double>::value)
        {
            kernel_axpy_double<<<grid_size, BLOCK_SIZE>>>(n, a, device_x, incx, device_y, incy);
        }

        cudaDeviceSynchronize();
    }

    inline void finalize(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {
        cudaMemcpy(x, device_x, sizeof(T) * n, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy(y, device_y, sizeof(T) * n, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        cudaFree(device_x);
        cudaFree(device_y);
    }
};

void run_test_cuda(size_t size, size_t runtimes);
