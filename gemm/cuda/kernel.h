#pragma once
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../matrix.h"

template<class T>
__global__ void kernel_gemm(T* device_a, T* device_b, T* device_c, T alpha, T beta, size_t size)
{
    const auto block_shift_x = blockDim.x * blockIdx.x;
    const auto block_shift_y = blockDim.y * blockIdx.y;

    const auto thread_x = threadIdx.x;
    const auto thread_y = threadIdx.y;

    const auto x = block_shift_x + thread_x;
    const auto y = block_shift_y + thread_y;

    auto a = raw_memory_matrix_view<T>(device_a, size, size, 1, 1);
    auto b = raw_memory_matrix_view<T>(device_b, size, size, 1, 1);
    auto c = raw_memory_matrix_view<T>(device_c, size, size, 1, 1);

    T value = 0;
    for (size_t k = 0; k < b.properties().size_y(); k++)
    {
        value += a.at(k, y) * b.at(x, k);
    }

    c.at(x, y) = alpha * value + beta * c.at(x, y);
}

template<class T>
__global__ void kernel_gemm_optimized(T* device_a, T* device_b, T* device_c, T alpha, T beta, size_t size)
{
    constexpr auto BLOCK_SIZE = 16;// gemm_cuda<T, true>::BLOCK_SIZE;

    const auto block_shift_x = blockDim.x * blockIdx.x;
    const auto block_shift_y = blockDim.y * blockIdx.y;

    const auto thread_x = threadIdx.x;
    const auto thread_y = threadIdx.y;

    const auto x = block_shift_x + thread_x;
    const auto y = block_shift_y + thread_y;

    __shared__ T a_shared_cache_data[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T b_shared_cache_data[BLOCK_SIZE * BLOCK_SIZE];

    auto a = raw_memory_matrix_view<T>(device_a, size, size, 1, 1);
    auto b = raw_memory_matrix_view<T>(device_b, size, size, 1, 1);
    auto c = raw_memory_matrix_view<T>(device_c, size, size, 1, 1);

    auto a_shared_cache = raw_memory_matrix_view<T>(a_shared_cache_data, blockDim.x, blockDim.y, 1, 1);
    auto b_shared_cache = raw_memory_matrix_view<T>(b_shared_cache_data, blockDim.x, blockDim.y, 1, 1);

    auto blocks_per_row = (a.properties().size_x() + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T value = 0;

    for (size_t block_idx = 0; block_idx < blocks_per_row; ++block_idx)
    {
        size_t shared_x = (block_idx * BLOCK_SIZE) + thread_x;
        size_t shared_y = (block_idx * BLOCK_SIZE) + thread_y;

        a_shared_cache.at(thread_x, thread_y) = a.at(shared_x, y);
        b_shared_cache.at(thread_x, thread_y) = b.at(x, shared_y);

        __syncthreads();

        for (size_t k = 0; k < BLOCK_SIZE; ++k)
        {
            value += a_shared_cache.at(k, thread_y) * b_shared_cache.at(thread_x, k);
        }

        __syncthreads();
    }

    c.at(x, y) = alpha * value + beta * c.at(x, y);
}

template<class T, bool is_optimized = false>
struct gemm_cuda
{   
    constexpr static size_t BLOCK_SIZE = 16;

    T* device_a;
    T* device_b;
    T* device_c;

    template<class Tfunc>
    void cudaTrace(Tfunc&& eval)
    {
        auto result = eval();

        if (result != cudaError::cudaSuccess)
        {
            std::cerr << "CUDA Error: " << cudaGetErrorName(result) << std::endl;
            std::cerr << cudaGetErrorString(result) << std::endl;
            
            throw;
        }
    }

    std::string name() const
    {
        if constexpr (is_optimized)
        {
            return std::string("advanced GPU CUDA multi thread implementation (block size ") + std::to_string(BLOCK_SIZE) + ")";
        }
        else
        {
            return std::string("default GPU CUDA multi thread implementation (block size ") + std::to_string(BLOCK_SIZE) + ")";
        }
    }

    inline void prepare(std::vector<raw_memory_matrix_view<T>>& matrix)
    {
        assert(matrix.size() == 3U);

        cudaTrace([&]() {
            return cudaMalloc(&device_a, sizeof(T) * matrix[0].properties().length());
        });

        cudaTrace([&]() {
            return cudaMalloc(&device_b, sizeof(T) * matrix[1].properties().length());
        });

        cudaTrace([&]() {
            return cudaMalloc(&device_c, sizeof(T) * matrix[2].properties().length());
        });

        ///

        cudaTrace([&]() {
            return cudaMemcpy(device_a, matrix[0].data(), matrix[0].size_of_data(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        });
        
        cudaTrace([&]() {
            return cudaMemcpy(device_b, matrix[1].data(), matrix[1].size_of_data(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        });

        cudaTrace([&]() {
            return cudaMemcpy(device_c, matrix[2].data(), matrix[2].size_of_data(), cudaMemcpyKind::cudaMemcpyHostToDevice);
        });
    }

    inline void main(std::vector<raw_memory_matrix_view<T>>& matrix, T alpha, T beta)
    {
        assert(matrix[0].properties().size_x() % BLOCK_SIZE == 0);
        assert(matrix[0].properties().size_y() % BLOCK_SIZE == 0);
        assert(matrix[1].properties().size_x() % BLOCK_SIZE == 0);
        assert(matrix[1].properties().size_y() % BLOCK_SIZE == 0);

        const dim3 grid_block_size = dim3{ BLOCK_SIZE, BLOCK_SIZE };
        const dim3 grid_size = dim3{ ((unsigned int)matrix[2].properties().size_x() + BLOCK_SIZE - 1) / BLOCK_SIZE, ((unsigned int)matrix[2].properties().size_y() + BLOCK_SIZE - 1) / BLOCK_SIZE };

        if constexpr (is_optimized)
        {
            kernel_gemm_optimized<T><<<grid_size, grid_block_size>>> (device_a, device_b, device_c, alpha, beta, matrix[0].properties().size_x());
        }
        else
        {
            kernel_gemm<T><<<grid_size, grid_block_size>>> (device_a, device_b, device_c, alpha, beta, matrix[0].properties().size_x());
        }

        cudaTrace([&]() {
            return cudaDeviceSynchronize();
        });
    }

    inline void finalize(std::vector<raw_memory_matrix_view<T>>& matrix)
    {
        cudaTrace([&]() {
            return cudaMemcpy(matrix[0].data(), device_a, matrix[0].size_of_data(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        });

        cudaTrace([&]() {
            return cudaMemcpy(matrix[1].data(), device_b, matrix[1].size_of_data(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        });

        cudaTrace([&]() {
            return cudaMemcpy(matrix[2].data(), device_c, matrix[2].size_of_data(), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        });

        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
    }
};

void run_test_cuda(size_t size, size_t runtimes);
