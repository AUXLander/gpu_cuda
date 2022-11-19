#include "kernel.h"
#include "../benchmark.h"

__global__ void kernel_axpy_float(size_t n, float alpha, float *x, size_t incx, float *y, size_t incy)
{
    auto idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < n) 
    {
        y[idx * incy] += alpha * x[idx *incx];
    }
}

__global__ void kernel_axpy_double(size_t n, double alpha, double *x, size_t incx, double *y, size_t incy)
{
    auto idx = blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < n) 
    {
        y[idx * incy] += alpha * x[idx *incx];
    }
}

template<class T> using axpy_cuda_8 = typename axpy_cuda<T, 8U>;
template<class T> using axpy_cuda_16 = typename axpy_cuda<T, 16U>;
template<class T> using axpy_cuda_32 = typename axpy_cuda<T, 32U>;
template<class T> using axpy_cuda_64 = typename axpy_cuda<T, 64U>;
template<class T> using axpy_cuda_128 = typename axpy_cuda<T, 128U>;
template<class T> using axpy_cuda_256 = typename axpy_cuda<T, 256U>;

template<class T>
void run_test_cuda_typed(size_t size, size_t runtimes)
{
    benchmark<T, axpy_cuda_8>{}.run(size, runtimes);
    benchmark<T, axpy_cuda_16>{}.run(size, runtimes);
    benchmark<T, axpy_cuda_32>{}.run(size, runtimes);
    benchmark<T, axpy_cuda_64>{}.run(size, runtimes);
    benchmark<T, axpy_cuda_128>{}.run(size, runtimes);
    benchmark<T, axpy_cuda_256>{}.run(size, runtimes);
}

void run_test_cuda(size_t size, size_t runtimes)
{
    run_test_cuda_typed<float>(size, runtimes);
    run_test_cuda_typed<double>(size, runtimes);
}