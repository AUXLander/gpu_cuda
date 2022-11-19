#pragma once
#include <string>

template<class T>
struct axpy_cpu
{
    std::string name() const
    {
        return "default CPU single thread implementation";
    }

    inline void prepare(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {}

    inline void main(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {
        for (size_t i = 0; i < n; i++)
        {
            y[i * incy] += a * x[i * incx];
        }
    }

    inline void finalize(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {}
};

void run_test_cpu(size_t size, size_t runtimes);
