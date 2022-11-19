#pragma once
#include <string>
#include <omp.h>

template<class T>
struct axpy_omp
{
    std::string name() const
    {
        return "advanced CPU OpenMP multi thread implementation";
    }

    inline void prepare(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {}

    inline void main(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {
        int i;

#pragma omp parallel for private(i)
        for (i = 0; i < n; i++)
        {
            y[i * incy] += a * x[i * incx];
        }
    }
    
    inline void finalize(size_t n, T a, T* x, size_t incx, T* y, size_t incy)
    {}
};

void run_test_omp(size_t size, size_t runtimes);
