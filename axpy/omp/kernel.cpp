#include "kernel.h"
#include "../benchmark.h"

void run_test_omp(size_t size, size_t runtimes)
{
    benchmark<float, axpy_omp>{}.run(size, runtimes);
    benchmark<double, axpy_omp>{}.run(size, runtimes);
}