#include "kernel.h"
#include "../benchmark.h"

void run_test_omp(size_t size, size_t runtimes)
{
    benchmark<float, gemm_omp>{}.run(size, runtimes);
    benchmark<double, gemm_omp>{}.run(size, runtimes);
}