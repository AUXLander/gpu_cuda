#include "kernel.h"
#include "../benchmark.h"

void run_test_cpu(size_t size, size_t runtimes)
{
    benchmark<float, axpy_cpu>{}.run(size, runtimes);
    benchmark<double, axpy_cpu>{}.run(size, runtimes);
}