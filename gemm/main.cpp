#include <iostream>
#include "matrix.h"
#include "benchmark.h"

#include "cpu/kernel.h"
#include "omp/kernel.h"
#include "cuda/kernel.h"

int main()
{
    size_t runtimes = 2;
    size_t matrix_size = 4096;

    //run_test_cpu(matrix_size, runtimes);
    std::cout << std::endl;

    //run_test_omp(matrix_size, runtimes);
    std::cout << std::endl;

    run_test_cuda(matrix_size, runtimes);
    std::cout << std::endl;

    return 0;
}
