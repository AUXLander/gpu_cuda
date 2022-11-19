#include "benchmark.h"

#include "cpu/kernel.h"
#include "omp/kernel.h"
#include "cuda/kernel.h"

int main()
{
    const std::vector<size_t> size_set = { 50'000'000, 100'000'000, 200'000'000, 300'000'000, 400'000'000, 500'000'000};

    // const std::vector<size_t> size_set = { 1024'0 };

    size_t runtimes = 100;

    for (size_t size : size_set)
    {
        std::cout << "Set size = " << size << std::endl;

        run_test_cpu(size, runtimes);
        std::cout << std::endl;

        run_test_omp(size, runtimes);
        std::cout << std::endl;

        run_test_cuda(size, runtimes);
        std::cout << std::endl;

        std::cout << std::endl;
    }

    return 0;
}
