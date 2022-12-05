#include "kernel.h"
#include "../benchmark.h"
#include "../cpu/kernel.h"

template<class T> using gemm_cuda_16 = typename gemm_cuda<T, false>;
template<class T> using gemm_cuda_16_optimized = typename gemm_cuda<T, true>;

template<class T>
void run_test_cuda_typed(size_t size, size_t runtimes)
{
    benchmark<T, gemm_cpu>               cpu;
    benchmark<T, gemm_cuda_16>           cuda_defaut;
    benchmark<T, gemm_cuda_16_optimized> cuda_optimized;

    if (runtimes == 1)
    {
        cpu.run(size, runtimes);
        std::cout << std::endl;
    }

    cuda_defaut.run(size, runtimes);

    if (runtimes == 1)
    {
        cpu.export_data(
            [&](const raw_memory_matrix_view<T>& cpu_matrix)
            {
                cuda_defaut.export_data(
                    [&](const raw_memory_matrix_view<T>& cuda_matrix)
                    {
                        compare_matrix_result(cpu_matrix, cuda_matrix);
                    });
            });
    }

    std::cout << std::endl;

    cuda_optimized.run(size, runtimes);

    if (runtimes == 1)
    {
        cpu.export_data(
            [&](const raw_memory_matrix_view<T>& cpu_matrix)
            {
                cuda_optimized.export_data(
                    [&](const raw_memory_matrix_view<T>& cuda_matrix)
                    {
                        compare_matrix_result(cpu_matrix, cuda_matrix);
                    });
            });
    }

    std::cout << std::endl;
}

void run_test_cuda(size_t size, size_t runtimes)
{
    run_test_cuda_typed<float>(size, runtimes);
    run_test_cuda_typed<double>(size, runtimes);
}