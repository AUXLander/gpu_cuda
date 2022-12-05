#pragma once
#include <cassert>
#include <string>
#include <vector>

#include "../matrix.h"

template<class T>
struct gemm_omp
{
    std::string name() const
    {
        return "default CPU single thread implementation";
    }

    inline void prepare([[maybe_unused]] std::vector<raw_memory_matrix_view<T>>&)
    {}

    inline void main(std::vector<raw_memory_matrix_view<T>>& matrix, T alpha, T beta)
    {
        assert(matrix.size() == 3U);

        auto &a = matrix[0];
        auto &b = matrix[1];
        auto &c = matrix[2];

        #pragma omp parallel for
        for (int i = 0; i < c.properties().size_x(); i++)
        {
            for (size_t j = 0; j < c.properties().size_y(); j++)
            {
                T value = 0;

                for (size_t k = 0; k < b.properties().size_y(); k++)
                {
                    value += a.at(i, k) * b.at(k, j);
                }

                c.at(i, j) = alpha * value + beta * c.at(i, j);
            }
        }
    }

    inline void finalize([[maybe_unused]] std::vector<raw_memory_matrix_view<T>>&)
    {}
};

void run_test_omp(size_t size, size_t runtimes);
