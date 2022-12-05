#pragma once
#include <string>
#include <vector>
#include <cassert>

#include "../matrix.h"

template<class T>
struct gemm_cpu
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

        for (size_t x = 0; x < c.properties().size_x(); ++x)
        {
            for (size_t y = 0; y < c.properties().size_y(); ++y)
            {
                T value = 0;

                for (size_t k = 0; k < b.properties().size_y(); ++k)
                {
                    value += a.at(k, y) * b.at(x, k);
                }

                c.at(x, y) = alpha * value + beta * c.at(x, y);
            }
        }
    }

    inline void finalize([[maybe_unused]] std::vector<raw_memory_matrix_view<T>>&)
    {}
};

void run_test_cpu(size_t size, size_t runtimes);
