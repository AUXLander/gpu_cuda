#pragma once

#include "matrix.h"

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <algorithm>

template<class T, template<class> class kernel>
class benchmark : public kernel<T>
{
    constexpr static size_t N = 100;
    constexpr static size_t MATRIX_COUNT = 3;

    T* matrix_data { nullptr };

    std::vector<raw_memory_matrix_view<T>> matrix;
    
    void fill_matrix(raw_memory_matrix_view<T>& view)
    {
        float inc = 0;

        view.for_each(
            [&](size_t x, size_t y, size_t z, size_t l) 
            {
                auto value = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);

                view.at(x, y, z, l) = value;
                // view.at(x, y, z, l) = inc++;
            });
    }

public:

    benchmark() {;}

    std::string name() const
    {
        return kernel<T>::name();
    }

    double mark(size_t size_x, size_t size_y, size_t number_of_tests)
    {
        srand(42);

        const size_t single_matrix_size = size_x * size_y;

        matrix_data = new T[single_matrix_size * MATRIX_COUNT] { 0.0 };

        for(size_t index = 0; index < MATRIX_COUNT; ++index)
        {
            matrix.emplace_back(matrix_data + single_matrix_size * index, size_x, size_y, 1U, 1U);

            fill_matrix(matrix[index]);
        }

        kernel<T>::prepare(matrix);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        for (int i = 0; i < number_of_tests; i++) 
        {
            kernel<T>::main(matrix, 2.412f, 3.14f);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        kernel<T>::finalize(matrix);

        //delete[] matrix_data;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        auto time_ms  = static_cast<double>(duration.count()) / 1000.0;

        return time_ms / static_cast<double>(number_of_tests);
    }

    benchmark<T, kernel>& run(size_t size, size_t number_of_tests)
    {
        std::cout << "Run for ";

        if constexpr (std::is_same<T, float>::value)
        {
            std::cout << "FLOAT ";
        }

        if constexpr (std::is_same<T, double>::value)
        {
            std::cout << "DOUBLE";
        }

        std::cout << " " << name() << ": " << mark(size, size, number_of_tests) << " ms" << std::endl;

        return *this;
    }

    template<class Tfunc>
    void export_data(Tfunc&& func) const
    {
        func(matrix[2]);
    }
};

template<class T>
void compare_matrix_result(const raw_memory_matrix_view<T>& lhs, const raw_memory_matrix_view<T>& rhs)
{
    assert(lhs.properties().size_x() == rhs.properties().size_x());
    assert(lhs.properties().size_y() == rhs.properties().size_y());
    assert(lhs.properties().size_z() == rhs.properties().size_z());
    assert(lhs.properties().size_l() == rhs.properties().size_l());

    double max_diff = -1.0;

    lhs.for_each([&](size_t x, size_t y, size_t z, size_t l) {

        auto lv = lhs.at(x, y, z, l);
        auto rv = rhs.at(x, y, z, l);

        auto diff = std::abs(lv - rv);

        if (diff > max_diff)
        {
            max_diff = diff;
        }

        });

    std::cout << "Max absolute difference: " << max_diff << std::endl;
}