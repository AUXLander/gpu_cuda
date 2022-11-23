#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

template<class T, template<class> class kernel>
class benchmark : public kernel<T>
{
    constexpr static size_t N = 100;

    std::vector<T> x;
    std::vector<T> y;

    void fill_container(std::vector<T>& container, size_t size)
    {
        container.resize(size);

        std::generate(container.begin(), container.end(), []() {
            return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        });
    }

public:

    std::string name() const
    {
        return kernel<T>::name();
    }

    double mark(size_t size, size_t number_of_tests)
    {
        T a = rand();

        fill_container(x, size);
        fill_container(y, size);

        kernel<T>::prepare(size, a, x.data(), 1, y.data(), 1);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        for (int i = 0; i < number_of_tests; i++) 
        {
            kernel<T>::main(size, a, x.data(), 1, y.data(), 1);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        kernel<T>::finalize(size, a, x.data(), 1, y.data(), 1);

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        auto time_ms  = static_cast<double>(duration.count()) / 1000.0;

        return time_ms / static_cast<double>(number_of_tests);
    }

    void run(size_t size, size_t number_of_tests)
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

        std::cout << " " << name() << ": " << mark(size, number_of_tests) << " ms" << std::endl;
    }
};
