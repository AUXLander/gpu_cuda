cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
project(kernel_omp CXX)

find_package(OpenMP)

add_library(kernel_omp STATIC kernel.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kernel_omp PUBLIC OpenMP::OpenMP_CXX)
endif()