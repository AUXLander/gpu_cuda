cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
project(kernel_cpu CXX)

add_library(kernel_cpu STATIC kernel.cpp)