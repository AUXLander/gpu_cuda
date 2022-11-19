cmake_minimum_required(VERSION 3.14)
set(CMAKE_CUDA_STANDARD 17)
project(hello_cuda CUDA)

add_executable(hello_cuda kernel.cu)