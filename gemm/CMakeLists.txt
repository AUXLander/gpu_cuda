cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)
project(gemm CUDA)

add_executable(gemm main.cpp)

add_subdirectory(cpu)
add_subdirectory(omp)
add_subdirectory(cuda)

target_link_libraries(gemm
                      PRIVATE kernel_cpu
                      PRIVATE kernel_omp
                      PRIVATE kernel_cuda
                      )

set_target_properties(gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
