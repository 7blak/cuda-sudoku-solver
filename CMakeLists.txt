cmake_minimum_required(VERSION 4.0)
project(cuda_sudoku_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

set(HOST_SOURCES src/main.cpp)

set(CUDA_SOURCES
        src/sudoku.cuh
        src/sudoku.cu
        src/sudoku_cpu.h
        src/sudoku_cpu.cpp)

add_executable(cuda_sudoku_solver ${HOST_SOURCES} ${CUDA_SOURCES})

target_include_directories(cuda_sudoku_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}}/src)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.txt ${CMAKE_CURRENT_BINARY_DIR}/config.txt COPYONLY)