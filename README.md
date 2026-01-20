# CUDA Sudoku Solver

A high-performance implementation of a Sudoku solver using CUDA. This project utilizes a Breadth-First Search (BFS) approach on the GPU and compares it against a recursive backtracking algorithm on the CPU.

### To skip CPU Sudoku implementation, use flag --no-cpu

## Prerequisites
* **NVIDIA GPU** with CUDA support.
* **CUDA Toolkit** (installed and configured in your PATH).
* **C++ Compiler** with C++17 support
* **CMake** version 4.0 or higher.

## Setup and Build

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/7blak/cuda-sudoku-solver
    cd cuda-sudoku-solver
    ```

2.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```

3.  **Generate build files with CMake:**
    ```bash
    cmake ..
    ```

4.  **Compile the project:**
    ```bash
    make
    ```

## Execution

Run the executable generated in the build directory:

```bash
./cuda_sudoku_solver
```

## Configuration
To change the input board, modify the config.txt file located in the build directory
(copied from root during configuration). The file should contain 81 digits representing the Sudoku board,
with 0 denoting empty cells - 9 digits per line and 9 rows. Some example data is included in /input.