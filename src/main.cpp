#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "sudoku.cuh"
#include "sudoku_cpu.h"

bool loadBoard(const std::string &filename, unsigned char *h_board) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    char c;
    int index = 0;
    while (file >> c && index < 81) {
        if (c >= '0' && c <= '9') {
            h_board[index++] = static_cast<unsigned char>(c - '0');
        }
    }
    file.close();

    if (index != 81) {
        std::cerr << "Error: File did not contain 81 digits (found " << index << ")" << std::endl;
        return false;
    }
    return true;
}

void printBoard(std::vector<unsigned char> &h_board) {
    for (int i = 0; i < 81; i++) {
        if (i % 9 == 0 && i != 0)
            std::cout << std::endl;
        std::cout << static_cast<int>(h_board[i]) << " ";
    }
}

int main(int argc, char **argv) {
    bool runCPU = true;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--no-cpu") {
            runCPU = false;
            std::cout << "Flag --no-cpu detected: Skipping CPU implementation." << std::endl;
        }
    }

    SudokuConfig config;
    config.max_boards = 5000000;
    config.board_stride = 81;
    config.input_file = "config.txt";


    std::vector<unsigned char> h_board(config.board_stride);

    if (!loadBoard(config.input_file, h_board.data())) {
        return -1;
    }

    std::vector<unsigned char> h_board_cpu = h_board;

    std::cout << "Initial Board Loaded." << std::endl;
    printBoard(h_board);
    std::cout << "\n------------------------\n";

    std::cout << ">>>>>> Starting CUDA Sudoku Solver..." << std::endl;
    solveSudokuCUDA(h_board.data(), config);
    std::cout << ">>>>>> CUDA Solver Finished." << std::endl << std::endl;

    if (runCPU) {
        std::cout << ">>>>>> Starting CPU Sudoku Solver..." << std::endl;
        solveSudokuCPU(h_board_cpu.data(), config);
        std::cout << ">>>>>> CPU Solver Finished." << std::endl;
    }

    std::cout << "\n--- Final Result ---" << std::endl;
    printBoard(h_board);
    std::cout << std::endl;

    return 0;
}
