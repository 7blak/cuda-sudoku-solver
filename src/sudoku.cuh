#pragma once

#include <string>

struct SudokuConfig {
    int max_boards;
    int board_stride;
    std::string input_file;
};

void solveSudokuCUDA(unsigned char *h_board, const SudokuConfig &config);