#include "sudoku_cpu.h"
#include <iostream>
#include <chrono>

// Helper: Checks if placing 'num' at board[row][col] is valid
// according to Sudoku rules (Row, Column, 3x3 Box).
bool isSafeCPU(const unsigned char *board, int row, int col, unsigned char num) {
    // 1. Check Row
    for (int x = 0; x < 9; x++) {
        if (board[row * 9 + x] == num) {
            return false;
        }
    }

    // 2. Check Column
    for (int x = 0; x < 9; x++) {
        if (board[x * 9 + col] == num) {
            return false;
        }
    }

    // 3. Check 3x3 Box
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[(startRow + i) * 9 + (startCol + j)] == num) {
                return false;
            }
        }
    }

    return true;
}

// Recursive Backtracking Function
bool solveBacktracking(unsigned char *board) {
    int row = -1;
    int col = -1;
    bool isEmpty = false;

    // Find the first empty cell (marked with 0)
    for (int i = 0; i < 81; i++) {
        if (board[i] == 0) {
            row = i / 9;
            col = i % 9;
            isEmpty = true;
            break;
        }
    }

    // If no empty location is found, the board is solved
    if (!isEmpty) {
        return true;
    }

    for (unsigned char num = 1; num <= 9; num++) {
        if (isSafeCPU(board, row, col, num)) {
            // Place the digit
            board[row * 9 + col] = num;

            // Recurse
            if (solveBacktracking(board)) {
                return true;
            }

            // Backtrack
            board[row * 9 + col] = 0;
        }
    }

    return false;
}

void solveSudokuCPU(unsigned char *h_board, const SudokuConfig &config) {
    std::cout << "--- CPU Backtracking Search ---" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    bool solved = solveBacktracking(h_board);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;

    if (solved) {
        std::cout << "CPU Solution Found!" << std::endl;
    } else {
        std::cout << "No solution found by CPU." << std::endl;
    }

    std::cout << "CPU Execution Time:   " << elapsed.count() << " ms" << std::endl;
    std::cout << "-------------------------------" << std::endl;
}