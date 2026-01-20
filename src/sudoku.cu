#include "sudoku.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// Macro for checking CUDA errors
#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

typedef unsigned char u8;

// --- DEVICE HELPER FUNCTIONS ---

__device__ __forceinline__ u8 getCell(const u8 *d_boards, int board_idx, int cell_idx, int stride) {
    return d_boards[cell_idx * stride + board_idx];
}

__device__ __forceinline__ void setCell(u8 *d_boards, int board_idx, int cell_idx, int stride, u8 val) {
    d_boards[cell_idx * stride + board_idx] = val;
}

// Returns a bitmask of valid numbers for a specific cell index (0-80).
// Bit 0 = Number 1, Bit 1 = Number 2 ... Bit 8 = Number 9.
// 1 = Valid Candidate, 0 = Invalid (already present in row/col/box)
__device__ int getCandidateMask(const u8 *d_boards, int board_idx, int cell_idx, int stride) {
    int row = cell_idx / 9;
    int col = cell_idx % 9;
    int box_start_row = row / 3 * 3;
    int box_start_col = col / 3 * 3;

    // Start with all 9 bits set (binary 111111111 = 0x1FF)
    int valid_mask = 0x1FF;

    // Check Row
    int row_start = row * 9;
    for (int c = 0; c < 9; c++) {
        u8 val = getCell(d_boards, board_idx, row_start + c, stride);
        // Ex. If val is 5, we shift 1 by 4 (5-1) and invert to clear the 5th bit
        if (val != 0)
            valid_mask &= ~(1 << (val - 1));
    }

    // Check Column
    for (int r = 0; r < 9; r++) {
        u8 val = getCell(d_boards, board_idx, r * 9 + col, stride);
        if (val != 0)
            valid_mask &= ~(1 << (val - 1));
    }

    // Check 3x3 Box
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            u8 val = getCell(d_boards, board_idx, (box_start_row + r) * 9 + (box_start_col + c), stride);
            if (val != 0)
                valid_mask &= ~(1 << (val - 1));
        }
    }

    return valid_mask;
}

// --- SUDOKU SOLVER PART 1: BOARD ANALYSIS ---

// For each board, find the empty cell with the FEWEST possible candidates.
__global__ void analysisKernel(u8 *d_boards, int *d_candidate_counts, int *d_best_indices, int num_boards, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boards)
        return;

    // Initialize with a value larger than max candidates (9)
    int min_candidates = 100;
    int best_cell = -1;
    bool solved_or_full = true;

    // Linear scan of the board to find the best cell to guess
    for (int i = 0; i < 81; i++) {
        u8 val = getCell(d_boards, idx, i, stride);
        if (val == 0) {
            solved_or_full = false;

            int mask = getCandidateMask(d_boards, idx, i, stride);
            int count = __popc(mask);

            // If a cell has 0 options, this board is a dead end.
            if (count == 0) {
                min_candidates = 0;
                best_cell = -1;
                break;
            }

            if (count < min_candidates) {
                min_candidates = count;
                best_cell = i;

                // If we find a cell with only 1 option, that's the best move. Stop searching.
                if (count == 1)
                    break;
            }
        }
    }

    // If board is full (no 0s), it doesn't generate children here.
    // The validation step handles detecting the solution.
    if (solved_or_full)
        min_candidates = 0;

    d_candidate_counts[idx] = min_candidates;
    d_best_indices[idx] = best_cell;
}

// --- SUDOKU SOLVER PART 2: EXPANSION ---

// Reads 'num_parents' boards -> Uses 'd_offsets' to determine where to write the new children ->
// Copies the parent board, fills in the guess, and saves to 'd_output'.
__global__ void expansionKernel(u8 *d_input, u8 *d_output, int *d_offsets, int *d_best_indices, int num_parents,
                                int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parents)
        return;

    int count = 0;
    int best_cell = d_best_indices[idx];
    int start_offset = d_offsets[idx];

    // If best_cell is -1 (dead end or solved), we do nothing.
    if (best_cell < 0 || best_cell >= 81)
        return;

    // Re-calculate valid candidates for the chosen cell
    int mask = getCandidateMask(d_input, idx, best_cell, stride);

    // Iterate through possible numbers 1..9
    for (int val = 1; val <= 9; val++) {
        // If bit is set, this is a valid candidate
        if (mask & 1 << (val - 1)) {
            // Calculate destination address
            int global_child_idx = start_offset + count;

            // Copy parent state (manual loop for speed in kernel)
            for (int k = 0; k < 81; k++) {
                u8 p_val = getCell(d_input, idx, k, stride);
                setCell(d_output, global_child_idx, k, stride, p_val);
            }

            // Apply the guess
            setCell(d_output, global_child_idx, best_cell, stride, static_cast<u8>(val));

            count++;
        }
    }
}

// --- SUDOKU SOLVER PART 3: VALIDATION ---

// Checks if any of the boards are fully solved (no zeros).
__global__ void validationKernel(u8 *d_boards, u8 *d_final_solution, bool *d_solved_flag, int num_boards, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boards)
        return;

    if (*d_solved_flag)
        return;

    bool is_full = true;

    // Check for any empty cells
    for (int i = 0; i < 81; i++) {
        if (getCell(d_boards, idx, i, stride) == 0) {
            is_full = false;
            break;
        }
    }

    if (is_full) {
        *d_solved_flag = true;

        // Copy to the final solution buffer
        for (int k = 0; k < 81; k++) {
            d_final_solution[k] = getCell(d_boards, idx, k, stride);
        }
    }
}

// --- HOST WRAPPERS ---

// Calculates offsets using Thrust Exclusive Scan
int performPrefixSum(int *d_candidate_counts, int *d_offsets, int num_boards) {
    thrust::device_ptr<int> t_counts(d_candidate_counts);
    thrust::device_ptr<int> t_offsets(d_offsets);

    // Perform exclusive scan
    // Input: [2, 3, 0, 1] -> Offsets: [0, 2, 5, 5]
    thrust::exclusive_scan(t_counts, t_counts + num_boards, t_offsets);

    // Calculate total sum (last offset + last count)
    int last_count = 0;
    int last_offset = 0;

    CHECK_CUDA(cudaMemcpy(&last_count, d_candidate_counts + num_boards - 1, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&last_offset, d_offsets + num_boards - 1, sizeof(int), cudaMemcpyDeviceToHost));

    return last_offset + last_count;
}

// --- MAIN SOLVER FUNCTION ---

void solveSudokuCUDA(unsigned char *h_board, const SudokuConfig &config) {
    u8 *d_boards_buffer_A, *d_boards_buffer_B, *d_final_solution;
    int *d_candidate_counts, *d_offsets, *d_best_indices;
    bool *d_solved_flag;

    int stride = config.max_boards;

    size_t board_mem_size = config.max_boards * config.board_stride * sizeof(u8);
    size_t int_mem_size = config.max_boards * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_boards_buffer_A, board_mem_size));
    CHECK_CUDA(cudaMalloc(&d_boards_buffer_B, board_mem_size));
    CHECK_CUDA(cudaMalloc(&d_candidate_counts, int_mem_size));
    CHECK_CUDA(cudaMalloc(&d_best_indices, int_mem_size));
    CHECK_CUDA(cudaMalloc(&d_offsets, int_mem_size));
    CHECK_CUDA(cudaMalloc(&d_final_solution, config.board_stride * sizeof(u8)));
    CHECK_CUDA(cudaMalloc(&d_solved_flag, sizeof(bool)));

    int num_active_boards = 1;
    bool h_solved = false;

    CHECK_CUDA(cudaMemset(d_solved_flag, 0, sizeof(bool)));
    CHECK_CUDA(cudaMemcpy2D(
        d_boards_buffer_A,
        stride,
        h_board,
        1,
        1,
        81,
        cudaMemcpyHostToDevice
    ));

    u8 *d_in = d_boards_buffer_A;
    u8 *d_out = d_boards_buffer_B;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- PRE-CHECK: Is the initial board already solved? ---
    validationKernel<<<1, 1>>>(d_in, d_final_solution, d_solved_flag, 1, stride);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(&h_solved, d_solved_flag, sizeof(bool), cudaMemcpyDeviceToHost));

    if (h_solved) {
        std::cout << "Input board was already solved." << std::endl;
    } else {
        std::cout << "Starting BFS Search..." << std::endl;
    }

    // --- MAIN LOOP ---
    int iteration = 0;
    while (num_active_boards > 0 && !h_solved) {
        iteration++;

        // Part 1: Analysis
        int grid_size = (num_active_boards + 255) / 256;

        analysisKernel<<<grid_size, 256>>>(d_in, d_candidate_counts, d_best_indices, num_active_boards, stride);
        CHECK_CUDA(cudaGetLastError());

        // Scan new boards (Prefix Sum)
        int total_new_boards = performPrefixSum(d_candidate_counts, d_offsets, num_active_boards);

        if (total_new_boards == 0)
            break;

        if (total_new_boards > config.max_boards) {
            std::cerr << "!!! Memory Limit Exceeded (" << total_new_boards << " boards). Please try an easier board !!!"
                    << std::endl;
            break;
        }

        // Part 2: Expansion
        expansionKernel<<<grid_size, 256>>>(d_in, d_out, d_offsets, d_best_indices, num_active_boards, stride);
        CHECK_CUDA(cudaGetLastError());

        // Part 3: Validation
        int child_grid_size = (total_new_boards + 255) / 256;
        validationKernel<<<child_grid_size, 256>>>(d_out, d_final_solution, d_solved_flag, total_new_boards, stride);
        CHECK_CUDA(cudaGetLastError());

        CHECK_CUDA(cudaMemcpy(&h_solved, d_solved_flag, sizeof(bool), cudaMemcpyDeviceToHost));
        if (h_solved)
            break;

        // Swap Buffers
        // The children become the new parents
        num_active_boards = total_new_boards;
        std::swap(d_in, d_out);

        if (iteration % 5 == 0 || num_active_boards > 100000) {
            std::cout << "Iter: " << iteration << " | Active Boards: " << num_active_boards << std::endl;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Total Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Total Iterations:     " << iteration << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    if (h_solved) {
        CHECK_CUDA(cudaMemcpy(h_board, d_final_solution, config.board_stride * sizeof(u8), cudaMemcpyDeviceToHost));
    } else {
        std::cout << "No solution found (or memory limit reached)." << std::endl;
    }

    cudaFree(d_boards_buffer_A);
    cudaFree(d_boards_buffer_B);
    cudaFree(d_candidate_counts);
    cudaFree(d_best_indices);
    cudaFree(d_offsets);
    cudaFree(d_final_solution);
    cudaFree(d_solved_flag);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
