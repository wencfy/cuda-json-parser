#include <string>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <unordered_map>

#include "lexer.cuh"

__global__ void map_trans_kernel(
    lexer::ParallelLexer::Transition *trans,
    lexer::ParallelLexer::Transition *initial_states,
    char *input,
    size_t input_length,
    size_t N_THREADS
) {
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < input_length; idx += N_THREADS) {
        trans[idx] = initial_states[input[idx]];
        // printf("%c %lu %d\n", input[idx], initial_states[input[idx]].result_state, initial_states[input[idx]].produces_lexeme);
    }
}

__global__ void prefix_step_kernel(
    lexer::ParallelLexer::Transition *start,
    lexer::ParallelLexer::Transition *prefix,
    lexer::ParallelLexer::Transition *merge_table,
    size_t num_states,
    size_t step_size,
    size_t input_length,
    size_t N_THREADS
) {
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx <= input_length; idx += N_THREADS) {
        if (idx < step_size) {
            prefix[idx] = start[idx];
        } else if (idx < input_length) {
            // start[idx - step_size] + start[idx]
            size_t table_index = (start[idx - step_size].result_state) + start[idx].result_state * num_states;
            prefix[idx] = merge_table[table_index];
            // printf("adding %lu: %lu; %lu: %lu, res: %lu\n", idx - step_size, start[idx - step_size].result_state, idx, start[idx].result_state, prefix[idx].result_state);
        } else if (idx == input_length) {
            size_t table_index = (prefix[idx - 1].result_state);
            prefix[idx] = merge_table[table_index];
        }
    }
    if (step_size == input_length && threadIdx.x + blockIdx.x * blockDim.x == 0) {
        start[input_length] = merge_table[start[input_length - 1].result_state];
    }
}

__global__ void extract_final_kernel(
    lexer::ParallelLexer::Transition *trans,
    lexer::Lexeme **final_states,
    lexer::Lexeme **res,
    bool *res_is_token,
    size_t input_length,
    size_t N_THREADS
) {
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx <= input_length; idx += N_THREADS) {
        // printf("%d: %lu, %d\n", idx, trans[idx].result_state, trans[idx].produces_lexeme);
        res[idx] = final_states[trans[idx].result_state];
        res_is_token[idx] = trans[idx].produces_lexeme;
    }
}

void CudaLexer::map_trans()
{
    char *d_input;
    lexer::ParallelLexer::Transition *d_initial_states;
    size_t d_initial_states_size = initial_states.size() * sizeof(lexer::ParallelLexer::Transition);
    size_t d_trans_size = (input.length() + 1) * sizeof(lexer::ParallelLexer::Transition);
    size_t d_input_size = input.length() * sizeof(char);
    cudaMalloc(&d_initial_states, d_initial_states_size);
    cudaMalloc(&d_trans, d_trans_size);
    cudaMalloc(&d_input, d_input_size);

    cudaMemcpy(d_input, input.c_str(), d_input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_initial_states, initial_states.data(), d_initial_states_size, cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 num_blocks(1200);
    size_t N_THREADS = block_size.x * num_blocks.x;
    map_trans_kernel<<<num_blocks, block_size>>>(d_trans, d_initial_states, d_input, input.length(), N_THREADS);

    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_initial_states);
}

void CudaLexer::compute_prefix()
{
    lexer::ParallelLexer::Transition *d_prefix;
    cudaMalloc(&d_prefix, (input.length() + 1) * sizeof(lexer::ParallelLexer::Transition));

    lexer::ParallelLexer::Transition *d_merge_table;
    size_t d_merge_table_size = this->num_states * this->num_states * sizeof(lexer::ParallelLexer::Transition);
    cudaMalloc(&d_merge_table, d_merge_table_size);

    cudaMemcpy(d_merge_table, this->merge_table, d_merge_table_size, cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 num_blocks(1200);
    size_t N_THREADS = block_size.x * num_blocks.x;
    // __global__ void prefix_step_kernel(
    //     lexer::ParallelLexer::Transition *start,
    //     lexer::ParallelLexer::Transition *prefix,
    //     lexer::ParallelLexer::Transition *merge_table,
    //     size_t num_states,
    //     size_t step_size,
    //     size_t input_length,
    //     size_t N_THREADS,
    // )
    for (size_t step_size = 1; step_size < input.length(); step_size <<= 1) {
        prefix_step_kernel<<<num_blocks, block_size>>>(
            d_trans, d_prefix, d_merge_table, num_states, step_size, input.length(), N_THREADS
        );
        cudaDeviceSynchronize();
        auto tmp = d_trans;
        d_trans = d_prefix;
        d_prefix = tmp;
    }

    prefix_step_kernel<<<num_blocks, block_size>>>(
        d_trans, d_prefix, d_merge_table, num_states, input.length(), input.length(), N_THREADS
    );

    cudaFree(d_prefix);
    cudaFree(d_merge_table);
}

void CudaLexer::extract_results()
{
    // __global__ void extract_final_kernel(
    //     lexer::ParallelLexer::Transition *trans,
    //     lexer::Lexeme **final_states,
    //     lexer::Lexeme **res,
    //     bool *res_is_token
    //     unsigned long long *count,
    //     size_t input_length,
    //     size_t N_THREADS
    // )
    lexer::Lexeme **d_final_states;
    lexer::Lexeme **d_res;
    bool *d_res_is_token;
    size_t d_final_states_size = final_states.size() * sizeof(lexer::Lexeme*);
    size_t d_res_size = (input.length() + 1) * sizeof(lexer::Lexeme*);
    size_t d_res_is_token_size = (input.length() + 1) * sizeof(bool);
    cudaMalloc(&d_final_states, d_final_states_size);
    cudaMalloc(&d_res, d_res_size);
    cudaMalloc(&d_res_is_token, d_res_is_token_size);

    cudaMemcpy(d_final_states, final_states.data(), d_final_states_size, cudaMemcpyHostToDevice);

    dim3 block_size(256);
    dim3 num_blocks(1200);
    size_t N_THREADS = block_size.x * num_blocks.x;
    extract_final_kernel<<<num_blocks, block_size>>>(
        d_trans,
        d_final_states,
        d_res,
        d_res_is_token,
        input.length(),
        N_THREADS
    );
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    res = (lexer::Lexeme **) malloc((input.length() + 1) * sizeof(lexer::Lexeme *));
    res_is_token = (bool *) malloc((input.length() + 1) * sizeof(bool));

    cudaMemcpy(res, d_res, input.length() * sizeof(lexer::Lexeme *), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_is_token, d_res_is_token, (input.length() + 1) * sizeof(bool), cudaMemcpyDeviceToHost);

    res_is_token = res_is_token + 1;

    cudaFree(d_final_states);
    cudaFree(d_res);
    cudaFree(d_res_is_token);

    cudaFree(d_trans);
}

CudaLexer::CudaLexer(lexer::ParallelLexer &lexer) {
    this->initial_states = lexer.initial_states;
    this->num_states = lexer.merge_table.states();
    this->merge_table = (lexer::ParallelLexer::Transition*) malloc(this->num_states * this->num_states * sizeof(lexer::ParallelLexer::Transition));
    for (size_t i = 0; i < this->num_states; i++) {
        for (size_t j = 0; j < this->num_states; j++) {
            size_t idx = i + j * this->num_states;
            this->merge_table[idx] = lexer.merge_table(i, j);
        }
    }
    this->final_states = lexer.final_states;
}

void CudaLexer::lex_cuda(std::string input)
{
    this->input = input;
    
    clock_t start = clock();
    map_trans();

    compute_prefix();

    extract_results();
    clock_t end = clock();

    printf("CUDA Running Time: %lf s\n", ((double)(end - start))/ CLOCKS_PER_SEC);

    print_token_table();
}

void CudaLexer::print_token_table() {
    std::unordered_map<lexer::Lexeme *, int> mp;
    for (int i = 0; i < input.length(); i++) {
        if (res_is_token[i]) {
            if (mp.find(res[i]) != mp.end()) {
                mp[res[i]]++;
            } else {
                mp[res[i]] = 1;
            }
        }
    }

    printf("lexeme\t\tcount\n");
    for (auto p: mp) {
        printf("%-20s\t%5d\n", p.first->name.c_str(), p.second);
    }
}
