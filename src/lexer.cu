#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include "lexer/lexer_parser.hpp"
#include "lexer/parallel_lexer.hpp"

__global__ void map_trans(
    lexer::ParallelLexer::Transition *trans,
    lexer::ParallelLexer::Transition *initial_states,
    char *input,
    int input_length
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < input_length) {
        trans[idx] = initial_states[input[idx]];
        std::cout << idx << "---" << trans[idx] << endl;
    }
}

class CudaLexer
{
    lexer::ParallelLexer lexer;
    std::string input;

    lexer::ParallelLexer::Transition *d_trans;
    lexer::ParallelLexer::Transition *d_prefix;

    void map_trans()
    {
        auto initial_states = lexer.initial_states;
        char *d_input;
        lexer::ParallelLexer::Transition *d_initial_states;
        size_t d_initial_states_size = initial_states.size() * sizeof(lexer::ParallelLexer::Transition);
        size_t d_trans_size = input.length() * sizeof(lexer::ParallelLexer::Transition);
        size_t d_input_size = input.length() * sizeof(char);
        cudaMalloc(&d_initial_states, d_initial_states);
        cudaMalloc(&d_trans, d_trans_size);
        cudaMalloc(&d_input, d_input_size);

        cudaMemcpy(d_input, input.c_str(), d_input_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_initial_states, initial_states.data(), d_initial_states_size, cudaMemcpyHostToDevice);

        dim3 block_size(256);
        dim3 num_blocks(1200);
        map_trans<<<num_blocks, block_size>>>(d_trans, d_initial_states, d_input, input.length());

        cudaFree(d_input);
        cudaFree(d_initial_states);
    }

    void compute_prefix()
    {
    }

    void copy_results() {
        
    }

public:
    CudaLexer(lexer::ParallelLexer lexer, std::string input) : lexer(lexer), input(input) {}

    void run_lexer() {
        map_trans();

        copy_results();
    }
}
