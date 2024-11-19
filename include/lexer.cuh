#ifndef _CUDA_LEXER
#define _CUDA_LEXER

#include <cuda_runtime.h>

#include "lexer/lexer_parser.hpp"
#include "lexer/parallel_lexer.hpp"

__global__ void map_trans_kernel(
    lexer::ParallelLexer::Transition *trans,
    lexer::ParallelLexer::Transition *initial_states,
    char *input,
    size_t input_length,
    size_t N_THREADS
);

__global__ void prefix_step_kernel(
    lexer::ParallelLexer::Transition *start,
    lexer::ParallelLexer::Transition *prefix,
    lexer::ParallelLexer::Transition *merge_table,
    size_t num_states,
    size_t step_size,
    size_t input_length,
    size_t N_THREADS
);

__global__ void extract_final_kernel(
    lexer::ParallelLexer::Transition *trans,
    lexer::Lexeme **final_states,
    lexer::Lexeme **res,
    bool *res_is_token,
    size_t input_length,
    size_t N_THREADS
);

class CudaLexer {
    std::string input;

    std::vector<lexer::ParallelLexer::Transition> initial_states;
    std::vector<const lexer::Lexeme*> final_states;

    lexer::ParallelLexer::Transition *merge_table;
    size_t num_states;

    lexer::ParallelLexer::Transition *d_trans;

    lexer::Lexeme **res;
    bool *res_is_token;

    void map_trans();
    void compute_prefix();

    void extract_results();
    void print_token_table();

public:
    CudaLexer(lexer::ParallelLexer &lexer);
    void lex_cuda(std::string input);
};

#endif