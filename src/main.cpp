#include <iostream>
#include <fstream>
#include <string_view>
#include <string>
#include <iterator>
#include <stdexcept>
#include <optional>
#include <cstdlib>
#include <cassert>

#include "parser.hpp"
#include "token_mapping.hpp"
#include "lexer/lexer_parser.hpp"
#include "lexer/parallel_lexer.hpp"
#include "lexer/interpreter.hpp"
#include "lexer.cuh"

std::optional<std::string> read_input(const char *filename)
{
    auto in = std::ifstream(filename, std::ios::binary);
    if (!in)
    {
        printf("Error: Failed to open input file '%s'\n", filename);
        return std::nullopt;
    }

    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct LexerGeneration
{
    lexer::LexicalGrammar grammar;
    lexer::ParallelLexer parallel_lexer;
};

std::optional<LexerGeneration> generate_lexer(TokenMapping &tm, const char *lexer_src)
{
    std::string input;
    if (auto maybe_input = read_input(lexer_src))
    {
        input = std::move(maybe_input.value());
    }
    else
    {
        return std::nullopt;
    }

    try
    {
        auto parser = Parser(input);
        auto lexer_parser = lexer::LexerParser(&parser);

        auto g = lexer_parser.parse();
        g.validate();

        auto parallel_lexer = lexer::ParallelLexer(&g);

        g.add_tokens(tm);

        return {{std::move(g), std::move(parallel_lexer)}};
    }
    catch (const std::runtime_error &e)
    {
        printf("Failed to generate lexer: %s\n", e.what());
        return std::nullopt;
    }
}

int main() {
    auto tm = TokenMapping();

    char lexer_src[] = "json.lex";

    auto lexer = generate_lexer(tm, lexer_src);

    std::string input;

    if (lexer.has_value()) {
        lexer->parallel_lexer.dump_sizes(std::cout);

        auto cuda_lexer = new CudaLexer(lexer->parallel_lexer);
        lexer::LexerInterpreter *interpreter = new lexer::LexerInterpreter(&lexer->parallel_lexer);
        
        while (true) {
            printf("Please input your filename: ");
            char filename[256];
            scanf("%255s", filename);

            if (auto maybe_input = read_input(filename)) {
                input = std::move(maybe_input.value());
                printf("--------------------------------------------------\n");
                printf("Lexing %s (%fkb) using cuda and cpu\n", filename, input.length() / 1024.0);
                printf("--------------------------------------------------\n");

                cuda_lexer->lex_cuda(input);
                interpreter->lex_linear(input);
            }
        }
    }
}