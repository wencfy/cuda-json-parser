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
    if (auto maybe_input = read_input("test.json"))
    {
        input = std::move(maybe_input.value());
    }

    std::cout << input << std::endl;

    if (lexer.has_value()) {
        lexer->parallel_lexer.dump_sizes(std::cout);

        auto initial_states = lexer->parallel_lexer.initial_states;
        // lexer->parallel_lexer.merge_table;
        auto final_states = lexer->parallel_lexer.final_states;
        
        std::vector<const lexer::ParallelLexer::Transition *> trans(input.length());
        for (int i = 0; i < trans.size(); i++) {
            trans[i] = &initial_states[input[i]];
        }


        std::vector<const lexer::ParallelLexer::Transition *> prefix(input.length() + 1);
        std::vector<const lexer::Lexeme *> res(input.length());
        prefix[0] = trans[0];
        for (int i = 1; i < trans.size(); i++) {
            prefix[i] = &lexer->parallel_lexer.merge_table(prefix[i - 1]->result_state, trans[i]->result_state);
        }

        prefix[trans.size()] = &lexer->parallel_lexer.merge_table(prefix[trans.size() - 1]->result_state, 0);

        for (auto a: prefix) {
            std::cout << a->produces_lexeme << " " << a->result_state << std::endl;
        }
        std::cout << final_states[5]->name << std::endl;
        std::cout << final_states[24]->name << std::endl;
        std::cout << final_states[32]->name << std::endl;
    }
}