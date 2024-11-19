#ifndef _LEXER_INTERPRETER
#define _LEXER_INTERPRETER

#include <string_view>
#include <unordered_map>

#include "lexer/parallel_lexer.hpp"

namespace lexer
{
    struct LexerInterpreter
    {
        const ParallelLexer *lexer;

        std::unordered_map<const lexer::Lexeme *, int> mp;

        LexerInterpreter(const ParallelLexer *lexer);

        void lex_linear(std::string_view input);

        void add_token(const lexer::Lexeme *t);

        void print_token_table();
    };
}

#endif