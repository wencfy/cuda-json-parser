#ifndef _LEXER_INTERPRETER
#define _LEXER_INTERPRETER

#include <string_view>

#include "lexer/parallel_lexer.hpp"

namespace lexer
{
    struct LexerInterpreter
    {
        const ParallelLexer *lexer;

        LexerInterpreter(const ParallelLexer *lexer);

        void lex_linear(std::string_view input) const;
    };
}

#endif