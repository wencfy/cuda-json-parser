#ifndef _LEXER_LEXICAL_GRAMMAR
#define _LEXER_LEXICAL_GRAMMAR

#include <string>
#include <stdexcept>

#include "lexer/fsa.hpp"
#include "lexer/regex.hpp"
#include "token_mapping.hpp"

namespace lexer
{
    struct LexemeMatchesEmptyError : std::runtime_error
    {
        LexemeMatchesEmptyError() : std::runtime_error("Lexeme matches the empty string") {}
    };

    struct Lexeme
    {
        std::string name;
        UniqueRegexNode regex;
        std::vector<const Lexeme *> preceded_by;

        Token as_token() const;
    };

    struct LexicalGrammar
    {
        std::vector<Lexeme> lexemes;

        size_t lexeme_id(const Lexeme *lexeme) const;

        void add_tokens(TokenMapping &tm) const;

        void validate() const;
    };
}

#endif
