#include "lexer/lexical_grammar.hpp"

#include <cassert>
#include <iostream>

namespace lexer {
    Token Lexeme::as_token() const {
        return {Token::Type::USER_DEFINED, this->name};
    }

    size_t LexicalGrammar::lexeme_id(const Lexeme* lexeme) const {
        assert(lexeme >= this->lexemes.data() && lexeme < &this->lexemes.data()[this->lexemes.size()]);
        return lexeme - this->lexemes.data();
    }

    void LexicalGrammar::add_tokens(TokenMapping& tm) const {
        tm.insert(Token::INVALID);
        for (const auto& lexeme : this->lexemes) {
            tm.insert(lexeme.as_token());
        }
    }

    void LexicalGrammar::validate() const {
        // Empty tokens will mess up the lexer, so check here that there are none.
        // Checking here will allow us to catch all of them at once.
        bool error = false;

        for (const auto& lexeme : this->lexemes) {
            printf("\n%s: ", lexeme.name.c_str());
            lexeme.regex->print(std::cout);
            if (!lexeme.regex->matches_empty())
                continue;

            printf("Lexeme matches the empty string\n");
            error = true;
        }

        if (error)
            throw LexemeMatchesEmptyError();
    }
}
