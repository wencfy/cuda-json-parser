#ifndef _LEXER_LEXER_PARSER
#define _LEXER_LEXER_PARSER

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <string_view>
#include <cstddef>

#include "parser.hpp"
#include "lexer/lexical_grammar.hpp"
#include "lexer/regex.hpp"

namespace lexer {
    struct LexerParseError: std::runtime_error {
        LexerParseError(): std::runtime_error("Parse error") {}
    };

    class LexerParser {
        struct LexemeDefinition {
            size_t index;
            std::unordered_set<std::string_view> preceded_by;
        };

        Parser* parser;

        std::vector<Lexeme> lexemes;
        std::unordered_map<std::string_view, LexemeDefinition> lexeme_definitions;

    public:
        LexerParser(Parser* parser);
        LexicalGrammar parse();

    private:
        bool lexeme_decl();
        bool precede_list(std::unordered_set<std::string_view>& preceded_by);
        bool insert_precedes();
    };
}

#endif