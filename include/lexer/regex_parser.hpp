#include <stdexcept>
#include <cstdint>

#include "parser.hpp"
#include "lexer/regex.hpp"

namespace lexer
{
    struct RegexParseError : std::runtime_error
    {
        RegexParseError() : std::runtime_error("Parse error") {}
    };

    class RegexParser
    {
        Parser *parser;

    public:
        RegexParser(Parser *parser);
        UniqueRegexNode parse();

    private:
        UniqueRegexNode alternation();
        UniqueRegexNode sequence();
        UniqueRegexNode maybe_repeat();
        UniqueRegexNode maybe_atom();
        UniqueRegexNode group();
        uint8_t escaped_char();
    };
}