#ifndef _PARSER
#define _PARSER

#include <string_view>
#include <optional>
#include <cstddef>
#include <cstdint>

struct Parser
{
    std::string_view source;
    size_t offset;

    Parser(std::string_view source);

    std::optional<uint8_t> peek() const;
    std::optional<uint8_t> consume();

    bool test(uint8_t c);
    bool eat(uint8_t c);
    bool expect(uint8_t c);
    bool eat_delim(bool eat_newlines = true);

    std::string_view word(); // [a-zA-Z_][a-zA-Z0-9]*

    void skip_until(uint8_t end);

    bool is_word_start_char(uint8_t c) const;
    bool is_word_continue_char(uint8_t c) const;
};

#endif