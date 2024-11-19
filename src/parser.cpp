#include "parser.hpp"

#include <cctype>
#include <cstdlib>
#include <cassert>
#include <cstdio>

Parser::Parser(std::string_view source) : source(source), offset(0) {}

std::optional<uint8_t> Parser::peek() const
{
    if (this->offset < this->source.size())
        return this->source[this->offset];
    return std::nullopt;
}

std::optional<uint8_t> Parser::consume()
{
    auto maybe_next = this->peek();
    if (maybe_next.has_value())
        ++this->offset;
    return maybe_next;
}

bool Parser::test(uint8_t c)
{
    if (auto next = this->peek())
        return next == c;
    return false;
}

bool Parser::eat(uint8_t c)
{
    if (this->test(c))
    {
        this->consume();
        return true;
    }
    return false;
}

bool Parser::expect(uint8_t c)
{
    if (!this->eat(c))
    {
        if (auto actual = this->peek())
        {
            printf("Unexpected character '%d', expected '%d'\n", actual.value(), c);
        }
        else
        {
            printf("Unexpected EOF, expected '%d'\n", c);
        }
        return false;
    }
    return true;
}

bool Parser::eat_delim(bool eat_newlines)
{
    // Eat any delimiter, such as whitespace and comments
    bool delimited = false;

    while (true)
    {
        auto c = this->peek();
        if (!c.has_value())
            return delimited;
        switch (c.value())
        {
        case ' ':
        case '\t':
        case '\r':
            this->consume();
            break;
        case '#':
            while (!this->test('\n'))
            {
                this->consume();
            }
            break;
        case '\n':
            if (eat_newlines)
            {
                this->consume();
                break;
            }
        default:
            return delimited;
        }
        delimited = true;
    }
}

std::string_view Parser::word()
{
    size_t start = this->offset;

    if (auto c = this->peek())
    {
        if (!this->is_word_start_char(c.value()))
        {
            printf("Invalid character '{}', expected <word>\n");
            return "";
        }
    }
    else
    {
        printf("Unexpected EOF, expected <word>\n");
        return "";
    }

    this->consume();

    while (auto c = this->peek())
    {
        if (!this->is_word_continue_char(c.value()))
            break;
        this->consume();
    }

    return this->source.substr(start, this->offset - start);
}

void Parser::skip_until(uint8_t end)
{
    while (true)
    {
        this->eat_delim(); // make sure to skip comments
        auto maybe_next = this->consume();
        if (!maybe_next.has_value() || maybe_next == end)
            break;
    }
}

bool Parser::is_word_start_char(uint8_t c) const
{
    return std::isalpha(c) || c == '_';
}

bool Parser::is_word_continue_char(uint8_t c) const
{
    return std::isalnum(c) || c == '_';
}
