#ifndef _LEXER_CHAR_RANGE
#define _LEXER_CHAR_RANGE

#include <cstdint>

namespace lexer
{
    struct CharRange
    {
        uint8_t min;
        uint8_t max;

        bool contains(uint8_t c) const;
        bool intersecting_or_adjacent(const CharRange &other) const;
        void merge(const CharRange &other);
    };
}

#endif