#ifndef _HASH_UTIL
#define _HASH_UTIL

#include <functional>
#include <cstddef>
#include <cassert>
#include <bit>

constexpr size_t hash_combine(size_t lhs, size_t rhs)
{
    // Ye olde boost hash combine
    return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2));
}

template <typename It, typename Hasher>
size_t hash_range(It begin, It end, Hasher hasher)
{
    size_t hash = 0;
    while (begin != end)
    {
        hash = hash_combine(hash, hasher(*begin++));
    }
    return hash;
}

template <typename It, typename Hasher>
size_t hash_order_independent_range(It begin, It end, Hasher hasher)
{
    size_t hash = 0;
    while (begin != end)
    {
        hash ^= hasher(*begin++);
    }
    return hash;
}

template <typename T>
constexpr T int_bit_width(T x)
{
    auto width = std::max(T{8}, std::bit_ceil(static_cast<T>(std::bit_width(x))));
    // This should only happen if there are a LOT of rules anyway.
    assert(width <= 64);
    return width;
}

#endif