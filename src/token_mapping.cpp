#include "token_mapping.hpp"
#include "hash_util.hpp"

#include <algorithm>
#include <string_view>
#include <vector>
#include <utility>
#include <ostream>

const Token Token::INVALID = {Type::INVALID, "invalid"};
const Token Token::START_OF_INPUT = {Type::START_OF_INPUT, "soi"};
const Token Token::END_OF_INPUT = {Type::END_OF_INPUT, "eoi"};

size_t Token::Hash::operator()(const Token &token) const
{
    return hash_combine(std::hash<Type>{}(token.type), std::hash<std::string>{}(token.name));
};

bool operator==(const Token &rhs, const Token &lhs)
{
    Token::Hash hasher;
    return hasher(lhs) == hasher(rhs);
    // return rhs.type == lhs.type && rhs.name == rhs.name;
}

void TokenMapping::insert(const Token &token)
{
    auto result = this->tokens.insert({token, this->num_tokens()});
    if (!result.second)
    {
        printf("Insertion failed, element with key %s already exists.\n", token.name.c_str());
    }
}

bool TokenMapping::contains(const Token &token) const
{
    return this->tokens.find(token) != this->tokens.end();
}

size_t TokenMapping::backing_type_bits() const
{
    return int_bit_width(this->tokens.size() - 1);
}

size_t TokenMapping::token_id(const Token &token) const
{
    return this->tokens.at(token);
}

size_t TokenMapping::num_tokens() const
{
    return this->tokens.size();
}

void TokenMapping::print_tokens() const
{
    for (auto t : tokens)
    {
        printf("%s\n", t.first.name.c_str());
    }
}
