#include <vector>

#include "lexer/interpreter.hpp"
#include "lexer/lexical_grammar.hpp"

namespace lexer
{
    LexerInterpreter::LexerInterpreter(const ParallelLexer *lexer) : lexer(lexer) {}

    void LexerInterpreter::lex_linear(std::string_view input) const
    {
        auto states = std::vector<ParallelLexer::StateIndex>();

        for (auto c : input)
        {
            auto state = this->lexer->initial_states[c];
            states.push_back(state.result_state);
            if (state.produces_lexeme)
            {
                auto t = this->lexer->final_states[ParallelLexer::START];
                printf("%s{}\n", t ? t->name.c_str() : "(internal error)");
            }
        }

        for (size_t i = 1; i < input.size(); ++i)
        {
            auto prev = states[i - 1];
            auto state = this->lexer->merge_table(states[i - 1], states[i]);
            states[i] = state.result_state;
            if (state.produces_lexeme)
            {
                auto t = this->lexer->final_states[prev];
                printf("%s{}\n", t ? t->name.c_str() : "(internal error)");
            }
        }

        auto t = this->lexer->final_states[states.back()];
        printf("%s{}\n", t ? t->name.c_str() : "(input error)");
    }
}
