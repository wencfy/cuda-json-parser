#include <vector>
#include <time.h>

#include "lexer/interpreter.hpp"
#include "lexer/lexical_grammar.hpp"

namespace lexer
{
    LexerInterpreter::LexerInterpreter(const ParallelLexer *lexer) : lexer(lexer) {}

    void LexerInterpreter::lex_linear(std::string_view input)
    {
        clock_t start = clock();
        auto states = std::vector<ParallelLexer::StateIndex>();

        for (auto c : input)
        {
            auto state = this->lexer->initial_states[c];
            states.push_back(state.result_state);
            if (state.produces_lexeme)
            {
                auto t = this->lexer->final_states[ParallelLexer::START];
                // printf("%s\n", t ? t->name.c_str() : "(internal error)");
                add_token(t);
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
                // printf("%s\n", t ? t->name.c_str() : "(internal error)");
                add_token(t);
            }
        }

        auto t = this->lexer->final_states[states.back()];
        // printf("%s\n", t ? t->name.c_str() : "(input error)");
        add_token(t);

        clock_t end = clock();

        printf("CPU Running Time: %lf s\n", ((double)(end - start))/ CLOCKS_PER_SEC);
        print_token_table();
    }

    void LexerInterpreter::add_token(const lexer::Lexeme *t) {
        if (t) {
            if (mp.find(t) != mp.end()) {
                mp[t]++;
            } else {
                mp[t] = 1;
            }
        } else {
            printf("%s\n", "(internal error)");
        }
    }

    void LexerInterpreter::print_token_table() {
        printf("lexeme\t\tcount\n");
        for (auto p: mp) {
            printf("%-20s\t%5d\n", p.first->name.c_str(), p.second);
        }
    }
}
