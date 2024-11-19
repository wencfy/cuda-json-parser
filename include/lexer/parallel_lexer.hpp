#ifndef _LEXER_PARALLEL_LEXER
#define _LEXER_PARALLEL_LEXER

#include <memory>

#include "lexer/fsa.hpp"

namespace lexer
{
    struct ParallelLexer
    {
        using StateIndex = FiniteStateAutomaton::StateIndex;

        constexpr const static StateIndex REJECT = FiniteStateAutomaton::REJECT;
        constexpr const static StateIndex START = FiniteStateAutomaton::START;

        struct Transition
        {
            StateIndex result_state;
            bool produces_lexeme;

            Transition();
            Transition(StateIndex result_state, bool produces_lexe);
        };

        class MergeTable
        {
            constexpr const static size_t GROW_FACTOR = 2;
            constexpr const static size_t MIN_SIZE = 16;

            size_t num_states;
            size_t capacity;
            std::unique_ptr<Transition[]> merge_table;

        public:
            MergeTable();

            void resize(size_t num_states);

            size_t index(StateIndex first, StateIndex second) const;

            Transition &operator()(StateIndex first, StateIndex second);
            const Transition &operator()(StateIndex first, StateIndex second) const;

            size_t states() const;
        };

        std::vector<Transition> initial_states;

        MergeTable merge_table;

        std::vector<const Lexeme*> final_states;

        StateIndex identity_state_index;

        ParallelLexer(const LexicalGrammar* g);

        void dump_sizes(std::ostream& out) const;
    };
}

#endif