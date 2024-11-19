#ifndef _LEXER_FSA
#define _LEXER_FSA

#include <vector>
#include <iosfwd>
#include <optional>
#include <limits>
#include <cstddef>
#include <cstdint>

namespace lexer {
    struct Lexeme;
    struct LexicalGrammar;

    struct FiniteStateAutomaton {
        using Symbol = uint8_t;
        using StateIndex = size_t;

        static constexpr const size_t MAX_SYM = std::numeric_limits<Symbol>::max();

        static constexpr const StateIndex REJECT = 0;
        static constexpr const StateIndex START = 1;

        struct Transition {
            /**
             * Symbol of Transition
             */
            std::optional<uint8_t> maybe_sym;
            StateIndex dst;
            bool produces_lexeme;
        };

        struct State {
            const Lexeme* lexeme;
            std::vector<Transition> transitions;
        };

        std::vector<State> states;

        FiniteStateAutomaton();

        size_t num_states() const;

        StateIndex add_state();

        void add_transition(StateIndex src, StateIndex dst, std::optional<uint8_t> sym, bool produces_lexeme = false);
        void add_epsilon_transition(StateIndex src, StateIndex dst, bool produces_lexeme = false);

        std::optional<StateIndex> find_first_transition_dst(StateIndex src, std::optional<uint8_t> sym) const;

        State& operator[](StateIndex state);
        const State& operator[](StateIndex state) const;

        void dump_dot(std::ostream& os) const;

        void to_dfa(const LexicalGrammar* g, FiniteStateAutomaton& dfa, StateIndex nfa_start, StateIndex dfa_start) const;

        static FiniteStateAutomaton build_lexer_dfa(const LexicalGrammar* g);
    };
}

#endif