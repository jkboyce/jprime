//
// Graph.h
//
// Data structures related to the juggling graph for B objects, max throw H.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_GRAPH_H_
#define JPRIME_GRAPH_H_

#include "State.h"

#include <string>
#include <vector>
#include <tuple>
#include <cstdint>
#include <stdexcept>
#include <format>


class Graph {
 public:
  Graph(unsigned b, unsigned h);
  Graph(unsigned b, unsigned h, const std::vector<bool>& xa, unsigned n = 0);
  Graph() = default;

  unsigned b = 0;  // number of objects
  unsigned h = 0;  // maximum throw value
  unsigned n = 0;  // if nonzero then single-period graph
  std::vector<bool> xarray;  // throw values to exclude

  // information about states
  unsigned numstates = 0;
  std::vector<State> state;
  std::vector<unsigned> cyclenum;  // indexed by state number
  std::vector<unsigned> max_startstate_usable;

  // information about shift cycles
  unsigned numcycles = 0;
  unsigned numshortcycles = 0;
  std::vector<unsigned> cycleperiod;  // indexed by cycle number

  // graph transition matrices
  unsigned maxoutdegree = 0;
  std::vector<unsigned> outdegree;
  std::vector<std::vector<unsigned>> outmatrix;
  std::vector<std::vector<unsigned>> outthrowval;

 private:
  void initialize();
  static void gen_states_all(std::vector<State>& s, unsigned b, unsigned h);
  static void gen_states_for_period(std::vector<State>& s, unsigned b,
    unsigned h, unsigned );
  unsigned find_shift_cycles();
  void build_graph_matrix();
  void find_max_startstate_usable();
  void update_usable_states(std::vector<bool>& state_usable) const;

 public:
  void validate_graph();
  std::vector<int> get_exit_cycles(unsigned start_state) const;
  std::tuple<std::vector<std::vector<unsigned>>,
    std::vector<std::vector<unsigned>>> get_exclude_states() const;
  unsigned prime_period_bound(unsigned start_state) const;
  unsigned superprime_period_bound(unsigned start_state, unsigned shifts = -1U)
    const;
  unsigned get_statenum(const State& s) const;
  unsigned advance_state(unsigned statenum, unsigned throwval) const;
  unsigned reverse_state(unsigned statenum) const;
  unsigned downstream_state(unsigned statenum) const;
  unsigned upstream_state(unsigned statenum) const;
  std::string state_string(unsigned statenum) const;
  std::string to_string() const;

  constexpr static std::uint64_t combinations(unsigned a, unsigned b);
  constexpr static std::uint64_t shift_cycle_count(unsigned b, unsigned h,
    unsigned p);
  constexpr static std::uint64_t ordered_partitions(unsigned b, unsigned h,
    unsigned n);
};

std::ostream& operator<<(std::ostream& ost, const Graph& g);

//------------------------------------------------------------------------------
// Static methods for calculating graph size
//------------------------------------------------------------------------------

// Compute (a choose b).
//
// The number of states (vertices) in juggling graph (b,h) is (h choose b).
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message.

constexpr std::uint64_t Graph::combinations(unsigned a, unsigned b)
{
  if (a < b) {
    return 0;
  }

  std::uint64_t result = 1;
  constexpr auto MAX_UINT64 = std::numeric_limits<std::uint64_t>::max();

  for (unsigned denom = 1; denom <= std::min(b, a - b); ++denom) {
    if ((a - denom + 1) > MAX_UINT64 / result) {
      throw std::overflow_error(
          std::format("Overflow in combinations({},{})", a, b));
    }
    result = (result * (a - denom + 1)) / denom;
  }
  return result;
}

// Compute the number of shift cycles with `b` objects, max throw `h`, with
// exact period `p`.
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message.

constexpr std::uint64_t Graph::shift_cycle_count(unsigned b, unsigned h,
    unsigned p)
{
  if (h % p != 0) {
    return 0;
  }
  if (b % (h / p) != 0) {
    return 0;
  }
  if (p < h) {
    return shift_cycle_count(b * p / h, p, p);
  }

  std::uint64_t val = combinations(h, b);
  for (unsigned p2 = 1; p2 <= h / 2; ++p2) {
    val -= p2 * shift_cycle_count(b, h, p2);
  }
  return (val / h);
}

// Compute the number of states for a single-period graph.
//
// Not every state in graph (b,h) can be part of a pattern of period `n`. In
// particular we have the restriction that within the state, position[i] >=
// position[i + n] for all i.
//
// To count states, we partition each state into `n` slots, where slot `i` is
// associated with positions i, i + n, i + 2*n, ... in the state, up to a
// maximum of h - 1. The only degree of freedom is how many objects to put into
// each slot; the state positions must be filled from the bottom up in order to
// be part of a period `n` pattern.
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message.

constexpr std::uint64_t Graph::ordered_partitions(unsigned b, unsigned h,
    unsigned n)
{
  if (n == 0) {
    return 0;
  }
  std::vector<std::uint64_t> options((b + 1) * n, 0);
  constexpr auto MAX_UINT64 = std::numeric_limits<std::uint64_t>::max();

  // calculate the number of ways to fill slots `pos` and higher using `left`
  // balls, working backward from the end
  for (unsigned pos = n; pos-- > 0; ) {
    // upper bound on number of balls in slot `pos`
    unsigned max_fill = 0;
    for (unsigned i = pos; i < h; i += n) {
      ++max_fill;
    }

    for (unsigned left = (pos == 0 ? b : 0); left <= b; ++left) {
      const unsigned index = pos + left * n;  // index into array
      if (pos == n - 1) {
        options.at(index) = (left <= max_fill ? 1 : 0);  // last slot
      } else {
        for (unsigned i = 0; i <= max_fill && i <= left; ++i) {
          // put `i` balls into slot `pos`
          const unsigned index2 = (pos + 1) + (left - i) * n;
          if (options.at(index2) > MAX_UINT64 - options.at(index)) {
            throw std::overflow_error(
              std::format("Overflow in ordered_partitions({},{},{})", b, h, n));
          }
          options.at(index) += options.at(index2);
        }
      }
    }
  }

  return options.at(b * n);  // pos = 0, left = b
}

#endif
