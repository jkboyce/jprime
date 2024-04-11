//
// Graph.cc
//
// Data structures related to the juggling graph for N objects, max throw H.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Graph.h"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <limits>


Graph::Graph(unsigned int n, unsigned int h, const std::vector<bool>& xa,
    bool ltwc, unsigned int l)
    : n(n), h(h), l(l), xarray(xa), linkthrows_within_cycle(ltwc) {
  init();
}

Graph::Graph(unsigned int n, unsigned int h)
    : n(n), h(h), l(0), xarray(h + 1, false), linkthrows_within_cycle(true) {
  init();
}

Graph::Graph(const Graph& g)
    : Graph(g.n, g.h, g.xarray, g.linkthrows_within_cycle, g.l) {
}

Graph& Graph::operator=(const Graph& g) {
  if (this == &g)
    return *this;

  delete_arrays();
  n = g.n;
  h = g.h;
  l = g.l;
  xarray = g.xarray;
  linkthrows_within_cycle = g.linkthrows_within_cycle;
  init();

  return *this;
}

Graph::~Graph() {
  delete_arrays();
}

//------------------------------------------------------------------------------
// Prep core data structures during construction
//------------------------------------------------------------------------------

void Graph::init() {
  // calculate the number of states in the graph; fail if the number of states
  // cannot fit into an unsigned int
  const std::uint64_t num = (l == 0 ? combinations(h, n) :
      ordered_partitions(n, h, l));
  assert(num <= std::numeric_limits<unsigned int>::max());
  numstates = static_cast<unsigned int>(num);

  // enumerate the states
  state.push_back({0, h});  // state at index 0 is unused
  if (l == 0) {
    gen_states_all(state, n, h);
    // states are generated in sorted order
  } else {
    gen_states_for_period(state, n, h, l);
    if (state.size() > 1) {
      std::sort(state.begin() + 1, state.end(), state_compare);
    }
  }
  assert(state.size() == numstates + 1);

  for (size_t i = 0; i <= h; ++i) {
    if (!xarray.at(i)) {
      ++maxoutdegree;
    }
  }
  maxoutdegree = std::min(maxoutdegree, h - n + 1);
  allocate_arrays();

  find_shift_cycles();
  state_active.assign(numstates + 1, true);
  build_graph();
}

// Allocate all arrays used by the graph and initialize to default values.

void Graph::allocate_arrays() {
  outdegree = new unsigned int[numstates + 1];
  cyclenum = new unsigned int[numstates + 1];
  cycleperiod = new unsigned int[numstates + 1];
  isexitcycle = new bool[numstates + 1];

  for (size_t i = 0; i <= numstates; ++i) {
    outdegree[i] = 0;
    cyclenum[i] = 0;
    cycleperiod[i] = 0;
    isexitcycle[i] = false;
  }

  outmatrix = new unsigned int*[numstates + 1];
  outthrowval = new unsigned int*[numstates + 1];
  excludestates_throw = new unsigned int*[numstates + 1];
  excludestates_catch = new unsigned int*[numstates + 1];

  for (size_t i = 0; i <= numstates; ++i) {
    outmatrix[i] = new unsigned int[maxoutdegree];
    outthrowval[i] = new unsigned int[maxoutdegree];
    excludestates_throw[i] = new unsigned int[h];
    excludestates_catch[i] = new unsigned int[h];

    for (size_t j = 0; j < maxoutdegree; ++j) {
      outmatrix[i][j] = 0;
      outthrowval[i][j] = 0;
    }
    for (size_t j = 0; j < h; ++j) {
      excludestates_throw[i][j] = 0;
      excludestates_catch[i][j] = 0;
    }
  }
}

void Graph::delete_arrays() {
  for (size_t i = 0; i <= numstates; ++i) {
    if (outmatrix) {
      delete[] outmatrix[i];
      outmatrix[i] = nullptr;
    }
    if (outthrowval) {
      delete[] outthrowval[i];
      outthrowval[i] = nullptr;
    }
    if (excludestates_throw) {
      delete[] excludestates_throw[i];
      excludestates_throw[i] = nullptr;
    }
    if (excludestates_catch) {
      delete[] excludestates_catch[i];
      excludestates_catch[i] = nullptr;
    }
  }

  delete[] outmatrix;
  delete[] outthrowval;
  delete[] excludestates_throw;
  delete[] excludestates_catch;
  delete[] outdegree;
  delete[] cyclenum;
  delete[] cycleperiod;
  delete[] isexitcycle;
  outmatrix = nullptr;
  outthrowval = nullptr;
  excludestates_throw = nullptr;
  excludestates_catch = nullptr;
  outdegree = nullptr;
  cyclenum = nullptr;
  cycleperiod = nullptr;
  isexitcycle = nullptr;
}

//------------------------------------------------------------------------------
// Generate the states in the graph
//------------------------------------------------------------------------------

// Generate all possible states into the vector `s`.

void Graph::gen_states_all(std::vector<State>& s, unsigned int n,
    unsigned int h) {
  s.push_back({n, h});
  gen_states_all_helper(s, h - 1, n);
  s.pop_back();
}

// Helper function to generate states in the general case. Recursively insert
// 1s into successive slots, and when all 1s are used up append a new state
// to the list.

void Graph::gen_states_all_helper(std::vector<State>& s, unsigned int pos,
    unsigned int left) {
  if (left > (pos + 1))
    return;  // no way to succeed

  if (pos == 0) {
    s.back().slot.at(0) = left;
    // success: duplicate state at the end and continue
    s.push_back(s.back());
    return;
  }

  // try a '-' at position `pos`
  s.back().slot.at(pos) = 0;
  gen_states_all_helper(s, pos - 1, left);

  // then try a 'x' at position `pos`
  if (left > 0) {
    s.back().slot.at(pos) = 1;
    gen_states_all_helper(s, pos - 1, left - 1);
  }
}

// Generate all possible states that can be part of a pattern of period `l`.

void Graph::gen_states_for_period(std::vector<State>& s, unsigned int n,
    unsigned int h, unsigned int l) {
  s.push_back({n, h});
  gen_states_for_period_helper(s, 0, n, h, l);
  s.pop_back();
}

// Helper function to generate states in the single-period case. The states are
// enumerated by partitioning the `n` objects into `l` different buckets.

void Graph::gen_states_for_period_helper(std::vector<State>& s,
    unsigned int pos, unsigned int left, const unsigned int h,
    const unsigned int l) {
  if (pos == l) {
    if (left == 0) {
      // success: duplicate state at the end and continue
      s.push_back(s.back());
    }
    return;
  }

  // empty all the slots at position `pos`
  for (size_t i = pos; i < h; i += l) {
    s.back().slot[i] = 0;
  }

  // work out the maximum number that can go into later slots, and the min
  // for this slot
  unsigned int max_later = 0;
  for (size_t pos2 = pos + 1; pos2 < l; ++pos2) {
    for (size_t i = pos2; i < h; i += l) {
      ++max_later;
    }
  }
  unsigned int min_fill = (left > max_later ? left - max_later : 0);

  if (min_fill == 0)
    gen_states_for_period_helper(s, pos + 1, left, h, l);

  // successively fill slots at `pos`
  unsigned int filled = 0;
  for (size_t i = pos; i < h && filled < left; i += l) {
    s.back().slot[i] = 1;
    ++filled;
    if (filled >= min_fill)
      gen_states_for_period_helper(s, pos + 1, left - filled, h, l);
  }
}

// Compute the number of ways of building states for a single-period graph.
//
// We partition each state into `l` slots, where slot `i` is associated with
// positions i, i + l, i + 2*l, ... in the state, up to a maximum of h - 1.
// The only degree of freedom is how many objects to put into each slot; the
// state positions must be filled from the bottom up in order to be part of a
// period `l` pattern.

std::uint64_t Graph::ordered_partitions(unsigned int n, unsigned int h,
    unsigned int l) {
  std::map<op_key_type, std::uint64_t> cache;
  return ordered_partitions_helper(0, n, h, l, cache);
}

// Compute the number of ways of filling slot `pos` through slot `l-1`, given
// `left` remaining objects.

std::uint64_t Graph::ordered_partitions_helper(unsigned int pos,
    unsigned int left, const unsigned int h, const unsigned int l,
    std::map<op_key_type, std::uint64_t>& cache) {
  op_key_type key{pos, left};
  if (cache.find(key) != cache.end())
    return cache[key];

  unsigned int max_fill = 0;
  for (unsigned int i = pos; i < h; i += l) {
    ++max_fill;
  }
  max_fill = std::min(max_fill, left);

  std::uint64_t result = 0;
  if (pos == l - 1) {
    result = (left <= max_fill ? 1 : 0);
  } else {
    for (unsigned int i = 0; i <= max_fill; ++i) {
      result += ordered_partitions_helper(pos + 1, left - i, h, l, cache);
    }
  }

  cache[key] = result;
  return result;
}

//------------------------------------------------------------------------------
// Prep core data structures for search
//------------------------------------------------------------------------------

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period

void Graph::find_shift_cycles() {
  unsigned int cycleindex = 0;
  std::vector<unsigned int> cyclestates(h);

  for (size_t i = 0; i <= numstates; ++i) {
    cyclenum[i] = 0;
    cycleperiod[i] = 0;
  }

  for (size_t i = 1; i <= numstates; ++i) {
    State s = state.at(i);
    bool periodfound = false;
    bool newshiftcycle = true;
    unsigned int cycleper = h;

    for (size_t j = 0; j < h; ++j) {
      s = s.upstream();
      unsigned int k = get_statenum(s);
      cyclestates.at(j) = k;
      if (k == 0)
        continue;

      if (k == i && !periodfound) {
        cycleper = j + 1;
        periodfound = true;
      } else if (k < i) {
        newshiftcycle = false;
      }
    }
    assert(cyclestates.at(h - 1) == i);

    if (newshiftcycle) {
      for (size_t j = 0; j < h; j++) {
        if (cyclestates.at(j) > 0)
          cyclenum[cyclestates.at(j)] = cycleindex;
      }
      cycleperiod[cycleindex] = cycleper;
      if (cycleper < h)
        ++numshortcycles;
      ++cycleindex;
    }
  }
  numcycles = cycleindex;
}

// Build the core data structures used during pattern search. This takes into
// account whether states are active; transitions in and out of inactive states
// are pruned from the graph.

void Graph::build_graph() {
  while (true) {
    gen_matrices();

    // deactivate any states with 0 outdegree or indegree
    std::vector<unsigned int> indegree(numstates + 1, 0);
    bool changed = false;

    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_active.at(i))
        continue;
      if (outdegree[i] == 0) {
        state_active.at(i) = false;
        changed = true;
        continue;
      }

      for (size_t j = 0; j < outdegree[i]; ++j) {
        ++indegree.at(outmatrix[i][j]);
      }
    }

    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_active.at(i))
        continue;
      if (indegree.at(i) == 0) {
        state_active.at(i) = false;
        changed = true;
      }
    }

    if (!changed)
      break;
  }

  find_exit_cycles();
}

// Generate matrices describing the structure of the juggling graph:
//
// - Outward degree from each state (vertex) in the graph:
//         outdegree[statenum] --> degree
// - Outward connections from each state:
//         outmatrix[statenum][col] --> statenum  (where col < outdegree)
// - Throw values corresponding to outward connections from each state:
//         outthrowval[statenum][col] --> throw
//
// outmatrix[][] == 0 indicates no connection.

void Graph::gen_matrices() {
  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i)) {
      outdegree[i] = 0;
      continue;
    }

    // each row is calculated once per WorkAssignment
    if (outdegree[i] == 0) {
      unsigned int outthrownum = 0;
      for (int throwval = h; throwval >= 0; --throwval) {
        const unsigned int tv = static_cast<unsigned int>(throwval);
        if (xarray.at(tv))
          continue;

        unsigned int k = advance_state(i, tv);
        if (k == 0)
          continue;
        if (!state_active.at(k))
          continue;
        if (tv > 0 && tv < h && !linkthrows_within_cycle &&
            cyclenum[i] == cyclenum[k])
          continue;

        outmatrix[i][outthrownum] = k;
        outthrowval[i][outthrownum] = tv;
        ++outthrownum;
      }
      outdegree[i] = outthrownum;
      continue;
    }

    // remove inactive states from row
    unsigned int outthrownum = 0;
    for (size_t j = 0; j < outdegree[i]; ++j) {
      if (state_active.at(outmatrix[i][j])) {
        if (outthrownum != j) {
          outmatrix[i][outthrownum] = outmatrix[i][j];
          outthrowval[i][outthrownum] = outthrowval[i][j];
        }
        ++outthrownum;
      }
    }
    outdegree[i] = outthrownum;
  }
}

// Fill in array `isexitcycle` that indicates which shift cycles can exit
// directly to the start state, assumed to be the lowest active state number.

void Graph::find_exit_cycles() {
  for (size_t i = 0; i <= numstates; ++i)
    isexitcycle[i] = false;

  unsigned int lowest_active_state = 0;

  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i))
      continue;
    if (lowest_active_state == 0) {
      lowest_active_state = i;
      continue;
    }

    for (size_t j = 0; j < outdegree[i]; ++j) {
      if (outmatrix[i][j] == lowest_active_state)
        isexitcycle[cyclenum[i]] = true;
    }
  }
}

// Generate arrays that are used for marking excluded states during NORMAL
// mode search with marking.

void Graph::find_exclude_states() {
  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i)) {
      excludestates_throw[i][0] = 0;
      excludestates_catch[i][0] = 0;
      continue;
    }

    // Find states that are excluded by a link throw from state `i`. These are
    // the states downstream in i's shift cycle that end in 'x'.
    State s = state.at(i).downstream();
    unsigned int j = 0;
    while (s.slot.at(s.h - 1) != 0 && j < h) {
      unsigned int statenum = get_statenum(s);
      if (statenum == 0 || !state_active.at(statenum))
        break;
      excludestates_throw[i][j++] = statenum;
      s = s.downstream();
    }
    excludestates_throw[i][j] = 0;

    // Find states that are excluded by a link throw into state `i`. These are
    // the states upstream in i's shift cycle that start with '-'.
    s = state.at(i).upstream();
    j = 0;
    while (s.slot.at(0) == 0 && j < h) {
      unsigned int statenum = get_statenum(s);
      if (statenum == 0 || !state_active.at(statenum))
        break;
      excludestates_catch[i][j++] = statenum;
      s = s.upstream();
    }
    excludestates_catch[i][j] = 0;
  }
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Compute (a choose b).

std::uint64_t Graph::combinations(unsigned int a, unsigned int b) {
  if (a < b)
    return 0;

  std::uint64_t result = 1;
  for (unsigned int denom = 1; denom <= std::min(b, a - b); ++denom)
    result = (result * (a - denom + 1)) / denom;
  return result;
}

// Compute the number of shift cycles with `n` objects, max throw `h`, with
// exact period `p`.

std::uint64_t Graph::shift_cycle_count(unsigned int n, unsigned int h,
    unsigned int p) {
  if (h % p != 0)
    return 0;
  if (n % (h / p) != 0)
    return 0;
  if (p < h)
    return shift_cycle_count(n * p / h, p, p);

  std::uint64_t val = combinations(h, n);
  for (unsigned int p2 = 1; p2 <= h / 2; ++p2) {
    val -= p2 * shift_cycle_count(n, h, p2);
  }
  assert(val % h == 0);
  return (val / h);
}

// Calculate an upper bound on the length of prime patterns in the graph.

unsigned int Graph::prime_length_bound() const {
  // when there is more than one shift cycle, a prime pattern has to miss at
  // least one state in each shift cycle it visits

  unsigned int result = 0;
  std::vector<unsigned int> num_active(numcycles, 0);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      ++result;
      ++num_active.at(cyclenum[i]);
    }
  }

  int cycles_active = numcycles -
      std::count(num_active.begin(), num_active.end(), 0);

  if (cycles_active > 1) {
    for (size_t i = 0; i < numcycles; ++i) {
      if (num_active.at(i) == cycleperiod[i]) {
        --result;
      }
    }
  }
  return result;
}

// Calculate an upper bound on the length of superprime patterns in the graph.

unsigned int Graph::superprime_length_bound() const {
  std::vector<bool> any_active(numcycles, false);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      any_active.at(cyclenum[i]) = true;
    }
  }

  return std::count(any_active.begin(), any_active.end(), true);
}

// Return the index in the `state` array that corresponds to a given state.
// Returns 0 if not found.
//
// Note this assumes the `state` vector is sorted!

unsigned int Graph::get_statenum(const State& s) const {
  if (state_compare(s, state.at(1)))
    return 0;
  if (state_compare(state.at(numstates), s))
    return 0;

  if (state.at(1) == s)
    return 1;
  if (state.at(numstates) == s)
    return numstates;

  size_t below = 1;
  size_t above = numstates;

  // loop invariant: state[below] < s < state[above]
  while (below < above - 1) {
    size_t mid = (below + above) / 2;
    if (state.at(mid) == s)
      return mid;
    if (state_compare(state.at(mid), s)) {
      below = mid;
    } else {
      above = mid;
    }
  }
  return 0;
}

// Return the state number that comes from advancing a given state by a single
// throw. Returns 0 if the throw results in a collision.

unsigned int Graph::advance_state(unsigned int statenum,
    unsigned int throwval) const {
  if (throwval < 0 || throwval > state.at(statenum).h)
    return 0;
  if (throwval > 0 && state.at(statenum).slot.at(0) == 0)
    return 0;
  if (throwval < state.at(statenum).h &&
      state.at(statenum).slot.at(throwval) != 0)
    return 0;

  return get_statenum(state.at(statenum).advance_with_throw(throwval));
}

// Return the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the `state` vector).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

unsigned int Graph::reverse_state(unsigned int statenum) const {
  return get_statenum(state.at(statenum).reverse());
}

// Return the next state downstream in the given state's shift cycle.

unsigned int Graph::downstream_state(unsigned int statenum) const {
  return get_statenum(state.at(statenum).downstream());
}

// Return the next state upstream in the given state's shift cycle.

unsigned int Graph::upstream_state(unsigned int statenum) const {
  return get_statenum(state.at(statenum).upstream());
}

// Return a text representation of a given state number.

std::string Graph::state_string(unsigned int statenum) const {
  return state.at(statenum).to_string();
}
