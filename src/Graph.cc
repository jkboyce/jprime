//
// Graph.cc
//
// Data structures related to the juggling graph for B objects, max throw H.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Graph.h"

#include <iostream>
#include <algorithm>
#include <cassert>
#include <limits>


// Full graph for `b` objects, max throw `h`.

Graph::Graph(unsigned b, unsigned h)
    : b(b), h(h), n(0), xarray(h + 1, false) {
  initialize();
}

// Initialize with two additional options:
//
// Parameter `xa` specifies a vector of size h+1 that determines whether a given
// throw value is excluded; when xa[val] is true then no graph links with throw
// value `val` are created.
//
// Parameter `n` is optional and specifies that a single-period graph is
// desired; a single-period graph contains only those states that may be part of
// a period-n pattern. We only permit values n < h. When n == 0 we generate
// the full graph.

Graph::Graph(unsigned b, unsigned h, const std::vector<bool>& xa, unsigned n)
    : b(b), h(h), n(n), xarray(xa) {
  assert(xa.size() == h + 1);
  assert(n < h);
  initialize();
}

//------------------------------------------------------------------------------
// Prepare graph data structures during construction
//------------------------------------------------------------------------------

// Initialize the Graph object.

void Graph::initialize() {
  // fail if the number of states cannot fit into an unsigned int
  const std::uint64_t num = (n == 0 ? combinations(h, b) :
      ordered_partitions(b, h, n));
  assert(num <= std::numeric_limits<unsigned>::max());
  numstates = static_cast<unsigned>(num);

  // generate the states
  state.reserve(numstates + 2);
  state.push_back({h});  // state at index 0 is unused
  if (n == 0) {
    gen_states_all(state, b, h);
    // states are generated in sorted order
  } else {
    gen_states_for_period(state, b, h, n);
    if (state.size() > 1) {
      std::sort(state.begin() + 1, state.end(), state_compare);
    }
  }
  assert(state.size() == numstates + 1);

  // generate the shift cycles
  numcycles = find_shift_cycles();

  // fill in graph matrices
  build_graph();
  find_max_startstate_usable();
}

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period
//
// Return the total number of shift cycles found.

unsigned Graph::find_shift_cycles() {
  const unsigned state_unused = -1;
  cyclenum.assign(numstates + 1, state_unused);
  assert(cycleperiod.size() == 0);

  unsigned cycles = 0;
  std::vector<unsigned> cyclestates(h);

  for (size_t i = 1; i <= numstates; ++i) {
    if (cyclenum.at(i) != state_unused)
      continue;

    State s = state.at(i);
    bool periodfound = false;
    bool newshiftcycle = true;
    unsigned cycleper = h;

    for (size_t j = 0; j < h; ++j) {
      s = s.upstream();
      const auto k = get_statenum(s);
      cyclestates.at(j) = k;
      if (k == 0)
        continue;

      if (k == i && !periodfound) {
        cycleper = static_cast<unsigned>(j + 1);
        periodfound = true;
      } else if (k < i) {
        newshiftcycle = false;
      }
    }
    assert(cyclestates.at(h - 1) == i);

    if (newshiftcycle) {
      for (size_t j = 0; j < h; j++) {
        if (cyclestates.at(j) > 0) {
          cyclenum.at(cyclestates.at(j)) = cycles;
        }
      }
      cycleperiod.push_back(cycleper);
      if (cycleper < h) {
        ++numshortcycles;
      }
      ++cycles;
    }
  }
  assert(cycleperiod.size() == cycles);
  return cycles;
}

//------------------------------------------------------------------------------
// Generate the states in the graph
//------------------------------------------------------------------------------

// Helper function to generate states in the general case. Recursively insert
// 1s into successive slots, and when all 1s are used up append a new state
// to the list.

void gen_states_all_helper(std::vector<State>& s, unsigned pos, unsigned left) {
  if (left > (pos + 1))
    return;  // no way to succeed

  if (pos == 0) {
    s.back().slot(0) = left;
    // success: duplicate state at the end and continue
    s.push_back(s.back());
    return;
  }

  // try a '-' at position `pos`
  s.back().slot(pos) = 0;
  gen_states_all_helper(s, pos - 1, left);

  // then try a 'x' at position `pos`
  if (left > 0) {
    s.back().slot(pos) = 1;
    gen_states_all_helper(s, pos - 1, left - 1);
  }
}

// Generate all possible states into the vector `s`.
//
// Note this has a recursion depth of `h`.

void Graph::gen_states_all(std::vector<State>& s, unsigned b, unsigned h) {
  s.push_back({h});
  gen_states_all_helper(s, h - 1, b);
  s.pop_back();
}

// Helper function to generate states in the single-period case. The states are
// enumerated by partitioning the `b` objects into `n` different buckets.

void gen_states_for_period_helper(std::vector<State>& s, unsigned pos,
    unsigned left, const unsigned h, const unsigned n) {
  if (pos == n) {
    if (left == 0) {
      // success: duplicate state at the end and continue
      s.push_back(s.back());
    }
    return;
  }

  // empty all the slots at position `pos`
  for (size_t i = pos; i < h; i += n) {
    s.back().slot(i) = 0;
  }

  // work out the maximum number that can go into later slots, and the min
  // for this slot
  unsigned max_later = 0;
  for (size_t pos2 = pos + 1; pos2 < n; ++pos2) {
    for (size_t i = pos2; i < h; i += n) {
      ++max_later;
    }
  }
  const unsigned min_fill = (left > max_later ? left - max_later : 0);

  if (min_fill == 0)
    gen_states_for_period_helper(s, pos + 1, left, h, n);

  // successively fill slots at `pos`
  unsigned filled = 0;
  for (size_t i = pos; i < h && filled < left; i += n) {
    s.back().slot(i) = 1;
    ++filled;
    if (filled >= min_fill)
      gen_states_for_period_helper(s, pos + 1, left - filled, h, n);
  }
}

// Generate all possible states that can be part of a pattern of period `n`.
//
// Note this has a recursion depth of `h`.

void Graph::gen_states_for_period(std::vector<State>& s, unsigned b, unsigned h,
    unsigned n) {
  s.push_back({h});
  gen_states_for_period_helper(s, 0, b, h, n);
  s.pop_back();
}

//------------------------------------------------------------------------------
// Operations on graph matrices
//------------------------------------------------------------------------------

// Construct matrices describing the structure of the juggling graph.
//
// - Outward degree from each state (vertex) in the graph:
//         outdegree[statenum] --> degree
// - Outward connections from each state:
//         outmatrix[statenum][col] --> statenum  (where col < outdegree)
// - Throw values corresponding to outward connections from each state:
//         outthrowval[statenum][col] --> throw
//
// outmatrix[][] == 0 indicates no connection.

void Graph::build_graph() {
  maxoutdegree = 0;
  for (size_t i = 0; i <= h; ++i) {
    if (!xarray.at(i)) {
      ++maxoutdegree;
    }
  }
  maxoutdegree = std::min(maxoutdegree, h - b + 1);
  outdegree.assign(numstates + 1, 0);
  outmatrix.resize(numstates + 1);
  outthrowval.resize(numstates + 1);
  for (size_t i = 0; i <= numstates; ++i) {
    outmatrix.at(i).assign(maxoutdegree, 0);
    outthrowval.at(i).assign(maxoutdegree, 0);
  }

  for (size_t i = 1; i <= numstates; ++i) {
    unsigned outthrownum = 0;
    for (unsigned throwval = h + 1; throwval-- > 0; ) {
      if (xarray.at(throwval))
        continue;
      const auto k = advance_state(i, throwval);
      if (k == 0)
        continue;

      outmatrix.at(i).at(outthrownum) = k;
      outthrowval.at(i).at(outthrownum) = throwval;
      ++outthrownum;
    }
    outdegree.at(i) = outthrownum;
  }
}

// Fill in the vector `max_startstate_usable`, which indicates when a given
// state is accessible in the graph.
//
// The state `i` is accessible to a pattern starting from `start_state` if and
// only if start_state <= max_startstate_usable[i].

void Graph::find_max_startstate_usable() {
  max_startstate_usable.resize(numstates + 1);
  max_startstate_usable.assign(numstates + 1, 0);
  std::vector<bool> state_usable(numstates + 1, true);

  for (unsigned start_state = 1; start_state <= numstates; ++start_state) {
    state_usable.at(start_state - 1) = false;

    update_usable_states(state_usable);

    for (size_t i = start_state; i <= numstates; ++i) {
      if (state_usable.at(i)) {
        // confirm we never switch from unusable back to usable
        assert(max_startstate_usable.at(i) + 1 == start_state);
        max_startstate_usable.at(i) = start_state;
      }
    }
  }
}

// Find any additional unusable states in the graph.
//
// Propagate unusable links and states through the graph, by applying two
// transformations until no further reductions are possible:
// - Remove links into states with outdegree zero
// - For states with indegree zero, set outdegree to zero

void Graph::update_usable_states(std::vector<bool>& state_usable) const {
  std::vector<unsigned> usable_outdegree(outdegree);
  std::vector<unsigned> usable_indegree(numstates + 1, 0);

  // Propagate unusable links and states through the graph
  while (true) {
    bool changed = false;

    // remove links into unusable states
    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_usable.at(i)) {
        usable_outdegree.at(i) = 0;
        continue;
      }
      unsigned outthrownum = 0;
      for (size_t j = 0; j < outdegree.at(i); ++j) {
        if (state_usable.at(outmatrix.at(i).at(j))) {
          ++outthrownum;
        }
      }
      if (usable_outdegree.at(i) != outthrownum) {
        usable_outdegree.at(i) = outthrownum;
        changed = true;
      }
    }

    // mark unusable any states with zero outdegree or indegree
    usable_indegree.assign(numstates + 1, 0);

    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_usable.at(i))
        continue;
      if (usable_outdegree.at(i) == 0) {
        state_usable.at(i) = false;
        changed = true;
        continue;
      }

      for (size_t j = 0; j < outdegree.at(i); ++j) {
        ++usable_indegree.at(outmatrix.at(i).at(j));
      }
    }

    for (size_t i = 1; i <= numstates; ++i) {
      if (state_usable.at(i) && usable_indegree.at(i) == 0) {
        state_usable.at(i) = false;
        changed = true;
      }
    }

    if (!changed)
      break;
  }
}

// Remove unusable links and states from the graph.
//
// NOTE: This function will in general renumber the states! The exception is
// state number 1, which is reserved for the ground state and is never removed
// or renumbered.

void Graph::reduce_graph() {
  std::vector<bool> state_usable(numstates + 1, true);
  update_usable_states(state_usable);

  // TODO update:
  //   unsigned numstates = 0;
  //   std::vector<State> state;
  //   std::vector<unsigned> cyclenum;
  //   std::vector<unsigned> max_startstate_usable;
  //   std::vector<unsigned> outdegree;
  //   std::vector<std::vector<unsigned>> outmatrix;
  //   std::vector<std::vector<unsigned>> outthrowval;
}

// Return vector `isexitcycle` that indicates which shift cycles can exit
// directly to the start state with a link throw. This is used in SUPER mode to
// cut off search when all exit cycles have been used.

std::vector<int> Graph::get_exit_cycles(unsigned start_state) const {
  std::vector<int> isexitcycle(numcycles, 0);

  for (size_t i = start_state + 1; i <= numstates; ++i) {
    for (size_t j = 0; j < outdegree.at(i); ++j) {
      if (outmatrix.at(i).at(j) == start_state) {
        isexitcycle.at(cyclenum.at(i)) = 1;
      }
    }
  }
  isexitcycle.at(cyclenum.at(start_state)) = 0;

  return isexitcycle;
}

// Generate arrays that are used for marking excluded states during NORMAL
// mode search with marking. If one of the non-marking versions of gen_loops()
// is used then these arrays are ignored.
//
// This should be called after reduce_graph() and before gen_loops().

std::tuple<std::vector<std::vector<unsigned>>, std::vector<std::vector<unsigned>>>
    Graph::get_exclude_states(unsigned start_state) const
{
  std::vector<std::vector<unsigned>> excludestates_throw;
  std::vector<std::vector<unsigned>> excludestates_catch;

  excludestates_throw.resize(numstates + 1);
  excludestates_catch.resize(numstates + 1);
  for (size_t i = 0; i <= numstates; ++i) {
    excludestates_throw.at(i).assign(h, 0);
    excludestates_catch.at(i).assign(h, 0);
  }

  for (size_t i = 1; i <= numstates; ++i) {
    if (start_state > max_startstate_usable.at(i)) {
      excludestates_throw.at(i).at(0) = 0;
      excludestates_catch.at(i).at(0) = 0;
      continue;
    }

    // Find states that are excluded by a link throw from state `i`. These are
    // the states downstream in i's shift cycle that end in 'x'.
    State s = state.at(i).downstream();
    unsigned j = 0;
    while (s.slot(s.size() - 1) != 0) {
      const auto statenum = get_statenum(s);
      if (statenum == 0 || start_state > max_startstate_usable.at(statenum) ||
          statenum == i) {
        break;
      }
      excludestates_throw.at(i).at(j++) = statenum;
      s = s.downstream();
    }
    excludestates_throw.at(i).at(j) = 0;

    // Find states that are excluded by a link throw into state `i`. These are
    // the states upstream in i's shift cycle that start with '-'.
    s = state.at(i).upstream();
    j = 0;
    while (s.slot(0) == 0) {
      const auto statenum = get_statenum(s);
      if (statenum == 0 || start_state > max_startstate_usable.at(statenum) ||
          statenum == i) {
        break;
      }
      excludestates_catch.at(i).at(j++) = statenum;
      s = s.upstream();
    }
    excludestates_catch.at(i).at(j) = 0;
  }

  return make_tuple(excludestates_throw, excludestates_catch);
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Calculate an upper bound on the period of prime patterns in the graph,
// starting from `start_state` as the root node.

unsigned Graph::prime_period_bound(unsigned start_state) const {
  // case 1: the pattern visits multiple shift cycles; it must miss at least one
  // state on each cycle it visits.
  // case 2: the pattern stays on a single shift cycle; find the cycle with the
  // most active states.
  unsigned result_multicycle = 0;
  unsigned result_onecycle = 0;

  std::vector<unsigned> num_active(numcycles, 0);
  for (size_t i = start_state; i <= numstates; ++i) {
    if (start_state <= max_startstate_usable.at(i)) {
      ++num_active.at(cyclenum.at(i));
    }
  }

  const int cycles_active = numcycles -
      static_cast<int>(std::count(num_active.cbegin(), num_active.cend(), 0));

  for (size_t i = 0; i < numcycles; ++i) {
    if (cycles_active > 1) {
      result_multicycle += std::min(num_active.at(i), cycleperiod.at(i) - 1);
    }
    result_onecycle = std::max(result_onecycle, num_active.at(i));
  }

  return std::max(result_multicycle, result_onecycle);
}

// Calculate an upper bound on the period of superprime patterns, starting from
// `start_state` as the root node.
//
// Optional parameter `shifts` specifies an upper limit on the number of shift
// throws allowed in the pattern.

unsigned Graph::superprime_period_bound(unsigned start_state, unsigned shifts)
    const {
  std::vector<bool> any_active(numcycles, false);
  for (size_t i = start_state; i <= numstates; ++i) {
    if (start_state <= max_startstate_usable.at(i)) {
      any_active.at(cyclenum.at(i)) = true;
    }
  }

  const auto cycles_active = static_cast<unsigned>(
      std::count(any_active.cbegin(), any_active.cend(), true));

  if (cycles_active < 2) {
    return 0;
  }
  if (shifts == -1u) {
    return prime_period_bound(start_state);
  }
  return std::min(prime_period_bound(start_state), cycles_active + shifts);
}

// Return the index in the `state` array that corresponds to a given state.
// Returns 0 if not found.
//
// Note this assumes the `state` vector is sorted!

unsigned Graph::get_statenum(const State& s) const {
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
      return static_cast<unsigned>(mid);
    if (state_compare(state.at(mid), s)) {
      below = mid;
    } else {
      above = mid;
    }
  }
  return 0;
}

// Return the state number that comes from advancing a given state by a single
// throw. Returns 0 if the throw is not allowed.

unsigned Graph::advance_state(unsigned statenum, unsigned throwval) const {
  const State& s = state.at(statenum);

  if (throwval > 0 && s.slot(0) == 0)  // no object to throw
    return 0;
  if (throwval < s.size() && s.slot(throwval) != 0)  // collision w/prev throw
    return 0;
  if (throwval > s.size())  // out of range
    return 0;

  return get_statenum(s.advance_with_throw(throwval));
}

// Return the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the `state` vector).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

unsigned Graph::reverse_state(unsigned statenum) const {
  return get_statenum(state.at(statenum).reverse());
}

// Return the next state downstream in the given state's shift cycle.

unsigned Graph::downstream_state(unsigned statenum) const {
  return get_statenum(state.at(statenum).downstream());
}

// Return the next state upstream in the given state's shift cycle.

unsigned Graph::upstream_state(unsigned statenum) const {
  return get_statenum(state.at(statenum).upstream());
}

// Return a text representation of a given state number.

std::string Graph::state_string(unsigned statenum) const {
  return state.at(statenum).to_string();
}
