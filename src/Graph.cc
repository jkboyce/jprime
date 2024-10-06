//
// Graph.cc
//
// Data structures related to the juggling graph for B objects, max throw H.
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


// Full graph for `b` objects, max throw `h`.

Graph::Graph(unsigned b, unsigned h)
    : b(b), h(h), l(0), xarray(h + 1, false) {
  init();
}

// Initialize with two additional options:
//
// Parameter `xa` specifies a vector of size h+1 that determines whether a given
// throw value is excluded; when xa[val] is true then no graph links with throw
// value `val` are created.
//
// Parameter `l` is optional and specifies that a single-period graph is
// desired; a single-period graph contains only those states that may be part of
// a period-l pattern. We only permit values l < h. When l == 0 we generate
// the full graph.

Graph::Graph(unsigned b, unsigned h, const std::vector<bool>& xa, unsigned l)
    : b(b), h(h), l(l), xarray(xa) {
  assert(xa.size() == h + 1);
  assert(l < h);
  init();
}

//------------------------------------------------------------------------------
// Prepare graph data structures during construction
//------------------------------------------------------------------------------

// Initialize the Graph object.
//
// NOTE: Call build_graph() after this to fully populate the graph arrays with
// data.

void Graph::init() {
  // fail if the number of states cannot fit into an unsigned int
  const std::uint64_t num = (l == 0 ? combinations(h, b) :
      ordered_partitions(b, h, l));
  assert(num <= std::numeric_limits<unsigned>::max());
  numstates = static_cast<unsigned>(num);

  // generate the states
  state.reserve(numstates + 2);
  state.push_back({h});  // state at index 0 is unused
  if (l == 0) {
    gen_states_all(state, b, h);
    // states are generated in sorted order
  } else {
    gen_states_for_period(state, b, h, l);
    if (state.size() > 1) {
      std::sort(state.begin() + 1, state.end(), state_compare);
    }
  }
  assert(state.size() == numstates + 1);
  state_active.assign(numstates + 1, true);

  // generate the shift cycles
  numcycles = find_shift_cycles();

  // allocate arrays to be filled in later by build_graph() and
  // find_exclude_states()
  unsigned maxoutdegree = 0;
  for (size_t i = 0; i <= h; ++i) {
    if (!xarray.at(i)) {
      ++maxoutdegree;
    }
  }
  maxoutdegree = std::min(maxoutdegree, h - b + 1);
  outdegree.assign(numstates + 1, 0);
  outmatrix.resize(numstates + 1);
  outthrowval.resize(numstates + 1);
  excludestates_throw.resize(numstates + 1);
  excludestates_catch.resize(numstates + 1);
  for (size_t i = 0; i <= numstates; ++i) {
    outmatrix.at(i).assign(maxoutdegree, 0);
    outthrowval.at(i).assign(maxoutdegree, 0);
    excludestates_throw.at(i).assign(h, 0);
    excludestates_catch.at(i).assign(h, 0);
  }
  isexitcycle.assign(numcycles, 0);
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
// enumerated by partitioning the `b` objects into `l` different buckets.

void gen_states_for_period_helper(std::vector<State>& s, unsigned pos,
    unsigned left, const unsigned h, const unsigned l) {
  if (pos == l) {
    if (left == 0) {
      // success: duplicate state at the end and continue
      s.push_back(s.back());
    }
    return;
  }

  // empty all the slots at position `pos`
  for (size_t i = pos; i < h; i += l) {
    s.back().slot(i) = 0;
  }

  // work out the maximum number that can go into later slots, and the min
  // for this slot
  unsigned max_later = 0;
  for (size_t pos2 = pos + 1; pos2 < l; ++pos2) {
    for (size_t i = pos2; i < h; i += l) {
      ++max_later;
    }
  }
  unsigned min_fill = (left > max_later ? left - max_later : 0);

  if (min_fill == 0)
    gen_states_for_period_helper(s, pos + 1, left, h, l);

  // successively fill slots at `pos`
  unsigned filled = 0;
  for (size_t i = pos; i < h && filled < left; i += l) {
    s.back().slot(i) = 1;
    ++filled;
    if (filled >= min_fill)
      gen_states_for_period_helper(s, pos + 1, left - filled, h, l);
  }
}

// Generate all possible states that can be part of a pattern of period `l`.
//
// Note this has a recursion depth of `h`.

void Graph::gen_states_for_period(std::vector<State>& s, unsigned b, unsigned h,
    unsigned l) {
  s.push_back({h});
  gen_states_for_period_helper(s, 0, b, h, l);
  s.pop_back();
}

//------------------------------------------------------------------------------
// Populate graph data structures describing connections between states
//------------------------------------------------------------------------------

// Construct matrices describing the structure of the juggling graph, for the
// states that are currently active.
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
  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i)) {
      outdegree.at(i) = 0;
      continue;
    }

    unsigned outthrownum = 0;
    for (unsigned throwval = h + 1; throwval-- > 0; ) {
      if (xarray.at(throwval))
        continue;
      auto k = advance_state(i, throwval);
      if (k == 0 || !state_active.at(k))
        continue;

      outmatrix.at(i).at(outthrownum) = k;
      outthrowval.at(i).at(outthrownum) = throwval;
      ++outthrownum;
    }
    outdegree.at(i) = outthrownum;
  }

  find_exit_cycles();
}

// Remove unusable links and states from the graph. Apply two transformations
// until no further reductions are possible:
// - Remove links into inactive states
// - Deactivate states with zero outdegree or indegree
//
// Finally update these data structures based on the reduced graph:
// - isexitcycle, excludestates_throw, excludestates_catch

void Graph::reduce_graph() {
  while (true) {
    bool changed = false;

    // Remove links into inactive states
    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_active.at(i))
        continue;
      unsigned outthrownum = 0;
      for (size_t j = 0; j < outdegree.at(i); ++j) {
        if (state_active.at(outmatrix.at(i).at(j))) {
          if (outthrownum != j) {
            outmatrix.at(i).at(outthrownum) = outmatrix.at(i).at(j);
            outthrowval.at(i).at(outthrownum) = outthrowval.at(i).at(j);
          }
          ++outthrownum;
        }
      }
      if (outdegree.at(i) != outthrownum) {
        outdegree.at(i) = outthrownum;
        changed = true;
      }
    }

    // Deactivate states with zero outdegree or indegree
    std::vector<unsigned> indegree(numstates + 1, 0);
    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_active.at(i))
        continue;
      if (outdegree.at(i) == 0) {
        state_active.at(i) = false;
        changed = true;
        continue;
      }

      for (size_t j = 0; j < outdegree.at(i); ++j) {
        ++indegree.at(outmatrix.at(i).at(j));
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

  // initialize to empty; call find_exclude_states() to fill in if needed
  for (size_t i = 0; i <= numstates; ++i) {
    excludestates_throw.at(i).assign(h, 0);
    excludestates_catch.at(i).assign(h, 0);
  }
}

// Fill in array `isexitcycle` that indicates which shift cycles can exit
// directly to the start state with a link throw. This is used in SUPER mode to
// cut off search when all exit cycles have been used.
//
// The start state is assumed to be the lowest active state number.

void Graph::find_exit_cycles() {
  isexitcycle.assign(numcycles, false);
  unsigned lowest_active_state = 0;

  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i))
      continue;
    if (lowest_active_state == 0) {
      lowest_active_state = static_cast<unsigned>(i);
      continue;
    }

    for (size_t j = 0; j < outdegree.at(i); ++j) {
      if (outmatrix.at(i).at(j) == lowest_active_state) {
        isexitcycle.at(cyclenum.at(i)) = true;
      }
    }
  }

  if (lowest_active_state != 0) {
    isexitcycle.at(cyclenum.at(lowest_active_state)) = false;
  }
}

// Generate arrays that are used for marking excluded states during NORMAL
// mode search with marking. If one of the non-marking versions of gen_loops()
// is used then these arrays are ignored.
//
// This should be called after reduce_graph() and before gen_loops().

void Graph::find_exclude_states() {
  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i)) {
      excludestates_throw.at(i).at(0) = 0;
      excludestates_catch.at(i).at(0) = 0;
      continue;
    }

    // Find states that are excluded by a link throw from state `i`. These are
    // the states downstream in i's shift cycle that end in 'x'.
    State s = state.at(i).downstream();
    unsigned j = 0;
    while (s.slot(s.size() - 1) != 0 && j < h) {
      auto statenum = get_statenum(s);
      if (statenum == 0 || !state_active.at(statenum) || statenum == i)
        break;
      excludestates_throw.at(i).at(j++) = statenum;
      s = s.downstream();
    }
    excludestates_throw.at(i).at(j) = 0;

    // Find states that are excluded by a link throw into state `i`. These are
    // the states upstream in i's shift cycle that start with '-'.
    s = state.at(i).upstream();
    j = 0;
    while (s.slot(0) == 0 && j < h) {
      auto statenum = get_statenum(s);
      if (statenum == 0 || !state_active.at(statenum) || statenum == i)
        break;
      excludestates_catch.at(i).at(j++) = statenum;
      s = s.upstream();
    }
    excludestates_catch.at(i).at(j) = 0;
  }
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Calculate an upper bound on the length of prime patterns in the graph, using
// states that are currently active.

unsigned Graph::prime_length_bound() const {
  // Case 1: The pattern visits multiple shift cycles; it must miss at least one
  // state on each cycle it visits.
  unsigned result_multicycle = 0;
  std::vector<unsigned> num_active(numcycles, 0);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      ++result_multicycle;
      ++num_active.at(cyclenum.at(i));
    }
  }

  int cycles_active = numcycles -
      static_cast<int>(std::count(num_active.cbegin(), num_active.cend(), 0));
  if (cycles_active > 1) {
    for (size_t i = 0; i < numcycles; ++i) {
      if (num_active.at(i) == cycleperiod.at(i)) {
        --result_multicycle;
      }
    }
  } else {
    result_multicycle = 0;
  }

  // Case 2: The pattern stays on a single shift cycle; find the cycle with the
  // most active states.
  unsigned result_onecycle = 0;
  for (size_t i = 0; i < numcycles; ++i) {
    result_onecycle = std::max(result_onecycle, num_active.at(i));
  }

  return std::max(result_multicycle, result_onecycle);
}

// Calculate an upper bound on the length of superprime patterns with `shifts`
// shift throws, using states in the graph that are currently active.

unsigned Graph::superprime_length_bound(unsigned shifts) const {
  std::vector<bool> any_active(numcycles, false);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      any_active.at(cyclenum.at(i)) = true;
    }
  }

  auto active_cycles = static_cast<unsigned>(
      std::count(any_active.cbegin(), any_active.cend(), true));

  return (active_cycles > 1 ? active_cycles + shifts : 0);
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
