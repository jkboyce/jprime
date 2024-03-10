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
#include <vector>
#include <cassert>


Graph::Graph(int n, int h, const std::vector<bool>& xa, bool ltwc)
    : n(n), h(h), xarray(xa), linkthrows_within_cycle(ltwc) {
  init();
}

Graph::Graph(int n, int h)
    : n(n), h(h), xarray(h + 1, false), linkthrows_within_cycle(true) {
  init();
}

Graph::Graph(const Graph& g)
    : Graph(g.n, g.h, g.xarray, g.linkthrows_within_cycle) {
}

Graph& Graph::operator=(const Graph& g) {
  if (this == &g)
    return *this;

  delete_arrays();
  n = g.n;
  h = g.h;
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
  numstates = num_states(n, h);
  for (int i = 0; i <= h; ++i) {
    if (!xarray.at(i))
      ++maxoutdegree;
  }
  maxoutdegree = std::min(maxoutdegree, h - n + 1);
  allocate_arrays();

  state.push_back({n, h});  // index 0 in state vector is unused
  state.push_back({n, h});
  int ns = gen_states(0, h - 1, n);
  state.pop_back();
  assert(ns == numstates);
  assert(state.size() == numstates + 1);

  find_shift_cycles();
  state_active.assign(numstates + 1, true);
  build_graph();
}

// Allocate all arrays used by the graph and initialize to default values.

void Graph::allocate_arrays() {
  outdegree = new int[numstates + 1];
  cyclenum = new int[numstates + 1];
  cycleperiod = new int[numstates + 1];
  isexitcycle = new bool[numstates + 1];

  for (size_t i = 0; i <= numstates; ++i) {
    outdegree[i] = 0;
    cyclenum[i] = 0;
    cycleperiod[i] = 0;
    isexitcycle[i] = false;
  }

  outmatrix = new int*[numstates + 1];
  outthrowval = new int*[numstates + 1];
  excludestates_throw = new int*[numstates + 1];
  excludestates_catch = new int*[numstates + 1];

  for (size_t i = 0; i <= numstates; ++i) {
    outmatrix[i] = new int[maxoutdegree];
    outthrowval[i] = new int[maxoutdegree];
    excludestates_throw[i] = new int[h];
    excludestates_catch[i] = new int[h];

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

// Find the number of states (vertices) in the juggling graph, for a given
// number of balls and maximum throw value. This is just (h choose n).

int Graph::num_states(int n, int h) {
  int result = 1;
  for (int denom = 1; denom <= std::min(n, h - n); ++denom)
    result = (result * (h - denom + 1)) / denom;
  return result;
}

// Generate the list of all possible states into the `state` vector.
//
// Returns the number of states found.

int Graph::gen_states(int num, int pos, int left) {
  if (left > (pos + 1))
    return num;

  if (pos == 0) {
    state.at(num + 1).slot.at(0) = left;
    state.push_back(state.at(num + 1));
    return (num + 1);
  }

  // try a '-' at position `pos`
  state.at(num + 1).slot.at(pos) = 0;
  num = gen_states(num, pos - 1, left);

  // then try a 'x' at position `pos`
  if (left > 0) {
    state.at(num + 1).slot.at(pos) = 1;
    num = gen_states(num, pos - 1, left - 1);
  }

  return num;
}

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period

void Graph::find_shift_cycles() {
  int cycleindex = 0;
  std::vector<int> cyclestates(h);

  for (size_t i = 0; i <= numstates; ++i) {
    cyclenum[i] = 0;
    cycleperiod[i] = 0;
  }

  for (size_t i = 1; i <= numstates; ++i) {
    State s = state.at(i);
    bool periodfound = false;
    bool newshiftcycle = true;
    int cycleper = h;

    for (size_t j = 0; j < h; ++j) {
      s = s.upstream();
      int k = get_statenum(s);
      assert(k > 0);

      cyclestates.at(j) = k;
      if (k == i && !periodfound) {
        cycleper = static_cast<int>(j + 1);
        periodfound = true;
      } else if (k < i)
        newshiftcycle = false;
    }
    assert(cyclestates[h - 1] == i);

    if (newshiftcycle) {
      for (size_t j = 0; j < h; j++)
        cyclenum[cyclestates.at(j)] = cycleindex;
      cycleperiod[cycleindex] = cycleper;
      if (cycleper < h)
        ++numshortcycles;
      ++cycleindex;
    }
  }
  numcycles = cycleindex;
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
    outdegree[i] = 0;
    if (!state_active.at(i))
      continue;

    int outthrownum = 0;

    for (int throwval = h; throwval >= 0; --throwval) {
      if (xarray.at(throwval))
        continue;

      int k = advance_state(i, throwval);
      if (k <= 0)
        continue;
      if (!state_active.at(k))
        continue;
      if (throwval > 0 && throwval < h && !linkthrows_within_cycle &&
          cyclenum[i] == cyclenum[k])
        continue;

      outmatrix[i][outthrownum] = k;
      outthrowval[i][outthrownum] = throwval;
      ++outthrownum;
      ++outdegree[i];
    }
  }
}

// Generate arrays that are used for marking excluded states during NORMAL
// mode search.

void Graph::find_exclude_states() {
  for (size_t i = 1; i <= numstates; ++i) {
    // Find states that are excluded by a link throw from state `i`. These are
    // the states downstream in i's shift cycle that end in 'x'.
    State s = state.at(i).downstream();
    int j = 0;
    while (s.slot.at(s.h - 1) != 0 && j < h) {
      excludestates_throw[i][j++] = get_statenum(s);
      s = s.downstream();
    }
    excludestates_throw[i][j] = 0;

    // Find states that are excluded by a link throw into state `i`. These are
    // the states upstream in i's shift cycle that start with '-'.
    s = state.at(i).upstream();
    j = 0;
    while (s.slot.at(0) == 0 && j < h) {
      excludestates_catch[i][j++] = get_statenum(s);
      s = s.upstream();
    }
    excludestates_catch[i][j] = 0;
  }
}

// Fill in array `isexitcycle` that indicates which shift cycles can exit
// directly to the start state, assumed to be the lowest active state number.

void Graph::find_exit_cycles() {
  for (size_t i = 0; i <= numstates; ++i)
    isexitcycle[i] = false;

  int lowest_active_state = 0;

  for (size_t i = 1; i <= numstates; ++i) {
    if (!state_active.at(i))
      continue;
    if (lowest_active_state == 0) {
      lowest_active_state = i;
      continue;
    }

    for (int j = 0; j < outdegree[i]; ++j) {
      if (outmatrix[i][j] == lowest_active_state)
        isexitcycle[cyclenum[i]] = true;
    }
  }
}

// Build the core data structures used during pattern search. This takes into
// account whether states are active; transitions in and out of inactive states
// are pruned from the graph.

void Graph::build_graph() {
  while (true) {
    gen_matrices();

    // deactivate any states with 0 outdegree or indegree
    std::vector<int> indegree(numstates + 1, 0);
    bool changed = false;

    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_active.at(i))
        continue;
      if (outdegree[i] == 0) {
        state_active.at(i) = false;
        changed = true;
        continue;
      }

      for (int j = 0; j < outdegree[i]; ++j) {
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

  find_exclude_states();
  find_exit_cycles();
}

// Calculate an upper bound on the length of prime patterns in the graph.

int Graph::prime_length_bound() const {
  // when there is more than one shift cycle, a prime pattern has to miss at
  // least one state in each shift cycle it visits

  int result = 0;
  std::vector<bool> all_active(numcycles, true);
  std::vector<bool> any_active(numcycles, false);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      ++result;
      any_active.at(cyclenum[i]) = true;
    } else {
      all_active.at(cyclenum[i]) = false;
    }
  }

  int cycles_active = std::count(any_active.begin(), any_active.end(), true);
  if (cycles_active > 1) {
    for (size_t i = 0; i < numcycles; ++i) {
      if (any_active.at(i) && all_active.at(i))
        --result;
    }
  }
  return result;
}

// Calculate an upper bound on the length of superprime patterns in the graph.

int Graph::superprime_length_bound() const {
  std::vector<bool> any_active(numcycles, false);

  for (size_t i = 1; i <= numstates; ++i) {
    if (state_active.at(i)) {
      any_active.at(cyclenum[i]) = true;
    }
  }

  return std::count(any_active.begin(), any_active.end(), true);
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Return the index in the `state` array that corresponds to a given state.
// Returns -1 if not found.

int Graph::get_statenum(const State& s) const {
  for (int i = 1; i <= numstates; ++i) {
    if (state.at(i) == s)
      return i;
  }
  return -1;
}

// Return the state number that comes from advancing a given state by a single
// throw. Returns -1 if the throw results in a collision.

int Graph::advance_state(int statenum, int throwval) const {
  if (throwval < 0 || throwval > state.at(statenum).h)
    return -1;
  if (throwval > 0 && state.at(statenum).slot.at(0) == 0)
    return -1;
  if (throwval < state.at(statenum).h &&
      state.at(statenum).slot.at(throwval) != 0)
    return -1;

  return get_statenum(state.at(statenum).advance_with_throw(throwval));
}

// Return the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the `state` vector).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

int Graph::reverse_state(int statenum) const {
  return get_statenum(state.at(statenum).reverse());
}

// Return the next state downstream in the given state's shift cycle

int Graph::downstream_state(int statenum) const {
  return get_statenum(state.at(statenum).downstream());
}

// Return the next state upstream in the given state's shift cycle

int Graph::upstream_state(int statenum) const {
  return get_statenum(state.at(statenum).upstream());
}

// Return a text representation of a given state number

std::string Graph::state_string(int statenum) const {
  return state.at(statenum).to_string();
}
