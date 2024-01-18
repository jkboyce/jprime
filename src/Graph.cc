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
#include <vector>
#include <cassert>


Graph::Graph(int n, int h, const std::vector<bool>& xa, bool ltwc, bool s0g)
    : n(n), h(h), xarray(xa), linkthrows_within_cycle(ltwc),
      super0ground(s0g) {
  init();
}

Graph::Graph(int n, int h)
    : n(n), h(h), xarray(h + 1, false), linkthrows_within_cycle(true),
      super0ground(false) {
  init();
}

Graph::Graph(const Graph& g)
    : Graph(g.n, g.h, g.xarray, g.linkthrows_within_cycle, g.super0ground) {
}

Graph& Graph::operator=(const Graph& g) {
  if (this == &g)
    return *this;

  delete_arrays();
  n = g.n;
  h = g.h;
  xarray = g.xarray;
  linkthrows_within_cycle = g.linkthrows_within_cycle;
  super0ground = g.super0ground;
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
    if (!xarray[i])
      ++maxoutdegree;
  }
  maxoutdegree = std::min(maxoutdegree, h - n + 1);
  maxindegree = n + 1;
  highestbit = static_cast<uint64_t>(1) << (h - 1);
  allbits = (static_cast<uint64_t>(1) << h) - 1;

  allocate_arrays();
  int ns = gen_states(state, 0, h - 1, n, h, numstates);
  assert(ns == numstates);
  find_shift_cycles();
  gen_matrices();
}

// Allocate all arrays used by the graph and initialize to default values.

void Graph::allocate_arrays() {
  outdegree = new int[numstates + 1];
  indegree = new int[numstates + 1];
  cyclenum = new int[numstates + 1];
  state = new std::uint64_t[numstates + 1];
  cycleperiod = new int[numstates + 1];
  isexitcycle = new bool[numstates + 1];

  for (int i = 0; i <= numstates; ++i) {
    outdegree[i] = 0;
    indegree[i] = 0;
    cyclenum[i] = 0;
    state[i] = 0;
    cycleperiod[i] = 0;
    isexitcycle[i] = false;
  }

  outmatrix = new int*[numstates + 1];
  outthrowval = new int*[numstates + 1];
  inmatrix = new int*[numstates + 1];
  cyclepartner = new int*[numstates + 1];

  for (int i = 0; i <= numstates; ++i) {
    outmatrix[i] = new int[maxoutdegree];
    outthrowval[i] = new int[maxoutdegree];
    inmatrix[i] = new int[maxindegree];
    cyclepartner[i] = new int[h];

    for (int j = 0; j < maxoutdegree; ++j) {
      outmatrix[i][j] = 0;
      outthrowval[i][j] = 0;
    }
    for (int j = 0; j < maxindegree; ++j)
      inmatrix[i][j] = 0;
    for (int j = 0; j < h; ++j)
      cyclepartner[i][j] = 0;
  }
}

void Graph::delete_arrays() {
  for (int i = 0; i <= numstates; ++i) {
    if (outmatrix) {
      delete[] outmatrix[i];
      outmatrix[i] = nullptr;
    }
    if (outthrowval) {
      delete[] outthrowval[i];
      outthrowval[i] = nullptr;
    }
    if (inmatrix) {
      delete[] inmatrix[i];
      inmatrix[i] = nullptr;
    }
    if (cyclepartner) {
      delete[] cyclepartner[i];
      cyclepartner[i] = nullptr;
    }
  }

  delete[] outmatrix;
  delete[] outthrowval;
  delete[] inmatrix;
  delete[] cyclepartner;
  delete[] outdegree;
  delete[] indegree;
  delete[] cyclenum;
  delete[] state;
  delete[] cycleperiod;
  delete[] isexitcycle;
  outmatrix = nullptr;
  outthrowval = nullptr;
  inmatrix = nullptr;
  cyclepartner = nullptr;
  outdegree = nullptr;
  indegree = nullptr;
  cyclenum = nullptr;
  state = nullptr;
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

// Generate the list of all possible states into the state[] array.
//
// Returns the number of states found.

int Graph::gen_states(std::uint64_t* state, int num, int pos, int left, int h,
      int ns) {
  if (left > (pos + 1))
    return num;

  if (pos == 0) {
    if (left)
      state[num + 1] |= 1L;
    else
      state[num + 1] &= ~1L;

    if (num < (ns - 1))
      state[num + 2] = state[num + 1];
    return (num + 1);
  }

  state[num + 1] &= ~(1L << pos);
  num = gen_states(state, num, pos - 1, left, h, ns);
  if (left > 0) {
    state[num + 1] |= (static_cast<uint64_t>(1) << pos);
    num = gen_states(state, num, pos - 1, left - 1, h, ns);
  }

  return num;
}

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period
// - The other states on a given state's shift cycle:
//         cyclepartner[statenum][i] --> statenum  (where i < h)
//   where by convention:
//         cyclepartner[statenum][0] = upstream_state(statenum)
//         cyclepartner[statenum][h - 1] = statenum

void Graph::find_shift_cycles() {
  const std::uint64_t lowerbits = highestbit - 1;
  int cycleindex = 0;

  for (int i = 1; i <= numstates; ++i) {
    std::uint64_t statebits = state[i];
    bool periodfound = false;
    bool newshiftcycle = true;
    int cycleper = h;

    for (int j = 0; j < h; ++j) {
      if (statebits & highestbit)
        statebits = (statebits & lowerbits) << 1 | 1L;
      else
        statebits <<= 1;

      int k = get_statenum(statebits);
      assert(k > 0);

      cyclepartner[i][j] = k;
      if (k == i && !periodfound) {
        cycleper = j + 1;
        periodfound = true;
      } else if (k < i)
        newshiftcycle = false;
    }
    assert(cyclepartner[i][h - 1] == i);

    if (newshiftcycle) {
      for (int j = 0; j < h; j++)
        cyclenum[cyclepartner[i][j]] = cycleindex;
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
// - Inward degree to each state in the graph:
//         indegree[statenum] --> degree
// - Inward connections to each state:
//         inmatrix[statenum][col] --> statenum  (where col < indegree)
//
// outmatrix[][] == 0 indicates no connection.

void Graph::gen_matrices() {
  for (int i = 1; i <= numstates; ++i) {
    int outthrownum = 0;
    int inthrownum = 0;

    // first take care of outgoing throws
    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      if (j == 0) {
        if (!(state[i] & 1L)) {
          std::uint64_t temp = state[i] >> 1;
          bool found = false;

          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              outmatrix[i][outthrownum] = k;
              outthrowval[i][outthrownum++] = j;
              ++outdegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else if (state[i] & 1L) {
        std::uint64_t temp = static_cast<std::uint64_t>(1) << (j - 1);
        std::uint64_t temp2 = (state[i] >> 1);

        if (!(temp2 & temp)) {
          temp |= temp2;

          bool found = false;
          int k = 1;
          for (; k <= numstates; ++k) {
            if (state[k] == temp) {
              found = true;
              break;
            }
          }
          assert(found);
          if (j == h || linkthrows_within_cycle || cyclenum[i] != cyclenum[k]) {
            outmatrix[i][outthrownum] = k;
            outthrowval[i][outthrownum++] = j;
            ++outdegree[i];
            if (k == 1 && cyclenum[i] != cyclenum[k])
              isexitcycle[cyclenum[i]] = true;
          }
        }
      }
    }

    // then take care of ingoing throws
    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      if (j == 0) {
        if (!(state[i] & (static_cast<uint64_t>(1) << (h - 1)))) {
          std::uint64_t temp = state[i] << 1;

          bool found = false;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              inmatrix[i][inthrownum++] = k;
              ++indegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else if (j == h) {
        if (state[i] & (static_cast<uint64_t>(1) << (h - 1))) {
          std::uint64_t temp = state[i] ^ (static_cast<uint64_t>(1) << (h - 1));
          temp = (temp << 1) | 1L;

          bool found = false;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              inmatrix[i][inthrownum++] = k;
              ++indegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else {
        if ((state[i] & (static_cast<uint64_t>(1) << (j - 1))) &&
            !(state[i] & (static_cast<uint64_t>(1) << (h - 1)))) {
          std::uint64_t temp = state[i] ^ (static_cast<uint64_t>(1) << (j - 1));
          temp = (temp << 1) | 1L;

          bool found = false;
          int k = 1;
          for (; k <= numstates; ++k) {
            if (state[k] == temp) {
              found = true;
              break;
            }
          }
          assert(found);
          if (linkthrows_within_cycle || cyclenum[i] != cyclenum[k]) {
            inmatrix[i][inthrownum++] = k;
            ++indegree[i];
          }
        }
      }
    }
  }

  prune_graph();
}

void Graph::prune_graph() {
  bool pruning = true;
  // int num_unusable = 0;
  std::vector<bool> unusable(numstates + 1, false);

  if (super0ground) {
    // optimization specific to "-super 0 -g" searches; all cycle partners of
    // the ground state are unusable
    for (int i = 0; i < h - 1; ++i) {
      if (cyclepartner[1][i] == 1)
        break;
      outdegree[cyclepartner[1][i]] = 0;
    }
  }

  while (pruning) {
    pruning = false;

    for (int i = 1; i <= numstates; ++i) {
      if (unusable[i])
        continue;

      if (outdegree[i] == 0) {
        unusable[i] = true;
        // ++num_unusable;
        pruning = true;
        continue;
      }

      for (int j = 0; j < outdegree[i]; ++j) {
        if (unusable[outmatrix[i][j]]) {
          for (int k = j; k < outdegree[i] - 1; ++k) {
            outmatrix[i][k] = outmatrix[i][k + 1];
            outthrowval[i][k] = outthrowval[i][k + 1];
          }
          --outdegree[i];
          --j;
          pruning = true;
        }
      }

      if (indegree[i] == 0) {
        unusable[i] = true;
        // ++num_unusable;
        pruning = true;
        continue;
      }

      for (int j = 0; j < indegree[i]; ++j) {
        if (unusable[inmatrix[i][j]]) {
          for (int k = j; k < indegree[i] - 1; ++k) {
            inmatrix[i][k] = inmatrix[i][k + 1];
          }
          --indegree[i];
          --j;
          pruning = true;
        }
      }
    }
  }
  // std::cout << num_unusable << " states pruned (out of "
  //           << numstates << ")" << std::endl;
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Return the index in the `state` array that corresponds to a given state
// (represented as a bit pattern). Returns -1 if not found.

int Graph::get_statenum(std::uint64_t st) const {
  for (int i = 1; i <= numstates; ++i) {
    if (state[i] == st)
      return i;
  }
  return -1;
}

// Return the state number that comes from advancing a given state by a single
// throw. Returns -1 if the throw results in a collision.

int Graph::advance_state(int statenum, int throwval) const {
  if ((state[statenum] & 1L) != 0 && throwval == 0)
    return -1;
  if ((state[statenum] & 1L) == 0 && throwval != 0)
    return -1;

  std::uint64_t new_state = state[statenum] >> 1;
  if (throwval > 0) {
    std::uint64_t mask = static_cast<uint64_t>(1) << (throwval - 1);
    if (new_state & mask)
      return -1;
    new_state |= mask;
  }

  return get_statenum(new_state);
}

// Return the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the state[] array).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

int Graph::reverse_state(int statenum) const {
  if (statenum <= 0 || statenum > numstates)
    std::cerr << "bad statenum: " << statenum << std::endl;
  assert(statenum > 0 && statenum <= numstates);

  std::uint64_t new_state = 0;
  std::uint64_t mask1 = 1L;
  std::uint64_t mask2 = highestbit;

  while (mask2) {
    if (state[statenum] & mask2)
      new_state |= mask1;
    mask1 <<= 1;
    mask2 >>= 1;
  }

  return get_statenum(new_state);
}

// Return the next state downstream in the given state's shift cycle

int Graph::downstream_state(int statenum) const {
  std::uint64_t new_state = state[statenum] >> 1;

  if (state[statenum] & 1L)
    new_state |= highestbit;

  return get_statenum(new_state);
}

// Return the next state upstream in the given state's shift cycle

int Graph::upstream_state(int statenum) const {
  std::uint64_t new_state = state[statenum] << 1;

  if (new_state > allbits) {
    new_state ^= allbits;
    new_state |= 1L;
  }

  return get_statenum(new_state);
}

// Return a text representation of a given state number

std::string Graph::state_string(int statenum) const {
  std::string result;
  std::uint64_t value = state[statenum];
  for (int i = 0; i < h; ++i) {
    result += (value & 1 ? 'x' : '-');
    value >>= 1;
  }
  return result;
}
