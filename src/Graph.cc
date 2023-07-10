//
// Graph.cc
//
// Data structures related to the juggling graph for N objects, max throw H.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Graph.h"

#include <iostream>
#include <vector>


Graph::Graph(int n, int h, const std::vector<bool>& xa, bool ltwc) :
      n(n), h(h), xarray(xa), linkthrows_within_cycle(ltwc) {
  numstates = num_states(n, h);
  for (int i = 0; i <= h; ++i) {
    if (!xarray[i])
      ++maxoutdegree;
  }
  maxoutdegree = std::min(maxoutdegree, h - n + 1);
  maxindegree = n + 1;
  highestbit = 1L << (h - 1);
  allbits = (1L << h) - 1;

  allocate_arrays();
  int ns = gen_states(state, 0, h - 1, n, h, numstates);
  assert(ns == numstates);
  find_shift_cycles();
  gen_matrices();
}

Graph::~Graph() {
  delete_arrays();
}

//------------------------------------------------------------------------------
// Prep core data structures during construction
//------------------------------------------------------------------------------

// Allocate all arrays used by the graph and initialize to default values.

void Graph::allocate_arrays() {
  outdegree = new int[numstates + 1];
  indegree = new int[numstates + 1];
  cyclenum = new int[numstates + 1];
  state = new unsigned long[numstates + 1];
  cycleperiod = new int[numstates + 1];

  for (int i = 0; i <= numstates; ++i) {
    outdegree[i] = 0;
    indegree[i] = 0;
    cyclenum[i] = 0;
    state[i] = 0L;
    cycleperiod[i] = 0;
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
  outmatrix = nullptr;
  outthrowval = nullptr;
  inmatrix = nullptr;
  cyclepartner = nullptr;
  outdegree = nullptr;
  indegree = nullptr;
  cyclenum = nullptr;
  state = nullptr;
  cycleperiod = nullptr;
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

int Graph::gen_states(unsigned long* state, int num, int pos, int left, int h,
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
    state[num + 1] |= (1L << pos);
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

void Graph::find_shift_cycles() {
  const unsigned long lowerbits = highestbit - 1;
  int cycleindex = 0;

  for (int i = 1; i <= numstates; ++i) {
    unsigned long statebits = state[i];
    bool periodfound = false;
    bool newshiftcycle = true;
    int cycleper = h;

    for (int j = 0; j < h; ++j) {
      if (statebits & highestbit)
        statebits = (statebits & lowerbits) << 1 | 1L;
      else
        statebits <<= 1;

      int k = 1;
      for (; k <= numstates; ++k) {
        if (state[k] == statebits)
          break;
      }
      assert(k <= numstates);

      cyclepartner[i][j] = k;
      if (k == i && !periodfound) {
        cycleper = j + 1;
        periodfound = true;
      } else if (k < i)
        newshiftcycle = false;
    }

    if (newshiftcycle) {
      for (int j = 0; j < h; j++)
        cyclenum[cyclepartner[i][j]] = cycleindex;
      cycleperiod[cycleindex++] = cycleper;
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
          unsigned long temp = state[i] >> 1;
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
        unsigned long temp = (unsigned long)1L << (j - 1);
        unsigned long temp2 = (state[i] >> 1);

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
          }
        }
      }
    }

    // then take care of ingoing throws
    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      if (j == 0) {
        if (!(state[i] & (1L << (h - 1)))) {
          unsigned long temp = state[i] << 1;

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
        if (state[i] & (1L << (h - 1))) {
          unsigned long temp = state[i] ^ (1L << (h - 1));
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
        if ((state[i] & (1L << (j - 1))) && !(state[i] & (1L << (h - 1)))) {
          unsigned long temp = state[i] ^ (1L << (j - 1));
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
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Find the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the state[] array).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

int Graph::reverse_state(int statenum) const {
  unsigned long temp = 0;
  unsigned long mask1 = 1L;
  unsigned long mask2 = 1L << (h - 1);

  while (mask2) {
    assert(statenum >= 0 && statenum < numstates);
    if (state[statenum] & mask2)
      temp |= mask1;
    mask1 <<= 1;
    mask2 >>= 1;
  }

  for (int i = 1; i <= numstates; ++i) {
    if (state[i] == temp)
      return i;
  }
  assert(false);
}

// Return a text representation of a given state number

std::string Graph::state_string(int statenum) const {
  std::string result;
  unsigned long value = state[statenum];
  for (int i = 0; i < h; ++i) {
    result += (value & 1 ? 'x' : '-');
    value >>= 1;
  }
  return result;
}
