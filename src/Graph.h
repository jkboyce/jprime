//
// Graph.h
//
// Data structures related to the juggling graph for N objects, max throw H.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_GRAPH_H_
#define JPRIME_GRAPH_H_

#include "State.h"

#include <string>
#include <vector>
#include <cstdint>


class Graph {
 public:
  Graph(int n, int h, const std::vector<bool>& xa, bool ltwc);
  Graph(int n, int h);
  Graph(const Graph& g);
  Graph(Graph&&) =delete;
  Graph& operator=(const Graph& g);
  Graph& operator=(Graph&&) =delete;
  ~Graph();

 public:
  // calculated at construction and do not change
  int n;
  int h;
  std::vector<State> state;
  std::vector<bool> xarray;
  bool linkthrows_within_cycle;
  int numstates = 0;
  int maxoutdegree = 0;
  int numcycles = 0;
  int numshortcycles = 0;
  int* cyclenum;
  int* cycleperiod;

  // updated as states are activated/deactivated
  std::vector<bool> state_active;
  int** outmatrix;
  int* outdegree;
  int** outthrowval;
  int** excludestates_throw;
  int** excludestates_catch;
  bool* isexitcycle;

 private:
  void init();
  void allocate_arrays();
  void delete_arrays();
  static int num_states(int n, int h);
  int gen_states(int num, int pos, int left);
  void find_shift_cycles();
  void gen_matrices();
  void find_exclude_states();
  void find_exit_cycles();

 public:
  void build_graph();
  int prime_length_bound() const;
  int superprime_length_bound() const;
  int get_statenum(const State& s) const;
  int advance_state(int statenum, int throwval) const;
  int reverse_state(int statenum) const;
  int downstream_state(int statenum) const;
  int upstream_state(int statenum) const;
  std::string state_string(int statenum) const;
};

#endif
