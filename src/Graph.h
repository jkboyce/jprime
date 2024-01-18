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

#include <string>
#include <vector>
#include <cstdint>


class Graph {
 public:
  Graph(int n, int h, const std::vector<bool>& xa, bool ltwc, bool s0g);
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
  std::vector<bool> xarray;
  bool linkthrows_within_cycle;
  bool super0ground;
  int numstates = 0;
  std::uint64_t* state;
  int** outmatrix;
  int* outdegree;
  int maxoutdegree = 0;
  int** outthrowval;
  int** inmatrix;
  int* indegree;
  int maxindegree = 0;
  int numcycles = 0;
  int numshortcycles = 0;
  int* cyclenum;
  int* cycleperiod;
  bool* isexitcycle;
  int** cyclepartner;
  std::uint64_t highestbit = 0;
  std::uint64_t allbits = 0;

 private:
  void init();
  void allocate_arrays();
  void delete_arrays();
  static int num_states(int n, int h);
  static int gen_states(std::uint64_t* state, int num, int pos, int left,
      int h, int ns);
  int cluster_count(std::uint64_t s);
  void find_shift_cycles();
  void gen_matrices();
  void prune_graph();

 public:
  int get_statenum(std::uint64_t st) const;
  int advance_state(int sstatenum, int throwval) const;
  int reverse_state(int statenum) const;
  int downstream_state(int statenum) const;
  int upstream_state(int statenum) const;
  std::string state_string(int statenum) const;
};

#endif
