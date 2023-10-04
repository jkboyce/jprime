//
// Graph.h
//
// Data structures related to the juggling graph for N objects, max throw H.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_GRAPH_H_
#define JPRIME_GRAPH_H_

#include <string>
#include <vector>


class Graph {
 public:
  Graph(int n, int h, const std::vector<bool>& xa, bool ltwc);
  Graph(int n, int h);
  ~Graph();

 public:
  // calculated at construction and do not change
  const int n;
  const int h;
  const std::vector<bool> xarray;
  const bool linkthrows_within_cycle;
  int numstates = 0;
  unsigned long* state;
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
  int** cyclepartner;
  unsigned long highestbit = 0L;
  unsigned long allbits = 0L;

 private:
  void init();
  void allocate_arrays();
  void delete_arrays();
  static int num_states(int n, int h);
  static int gen_states(unsigned long* state, int num, int pos, int left,
      int h, int ns);
  void find_shift_cycles();
  void gen_matrices();

 public:
  int get_statenum(unsigned long st) const;
  int advance_state(int sstatenum, int throwval) const;
  int reverse_state(int statenum) const;
  int downstream_state(int statenum) const;
  int upstream_state(int statenum) const;
  std::string state_string(int statenum) const;
};

#endif
