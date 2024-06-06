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
#include <map>
#include <tuple>
#include <cstdint>


class Graph {
 public:
  Graph(unsigned int b, unsigned int h, const std::vector<bool>& xa, bool ltwc,
      unsigned int l = 0);
  Graph(unsigned int b, unsigned int h);
  Graph() = default;

 public:
  // calculated at construction and do not change
  unsigned int b = 0;  // number of objects
  unsigned int h = 0;  // maximum throw value
  unsigned int l = 0;  // if nonzero then single-period graph
  std::vector<bool> xarray;
  bool linkthrows_within_cycle = true;
  unsigned int numstates = 0;
  unsigned int maxoutdegree = 0;
  unsigned int numcycles = 0;
  unsigned int numshortcycles = 0;
  std::vector<State> state;
  std::vector<unsigned int> cyclenum;
  std::vector<unsigned int> cycleperiod;

  // updated as states are activated/deactivated
  std::vector<bool> state_active;
  std::vector<std::vector<unsigned int>> outmatrix;
  std::vector<unsigned int> outdegree;
  std::vector<std::vector<unsigned int>> outthrowval;
  std::vector<std::vector<unsigned int>> excludestates_throw;
  std::vector<std::vector<unsigned int>> excludestates_catch;
  std::vector<int> isexitcycle;

 private:
  void init();
  void find_shift_cycles();
  void gen_matrices();
  void find_exit_cycles();

  using op_key_type = std::tuple<unsigned int, unsigned int>;
  static void gen_states_all(std::vector<State>& s, unsigned int n,
    unsigned int h);
  static void gen_states_all_helper(std::vector<State>& s, unsigned int pos,
    unsigned int left);
  static void gen_states_for_period(std::vector<State>& s, unsigned int n,
    unsigned int h, unsigned int l);
  static void gen_states_for_period_helper(std::vector<State>& s,
    unsigned int pos, unsigned int left, unsigned int h, unsigned int l);
  static std::uint64_t ordered_partitions_helper(unsigned int pos,
    unsigned int left, const unsigned int h, const unsigned int l,
    std::map<op_key_type, std::uint64_t>& cache);

 public:
  void build_graph();
  void find_exclude_states();
  static std::uint64_t combinations(unsigned int a, unsigned int b);
  static std::uint64_t shift_cycle_count(unsigned int n, unsigned int h,
    unsigned int p);
  static std::uint64_t ordered_partitions(unsigned int n, unsigned int h,
    unsigned int l);
  unsigned int prime_length_bound() const;
  unsigned int superprime_length_bound() const;
  unsigned int get_statenum(const State& s) const;
  unsigned int advance_state(unsigned int statenum, unsigned int throwval)
    const;
  unsigned int reverse_state(unsigned int statenum) const;
  unsigned int downstream_state(unsigned int statenum) const;
  unsigned int upstream_state(unsigned int statenum) const;
  std::string state_string(unsigned int statenum) const;
};

#endif
