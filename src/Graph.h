//
// Graph.h
//
// Data structures related to the juggling graph for B objects, max throw H.
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
  Graph(unsigned b, unsigned h, const std::vector<bool>& xa, bool ltwc,
      unsigned l = 0);
  Graph(unsigned b, unsigned h);
  Graph() = default;

 public:
  // calculated at construction and do not change
  unsigned b = 0;  // number of objects
  unsigned h = 0;  // maximum throw value
  unsigned l = 0;  // if nonzero then single-period graph
  std::vector<bool> xarray;
  bool linkthrows_within_cycle = true;
  unsigned numstates = 0;
  unsigned maxoutdegree = 0;
  unsigned numcycles = 0;
  unsigned numshortcycles = 0;
  std::vector<State> state;
  std::vector<unsigned> cyclenum;
  std::vector<unsigned> cycleperiod;

  // updated as states are activated/deactivated
  std::vector<bool> state_active;
  std::vector<std::vector<unsigned>> outmatrix;
  std::vector<unsigned> outdegree;
  std::vector<std::vector<unsigned>> outthrowval;
  std::vector<std::vector<unsigned>> excludestates_throw;
  std::vector<std::vector<unsigned>> excludestates_catch;
  std::vector<int> isexitcycle;

 private:
  void init();
  void find_shift_cycles();
  void find_exit_cycles();

  using op_key_type = std::tuple<unsigned, unsigned>;
  static void gen_states_all(std::vector<State>& s, unsigned b, unsigned h);
  static void gen_states_all_helper(std::vector<State>& s, unsigned pos,
    unsigned left);
  static void gen_states_for_period(std::vector<State>& s, unsigned b,
    unsigned h, unsigned l);
  static void gen_states_for_period_helper(std::vector<State>& s, unsigned pos,
    unsigned left, unsigned h, unsigned l);
  static std::uint64_t ordered_partitions_helper(unsigned pos, unsigned left,
    const unsigned h, const unsigned l,
    std::map<op_key_type, std::uint64_t>& cache);

 public:
  void build_graph();
  void reduce_graph();
  void find_exclude_states();
  static std::uint64_t combinations(unsigned a, unsigned b);
  static std::uint64_t shift_cycle_count(unsigned b, unsigned h, unsigned p);
  static std::uint64_t ordered_partitions(unsigned b, unsigned h, unsigned l);
  unsigned prime_length_bound() const;
  unsigned superprime_length_bound() const;
  unsigned get_statenum(const State& s) const;
  unsigned advance_state(unsigned statenum, unsigned throwval) const;
  unsigned reverse_state(unsigned statenum) const;
  unsigned downstream_state(unsigned statenum) const;
  unsigned upstream_state(unsigned statenum) const;
  std::string state_string(unsigned statenum) const;
};

#endif
