//
// WorkAssignment.h
//
// Defines a work assignment to hand off between workers. Together with
// SearchConfig these completely define the computation.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKASSIGNMENT_H_
#define JPRIME_WORKASSIGNMENT_H_

#include "Graph.h"

#include <iostream>
#include <list>
#include <vector>
#include <string>


class WorkAssignment {
 public:
  // lowest value of `start_state` for search; 0 auto-calculates based on
  // command-line flags
  unsigned start_state = 0;

  // highest value of `start_state` for search; 0 auto-calculates based on
  // command-line flags
  unsigned end_state = 0;

  // lowest value of `pos` that still has unexplored throw options in the
  // search tree; used for splitting work
  unsigned root_pos = 0;

  // set of unexplored throw options at `pos`==`root_pos`
  std::list<unsigned> root_throwval_options;

  // sequence of throws comprising the current position in the search tree
  std::vector<unsigned> partial_pattern;

 public:
  // methods to convert to/from string representation
  bool from_string(const std::string& str);
  std::string to_string() const;

  // methods for work splitting
  WorkAssignment split(const Graph& graph, unsigned split_alg = 0);

 private:
  WorkAssignment split_takestartstates();
  WorkAssignment split_takeall(const Graph& graph);
  WorkAssignment split_takehalf(const Graph& graph);
  WorkAssignment split_takefraction(const Graph& graph, double f,
      bool take_front);
};

bool work_assignment_compare(const WorkAssignment& wa1,
  const WorkAssignment& wa2);

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
