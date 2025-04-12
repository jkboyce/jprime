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


/*
Rules about WorkAssignments

A WorkAssignment encapsulates all information needed to define a portion of the
overall search tree. It is the unit of work that is handed off between workers,
and split as needed.

We start DFS from a particular state in the graph, `start_state`. At any given
point in the search:
- `partial_pattern` represents the sequence of throws from `start_state` to the
  current node.
- `root_pos` represents the lowest tree depth with unexplored options; it always
  has a value in the range [0, partial_pattern.size()].
- `root_throwval_options` represents the set of unexplored throw values at depth
  `root_pos`.
- `end_state` represents the largest value of `start_state` to search over.

A WorkAssignment is in one of two states: Uninitialized or initialized.
- An uninitialized assignment has empty `root_throwval_options`, empty
  `partial_pattern`, and `root_pos` == 0. It represents an entire search tree
  from `start_state`, where the search has not yet begun.
- An initialized assignment has nonempty `root_throwval_options`. To begin the
  search from `start_state`, the assignment is first initialized. From that
  point onward all assignments with that value of `start_state`, including
  splits, will be created in an initialized condition.

It is possible that an assignment cannot be initialized, when `start_state` has
no outgoing links. This only occurs when `start_state` is an inactive state in
the graph.

Splitting a WorkAssignment consists of taking either:
(a) values of `start_state` where the search has not started, or
(b) some or all of the `root_throwval_options` values at depth `root_pos`.

In case (a) the new assignment is uninitialized. In case (b) it is initialized,
with the same value of `root_pos` as the original assignment, and the same
`partial_pattern` up to index (root_pos - 1). In case (b) the original
assignment has its `root_throwval_options` updated, and potentially `root_pos`
as well (always at least as large as the prior value).

If splitting is done via method (b) above, not all WorkAssignments can be split!
In particular, if the operation would leave the original assignment with no
remaining options at `root_pos`, and no depth `d` with unexplored options
between root_pos and the current search depth (root_pos < d <= pos), then it
is considered unsplittable.
*/

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

 public:
  void build_rootpos_throw_options(const Graph& graph, unsigned from_state,
    unsigned start_column);
};

bool work_assignment_compare(const WorkAssignment& wa1,
  const WorkAssignment& wa2);

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
