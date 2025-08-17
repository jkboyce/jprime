//
// WorkAssignment.h
//
// Defines a work assignment to hand off between workers.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKASSIGNMENT_H_
#define JPRIME_WORKASSIGNMENT_H_

#include "Graph.h"
#include "WorkSpace.h"

#include <iostream>
#include <list>
#include <vector>
#include <string>


/*
About WorkAssignments
---------------------

A WorkAssignment is the unit of work that is given to an individual worker
thread to complete. WorkAssignments can be split as needed to distribute the
search over an arbitrary number of workers. A WorkAssignment specifies both
(a) a subset of the overall search tree, and (b) the current state of the search
over that subtree. When the user interrupts a search, the current
WorkAssignments for all workers are collected and saved to a checkpoint file,
which may be resumed later.

Invariants to maintain:
- The results of a search (patterns, pattern counts, node counts) are invariant
  with respect to: (a) how WorkAssignments are split, (b) how many workers
  execute them, (c) the timing and number of times the search is interrupted,
  saved to a checkpoint file, and resumed, and (d) the order in which
  WorkAssignments are executed. The order in which patterns are found during the
  search is not an invariant, however.
- All valid WorkAssignments should be unchanged by the following round trips:
  (a) saving and loading from a file, (b) saving and loading from a string, and
  (c) saving and loading from a WorkSpace (the latter excluding assignments of
  type STARTUP which cannot be written to a WorkSpace).
- We can determine the splittability of a WorkAssignment, and split a
  WorkAssignment, offline using only the Graph object.
- Search workers only interrupt the search process when the following conditions
  hold: (a) the sequence of throws in pp is a valid partial path, (b) the
  sequence is not a complete pattern, and (c) we are cleared to advance one
  level deeper in the search tree into an unconstrained set of options.

We start DFS from a particular state in the graph, `start_state`. At any given
point in the search:
- `end_state` represents the largest value of start_state to search over. We do
  independent full-tree searches for each subsequent value of start_state, up to
  and including end_state.
- `partial_pattern` (pp) specifies the current location in the search tree, as
  reached from start_state via the sequence of throws in pp. pp.size() is the
  current search depth.
- `root_pos` (rp) is the shallowest tree depth with unexplored options; it
  always has a value in the range [0, pp.size()]. The worker's search from
  start_state concludes as soon as it backtracks to a depth < rp.
- `root_throwval_options` (rto) is the set of unexplored throw values for
  advancing the search deeper from rp (except when rto is empty; see below).

There are three kinds of valid WorkAssignments:
- STARTUP assignment, signified by start_state == end_state == 0. This
  represents an entire search that has not yet begun; values of start_state and
  end_state need to be initialized based on the search config. In this case we
  must have pp.size() == rp == rto.size() == 0.
- SPLITTABLE assignment, signified by rp < pp.size(). This represents a search
  that is in progress, in particular the state of the search after advancing to
  depth pp.size() via the sequence of throws in pp, immediately before advancing
  to the next depth. In this case we must have rto.size() > 0.
- UNSPLITTABLE assignment, signified by rp == pp.size(). In this case we must
  have rto.size() == 0, which signifies "all possible values" at depth rp. In
  this case the only unexplored options in our subtree are the ones we are about
  to advance into; hence the assignment is unsplittable.

Splitting WorkAssignments. A WorkAssignment W can be split in two ways:
- By stealing unexplored values of start_state. If W has
  end_state > start_state, some the unexplored values of start_state can be
  split off into a new WorkAssignment.
- By stealing unexplored throw values in rto (SPLITTABLE type only). In this
  case one or more of the unexplored throw values is transferred to a new
  WorkAssignment, which has the same pp as W, up to and including depth rp.

*/

class WorkAssignment {
 public:
  enum class Type {
    INVALID,
    STARTUP,
    SPLITTABLE,
    UNSPLITTABLE,
  };

  // current value of `start_state` for search; 0 auto-calculates based on
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

  // sequence of throws from `start_state` comprising the current position in
  // the search tree
  std::vector<unsigned> partial_pattern;

  // utility methods
  Type get_type() const;
  bool is_valid() const;
  bool operator==(const WorkAssignment& wa2) const;
  bool operator!=(const WorkAssignment& wa2) const;
  void build_rootpos_throw_options(const Graph& graph, unsigned from_state,
    unsigned start_column);

  // converting to/from a string representation
  std::string to_string() const;
  bool from_string(const std::string& str);

  // loading into/reading from a WorkSpace
  void to_workspace(WorkSpace* ws, unsigned slot) const;
  void from_workspace(const WorkSpace* ws, unsigned slot);

  // work splitting
  bool is_splittable() const;
  WorkAssignment split(const Graph& graph, unsigned split_alg = 1);

 private:
  WorkAssignment split_takestartstates();
  WorkAssignment split_takeall(const Graph& graph);
  WorkAssignment split_takehalf(const Graph& graph);
  WorkAssignment split_takefraction(const Graph& graph, double f);
};

bool work_assignment_compare(const WorkAssignment& wa1,
  const WorkAssignment& wa2);

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
