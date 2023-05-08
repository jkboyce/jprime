
#ifndef JDEEP_WORKASSIGNMENT_H
#define JDEEP_WORKASSIGNMENT_H

#include <iostream>
#include <list>
#include <vector>
#include <string>

// Defines a work assignment to hand off between workers. Together with
// SearchConfig these completely define the computation.

struct WorkAssignment {
  // lowest value of `start_state` for search; -1 auto-calculates based on
  // command-line flags
  int start_state = -1;
  // highest value of `start_state` for search; -1 auto-calculates based on
  // command-line flags
  int end_state = -1;
  // lowest value of `pos` that still has unexplored throw options in the
  // search tree; used for splitting work
  int root_pos = 0;
  // set of unexplored throw options at `pos`==`root_pos`
  std::list<int> root_throwval_options;
  // sequence of throws comprising the current position in the search tree,
  // starting from root node to the latest valid search position
  std::vector<int> partial_pattern;

  // initialize from a string; return true on success, false otherwise
  bool from_string(std::string str);
};

// output a text representation
std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
