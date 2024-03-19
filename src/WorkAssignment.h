//
// WorkAssignment.h
//
// Defines a work assignment to hand off between workers. Together with
// SearchConfig these completely define the computation.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKASSIGNMENT_H_
#define JPRIME_WORKASSIGNMENT_H_

#include <iostream>
#include <list>
#include <vector>
#include <string>


struct WorkAssignment {
  // lowest value of `start_state` for search; 0 auto-calculates based on
  // command-line flags
  unsigned int start_state = 0;

  // highest value of `start_state` for search; 0 auto-calculates based on
  // command-line flags
  unsigned int end_state = 0;

  // lowest value of `pos` that still has unexplored throw options in the
  // search tree; used for splitting work
  unsigned int root_pos = 0;

  // set of unexplored throw options at `pos`==`root_pos`
  std::list<unsigned int> root_throwval_options;

  // sequence of throws comprising the current position in the search tree
  std::vector<unsigned int> partial_pattern;

  bool from_string(std::string str);
};

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
