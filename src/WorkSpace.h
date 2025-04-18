//
// WorkSpace.h
//
// Class that abstracts the concept of an array of work cells, used for loading
// and unloading WorkAssignments.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKSPACE_H_
#define JPRIME_WORKSPACE_H_

#include "Graph.h"

#include <tuple>


class WorkSpace {
 public:
  virtual const Graph& get_graph() const = 0;
  virtual void set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state) = 0;
  virtual std::tuple<unsigned, unsigned, unsigned> get_cell(unsigned slot,
    unsigned index) const = 0;
  virtual void set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos) = 0;
  virtual std::tuple<unsigned, unsigned, int> get_info(unsigned slot) const = 0;
};

#endif
