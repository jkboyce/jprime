//
// WorkAssignment.cc
//
// Functions for reading a WorkAssignment from a string representation, and
// printing to an output stream.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "WorkAssignment.h"

#include <iostream>
#include <string>
#include <sstream>
#include <regex>
#include <cassert>
#include <stdexcept>


// Initialize from a string.
//
// Return true on success, false otherwise.

bool WorkAssignment::from_string(const std::string& str) {
  std::regex rgx(
      "\\{ start_state:([0-9]+), end_state:([0-9]+), root_pos:([0-9]+), "
      "root_options:\\[([0-9,]+)\\], current:\\\"([0-9,]*)\\\" \\}"
  );
  std::smatch matches;

  if (!std::regex_search(str, matches, rgx) || matches.size() != 6) {
    return false;
  }

  start_state = std::stoi(matches[1].str());
  end_state = std::stoi(matches[2].str());
  root_pos = std::stoi(matches[3].str());

  root_throwval_options.clear();
  const std::string tvo{matches[4].str()};
  auto x = tvo.cbegin();
  while (true) {
    const auto y = std::find(x, tvo.cend(), ',');
    const std::string s{x, y};
    root_throwval_options.push_back(std::stoi(s));
    if (y == tvo.cend())
      break;
    x = y + 1;
  }

  partial_pattern.clear();
  const std::string pp{matches[5].str()};
  x = pp.cbegin();
  while (true) {
    const auto y = std::find(x, pp.cend(), ',');
    const std::string s{x, y};
    partial_pattern.push_back(std::stoi(s));
    if (y == pp.cend())
      break;
    x = y + 1;
  }

  // std::cout << "result:" << std::endl << *this << std::endl;
  return true;
}

// Return a text representation.

std::string WorkAssignment::to_string() const {
  std::ostringstream buffer;

  buffer << "{ start_state:" << start_state
         << ", end_state:" << end_state
         << ", root_pos:" << root_pos
         << ", root_options:[";
  for (const unsigned v : root_throwval_options) {
    if (v != root_throwval_options.front()) {
      buffer << ',';
    }
    buffer << v;
  }
  buffer << "], current:\"";
  for (size_t i = 0; i < partial_pattern.size(); ++i) {
    if (i > 0) {
      buffer << ',';
    }
    buffer << partial_pattern.at(i);
  }
  buffer << "\" }";
  return buffer.str();
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

// Return a work assignment that corresponds to a portion of the current work
// assignment, for handing off to another worker.
//
// If a work assignment cannot be split, throw a std::invalid_argument
// exception with a relevant error message.

WorkAssignment WorkAssignment::split(const Graph& graph, unsigned split_alg) {
  if (end_state > start_state) {
    return split_takestartstates();
  }

  switch (split_alg) {
    case 1:
      return split_takeall(graph);
    default:
      return split_takehalf(graph);
  }
}

// Return a work assignment that corresponds to giving away approximately half
// of the unexplored `start_state` values in the current assignment.

WorkAssignment WorkAssignment::split_takestartstates() {
  unsigned takenum = (end_state - start_state + 1) / 2;
  assert(takenum > 0);
  assert(end_state >= start_state + takenum);

  WorkAssignment wa;
  wa.start_state = end_state - takenum + 1;
  wa.end_state = end_state;
  wa.root_pos = 0;

  end_state -= takenum;
  return wa;
}

// Return a work assignment that gives away all of the unexplored throw options
// at root_pos.

WorkAssignment WorkAssignment::split_takeall(const Graph& graph) {
  return split_takefraction(graph, 1, false);
}

// Return a work assignment that gives away approximately half of the unexplored
// throw options at root_pos.

WorkAssignment WorkAssignment::split_takehalf(const Graph& graph) {
  return split_takefraction(graph, 0.5, false);
}

// Return a work assignment that gives away approximately the target fraction of
// the unexplored throw options at root_pos.

WorkAssignment WorkAssignment::split_takefraction(const Graph& graph, double f,
      bool take_front) {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  for (size_t i = 0; i < root_pos; ++i) {
    wa.partial_pattern.push_back(partial_pattern.at(i));
  }

  // ensure the throw value at `root_pos` isn't on the list of throw options
  auto iter = root_throwval_options.begin();
  auto end = root_throwval_options.end();
  while (iter != end) {
    if (*iter == partial_pattern.at(root_pos)) {
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
  }
  if (root_throwval_options.size() == 0) {
    throw std::invalid_argument(
        "Cannot split work assignment; root_throw_options_1");
  }

  // move `take_count` unexplored root_pos options to the new work assignment
  auto take_count =
      static_cast<size_t>(0.51 + f * root_throwval_options.size());
  take_count = std::min(std::max(take_count, static_cast<size_t>(1)),
      root_throwval_options.size());

  const auto take_begin_idx = static_cast<size_t>(take_front ?
        0 : root_throwval_options.size() - take_count);
  const auto take_end_idx = take_begin_idx + take_count;

  iter = root_throwval_options.begin();
  end = root_throwval_options.end();
  for (size_t index = 0; iter != end; ++index) {
    if (index >= take_begin_idx && index < take_end_idx) {
      wa.root_throwval_options.push_back(*iter);
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
  }

  // did we give away all our throw options at `root_pos`?
  if (root_throwval_options.size() == 0) {
    // Find the shallowest depth `new_root_pos` where there are unexplored throw
    // options. We have no more options at the current root_pos, so
    // new_root_pos > root_pos.
    //
    // We're also at a point in the search where we know there are unexplored
    // options remaining somewhere between `root_pos` and `pos`; see e.g.
    // Worker::iterative_can_split().
    //
    // So we know there must be a value of `new_root_pos` with the properties we
    // need in the range root_pos < new_root_pos <= pos.

    unsigned from_state = start_state;
    unsigned new_root_pos = -1;
    unsigned col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (unsigned i = 0; i < partial_pattern.size(); ++i) {
      const auto throwval = static_cast<unsigned>(partial_pattern.at(i));
      for (col = 0; col < graph.outdegree.at(from_state); ++col) {
        if (throwval == graph.outthrowval.at(from_state).at(col)) {
          break;
        }
      }
      // diagnostics if there's a problem
      if (col == graph.outdegree.at(from_state)) {
        std::cerr << "i = " << i
                  << ", from_state = " << from_state
                  << ", start_state = " << start_state
                  << ", root_pos = " << root_pos
                  << ", col = " << col
                  << ", throwval = " << throwval
                  << '\n';
      }
      assert(col != graph.outdegree.at(from_state));

      if (i > root_pos && col < graph.outdegree.at(from_state) - 1) {
        new_root_pos = i;
        break;
      }

      from_state = graph.outmatrix.at(from_state).at(col);
    }
    if (new_root_pos == -1u) {
      throw std::invalid_argument("Cannot split work assignment; root_pos");
    }

    root_pos = new_root_pos;

    // rebuild the list of throw options at `root_pos`
    root_throwval_options.clear();
    for (unsigned i = col + 1; i < graph.outdegree.at(from_state); ++i) {
      root_throwval_options.push_back(graph.outthrowval.at(from_state).at(i));
    }
    if (root_throwval_options.size() == 0) {
      throw std::invalid_argument(
          "Cannot split work assignment; root_throw_options_2");
    }
  }

  return wa;
}

//------------------------------------------------------------------------------
// Free functions
//------------------------------------------------------------------------------

// Standard library compliant Compare relation for work assignments.
//
// Returns true if the first argument appears before the second in a strict
// weak ordering, and false otherwise. When sorting this puts the best splitting
// targets at the beginning.

bool work_assignment_compare(const WorkAssignment& wa1,
      const WorkAssignment& wa2) {
  auto diff = (wa1.end_state - wa1.start_state) <=>
      (wa2.end_state - wa2.start_state);
  if (diff > 0) {
    return true;
  }
  if (diff < 0) {
    return false;
  }
  diff = wa1.root_pos <=> wa2.root_pos;
  if (diff < 0) {
    return true;
  }
  if (diff > 0) {
    return false;
  }
  diff = wa1.root_throwval_options.size() <=> wa2.root_throwval_options.size();
  if (diff > 0) {
    return true;
  }
  return false;
}

// Print to an output stream using the default text format.

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa) {
  ost << wa.to_string();
  return ost;
}
