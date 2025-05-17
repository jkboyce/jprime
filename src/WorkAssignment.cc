//
// WorkAssignment.cc
//
// Defines a work assignment to hand off between workers.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "WorkAssignment.h"
#include "WorkCell.h"

#include <iostream>
#include <string>
#include <set>
#include <sstream>
#include <regex>
#include <cassert>
#include <stdexcept>


// Return the WorkAssignment type, one of: INVALID, STARTUP, SPLITTABLE,
// UNSPLITTABLE.

WorkAssignment::Type WorkAssignment::get_type() const
{
  if (start_state == 0 && end_state == 0 && partial_pattern.size() == 0 &&
      root_pos == 0 && root_throwval_options.empty()) {
    return Type::STARTUP;
  }

  if (start_state == 0 || end_state < start_state ||
      root_pos > partial_pattern.size()) {
    return Type::INVALID;
  }

  if (root_pos == partial_pattern.size()) {
    return root_throwval_options.empty() ? Type::UNSPLITTABLE : Type::INVALID;
  }

  // throws in `rto`, and pp.at(root_pos), must all be distinct
  std::set s(root_throwval_options.begin(), root_throwval_options.end());
  s.insert(partial_pattern.at(root_pos));
  if (s.size() != root_throwval_options.size() + 1) {
    return Type::INVALID;
  }

  return root_throwval_options.empty() ? Type::INVALID : Type::SPLITTABLE;
}

// Determine if a WorkAssignment is valid.

bool WorkAssignment::is_valid() const
{
  return (get_type() != Type::INVALID);
}

// Perform comparisons on WorkAssignments.

bool WorkAssignment::operator==(const WorkAssignment& wa2) const
{
  if (start_state != wa2.start_state) {
    return false;
  }
  if (end_state != wa2.end_state) {
    return false;
  }
  if (root_pos != wa2.root_pos) {
    return false;
  }
  if (root_throwval_options != wa2.root_throwval_options) {
    return false;
  }
  if (partial_pattern != wa2.partial_pattern) {
    return false;
  }
  return true;
}

bool WorkAssignment::operator!=(const WorkAssignment& wa2) const
{
  return !(*this == wa2);
}

// Enumerate the set of throw options available at position `root_pos` in the
// pattern. This list of options is maintained in case we get a request to split
// work.

void WorkAssignment::build_rootpos_throw_options(const Graph& graph,
    unsigned from_state, unsigned start_column)
{
  root_throwval_options.clear();
  for (unsigned col = start_column; col < graph.outdegree.at(from_state);
      ++col) {
    root_throwval_options.push_back(graph.outthrowval.at(from_state).at(col));
  }
}

//------------------------------------------------------------------------------
// Converting to/from a string representation
//------------------------------------------------------------------------------

// Return a string representation.

std::string WorkAssignment::to_string() const
{
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

// Initialize from a string.
//
// Return true on success, false otherwise.

bool WorkAssignment::from_string(const std::string& str)
{
  static const std::regex rgx(
      "\\{ start_state:([0-9]+), end_state:([0-9]+), root_pos:([0-9]+), "
      "root_options:\\[([0-9,]*)\\], current:\\\"([0-9,]*)\\\" \\}"
  );
  std::smatch matches;

  if (!std::regex_search(str, matches, rgx) || matches.size() != 6) {
    return false;
  }

  start_state = std::stoi(matches[1].str());
  end_state = std::stoi(matches[2].str());
  root_pos = std::stoi(matches[3].str());

  root_throwval_options.clear();
  const std::string rto{matches[4].str()};
  auto x = rto.cbegin();
  while (x != rto.cend()) {
    const auto y = std::find(x, rto.cend(), ',');
    const std::string s{x, y};
    root_throwval_options.push_back(std::stoi(s));
    if (y == rto.cend())
      break;
    x = y + 1;
  }

  partial_pattern.clear();
  const std::string pp{matches[5].str()};
  x = pp.cbegin();
  while (x != pp.cend()) {
    const auto y = std::find(x, pp.cend(), ',');
    const std::string s{x, y};
    partial_pattern.push_back(std::stoi(s));
    if (y == pp.cend())
      break;
    x = y + 1;
  }

  // std::cout << "result:" << std::endl << *this << std::endl;
  return is_valid();
}

//------------------------------------------------------------------------------
// Loading into/reading from a WorkSpace
//------------------------------------------------------------------------------

// Load the work assignment into a workspace.
//
// If the work assignment cannot be loaded, throw a std::invalid_argument
// exception with a relevant error message.

void WorkAssignment::to_workspace(WorkSpace* ws, unsigned slot) const
{
  if (get_type() != Type::SPLITTABLE && get_type() != Type::UNSPLITTABLE) {
    std::ostringstream oss;
    oss << "Error: tried to load assignment:\n  " << to_string();
    throw std::invalid_argument(oss.str());
  }

  const Graph& graph = ws->get_graph();
  unsigned from_state = start_state;
  int pos = 0;

  for (size_t i = 0; i < partial_pattern.size(); ++i) {
    pos = static_cast<unsigned>(i);
    WorkCell wc;
    wc.from_state = from_state;
    wc.col_limit = graph.outdegree.at(wc.from_state);

    const auto tv = partial_pattern.at(i);
    for (wc.col = 0; wc.col < wc.col_limit; ++wc.col) {
      if (graph.outthrowval.at(wc.from_state).at(wc.col) == tv)
        break;
    }

    if (wc.col == wc.col_limit) {
      std::cerr << "ERROR in WorkAssignment::to_workspace():\n"
                << "work assignment: " << *this << '\n'
                << "start_state: " << start_state
                << " (" << graph.state.at(start_state) << ")\n"
                << "from_state: " << wc.from_state
                << " (" << graph.state.at(wc.from_state) << ")\n"
                << "partial_pattern: ";
      for (size_t j = 0; j < partial_pattern.size(); ++j) {
        if (j != 0) {
          std::cerr << ',';
        }
        std::cerr << partial_pattern.at(j);
      }
      std::cerr << '\n'
                << "couldn't make throw of value " << tv
                << " at pos " << pos << '\n'
                << "-------------------------\n"
                << "links from state " << wc.from_state << ":\n";
      for (size_t j = 0; j < graph.outdegree.at(wc.from_state); ++j) {
        std::cerr << "col=" << j << ", throwval="
                  << graph.outthrowval.at(wc.from_state).at(j)
                  << ", to_state=" << graph.outmatrix.at(wc.from_state).at(j)
                  << '\n';
      }
    }
    assert(wc.col < wc.col_limit);

    if (pos < static_cast<int>(root_pos)) {
      wc.col_limit = wc.col + 1;
    }

    ws->set_cell(slot, i, wc.col, wc.col_limit, wc.from_state);
    from_state = graph.outmatrix.at(wc.from_state).at(wc.col);
  }

  if (partial_pattern.size() == 0 || pos < static_cast<int>(root_pos)) {
    // loading a work assignment of type UNSPLITTABLE; we didn't initialize the
    // workcell at `root_pos` in the loop above, so do it here
    assert(partial_pattern.size() == 0 ||
        pos + 1 == static_cast<int>(root_pos));
    assert(get_type() == WorkAssignment::Type::UNSPLITTABLE);

    WorkCell rwc;
    rwc.col = 0;
    rwc.col_limit = graph.outdegree.at(from_state);
    rwc.from_state = from_state;
    ws->set_cell(slot, root_pos, rwc.col, rwc.col_limit, rwc.from_state);

    pos = partial_pattern.size() - 1;
  } else {
    // loading a work assignment of type SPLITTABLE; initialized `col` at
    // `root_pos` in loop above, now set `col_limit`
    auto [col, col_limit, from_state] = ws->get_cell(slot, root_pos);
    col_limit = 0;
    for (size_t i = 0; i < graph.outdegree.at(from_state); ++i) {
      const unsigned tv = graph.outthrowval.at(from_state).at(i);
      if (std::find(root_throwval_options.cbegin(),
          root_throwval_options.cend(), tv) != root_throwval_options.cend()) {
        col_limit = std::max(col_limit, static_cast<unsigned>(i + 1));
      }
    }
    assert(col < col_limit);
    assert(col < graph.outdegree.at(from_state));
    ws->set_cell(slot, root_pos, col, col_limit, from_state);
  }

  ws->set_info(slot, start_state, end_state, pos);
}

// Read the work assignment from a workspace.
//
// If the work assignment cannot be read, throw a std::invalid_argument
// exception with a relevant error message.

void WorkAssignment::from_workspace(const WorkSpace* ws, unsigned slot)
{
  const Graph& graph = ws->get_graph();

  // find `start_state` and `end_state`
  const auto [sstate, estate, pos] = ws->get_info(slot);
  start_state = sstate;
  end_state = estate;

  // find `partial_pattern` and `root_pos`
  unsigned from_state = start_state;
  root_pos = -1u;
  partial_pattern.clear();

  for (int i = 0; i <= pos; ++i) {
    auto [col, col_limit, fr_state] = ws->get_cell(slot, i);
    assert(fr_state == from_state);

    partial_pattern.push_back(graph.outthrowval.at(from_state).at(col));

    if (graph.outdegree.at(from_state) < col_limit) {
      col_limit = graph.outdegree.at(from_state);
    }
    if (root_pos == -1u && col < col_limit - 1) {
      root_pos = i;
    }
    from_state = graph.outmatrix.at(from_state).at(col);
  }

  // find `root_throwval_options`
  root_throwval_options.clear();

  if (root_pos == -1u) {
    // current WorkAssignment is UNSPLITTABLE
    root_pos = pos + 1;
    assert(get_type() == Type::UNSPLITTABLE);
  } else {
    // current WorkAssignment is SPLITTABLE
    auto [rwc_col, rwc_col_limit, rwc_fr_state] = ws->get_cell(slot, root_pos);

    if (graph.outdegree.at(rwc_fr_state) < rwc_col_limit) {
      rwc_col_limit = graph.outdegree.at(rwc_fr_state);
    }

    for (size_t col = rwc_col + 1; col < rwc_col_limit; ++col) {
      root_throwval_options.push_back(
          graph.outthrowval.at(rwc_fr_state).at(col));
    }
    assert(get_type() == Type::SPLITTABLE);
  }
}

//------------------------------------------------------------------------------
// Work splitting
//------------------------------------------------------------------------------

// Determine if a work assignment can be split.

bool WorkAssignment::is_splittable() const
{
  return is_valid() &&
      (end_state > start_state || get_type() == Type::SPLITTABLE);
}

// Return a work assignment that corresponds to a portion of the current work
// assignment, for handing off to another worker.
//
// If a work assignment cannot be split, leave the WorkAssignment unchanged and
// throw a std::invalid_argument exception with a relevant error message.

WorkAssignment WorkAssignment::split(const Graph& graph, unsigned split_alg)
{
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

WorkAssignment WorkAssignment::split_takestartstates()
{
  unsigned takenum = (end_state - start_state + 1) / 2;
  assert(takenum > 0);
  assert(end_state >= start_state + takenum);

  WorkAssignment wa;
  wa.start_state = end_state - takenum + 1;
  wa.end_state = end_state;
  assert(wa.get_type() == Type::UNSPLITTABLE);

  end_state -= takenum;
  return wa;
}

// Return a work assignment that gives away all of the unexplored throw options
// at root_pos.

WorkAssignment WorkAssignment::split_takeall(const Graph& graph)
{
  return split_takefraction(graph, 1);
}

// Return a work assignment that gives away approximately half of the unexplored
// throw options at root_pos.

WorkAssignment WorkAssignment::split_takehalf(const Graph& graph)
{
  return split_takefraction(graph, 0.5);
}

// Return a work assignment that gives away approximately the target fraction of
// the unexplored throw options at root_pos.
//
// If a work assignment cannot be split, leave the WorkAssignment unchanged and
// throw a std::invalid_argument exception with a relevant error message.

WorkAssignment WorkAssignment::split_takefraction(const Graph& graph, double f)
{
  if (get_type() != Type::SPLITTABLE) {
    std::cerr << "error trying to split assignment:\n  " << to_string()
              << '\n';
    throw std::invalid_argument("Cannot split work assignment");
  }

  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  for (size_t i = 0; i < root_pos; ++i) {
    wa.partial_pattern.push_back(partial_pattern.at(i));
  }

  // move `take_count` unexplored root_pos options to the new work assignment,
  // specifically the highest `col` values so that each resulting assignment has
  // consecutive values of `col` in its root_throwval_options
  auto take_count =
      static_cast<size_t>(0.51 + f * root_throwval_options.size());
  take_count = std::min(std::max(take_count, static_cast<size_t>(1)),
      root_throwval_options.size());

  const auto take_begin_idx = static_cast<size_t>(root_throwval_options.size() -
      take_count);
  const auto take_end_idx = take_begin_idx + take_count;

  auto iter = root_throwval_options.begin();
  auto end = root_throwval_options.end();
  for (size_t index = 0; iter != end; ++index) {
    if (index >= take_begin_idx && index < take_end_idx) {
      wa.root_throwval_options.push_back(*iter);
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
  }

  if (wa.root_throwval_options.empty()) {
    // diagnostic message if there's a problem
    std::cerr << "error 1 splitting " << *this
              << "\n  new assignment = " << wa
              << "\nstart_state = " << wa.start_state
              << " (" << graph.state.at(wa.start_state) << ")\n";
    for (unsigned i = 0; i < graph.outdegree.at(wa.start_state); ++i) {
      std::cerr << "  " << i << ": "
                << graph.outthrowval.at(wa.start_state).at(i)
                << " -> " << graph.outmatrix.at(wa.start_state).at(i) << '\n';
    }
  }
  assert(!wa.root_throwval_options.empty());

  // did we give away all our throw options at `root_pos`?
  if (root_throwval_options.empty()) {
    // Find the shallowest depth `new_root_pos` where there are unexplored throw
    // options. We have no more options at the current root_pos, so
    // new_root_pos > root_pos.
    //
    // If there is no such new_root_pos < partial_pattern.size(), then this
    // becomes an UNSPLITTABLE assignment and set
    // new_root_pos = partial_pattern.size().

    unsigned from_state = start_state;
    unsigned new_root_pos = -1;
    unsigned col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (unsigned i = 0; i < partial_pattern.size(); ++i) {
      const auto tv = static_cast<unsigned>(partial_pattern.at(i));
      for (col = 0; col < graph.outdegree.at(from_state); ++col) {
        if (graph.outthrowval.at(from_state).at(col) == tv) {
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
                  << ", throwval = " << tv
                  << '\n';
      }
      assert(col < graph.outdegree.at(from_state));

      if (i > root_pos && col < graph.outdegree.at(from_state) - 1) {
        new_root_pos = i;
        break;
      }

      from_state = graph.outmatrix.at(from_state).at(col);
    }

    if (new_root_pos == -1u) {
      root_pos = partial_pattern.size();
      // leave root_throwval_options empty
      assert(get_type() == Type::UNSPLITTABLE);
    } else {
      root_pos = new_root_pos;
      // rebuild the list of throw options at `root_pos`
      build_rootpos_throw_options(graph, from_state, col + 1);
      assert(get_type() == Type::SPLITTABLE);
    }
  } else {
    assert(get_type() == Type::SPLITTABLE);
  }

  // move the first element in wa.root_throwval_options to the end of
  // wa.partial_pattern
  wa.partial_pattern.push_back(wa.root_throwval_options.front());
  wa.root_throwval_options.pop_front();
  if (wa.root_throwval_options.empty()) {
    wa.root_pos = wa.partial_pattern.size();
    assert(wa.get_type() == Type::UNSPLITTABLE);
  } else {
    assert(wa.get_type() == Type::SPLITTABLE);
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
      const WorkAssignment& wa2)
{
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

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa)
{
  ost << wa.to_string();
  return ost;
}
