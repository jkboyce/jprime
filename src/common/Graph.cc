//
// Graph.cc
//
// Data structures related to the juggling graph for B objects, max throw H.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Graph.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>


// Full graph for `b` objects, max throw `h`.

Graph::Graph(unsigned b, unsigned h)
    : b(b), h(h), xarray(h + 1, false)
{
  initialize();
}

// Initialize with two additional options:
//
// Parameter `xa` specifies a vector of size h+1 that determines whether a given
// throw value is excluded; when xa[val] is true then no graph links with throw
// value `val` are created.
//
// Parameter `n` is optional and specifies that a single-period graph is
// desired; a single-period graph contains only those states that may be part of
// a period-n pattern. We only permit values n < h. When n == 0 we generate
// the full graph.

Graph::Graph(unsigned b, unsigned h, const std::vector<bool>& xa, unsigned n)
    : b(b), h(h), n(n), xarray(xa)
{
  assert(xa.size() == h + 1);
  assert(n < h);
  initialize();
}

//------------------------------------------------------------------------------
// Prepare graph data structures during construction
//------------------------------------------------------------------------------

// Initialize the Graph object.
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message.

void Graph::initialize()
{
  // fail if the number of states cannot fit into an unsigned int
  const std::uint64_t num = (n == 0 ? combinations(h, b) :
      ordered_partitions(b, h, n));
  if (num > std::numeric_limits<unsigned>::max()) {
    throw std::overflow_error("Graph initialization error: too many states");
  }
  numstates = static_cast<unsigned>(num);

  // generate the states
  state.reserve(numstates + 2);
  state.emplace_back(h);  // state at index 0 is unused
  if (n == 0) {
    gen_states_all(state, b, h);
    // states are generated in sorted order
  } else {
    gen_states_for_period(state, b, h, n);
    if (state.size() > 1) {
      std::sort(state.begin() + 1, state.end(), state_compare);
    }
  }
  assert(state.size() == numstates + 1);

  build_graph_matrix();
}

//------------------------------------------------------------------------------
// Generate the states in the graph
//------------------------------------------------------------------------------

// Helper function to generate states in the general case. Recursively insert
// 1s into successive slots, and when all 1s are used up append a new state
// to the list.

void gen_states_all_helper(std::vector<State>& s, unsigned pos, unsigned left)
{
  if (left > (pos + 1)) {
    return;  // no way to succeed
  }

  if (pos == 0) {
    s.back().slot(0) = left;
    // success: duplicate state at the end and continue
    s.push_back(s.back());
    return;
  }

  // try a '-' at position `pos`
  s.back().slot(pos) = 0;
  gen_states_all_helper(s, pos - 1, left);

  // then try a 'x' at position `pos`
  if (left > 0) {
    s.back().slot(pos) = 1;
    gen_states_all_helper(s, pos - 1, left - 1);
  }
}

// Generate all possible states into the vector `s`.
//
// Note this has a recursion depth of `h`.

void Graph::gen_states_all(std::vector<State>& s, unsigned b, unsigned h)
{
  s.emplace_back(h);
  gen_states_all_helper(s, h - 1, b);
  s.pop_back();
}

// Helper function to generate states in the single-period case. The states are
// enumerated by partitioning the `b` objects into `n` different buckets.

void gen_states_for_period_helper(std::vector<State>& s, unsigned pos,
    unsigned left, const unsigned h, const unsigned n)
{
  if (pos == n) {
    if (left == 0) {
      // success: duplicate state at the end and continue
      s.push_back(s.back());
    }
    return;
  }

  // empty all the slots at position `pos`
  for (size_t i = pos; i < h; i += n) {
    s.back().slot(i) = 0;
  }

  // work out the maximum number that can go into later slots, and the min
  // for this slot
  unsigned max_later = 0;
  for (size_t pos2 = pos + 1; pos2 < n; ++pos2) {
    for (size_t i = pos2; i < h; i += n) {
      ++max_later;
    }
  }
  const unsigned min_fill = (left > max_later ? left - max_later : 0);

  if (min_fill == 0) {
    gen_states_for_period_helper(s, pos + 1, left, h, n);
  }

  // successively fill slots at `pos`
  unsigned filled = 0;
  for (size_t i = pos; i < h && filled < left; i += n) {
    s.back().slot(i) = 1;
    ++filled;
    if (filled >= min_fill) {
      gen_states_for_period_helper(s, pos + 1, left - filled, h, n);
    }
  }
}

// Generate all possible states that can be part of a pattern of period `n`.
//
// Note this has a recursion depth of `h`.

void Graph::gen_states_for_period(std::vector<State>& s, unsigned b, unsigned h,
    unsigned n)
{
  s.emplace_back(h);
  gen_states_for_period_helper(s, 0, b, h, n);
  s.pop_back();
}

//------------------------------------------------------------------------------
// Find the shift cycles in the graph
//------------------------------------------------------------------------------

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period
//
// Return the total number of shift cycles found.

unsigned Graph::find_shift_cycles()
{
  const unsigned state_unused = -1U;
  cyclenum.assign(numstates + 1, state_unused);
  cycleperiod.clear();
  numshortcycles = 0;

  unsigned cycles = 0;
  std::vector<unsigned> cyclestates(h);
  State s(h);

  for (size_t i = 1; i <= numstates; ++i) {
    if (cyclenum.at(i) != state_unused) {
      continue;
    }

    s = state.at(i);
    unsigned cycleper = h;

    for (size_t j = 0; j < h; ++j) {
      s.shift_upstream();
      const auto k = get_statenum(s);
      cyclestates.at(j) = k;

      if (k == i) {
        cycleper = static_cast<unsigned>(j + 1);
        break;
      }
    }

    for (size_t j = 0; j < cycleper; ++j) {
      if (cyclestates.at(j) > 0) {
        cyclenum.at(cyclestates.at(j)) = cycles;
      }
    }
    cycleperiod.push_back(cycleper);
    if (cycleper < h) {
      ++numshortcycles;
    }
    ++cycles;
  }
  assert(cycleperiod.size() == cycles);
  numcycles = cycles;
  return cycles;
}

//------------------------------------------------------------------------------
// Operations on graph matrices
//------------------------------------------------------------------------------

// Construct matrices describing the structure of the juggling graph.
//
// - Outward degree from each state (vertex) in the graph:
//         outdegree[statenum] --> degree
// - Outward connections from each state:
//         outmatrix[statenum][col] --> statenum  (where col < outdegree)
// - Throw values corresponding to outward connections from each state:
//         outthrowval[statenum][col] --> throw
//
// outmatrix[][] == 0 indicates no connection.

void Graph::build_graph_matrix()
{
  maxoutdegree = 0;
  for (size_t i = 0; i <= h; ++i) {
    if (!xarray.at(i)) {
      ++maxoutdegree;
    }
  }
  maxoutdegree = std::min(maxoutdegree, h - b + 1);
  outdegree.assign(numstates + 1, 0);
  const size_t total_elements = static_cast<size_t>(numstates + 1) * maxoutdegree;
  outmatrix.assign(total_elements, 0);
  outthrowval.assign(total_elements, 0);

  auto process_range = [this](size_t start, size_t end) {
    State temp(h);
    const size_t stride = maxoutdegree;
    for (size_t i = start; i < end; ++i) {
      const auto& s = state.at(i);
      unsigned outthrownum = 0;
      const size_t row_offset = i * stride;

      if (s.slot(0) == 0) {
        if (!xarray.at(0)) {
          temp = s;
          temp.shift_downstream_zero();
          const auto k = get_statenum(temp);
          if (k != 0) {
            outmatrix[row_offset + outthrownum] = k;
            outthrowval[row_offset + outthrownum] = 0;
            ++outthrownum;
          }
        }
      } else {
        temp = s;
        temp.shift_downstream_zero();

        for (unsigned throwval = h; throwval > 0; --throwval) {
          if (xarray.at(throwval)) {
            continue;
          }
          if (throwval < h && s.slot(throwval) != 0) {
            continue;
          }

          temp.slot(throwval - 1) = 1;
          const auto k = get_statenum(temp);
          temp.slot(throwval - 1) = 0;

          if (k != 0) {
            outmatrix[row_offset + outthrownum] = k;
            outthrowval[row_offset + outthrownum] = throwval;
            ++outthrownum;
          }
        }
      }
      outdegree.at(i) = outthrownum;
    }
  };

  unsigned num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0 || numstates < 1000) {
    num_threads = 1;
  }

  if (num_threads == 1) {
    process_range(1, numstates + 1);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const size_t total = numstates;
    for (unsigned t = 0; t < num_threads; ++t) {
      const size_t start = 1 + (t * total) / num_threads;
      const size_t end = 1 + ((t + 1) * total) / num_threads;
      workers.emplace_back(process_range, start, end);
    }
    for (auto& w : workers) {
      w.join();
    }
  }
}

// Fill in the vector `max_startstate_usable`, which indicates when a given
// state is accessible in the graph.
//
// The state `i` is accessible to a pattern starting from `start_state` if and
// only if start_state <= max_startstate_usable[i].

void Graph::find_max_startstate_usable()
{
  max_startstate_usable.assign(numstates + 1, 0);

  std::vector<unsigned> in_degree(numstates + 1, 0);
  std::vector<unsigned> out_degree(numstates + 1, 0);
  std::vector<std::vector<unsigned>> in_matrix(numstates + 1);

  for (unsigned u = 1; u <= numstates; ++u) {
    out_degree.at(u) = outdegree.at(u);
    for (unsigned col = 0; col < outdegree.at(u); ++col) {
      const auto v = outmatrix_at(u, col);
      if (v != 0) {
        ++in_degree.at(v);
        in_matrix.at(v).push_back(u);
      }
    }
  }

  std::vector<bool> removed(numstates + 1, false);
  std::queue<unsigned> q;

  auto eliminate = [&](unsigned u, unsigned current_start_state) {
    if (removed.at(u)) {
      return;
    }
    removed.at(u) = true;
    max_startstate_usable.at(u) = current_start_state - 1;
    q.push(u);

    while (!q.empty()) {
      const auto curr = q.front();
      q.pop();

      for (unsigned col = 0; col < outdegree.at(curr); ++col) {
        const auto v = outmatrix_at(curr, col);
        if (v != 0 && !removed.at(v)) {
          if (--in_degree.at(v) == 0) {
            removed.at(v) = true;
            max_startstate_usable.at(v) = current_start_state - 1;
            q.push(v);
          }
        }
      }

      for (const auto p : in_matrix.at(curr)) {
        if (!removed.at(p)) {
          if (--out_degree.at(p) == 0) {
            removed.at(p) = true;
            max_startstate_usable.at(p) = current_start_state - 1;
            q.push(p);
          }
        }
      }
    }
  };

  for (unsigned start_state = 2; start_state <= numstates; ++start_state) {
    eliminate(start_state - 1, start_state);
  }

  for (unsigned u = 1; u <= numstates; ++u) {
    if (!removed.at(u)) {
      max_startstate_usable.at(u) = u;
    }
  }
}

// Identify states in the graph that are unusable.
//
// Propagate unusable links and states through the graph by applying two
// transformations, until no further reductions are possible:
// - Remove links into states with outdegree zero
// - For states with indegree zero, set outdegree to zero
//
// Updates the input vector `state_usable` to false for any state numbers found
// to be unusable.

void Graph::update_usable_states(std::vector<bool>& state_usable) const
{
  std::vector<uint16_t> usable_outdegree(outdegree);
  std::vector<unsigned> usable_indegree(numstates + 1, 0);

  // Propagate unusable links and states through the graph
  while (true) {
    bool changed = false;

    // remove links into unusable states
    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_usable.at(i)) {
        usable_outdegree.at(i) = 0;
        continue;
      }
      uint16_t outthrownum = 0;
      for (size_t j = 0; j < outdegree.at(i); ++j) {
        if (state_usable.at(outmatrix_at(i, j))) {
          ++outthrownum;
        }
      }
      if (usable_outdegree.at(i) != outthrownum) {
        usable_outdegree.at(i) = outthrownum;
        changed = true;
      }
    }

    // mark unusable any states with zero outdegree or indegree
    usable_indegree.assign(numstates + 1, 0);

    for (size_t i = 1; i <= numstates; ++i) {
      if (!state_usable.at(i)) {
        continue;
      }
      if (usable_outdegree.at(i) == 0) {
        state_usable.at(i) = false;
        changed = true;
        continue;
      }

      for (size_t j = 0; j < outdegree.at(i); ++j) {
        ++usable_indegree.at(outmatrix_at(i, j));
      }
    }

    for (size_t i = 1; i <= numstates; ++i) {
      if (state_usable.at(i) && usable_indegree.at(i) == 0) {
        state_usable.at(i) = false;
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }
}

//------------------------------------------------------------------------------
// Public methods
//------------------------------------------------------------------------------

// Update graph data structures after an edit operation, by removing states and
// links that are unusable.
//
// NOTE: This function must be called after any kind of edit operation that
// removes links and/or states, to ensure the internal state of the graph is
// consistent.
//
// NOTE: This function will in general renumber the states! The exception is
// state number 1, which is reserved for the ground state and is never removed
// or renumbered.

void Graph::validate_graph(bool ground_only)
{
  std::vector<bool> state_usable(numstates + 1, true);
  update_usable_states(state_usable);

  // work out new state numbers after unusable ones are removed
  std::vector<unsigned> newstatenum(numstates + 1, 0);
  newstatenum.at(1) = 1;
  unsigned statenum = 2;
  for (size_t i = 2; i <= numstates; ++i) {
    if (state_usable.at(i)) {
      newstatenum.at(i) = statenum;
      ++statenum;
    }
  }

  // Items to update:
  //   unsigned numstates;
  //   std::vector<State> state;
  //   std::vector<unsigned> cyclenum;
  //   std::vector<unsigned> max_startstate_usable;
  //   unsigned maxoutdegree;
  //   std::vector<unsigned> outdegree;
  //   std::vector<std::vector<unsigned>> outmatrix;
  //   std::vector<std::vector<unsigned>> outthrowval;

  unsigned new_maxoutdegree = 0;

  for (size_t i = 1; i <= numstates; ++i) {
    const auto newi = newstatenum.at(i);
    if (newi == 0) {
      continue;
    }
    assert(newi <= i);

    // copy graph data from row `i` to row `newi`
    state.at(newi) = state.at(i);
    if (!cyclenum.empty()) {
      cyclenum.at(newi) = cyclenum.at(i);
    }

    unsigned outthrownum = 0;
    for (size_t j = 0; j < outdegree.at(i); ++j) {
      const auto newsn = newstatenum.at(outmatrix_at(i, j));
      if (newsn != 0) {
        outmatrix_at(newi, outthrownum) = newsn;
        outthrowval_at(newi, outthrownum) = outthrowval_at(i, j);
        ++outthrownum;
      }
    }
    assert(outthrownum != 0);
    outdegree.at(newi) = outthrownum;
    new_maxoutdegree = std::max(new_maxoutdegree, outthrownum);

    for (size_t j = outthrownum; j < maxoutdegree; ++j) {
      outmatrix_at(newi, j) = 0;
      outthrowval_at(newi, j) = 0;
    }
  }

  numstates = statenum - 1;
  const unsigned old_maxoutdegree = maxoutdegree;
  maxoutdegree = new_maxoutdegree;
  // State is not default constructable
  state.erase(state.begin() + numstates + 1, state.end());
  assert(state.size() == numstates + 1);
  if (!cyclenum.empty()) {
    cyclenum.resize(numstates + 1);
  }
  outdegree.resize(numstates + 1);

  if (new_maxoutdegree < old_maxoutdegree) {
    for (size_t k = 1; k <= numstates; ++k) {
      for (size_t col = 0; col < new_maxoutdegree; ++col) {
        outmatrix[k * new_maxoutdegree + col] = outmatrix[k * old_maxoutdegree + col];
        outthrowval[k * new_maxoutdegree + col] = outthrowval[k * old_maxoutdegree + col];
      }
    }
  }
  const size_t total_elements = static_cast<size_t>(numstates + 1) * maxoutdegree;
  outmatrix.resize(total_elements);
  outthrowval.resize(total_elements);

  if (!ground_only) {
    find_max_startstate_usable();
  }
}

// Return vector `isexitcycle` that indicates which shift cycles can exit
// directly to the start state with a link throw. This is used in SUPER mode to
// cut off search when all exit cycles have been used.

std::vector<int> Graph::get_exit_cycles(unsigned start_state) const
{
  assert(has_shift_cycles());

  std::vector<int> isexitcycle(numcycles, 0);

  for (size_t i = start_state + 1; i <= numstates; ++i) {
    for (size_t j = 0; j < outdegree.at(i); ++j) {
      if (outmatrix_at(i, j) == start_state) {
        isexitcycle.at(cyclenum.at(i)) = 1;
      }
    }
  }
  isexitcycle.at(cyclenum.at(start_state)) = 0;

  return isexitcycle;
}

// Generate arrays that are used for marking excluded states during search. If
// one of the non-marking versions of gen_loops() is used then these arrays are
// unneeded.
//
// This should be called after validate_graph() and before gen_loops().

void Graph::build_exclude_states()
{
  assert(has_shift_cycles());
  excludestates_tail.resize(numstates + 1);
  excludestates_head.resize(numstates + 1);
  for (size_t i = 0; i <= numstates; ++i) {
    excludestates_tail.at(i).assign(h, 0);
    excludestates_head.at(i).assign(h, 0);
  }

  for (size_t i = 1; i <= numstates; ++i) {
    // Find states that are excluded by a link throw from state `i`. These are
    // the states downstream in i's shift cycle that end in 'x'.
    State s = state.at(i).downstream();
    unsigned j = 0;
    while (s.slot(s.size() - 1) != 0) {
      const auto statenum = get_statenum(s);
      if (statenum == 0 || statenum == i) {
        break;
      }

      // assert that state numbers are monotonically increasing; see note
      // on Worker::mark_unreachable_states_throw()
      assert((j == 0 && statenum > i) ||
          (j > 0 && statenum > excludestates_tail.at(i).at(j - 1)));

      excludestates_tail.at(i).at(j++) = statenum;
      s = s.downstream();
    }
    excludestates_tail.at(i).at(j) = 0;

    // Find states that are excluded by a link throw into state `i`. These are
    // the states upstream in i's shift cycle that start with '-'.
    s = state.at(i).upstream();
    j = 0;
    while (s.slot(0) == 0) {
      const auto statenum = get_statenum(s);
      if (statenum == 0 || statenum == i) {
        break;
      }
      assert((j == 0 && statenum > i) ||
          (j > 0 && statenum > excludestates_head.at(i).at(j - 1)));
      excludestates_head.at(i).at(j++) = statenum;
      s = s.upstream();
    }
    excludestates_head.at(i).at(j) = 0;
  }
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Calculate an upper bound on the period of prime patterns in the graph,
// starting from `start_state` as the root node.

unsigned Graph::prime_period_bound(unsigned start_state) const
{
  assert(has_shift_cycles());

  // case 1: the pattern visits multiple shift cycles; it must miss at least one
  // state on each cycle it visits.
  // case 2: the pattern stays on a single shift cycle; find the cycle with the
  // most active states.
  unsigned result_multicycle = 0;
  unsigned result_onecycle = 0;

  std::vector<unsigned> num_active(numcycles, 0);
  for (size_t i = start_state; i <= numstates; ++i) {
    if (is_usable_for_startstate(i, start_state)) {
      ++num_active.at(cyclenum.at(i));
    }
  }

  const int cycles_active = static_cast<int>(numcycles) -
      static_cast<int>(std::count(num_active.cbegin(), num_active.cend(), 0));

  for (size_t i = 0; i < numcycles; ++i) {
    if (cycles_active > 1) {
      result_multicycle += std::min(num_active.at(i), cycleperiod.at(i) - 1);
    }
    result_onecycle = std::max(result_onecycle, num_active.at(i));
  }

  return std::max(result_multicycle, result_onecycle);
}

// Calculate an upper bound on the period of superprime patterns, starting from
// `start_state` as the root node.
//
// Optional parameter `shifts` specifies an upper limit on the number of shift
// throws allowed in the pattern.

unsigned Graph::superprime_period_bound(unsigned start_state, unsigned shifts)
    const
{
  assert(has_shift_cycles());

  std::vector<bool> any_active(numcycles, false);
  for (size_t i = start_state; i <= numstates; ++i) {
    if (is_usable_for_startstate(i, start_state)) {
      any_active.at(cyclenum.at(i)) = true;
    }
  }

  const auto cycles_active = static_cast<unsigned>(
      std::count(any_active.cbegin(), any_active.cend(), true));

  if (cycles_active < 2) {
    return 0;
  }
  if (shifts == -1U) {
    return prime_period_bound(start_state);
  }
  return std::min(prime_period_bound(start_state), cycles_active + shifts);
}

// Return the index in the `state` array that corresponds to a given state.
// Returns 0 if not found.
//
// Note this assumes the `state` vector is sorted!

unsigned Graph::get_statenum(const State& s) const
{
  if (s.size() != h) {
    return 0;
  }

  const int cmp1 = s.compare_same_size(state.at(1));
  if (cmp1 < 0) {
    return 0;
  }
  if (cmp1 == 0) {
    return 1;
  }

  const int cmpn = state.at(numstates).compare_same_size(s);
  if (cmpn < 0) {
    return 0;
  }
  if (cmpn == 0) {
    return numstates;
  }

  size_t below = 1;
  size_t above = numstates;

  // loop invariant: state[below] < s < state[above]
  while (below < above - 1) {
    size_t mid = (below + above) / 2;
    const int cmp = state.at(mid).compare_same_size(s);
    if (cmp == 0) {
      return static_cast<unsigned>(mid);
    }
    if (cmp < 0) {
      below = mid;
    } else {
      above = mid;
    }
  }
  return 0;
}

// Return the state number that comes from advancing a given state by a single
// throw. Returns 0 if the throw is not allowed.

unsigned Graph::advance_state(unsigned statenum, unsigned throwval) const
{
  const auto& s = state.at(statenum);

  if (throwval > 0 && s.slot(0) == 0) {  // no object to throw
    return 0;
  }
  if (throwval < s.size() && s.slot(throwval) != 0) {  // collision w/prev throw
    return 0;
  }
  if (throwval > s.size()) {  // out of range
    return 0;
  }

  return get_statenum(s.advance_with_throw(throwval));
}

// Return the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the `state` vector).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

unsigned Graph::reverse_state(unsigned statenum) const
{
  return get_statenum(state.at(statenum).reverse());
}

// Return the next state downstream in the given state's shift cycle.

unsigned Graph::downstream_state(unsigned statenum) const
{
  return get_statenum(state.at(statenum).downstream());
}

// Return the next state upstream in the given state's shift cycle.

unsigned Graph::upstream_state(unsigned statenum) const
{
  return get_statenum(state.at(statenum).upstream());
}

// Return a text representation of a given state number.

std::string Graph::state_string(unsigned statenum) const
{
  return state.at(statenum).to_string();
}

// Return the minimum number of graph hops required to transition from state
// `from` to state `to`.
//
// Note this currently ignores any edits to the graph; it assumes the full
// graph (b, h). So this represents a lower bound on the hop distance.

unsigned Graph::min_distance(const State& from, const State& to)
{
  assert(from.size() == to.size());
  const auto h = static_cast<unsigned>(from.size());

  if (h <= 64) {
    const uint64_t f0 = from.word(0);
    const uint64_t inv_t0 = ~to.word(0);
    for (unsigned d = 0; d < h; ++d) {
      if (((f0 >> d) & inv_t0) == 0ULL) {
        return d;
      }
    }
    return h;
  }

  const size_t words_used = (h + 63) / 64;
  uint64_t f[State::NUM_WORDS];
  uint64_t inv_t[State::NUM_WORDS];
  for (size_t w = 0; w < words_used; ++w) {
    f[w] = from.word(w);
    inv_t[w] = ~to.word(w);
  }

  for (unsigned d = 0; d <= h; ++d) {
    bool match = true;
    for (size_t w = 0; w < words_used; ++w) {
      if ((f[w] & inv_t[w]) != 0ULL) {
        match = false;
        break;
      }
    }
    if (match) {
      return d;
    }

    for (size_t w = 0; w < words_used; ++w) {
      f[w] >>= 1;
      if (w + 1 < words_used) {
        f[w] |= (f[w + 1] << 63);
      }
    }
  }
  return h;
}

// Return a text representation of the graph.

std::string Graph::to_string() const
{
  std::ostringstream buffer;

  buffer << "graph:\n"
         << "  b = " << b << "\n"
         << "  h = " << h << "\n"
         << "  n = " << n << "\n"
         << "  numstates = " << numstates << "\n"
         << "  states[] = { unused";
  for (size_t i = 1; i < numstates; ++i) {
    buffer << ", " << state_string(static_cast<unsigned>(i));
  }
  buffer << " }\n";
  if (!cyclenum.empty()) {
    buffer << "  numcycles = " << numcycles << "\n"
           << "  cyclenum[] = { unused, ";
    for (size_t i = 1; i <= numstates; ++i) {
      buffer << cyclenum.at(i);
      if (i < numstates) {
        buffer << ", ";
      }
    }
    buffer << " }\n"
           << "  cycleperiod[] = { ";
    for (size_t i = 0; i < numcycles; ++i) {
      buffer << cycleperiod.at(i);
      if (i < numcycles - 1) {
        buffer << ", ";
      }
    }
    buffer << " }\n"
           << "  numshortcycles = " << numshortcycles << "\n";
  }
  buffer << "  maxoutdegree = " << maxoutdegree << "\n"
         << "  outdegree[] = { unused, ";
  for (size_t i = 1; i <= numstates; ++i) {
    buffer << outdegree.at(i);
    if (i < numstates) {
      buffer << ", ";
    }
  }
  buffer << " }\n"
         << "  outmatrix[][] = {\n"
         << "  0 -> unused,\n";
  for (size_t i = 1; i <= numstates; ++i) {
    buffer << "  " << i << " -> { ";
    for (size_t j = 0; j < outdegree.at(i); ++j) {
      buffer << outmatrix_at(i, j);
      if (j + 1 < outdegree.at(i)) {
        buffer << ", ";
      }
    }
    buffer << " }";
    if (i < numstates) {
      buffer << ", ";
    }
    buffer << '\n';
  }
  buffer << "}\n";
  buffer << "  outthrowval[][] = {\n"
         << "  0 -> unused,\n";
  for (size_t i = 1; i <= numstates; ++i) {
    buffer << "  " << i << " -> { ";
    for (size_t j = 0; j < outdegree.at(i); ++j) {
      buffer << outthrowval_at(i, j);
      if (j + 1 < outdegree.at(i)) {
        buffer << ", ";
      }
    }
    buffer << " }";
    if (i < numstates) {
      buffer << ", ";
    }
    buffer << '\n';
  }
  buffer << "}\n";

  return buffer.str();
}

// Print to an output stream using the default text format.

std::ostream& operator<<(std::ostream& ost, const Graph& g)
{
  ost << g.to_string();
  return ost;
}
