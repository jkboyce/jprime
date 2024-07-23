//
// GenLoopsIterative.cc
//
// Core graph search routines, implemented as iterative functions that are drop-
// in replacements for recursive versions in GenLoopsRecursive.cc. These
// routines are by far the most performance-critical portions of jprime.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"
#include "Graph.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>


// This is a non-recursive version of get_loops_normal(), with identical
// interface and behavior.
//
// It is slightly faster than the recursive version and also avoids potential
// stack overflow on deeper searches.
//
// Template parameter `REPORT` specifies whether the patterns found are reported
// to the Coordinator (true), or merely counted (false).

template<bool REPORT>
void Worker::iterative_gen_loops_normal() {
  if (!iterative_init_workspace(false)) {
    assert(false);
  }

  // temp storage to access matrix elements quickly
  std::vector<unsigned int*> om_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
  }
  unsigned int** const outmatrix = om_row.data();

  // local variables to improve performance
  int p = pos;
  uint64_t nn = nnodes;
  const int lmax = l_max;
  int* const u = used.data();
  unsigned int steps = 0;
  unsigned int steps_limit = steps_per_inbox_check;
  const unsigned int st_state = start_state;
  unsigned int* const outdegree = graph.outdegree.data();
  std::uint64_t* const c = count.data();

  // Note that beat 0 is stored at index 1 in the `beat` array. We do this to
  // provide a guard since beat[0].col is modified at the end of the search.
  SearchState* ss = &beat.at(pos + 1);

  while (p >= 0) {
    // begin with any necessary cleanup from previous marking operations
    if (ss->to_state != 0) {
      u[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking:
    if (ss->col == ss->col_limit) {
      --p;
      --ss;
      ++ss->col;
      ++nn;
      continue;
    }

    const unsigned int to_state = ss->outmatrix[ss->col];
    if (to_state == st_state) {
      if (REPORT) {
        pos = p;
        iterative_handle_finished_pattern();
      } else {
        ++c[p + 1];
      }
      ++ss->col;
      goto skip_unmarking;
    }

    if (u[to_state]) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (p + 1 == lmax) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (++steps >= steps_limit) {
      steps = 0;

      pos = p;
      if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
        for (size_t i = 0; i <= pos; ++i) {
          pattern.at(i) = graph.outthrowval.at(beat.at(i + 1).from_state)
                                           .at(beat.at(i + 1).col);
        }
        pattern.at(pos + 1) = -1;
        nnodes = nn;
        process_inbox_running();
        iterative_update_after_split();
        nn = nnodes;
        steps_limit = steps_per_inbox_check;
        // the above functions do not change `pos`
      }
    }

    // advance to next beat
    u[to_state] = 1;
    ss->to_state = to_state;
    ++p;
    ++ss;
    ss->col = 0;
    ss->col_limit = outdegree[to_state];
    ss->from_state = to_state;
    ss->to_state = 0;
    ss->outmatrix = outmatrix[to_state];
    goto skip_unmarking;
  }

  ++p;
  pos = p;
  nnodes = nn;
  assert(pos == 0);
}

// Non-recursive version of get_loops_normal_marking()

void Worker::iterative_gen_loops_normal_marking() {
  if (!iterative_init_workspace(true)) {
    assert(false);
  }

  std::vector<unsigned int*> om_row(graph.numstates + 1, nullptr);
  std::vector<unsigned int*> otv_row(graph.numstates + 1, nullptr);
  std::vector<unsigned int*> es_throw_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
    otv_row.at(i) = graph.outthrowval.at(i).data();
    es_throw_row.at(i) = graph.excludestates_throw.at(i).data();
  }
  unsigned int** const outmatrix = om_row.data();
  unsigned int** const outthrowval = otv_row.data();
  unsigned int** const es_throw = es_throw_row.data();

  int p = pos;
  uint64_t nn = nnodes;
  const int lmax = l_max;
  int* const u = used.data();
  unsigned int steps = 0;
  unsigned int steps_limit = steps_per_inbox_check;
  unsigned int st_state = start_state;
  unsigned int** const ds_bystate = deadstates_bystate.data();
  unsigned int* const outdegree = graph.outdegree.data();

  SearchState* ss = &beat.at(pos + 1);

  while (p >= 0) {
    // begin with any necessary cleanup from previous marking operations
    if (ss->to_state != 0) {
      u[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking1:
    if (ss->excludes_catch) {
      unsigned int* const ds = ss->deadstates_catch;
      unsigned int* es = ss->excludes_catch;
      for (unsigned int statenum; (statenum = *es); ++es) {
        if (--u[statenum] == 0 && --*ds > 0)
          ++max_possible;
      }
      ss->excludes_catch = nullptr;
    }

    skip_unmarking2:
    if (ss->col == ss->col_limit) {
      if (ss->excludes_throw) {
        unsigned int* const ds = ss->deadstates_throw;
        unsigned int* es = ss->excludes_throw;
        for (unsigned int statenum; (statenum = *es); ++es) {
          if (--u[statenum] == 0 && --*ds > 0)
            ++max_possible;
        }
      }

      --p;
      --ss;
      ++ss->col;
      ++nn;
      continue;
    }

    const unsigned int to_state = ss->outmatrix[ss->col];
    if (to_state == st_state) {
      pos = p;
      iterative_handle_finished_pattern();
      ++ss->col;
      goto skip_unmarking2;
    }

    if (u[to_state]) {
      ++ss->col;
      goto skip_unmarking2;
    }

    if (p + 1 == lmax) {
      ++ss->col;
      goto skip_unmarking2;
    }

    const unsigned int throwval = outthrowval[ss->from_state][ss->col];
    if (throwval != 0 && throwval != graph.h) {
      if (ss->excludes_throw == nullptr) {
        // mark states excluded by link throw; only need to do this once since
        // the link throws all come at the end of each row in `outmatrix`
        bool valid1 = true;
        unsigned int* const ds = ds_bystate[ss->from_state];
        unsigned int* es = es_throw[ss->from_state];
        ss->excludes_throw = es;  // save to clean up later
        ss->deadstates_throw = ds;

        for (unsigned int statenum; (statenum = *es); ++es) {
          if (++u[statenum] == 1 && ++*ds > 1 &&
              --max_possible < static_cast<int>(l_min))
            valid1 = false;
        }

        if (!valid1) {
          // undo marking operation and bail to previous beat
          es = ss->excludes_throw;
          for (unsigned int statenum; (statenum = *es); ++es) {
            if (--u[statenum] == 0 && --*ds > 0)
              ++max_possible;
          }

          --p;
          --ss;
          ++ss->col;
          ++nn;
          continue;
        }
      }

      // account for states excluded by link catch
      bool valid2 = true;
      unsigned int* const ds = ds_bystate[to_state];
      unsigned int* es = graph.excludestates_catch[to_state].data();
      ss->excludes_catch = es;
      ss->deadstates_catch = ds;

      for (unsigned int statenum; (statenum = *es); ++es) {
        if (++u[statenum] == 1 && ++*ds > 1 &&
            --max_possible < static_cast<int>(l_min))
          valid2 = false;
      }

      if (valid2) {
        // advance to next beat
        u[to_state] = 1;
        ss->to_state = to_state;
        ++p;
        ++ss;
        ss->col = 0;
        ss->col_limit = outdegree[to_state];
        ss->from_state = to_state;
        ss->to_state = 0;
        ss->outmatrix = outmatrix[to_state];
        ss->excludes_throw = nullptr;
        ss->excludes_catch = nullptr;
        goto skip_unmarking2;
      }

      // couldn't advance to next beat, so go to next column in this one
      ++ss->col;
      goto skip_unmarking1;
    } else {  // shift throw
      if (++steps >= steps_limit) {
        steps = 0;

        pos = p;
        if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
            pattern.at(i) = graph.outthrowval.at(beat.at(i + 1).from_state)
                                             .at(beat.at(i + 1).col);
          }
          pattern.at(pos + 1) = -1;
          nnodes = nn;
          process_inbox_running();
          iterative_update_after_split();
          nn = nnodes;
          steps_limit = steps_per_inbox_check;
        }
      }

      // advance to next beat
      u[to_state] = 1;
      ss->to_state = to_state;
      ++p;
      ++ss;
      ss->col = 0;
      ss->col_limit = outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = outmatrix[to_state];
      ss->excludes_throw = nullptr;
      ss->excludes_catch = nullptr;
      goto skip_unmarking2;
    }
  }

  ++p;
  pos = p;
  nnodes = nn;
  assert(pos == 0);
}

// Non-recursive version of get_loops_super()
//
// Template parameter `SUPER0` specifies whether no shift throws are allowed.
// When `SUPER0` == true then certain optimizations can be applied.

template<bool SUPER0>
void Worker::iterative_gen_loops_super() {
  if (!iterative_init_workspace(false)) {
    assert(false);
  }

  std::vector<std::vector<unsigned int>> is_linkthrow(graph.numstates + 1);
  if (!SUPER0) {
    for (size_t i = 1; i <= graph.numstates; ++i) {
      for (size_t j = 0; j < graph.outdegree[i]; ++j) {
        const unsigned int tv = graph.outthrowval[i][j];
        is_linkthrow.at(i).push_back(tv != 0 && tv != graph.h ? 1 : 0);
      }
    }
  }

  std::vector<unsigned int*> om_row(graph.numstates + 1, nullptr);
  std::vector<unsigned int*> ilt_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
    if (!SUPER0) {
      ilt_row.at(i) = is_linkthrow.at(i).data();
    }
  }
  unsigned int** const outmatrix = om_row.data();
  unsigned int** const islinkthrow = ilt_row.data();

  int p = pos;
  uint64_t nn = nnodes;
  const int lmax = l_max;
  int* const u = used.data();
  unsigned int steps = 0;
  int* const cu = cycleused.data();
  unsigned int* const outdegree = graph.outdegree.data();
  unsigned int* const cyclenum = graph.cyclenum.data();
  int* const isexitcycle = graph.isexitcycle.data();

  SearchState* ss = &beat.at(pos + 1);

  while (p >= 0) {
    // begin with any necessary cleanup from previous operations
    if (ss->to_cycle != -1) {
      cu[ss->to_cycle] = false;
      ss->to_cycle = -1;
    }
    if (!SUPER0) {
      if (ss->to_state != 0) {
        u[ss->to_state] = 0;
        ss->to_state = 0;
      }
    }

    skip_unmarking:
    if (ss->col == ss->col_limit) {
      --p;
      --ss;
      ++ss->col;
      ++nn;
      continue;
    }

    const unsigned int to_state = ss->outmatrix[ss->col];
    if (!SUPER0 && u[to_state]) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (SUPER0 || islinkthrow[ss->from_state][ss->col]) {
      if (to_state == start_state) {
        pos = p;
        iterative_handle_finished_pattern();
        ++ss->col;
        goto skip_unmarking;
      }

      const unsigned int to_cycle = cyclenum[to_state];
      if (cu[to_cycle]) {
        ++ss->col;
        goto skip_unmarking;
      }

      if ((SUPER0 || ss->shifts_remaining == 0) &&
          ss->exitcycles_remaining == 0) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (p + 1 == lmax) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (++steps >= steps_per_inbox_check) {
        steps = 0;

        pos = p;
        if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
            pattern.at(i) = graph.outthrowval.at(beat.at(i + 1).from_state)
                                             .at(beat.at(i + 1).col);
          }
          pattern.at(pos + 1) = -1;
          nnodes = nn;
          process_inbox_running();
          iterative_update_after_split();
          nn = nnodes;
        }
      }

      unsigned int next_shifts_remaining;
      if (!SUPER0) {
        u[to_state] = 1;
        next_shifts_remaining = ss->shifts_remaining;
      }
      cu[to_cycle] = true;
      ss->to_state = to_state;
      ss->to_cycle = to_cycle;
      const int next_exitcycles_remaining = (isexitcycle[to_cycle] ?
          ss->exitcycles_remaining - 1 : ss->exitcycles_remaining);
      ++p;
      ++ss;
      ss->col = 0;
      ss->col_limit = outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = outmatrix[to_state];
      ss->to_cycle = -1;
      if (!SUPER0) {
        ss->shifts_remaining = next_shifts_remaining;
      }
      ss->exitcycles_remaining = next_exitcycles_remaining;
      goto skip_unmarking;
    } else {  // shift throw
      const unsigned int next_shifts_remaining = ss->shifts_remaining;

      if (next_shifts_remaining == 0) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (to_state == start_state) {
        pos = p;
        iterative_handle_finished_pattern();
        ++ss->col;
        goto skip_unmarking;
      }

      if (p + 1 == lmax) {
        ++ss->col;
        goto skip_unmarking;
      }

      u[to_state] = 1;
      ss->to_state = to_state;
      // ss->to_cycle = -1;
      const int next_exitcycles_remaining = ss->exitcycles_remaining;
      ++p;
      ++ss;
      ss->col = 0;
      ss->col_limit = outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = outmatrix[to_state];
      ss->to_cycle = -1;
      ss->shifts_remaining = next_shifts_remaining - 1;
      ss->exitcycles_remaining = next_exitcycles_remaining;
      goto skip_unmarking;
    }
  }

  ++p;
  pos = p;
  nnodes = nn;
  assert(pos == 0);
}

// Explicit template instantiations since template method definition is not
// in `.h` file

template void Worker::iterative_gen_loops_normal<true>();
template void Worker::iterative_gen_loops_normal<false>();
template void Worker::iterative_gen_loops_super<true>();
template void Worker::iterative_gen_loops_super<false>();

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Set up the SearchState array with initial values.
//
// Leaves `pos` pointing to the last beat with loaded data, ready for the
// iterative algorithm to resume. Input parameter `marking` indicates whether
// to do marking operations during setup (the `excludes` and `deadstates`
// elements of SearchState).
//
// Returns true on success, false on failure.

bool Worker::iterative_init_workspace(bool marking) {
  if (!loading_work) {
    pos = 0;
    SearchState& ss = beat.at(pos + 1);
    ss.col = 0;
    ss.col_limit = graph.outdegree.at(start_state);
    ss.from_state = start_state;
    ss.to_state = 0;
    ss.outmatrix = graph.outmatrix.at(start_state).data();
    ss.excludes_throw = nullptr;
    ss.excludes_catch = nullptr;
    ss.to_cycle = -1;
    ss.shifts_remaining = config.shiftlimit;
    ss.exitcycles_remaining = exitcyclesleft;
    return true;
  }

  // When loading from a work assignment, load_work_assignment() will have
  // set up `pattern`, `root_pos`, and `root_throwval_options`

  loading_work = false;
  unsigned int last_from_state = start_state;
  int shifts_remaining = static_cast<int>(config.shiftlimit);
  int exitcycles_remaining = static_cast<int>(exitcyclesleft);

  for (size_t i = 0; pattern.at(i) != -1; ++i) {
    pos = static_cast<unsigned int>(i);
    SearchState& ss = beat.at(i + 1);
    ss.from_state = last_from_state;
    ss.col_limit = graph.outdegree.at(ss.from_state);

    unsigned int tv = static_cast<unsigned int>(pattern.at(i));

    for (ss.col = 0; ss.col < ss.col_limit; ++ss.col) {
      if (graph.outthrowval.at(ss.from_state).at(ss.col) == tv)
        break;
    }
    if (ss.col == ss.col_limit) {
      std::cerr << "error loading work assignment:\n"
                << "start_state: " << start_state
                << " (" << graph.state.at(start_state) << ")\n"
                << "pos: " << pos << '\n'
                << "pattern: ";
      for (size_t j = 0; pattern.at(j) != -1; ++j) {
        if (j != 0)
          std::cerr << ',';
        std::cerr << pattern.at(j);
      }
      std::cerr << '\n';
    }
    assert(ss.col < ss.col_limit);
    if (pos < root_pos) {
      ss.col_limit = ss.col + 1;
    }

    if (shifts_remaining < 0) {
      std::cerr << "shifts_remaining went negative during init\n";
      return false;
    }
    if (exitcycles_remaining < 0) {
      std::cerr << "exitcycles_remaining went negative during init\n";
      return false;
    }

    ss.to_state = graph.outmatrix.at(ss.from_state).at(ss.col);
    ss.outmatrix = graph.outmatrix.at(ss.from_state).data();
    ss.excludes_throw = nullptr;
    ss.excludes_catch = nullptr;
    ss.to_cycle = graph.cyclenum.at(ss.to_state);
    ss.shifts_remaining = static_cast<unsigned int>(shifts_remaining);
    ss.exitcycles_remaining = static_cast<unsigned int>(exitcycles_remaining);

    if (marking && tv != 0 && tv != graph.h) {
      // mark unreachable states due to link throw
      unsigned int* ds = deadstates_bystate.at(ss.from_state);
      unsigned int* es = graph.excludestates_throw.at(ss.from_state).data();
      ss.excludes_throw = es;
      ss.deadstates_throw = ds;

      for (unsigned int statenum; (statenum = *es); ++es) {
        if (++used.at(statenum) == 1 && ++*ds > 1 &&
            --max_possible < static_cast<int>(l_min)) {
          pos = 0;
          return false;
        }
      }

      // mark unreachable states due to link catch
      ds = deadstates_bystate.at(ss.to_state);
      es = graph.excludestates_catch.at(ss.to_state).data();
      ss.excludes_catch = es;
      ss.deadstates_catch = ds;

      for (unsigned int statenum; (statenum = *es); ++es) {
        if (++used.at(statenum) == 1 && ++*ds > 1 &&
            --max_possible < static_cast<int>(l_min)) {
          pos = 0;
          return false;
        }
      }
    }

    if (config.mode == RunMode::SUPER_SEARCH) {
      if (tv != 0 && tv != graph.h) {
        assert(!cycleused.at(ss.to_cycle));
        cycleused.at(ss.to_cycle) = true;
        if (graph.isexitcycle.at(ss.to_cycle)) {
          --exitcycles_remaining;
        }
      } else {
        if (shifts_remaining == 0)
          return false;
        --shifts_remaining;
        ss.to_cycle = -1;
      }
    }

    // mark next state as used
    assert(used.at(ss.to_state) == 0);
    if (config.mode == RunMode::NORMAL_SEARCH || config.shiftlimit > 0)
      used.at(ss.to_state) = 1;

    last_from_state = ss.to_state;
  }

  if (pattern.at(0) == -1 || pos < root_pos) {
    // loading a work assignment that was stolen from another worker
    assert(pattern.at(0) == -1 || pos + 1 == root_pos);
    SearchState& rss = beat.at(root_pos + 1);
    rss.from_state = last_from_state;
    rss.to_state = 0;
    rss.outmatrix = graph.outmatrix.at(rss.from_state).data();
    rss.excludes_throw = nullptr;
    rss.excludes_catch = nullptr;
    rss.to_cycle = -1;
    rss.shifts_remaining = shifts_remaining;
    rss.exitcycles_remaining = exitcycles_remaining;

    // set `col` at `root_pos`
    rss.col = graph.maxoutdegree;
    for (size_t i = 0; i < graph.outdegree.at(rss.from_state); ++i) {
      unsigned int throwval = graph.outthrowval.at(rss.from_state).at(i);
      if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
          throwval) != root_throwval_options.end()) {
        rss.col = std::min(rss.col, static_cast<unsigned int>(i));
      }
    }

    pos = root_pos;
  }

  // set `col_limit` at `root_pos`
  SearchState& rss = beat.at(root_pos + 1);
  rss.col_limit = 0;
  for (size_t i = 0; i < graph.outdegree.at(rss.from_state); ++i) {
    unsigned int throwval = graph.outthrowval.at(rss.from_state).at(i);
    if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
        throwval) != root_throwval_options.end()) {
      rss.col_limit = std::max(rss.col_limit, static_cast<unsigned int>(i + 1));
    }
  }
  assert(rss.col < rss.col_limit);
  assert(rss.col < graph.outdegree.at(rss.from_state));

  return true;
}

// Calculate `root_pos` and `root_throwval_options` during the middle of an
// iterative search.
//
// These elements are not updated during the search itself, so they need to be
// regenerated before we respond to incoming messages.
//
// Returns true on success, false on failure. Failure occurs when there are
// no positions < `pos` with unexplored options.

bool Worker::iterative_calc_rootpos_and_options() {
  unsigned int new_root_pos = 0;
  for (; new_root_pos < pos; ++new_root_pos) {
    SearchState& ss = beat.at(new_root_pos + 1);
    if (ss.col < ss.col_limit - 1)
      break;
  }
  if (new_root_pos == pos)
    return false;

  if (new_root_pos != root_pos) {
    root_pos = new_root_pos;
    notify_coordinator_update();
  }

  root_throwval_options.clear();
  SearchState& ss = beat.at(new_root_pos + 1);
  for (size_t col = ss.col + 1; col < ss.col_limit; ++col) {
    root_throwval_options.push_back(
        graph.outthrowval.at(ss.from_state).at(col));
  }
  return true;
}

// Determine whether we will be able to respond to a SPLIT_WORK request at
// our current point in iterative search.
//
// Needs an updated value of `root_pos`.

bool Worker::iterative_can_split() {
  for (size_t i = root_pos + 1; i <= pos; ++i) {
    SearchState& ss = beat.at(i + 1);
    if (ss.col < ss.col_limit - 1)
      return true;
  }
  return false;
}

// Update the state of the iterative search if a SPLIT_WORK request updated
// `root_pos` and/or `root_throwval_options`. Do nothing if there was no work
// split.

void Worker::iterative_update_after_split() {
  if (root_pos > pos) {
    std::cerr << "worker " << worker_id
              << ", root_pos: " << root_pos
              << ", pos: " << pos
              << '\n';
  }
  assert(root_pos <= pos);
  for (size_t i = 0; i < root_pos; ++i) {
    SearchState& ss = beat.at(i + 1);
    ss.col_limit = ss.col + 1;  // ensure no further iteration on this beat
  }
  SearchState& ss = beat.at(root_pos + 1);
  unsigned int new_col_limit = ss.col + 1;
  for (size_t i = ss.col + 1; i < graph.outdegree.at(ss.from_state); ++i) {
    unsigned int throwval = graph.outthrowval.at(ss.from_state).at(i);
    if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
        throwval) != root_throwval_options.end()) {
      new_col_limit = static_cast<unsigned int>(i + 1);
    }
  }
  ss.col_limit = new_col_limit;
}

inline void Worker::iterative_handle_finished_pattern() {
  ++count[pos + 1];

  if ((pos + 1) >= l_min && !config.countflag) {
    for (size_t i = 0; i <= pos; ++i) {
      pattern.at(i) = graph.outthrowval.at(beat.at(i + 1).from_state)
                                       .at(beat.at(i + 1).col);
    }
    pattern.at(pos + 1) = -1;
    report_pattern();
  }
}
