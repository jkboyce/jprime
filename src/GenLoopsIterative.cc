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
#include <cassert>


// This is a non-recursive version of get_loops_normal(), with identical
// interface and behavior.
//
// It is slightly faster than the recursive version and also avoids potential
// stack overflow on deeper searches.

void Worker::iterative_gen_loops_normal_marking() {
  if (!iterative_init_workspace()) {
    assert(false);
  }

  // Note that beat 0 is stored at index 1 in the `beat` array. We do this to
  // provide a guard since beat[0].col is modified at the end of the search.
  SearchState* ss = &beat[pos + 1];

  while (pos >= 0) {
    // begin with any necessary cleanup from previous marking operations
    if (ss->to_state != 0) {
      used[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking1:
    if (ss->excludes_catch) {
      int* const ds = ss->deadstates_catch;
      int* es = ss->excludes_catch;
      for (int statenum; (statenum = *es); ++es) {
        if (--used[statenum] == 0 && --*ds > 0)
          ++max_possible;
      }
      ss->excludes_catch = nullptr;
    }

    skip_unmarking2:
    if (ss->col == ss->col_limit) {
      if (ss->excludes_throw) {
        int* const ds = ss->deadstates_throw;
        int* es = ss->excludes_throw;
        for (int statenum; (statenum = *es); ++es) {
          if (--used[statenum] == 0 && --*ds > 0)
            ++max_possible;
        }
      }

      --pos;
      --ss;
      ++ss->col;
      ++nnodes;
      continue;
    }

    const int to_state = ss->outmatrix[ss->col];
    if (to_state == start_state) {
      iterative_handle_finished_pattern();
      ++ss->col;
      goto skip_unmarking2;
    }

    if (used[to_state]) {
      ++ss->col;
      goto skip_unmarking2;
    }
    if (pos + 1 == l_max) {
      ++ss->col;
      goto skip_unmarking2;
    }

    const int throwval = graph.outthrowval[ss->from_state][ss->col];
    if (throwval != 0 && throwval != graph.h) {
      if (ss->excludes_throw == nullptr) {
        // mark states excluded by link throw; only need to do this once since
        // the link throws all come at the end of each row in `outmatrix`
        bool valid1 = true;
        int* const ds = deadstates_bystate[ss->from_state];
        int* es = graph.excludestates_throw[ss->from_state];
        ss->excludes_throw = es;  // save to clean up later
        ss->deadstates_throw = ds;

        for (int statenum; (statenum = *es); ++es) {
          if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min)
            valid1 = false;
        }

        if (!valid1) {
          // undo marking operation and bail to previous beat
          es = ss->excludes_throw;
          for (int statenum; (statenum = *es); ++es) {
            if (--used[statenum] == 0 && --*ds > 0)
              ++max_possible;
          }

          --pos;
          --ss;
          ++ss->col;
          ++nnodes;
          continue;
        }
      }

      // account for states excluded by link catch
      bool valid2 = true;
      int* const ds = deadstates_bystate[to_state];
      int* es = graph.excludestates_catch[to_state];
      ss->excludes_catch = es;
      ss->deadstates_catch = ds;

      for (int statenum; (statenum = *es); ++es) {
        if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min)
          valid2 = false;
      }

      if (valid2) {
        // advance to next beat
        used[to_state] = 1;
        ss->to_state = to_state;
        ++pos;
        ++ss;
        ss->col = 0;
        ss->col_limit = graph.outdegree[to_state];
        ss->from_state = to_state;
        ss->to_state = 0;
        ss->outmatrix = graph.outmatrix[to_state];
        ss->excludes_throw = nullptr;
        ss->excludes_catch = nullptr;
        goto skip_unmarking2;
      }

      // couldn't advance to next beat, so go to next column in this one
      ++ss->col;
      goto skip_unmarking1;
    } else {  // shift throw
      if (++steps_taken >= steps_per_inbox_check) {
        steps_taken = 0;
        iterative_calc_rootpos_and_options();

        if (iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
            pattern[i] = graph.outthrowval[beat[i + 1].from_state]
                                          [beat[i + 1].col];
          }
          pattern[pos + 1] = -1;
          process_inbox_running();
          iterative_update_after_split();
        }
      }

      // advance to next beat
      used[to_state] = 1;
      ss->to_state = to_state;
      ++pos;
      ++ss;
      ss->col = 0;
      ss->col_limit = graph.outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = graph.outmatrix[to_state];
      ss->excludes_throw = nullptr;
      ss->excludes_catch = nullptr;
      goto skip_unmarking2;
    }
  }

  ++pos;
  assert(pos == 0);
}

// Non-recursive version of get_loops_super()

void Worker::iterative_gen_loops_super() {
  if (!iterative_init_workspace()) {
    assert(false);
  }

  SearchState* ss = &beat[pos + 1];

  while (pos >= 0) {
    // begin with any necessary cleanup from previous operations
    if (ss->to_cycle != -1) {
      cycleused[ss->to_cycle] = false;
      ss->to_cycle = -1;
    }
    if (ss->to_state != 0) {
      used[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking:
    if (ss->col == ss->col_limit) {
      --pos;
      --ss;
      ++ss->col;
      ++nnodes;
      continue;
    }

    const int to_state = ss->outmatrix[ss->col];
    if (used[to_state]) {
      ++ss->col;
      goto skip_unmarking;
    }

    const int throwval = graph.outthrowval[ss->from_state][ss->col];
    const bool linkthrow = (throwval != 0 && throwval != graph.h);
    const int shifts_remaining = ss->shifts_remaining;

    if (linkthrow) {
      if (to_state == start_state) {
        iterative_handle_finished_pattern();
        ++ss->col;
        goto skip_unmarking;
      }

      const int to_cycle = graph.cyclenum[to_state];
      if (cycleused[to_cycle]) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (pos + 1 == l_max) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (++steps_taken >= steps_per_inbox_check) {
        steps_taken = 0;
        iterative_calc_rootpos_and_options();

        if (iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
            pattern[i] = graph.outthrowval[beat[i + 1].from_state]
                                          [beat[i + 1].col];
          }
          pattern[pos + 1] = -1;
          process_inbox_running();
          iterative_update_after_split();
        }
      }

      used[to_state] = 1;
      cycleused[to_cycle] = true;
      ss->to_state = to_state;
      ss->to_cycle = to_cycle;
      ++pos;
      ++ss;
      ss->col = 0;
      ss->col_limit = graph.outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = graph.outmatrix[to_state];
      ss->to_cycle = -1;
      ss->shifts_remaining = shifts_remaining;
      goto skip_unmarking;
    } else {  // shift throw
      if (shifts_remaining == 0) {
        ++ss->col;
        goto skip_unmarking;
      }

      if (to_state == start_state) {
        iterative_handle_finished_pattern();
        ++ss->col;
        goto skip_unmarking;
      }

      if (pos + 1 == l_max) {
        ++ss->col;
        goto skip_unmarking;
      }

      used[to_state] = 1;
      ss->to_state = to_state;
      // ss->to_cycle = -1;
      ++pos;
      ++ss;
      ss->col = 0;
      ss->col_limit = graph.outdegree[to_state];
      ss->from_state = to_state;
      ss->to_state = 0;
      ss->outmatrix = graph.outmatrix[to_state];
      ss->to_cycle = -1;
      ss->shifts_remaining = shifts_remaining - 1;
      goto skip_unmarking;
    }
  }

  ++pos;
  assert(pos == 0);
}

// Non-recursive version of get_loops_super0()

void Worker::iterative_gen_loops_super0() {
  if (!iterative_init_workspace()) {
    assert(false);
  }

  SearchState* ss = &beat[pos + 1];

  while (pos >= 0) {
    // begin with any necessary cleanup from previous operations
    if (ss->to_cycle != -1) {
      cycleused[ss->to_cycle] = false;
      ss->to_cycle = -1;
    }

    skip_unmarking:
    if (ss->col == ss->col_limit) {
      --pos;
      --ss;
      ++ss->col;
      ++nnodes;
      continue;
    }

    const int to_state = ss->outmatrix[ss->col];
    if (to_state == start_state) {
      iterative_handle_finished_pattern();
      ++ss->col;
      goto skip_unmarking;
    }

    const int to_cycle = graph.cyclenum[to_state];
    if (cycleused[to_cycle]) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (ss->exitcycles_remaining == 0) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (pos + 1 == l_max) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (++steps_taken >= steps_per_inbox_check) {
      steps_taken = 0;
      iterative_calc_rootpos_and_options();

      if (iterative_can_split()) {
        for (size_t i = 0; i <= pos; ++i) {
          pattern[i] = graph.outthrowval[beat[i + 1].from_state]
                                        [beat[i + 1].col];
        }
        pattern[pos + 1] = -1;
        process_inbox_running();
        iterative_update_after_split();
      }
    }

    cycleused[to_cycle] = true;
    ss->to_state = to_state;
    ss->to_cycle = to_cycle;
    int exitcycles_remaining = (graph.isexitcycle[to_cycle] ?
        ss->exitcycles_remaining - 1 : ss->exitcycles_remaining);
    ++pos;
    ++ss;
    ss->col = 0;
    ss->col_limit = graph.outdegree[to_state];
    ss->from_state = to_state;
    ss->to_state = 0;
    ss->outmatrix = graph.outmatrix[to_state];
    ss->to_cycle = -1;
    ss->exitcycles_remaining = exitcycles_remaining;
    goto skip_unmarking;
  }

  ++pos;
  assert(pos == 0);
}

// Set up the SearchState array with initial values.
//
// Leaves `pos` pointing to the last beat with loaded data, ready for the
// iterative algorithm to resume.
//
// Returns true on success, false on failure.

bool Worker::iterative_init_workspace() {
  if (!loading_work) {
    pos = 0;
    SearchState& ss = beat[pos + 1];
    ss.col = 0;
    ss.col_limit = graph.outdegree[start_state];
    ss.from_state = start_state;
    ss.to_state = 0;
    ss.outmatrix = graph.outmatrix[start_state];
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
  pos = -1;
  int last_from_state = start_state;
  int shifts_remaining = config.shiftlimit;
  int exitcycles_remaining = exitcyclesleft;

  for (size_t i = 0; pattern[i] != -1; ++i) {
    pos = i;
    SearchState& ss = beat[i + 1];
    ss.from_state = last_from_state;
    ss.col_limit = graph.outdegree[ss.from_state];

    for (ss.col = 0; ss.col < ss.col_limit; ++ss.col) {
      if (graph.outthrowval[ss.from_state][ss.col] == pattern[i])
        break;
    }
    if (ss.col == ss.col_limit) {
      std::cerr << "error loading work assignment:\n"
                << "start_state: " << start_state
                << " (" << graph.state[start_state] << ")\n"
                << "pos: " << pos << '\n'
                << "pattern: ";
      for (size_t j = 0; pattern[j] != -1; ++j)
        std::cerr << throw_char(pattern[j]);
      std::cerr << '\n';
    }
    assert(ss.col < ss.col_limit);
    if (pos < root_pos)
      ss.col_limit = ss.col + 1;

    ss.to_state = graph.outmatrix[ss.from_state][ss.col];
    ss.outmatrix = graph.outmatrix[ss.from_state];
    ss.excludes_throw = nullptr;
    ss.excludes_catch = nullptr;
    ss.to_cycle = graph.cyclenum[ss.to_state];
    ss.shifts_remaining = shifts_remaining;
    ss.exitcycles_remaining = exitcycles_remaining;

    if (config.mode == RunMode::NORMAL_SEARCH) {
      if (pattern[i] != 0 && pattern[i] != graph.h) {
        // mark unreachable states due to link throw
        int* ds = deadstates_bystate[ss.from_state];
        int* es = graph.excludestates_throw[ss.from_state];
        ss.excludes_throw = es;
        ss.deadstates_throw = ds;

        for (int statenum; (statenum = *es); ++es) {
          if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min) {
            pos = 0;
            return false;
          }
        }

        // mark unreachable states due to link catch
        ds = deadstates_bystate[ss.to_state];
        es = graph.excludestates_catch[ss.to_state];
        ss.excludes_catch = es;
        ss.deadstates_catch = ds;

        for (int statenum; (statenum = *es); ++es) {
          if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min) {
            pos = 0;
            return false;
          }
        }
      }
    }

    if (config.mode == RunMode::SUPER_SEARCH) {
      if (pattern[i] != 0 && pattern[i] != graph.h) {
        assert(!cycleused[ss.to_cycle]);
        cycleused[ss.to_cycle] = true;
        if (graph.isexitcycle[ss.to_cycle])
          --exitcycles_remaining;
      } else {
        if (shifts_remaining == 0)
          return false;
        --shifts_remaining;
        ss.to_cycle = -1;
      }
    }

    // mark next state as used
    assert(used[ss.to_state] == 0);
    if (config.mode == RunMode::NORMAL_SEARCH || config.shiftlimit > 0)
      used[ss.to_state] = 1;

    last_from_state = ss.to_state;
  }

  if (pos < root_pos) {
    // loading a work assignment that was stolen from another worker
    assert(pos == root_pos - 1);

    SearchState& rss = beat[root_pos + 1];
    rss.from_state = last_from_state;
    rss.to_state = 0;
    rss.outmatrix = graph.outmatrix[rss.from_state];
    rss.excludes_throw = nullptr;
    rss.excludes_catch = nullptr;
    rss.to_cycle = -1;
    rss.shifts_remaining = shifts_remaining;
    rss.exitcycles_remaining = exitcycles_remaining;

    // set `col` at `root_pos`
    rss.col = graph.maxoutdegree;
    for (size_t i = 0; i < graph.outdegree[rss.from_state]; ++i) {
      int throwval = graph.outthrowval[rss.from_state][i];
      if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
          throwval) != root_throwval_options.end()) {
        rss.col = std::min<int>(rss.col, i);
      }
    }

    pos = root_pos;
  }

  // set `col_limit` at `root_pos`
  SearchState& rss = beat[root_pos + 1];
  rss.col_limit = 0;
  for (size_t i = 0; i < graph.outdegree[rss.from_state]; ++i) {
    int throwval = graph.outthrowval[rss.from_state][i];
    if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
        throwval) != root_throwval_options.end()) {
      rss.col_limit = std::max<int>(rss.col_limit, i + 1);
    }
  }
  assert(rss.col < rss.col_limit);
  assert(rss.col < graph.outdegree[rss.from_state]);

  return true;
}

// Calculate `root_pos` and `root_throwval_options` during the middle of an
// iterative search.
//
// These elements are not updated during the search itself, so they need to be
// regenerated before we respond to incoming messages.

void Worker::iterative_calc_rootpos_and_options() {
  int new_root_pos = 0;
  for (; new_root_pos < pos; ++new_root_pos) {
    SearchState& ss = beat[new_root_pos + 1];
    if (ss.col < ss.col_limit - 1)
      break;
  }
  assert(new_root_pos < pos);

  if (new_root_pos != root_pos) {
    root_pos = new_root_pos;
    notify_coordinator_rootpos();
  }

  root_throwval_options.clear();
  SearchState& ss = beat[new_root_pos + 1];
  for (size_t col = ss.col + 1; col < ss.col_limit; ++col) {
    root_throwval_options.push_back(graph.outthrowval[ss.from_state][col]);
  }
}

// Determine whether we will be able to respond to a SPLIT_WORK request at
// our current point in iterative search.
//
// Needs an updated value of `root_pos`.

bool Worker::iterative_can_split() {
  for (size_t i = root_pos + 1; i <= pos; ++i) {
    SearchState& ss = beat[i + 1];
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
    SearchState& ss = beat[i + 1];
    ss.col_limit = ss.col + 1;  // ensure no further iteration on this beat
  }
  SearchState& ss = beat[root_pos + 1];
  int new_col_limit = ss.col + 1;
  for (size_t i = ss.col + 1; i < graph.outdegree[ss.from_state]; ++i) {
    int throwval = graph.outthrowval[ss.from_state][i];
    if (std::find(root_throwval_options.begin(), root_throwval_options.end(),
        throwval) != root_throwval_options.end()) {
      new_col_limit = i + 1;
    }
  }
  ss.col_limit = new_col_limit;
}

inline void Worker::iterative_handle_finished_pattern() {
  ++count[pos + 1];

  if ((pos + 1) >= l_min && !config.countflag) {
    for (size_t i = 0; i <= pos; ++i) {
      pattern[i] = graph.outthrowval[beat[i + 1].from_state][beat[i + 1].col];
    }
    report_pattern();
  }
}