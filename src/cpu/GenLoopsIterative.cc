//
// GenLoopsIterative.cc
//
// Core graph search routines, implemented as iterative functions that are drop-
// in replacements for recursive versions in GenLoopsRecursive.cc. These
// routines are by far the most performance-critical portions of jprime.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cassert>


// This is a non-recursive version of gen_loops_normal(), with identical
// interface and behavior.
//
// It is generally faster than the recursive version and also avoids potential
// stack overflow on deeper searches.
//
// Template parameter `REPORT` specifies whether the patterns found are reported
// to the coordinator (true), or merely counted (false).
// Template parameter `REPLAY` specifies whether we want to stop execution at a
// particular value of `pos`; this is used for initialization.

template<bool REPORT, bool REPLAY>
void Worker::iterative_gen_loops_normal() {
  if (!REPLAY) {
    // initializing the working variables is a two-step process, starting with
    // setting up the workspace based on our current position in the search
    // tree, followed by a replay pass through the algorithm to initialize all
    // other working variables
    iterative_init_workspace();

    // replay back through the algorithm up to and including position `pos`.
    // this sets up variables like used[], etc.
    if (pos == -1) {
      // search is just beginning; no need to replay
      pos = 0;
    } else {
      const auto pos_orig = pos;
      replay_to_pos = pos;
      pos = 0;
      iterative_gen_loops_normal<false, true>();
      assert(pos == pos_orig);
    }
    // now we can resume
  }

  // temp storage to access matrix elements quickly
  std::vector<unsigned*> om_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
  }
  unsigned** const outmatrix = om_row.data();

  // local variables to improve performance
  unsigned p = pos;
  uint64_t nn = nnodes;
  const unsigned nmax = n_max;
  int* const u = used.data();
  unsigned steps = 0;
  unsigned steps_limit = steps_per_inbox_check;
  const unsigned st_state = start_state;
  unsigned* const outdegree = graph.outdegree.data();
  std::uint64_t* const c = count.data();

  WorkCell* wc = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = wc->from_state;

  // main search loop
  while (true) {
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (wc->col == wc->col_limit) {
      // beat is finished, go back to previous one
      if (REPLAY) {
        assert(false);
      }
      u[from_state] = 0;
      ++nn;

      if (p == 0) {
        break;
      }
      --p;
      --wc;
      from_state = wc->from_state;
      ++wc->col;
      continue;
    }

    const unsigned to_state = outmatrix[from_state][wc->col];

    if (to_state == st_state) {
      // found a pattern
      if (REPLAY) {
        assert(false);
      }
      if (REPORT && p + 1 >= n_min) {
        pos = p;
        iterative_handle_finished_pattern();
      }
      ++c[p + 1];
      ++wc->col;
      continue;
    }

    if (to_state < st_state) {
      ++wc->col;
      continue;
    }

    if (u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    if (p + 1 == nmax) {
      if (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    if (!REPLAY && ++steps >= steps_limit) {
      // check the inbox for incoming messages, after doing some prep work
      // to ensure we can respond to any type of message we might receive
      //
      // this code is rarely evaluated so it is not performance-critical
      steps = 0;

      pos = p;
      if (iterative_can_split()) {
        for (int i = 0; i <= pos; ++i) {
          pattern.at(i) = graph.outthrowval.at(beat.at(i).from_state)
                                           .at(beat.at(i).col);
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

    // current throw is valid, so advance to next beat
    u[to_state] = 1;

    ++p;
    ++wc;
    if (REPLAY) {
      assert(wc->from_state == to_state);
    } else {
      wc->col = 0;
      wc->col_limit = outdegree[to_state];
      wc->from_state = to_state;
    }
    from_state = to_state;
  }

  if (REPLAY) {
    assert(p == replay_to_pos);
    assert(nn == nnodes);
  } else {
    assert(p == 0);
  }

  pos = p;
  nnodes = nn;
}

// Non-recursive version of gen_loops_normal_marking().

template<bool REPLAY>
void Worker::iterative_gen_loops_normal_marking() {
  if (!REPLAY) {
    iterative_init_workspace();
    if (pos == -1) {
      pos = 0;
    } else {
      const auto pos_orig = pos;
      replay_to_pos = pos;
      pos = 0;
      iterative_gen_loops_normal_marking<true>();
      assert(pos == pos_orig);
    }
  }

  std::vector<unsigned*> om_row(graph.numstates + 1, nullptr);
  std::vector<unsigned*> otv_row(graph.numstates + 1, nullptr);
  std::vector<unsigned*> es_throw_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
    otv_row.at(i) = graph.outthrowval.at(i).data();
    es_throw_row.at(i) = graph.excludestates_throw.at(i).data();
  }
  unsigned** const outmatrix = om_row.data();
  unsigned** const outthrowval = otv_row.data();
  unsigned** const es_throw = es_throw_row.data();

  unsigned p = pos;
  uint64_t nn = nnodes;
  const unsigned nmax = n_max;
  int* const u = used.data();
  unsigned steps = 0;
  unsigned steps_limit = steps_per_inbox_check;
  const unsigned st_state = start_state;
  unsigned** const ds_bystate = deadstates_bystate.data();
  unsigned* const outdegree = graph.outdegree.data();

  WorkCell* wc = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = wc->from_state;

  while (true) {
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (wc->col == wc->col_limit) {
      // beat is finished, backtrack after cleaning up marking operations
      unsigned* const ds = ds_bystate[from_state];
      unmark(u, wc->excludes_throw, ds);
      u[from_state] = 0;
      ++nn;

      if (p == 0) {
        break;
      }
      --p;
      --wc;
      unmark(u, wc->excludes_catch, ds);
      from_state = wc->from_state;
      ++wc->col;
      if (REPLAY) {
        std::cerr << "---- worker " << worker_id << " backtracked to pos "
                  << p << '\n';
        assert(wc->col == wc->col_limit);
      }
      continue;
    }

    const unsigned to_state = outmatrix[from_state][wc->col];

    if (to_state == st_state) {
      if (REPLAY) {
        assert(false);
      }
      if (p + 1 >= n_min && !config.countflag) {
        pos = p;
        iterative_handle_finished_pattern();
      }
      ++count[p + 1];
      ++wc->col;
      continue;
    }

    if (to_state < st_state) {
      ++wc->col;
      continue;
    }

    if (u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    if (p + 1 == nmax) {
      if (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    const unsigned throwval = outthrowval[from_state][wc->col];

    if (throwval != 0 && throwval != graph.h) {  // link throw
      if (wc->excludes_throw == nullptr) {
        // mark states on the `from_state` shift cycle that are excluded by the
        // link throw; only need to do this once since the link throws all come
        // at the end of each row in `outmatrix` and the excluded states are
        // independent of link throw value
        unsigned* es = es_throw[from_state];
        unsigned* const ds = ds_bystate[from_state];
        wc->excludes_throw = es;  // save for backtracking

        if (!mark(u, es, ds)) {
          // not valid, undo marking operation and bail to previous beat
          unmark(u, wc->excludes_throw, ds);
          u[from_state] = 0;
          ++nn;

          if (p == 0) {
            break;
          }
          --p;
          --wc;
          unmark(u, wc->excludes_catch, ds);
          from_state = wc->from_state;
          ++wc->col;
          if (REPLAY) {
            // we might get here if we're loading a job that was stolen from
            // another worker, and the job's prefix throws cause us to
            // backtrack due to `max_possible`, i.e. we can't replay up to
            // `root_pos`.
            std::cerr << "---- worker " << worker_id << ", condition 1 ----\n"
                      << "---- worker " << worker_id << " backtracked to pos "
                      << pos << '\n';
            assert(wc->col == wc->col_limit);
          }
          continue;
        }
      }

      // mark states excluded by link catch
      unsigned* es = graph.excludestates_catch[to_state].data();
      unsigned* const ds = ds_bystate[to_state];
      wc->excludes_catch = es;

      if (mark(u, es, ds)) {
        // valid, advance to next beat
        u[to_state] = 1;

        ++p;
        ++wc;
        if (REPLAY) {
          assert(wc->from_state == to_state);
        } else {
          wc->col = 0;
          wc->col_limit = outdegree[to_state];
          wc->from_state = to_state;
        }
        wc->excludes_throw = nullptr;
        wc->excludes_catch = nullptr;
        from_state = to_state;
        continue;
      }

      // couldn't advance to next beat, so go to next column in this one after
      // undoing the catch marking operation done above
      unmark(u, wc->excludes_catch, ds);
      ++wc->col;
      if (REPLAY) {
        std::cerr << "---- worker " << worker_id << ", condition 2 ----\n";
        assert(wc->col == wc->col_limit);
      }
      continue;
    } else {  // shift throw
      if (!REPLAY && ++steps >= steps_limit) {
        steps = 0;

        pos = p;
        if (iterative_can_split()) {
          for (int i = 0; i <= pos; ++i) {
            pattern.at(i) = graph.outthrowval.at(beat.at(i).from_state)
                                             .at(beat.at(i).col);
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

      ++p;
      ++wc;
      if (REPLAY) {
        assert(wc->from_state == to_state);
      } else {
        wc->col = 0;
        wc->col_limit = outdegree[to_state];
        wc->from_state = to_state;
      }
      wc->excludes_throw = nullptr;
      wc->excludes_catch = nullptr;
      from_state = to_state;
      continue;
    }
  }

  if (REPLAY) {
    if (p != replay_to_pos) {
      std::cerr << "### worker " << worker_id << " couldn't replay to pos "
                << replay_to_pos << ", exited at pos " << p << '\n';
    }
    assert(p == replay_to_pos);
    assert(nn == nnodes);
  } else {
    assert(p == 0);
  }

  pos = p;
  nnodes = nn;
}

// Helpers for iterative_gen_loops_marking()

inline bool Worker::mark(int* const& u, unsigned*& es, unsigned* const& ds) {
  bool valid = true;
  for (unsigned statenum; (statenum = *es); ++es) {
    if (++u[statenum] == 1 && ++*ds > 1 &&
        --max_possible < static_cast<int>(n_min)) {
      valid = false;
    }
  }
  return valid;
}

inline void Worker::unmark(int* const& u, unsigned*& es, unsigned* const& ds) {
  if (es) {
    for (unsigned statenum; (statenum = *es); ++es) {
      if (--u[statenum] == 0 && --*ds > 0) {
        ++max_possible;
      }
    }
    es = nullptr;
  }
}

// Non-recursive version of gen_loops_super() and gen_loops_super0().
//
// Template parameter `SUPER0` specifies whether no shift throws are allowed.
// When `SUPER0` == true then certain optimizations can be applied.

template<bool SUPER0, bool REPLAY>
void Worker::iterative_gen_loops_super() {
  if (!REPLAY) {
    iterative_init_workspace();
    if (pos == -1) {
      pos = 0;
    } else {
      const auto pos_orig = pos;
      replay_to_pos = pos;
      pos = 0;
      iterative_gen_loops_super<SUPER0, true>();
      assert(pos == pos_orig);
    }
  }

  std::vector<unsigned*> om_row(graph.numstates + 1, nullptr);
  for (size_t i = 0; i <= graph.numstates; ++i) {
    om_row.at(i) = graph.outmatrix.at(i).data();
  }
  unsigned** const outmatrix = om_row.data();

  unsigned p = pos;
  uint64_t nn = nnodes;
  const unsigned nmax = n_max;
  int* const u = used.data();
  unsigned steps = 0;
  int* const cu = cycleused.data();
  unsigned* const outdegree = graph.outdegree.data();
  unsigned* const cyclenum = graph.cyclenum.data();
  int* const isexitcycle = graph.isexitcycle.data();

  WorkCell* wc = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = wc->from_state;
  unsigned from_cycle = cyclenum[from_state];

  while (true) {
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (wc->col == wc->col_limit) {
      // beat is finished, go back to previous one
      if (REPLAY) {
        assert(false);
      }
      if (!SUPER0) {
        u[from_state] = 0;
      }
      ++nn;

      if (p == 0) {
        break;
      }
      --p;
      --wc;
      const unsigned to_cycle = from_cycle;
      from_state = wc->from_state;
      from_cycle = cyclenum[from_state];
      if (from_cycle == to_cycle) {  // unwinding a shift throw
        --shiftcount;
      } else {  // link throw
        cu[to_cycle] = false;
        if (isexitcycle[to_cycle]) {
          ++exitcyclesleft;
        }
      }
      ++wc->col;
      continue;
    }

    const unsigned to_state = outmatrix[from_state][wc->col];

    if (to_state < start_state) {
      ++wc->col;
      continue;
    }

    if (!SUPER0 && u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    const unsigned to_cycle = cyclenum[to_state];

    if (SUPER0 || (to_cycle != from_cycle)) {  // link throw
      if (to_state == start_state) {
        if (REPLAY) {
          assert(false);
        }
        if (p + 1 >= n_min && !config.countflag) {
          pos = p;
          iterative_handle_finished_pattern();
        }
        ++count[p + 1];
        ++wc->col;
        continue;
      }

      if (cu[to_cycle]) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if ((SUPER0 || shiftcount == config.shiftlimit) && exitcyclesleft == 0) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if (p + 1 == nmax) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if (!REPLAY && ++steps >= steps_per_inbox_check) {
        steps = 0;

        pos = p;
        if (iterative_can_split()) {
          for (int i = 0; i <= pos; ++i) {
            pattern.at(i) = graph.outthrowval.at(beat.at(i).from_state)
                                             .at(beat.at(i).col);
          }
          pattern.at(pos + 1) = -1;
          nnodes = nn;
          process_inbox_running();
          iterative_update_after_split();
          nn = nnodes;
        }
      }

      if (!SUPER0) {
        u[to_state] = 1;
      }
      cu[to_cycle] = true;
      if (isexitcycle[to_cycle]) {
        --exitcyclesleft;
      }
    } else {  // shift throw
      if (shiftcount == config.shiftlimit) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if (to_state == start_state) {
        if (REPLAY) {
          assert(false);
        }
        if (shiftcount < p) {
          // don't allow all shift throws
          if (p + 1 >= n_min && !config.countflag) {
            pos = p;
            iterative_handle_finished_pattern();
          }
          ++count[p + 1];
        }
        ++wc->col;
        continue;
      }

      if (p + 1 == nmax) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      u[to_state] = 1;
      ++shiftcount;
    }

    // advance to next beat
    ++p;
    ++wc;
    if (REPLAY) {
      assert(wc->from_state == to_state);
    } else {
      wc->col = 0;
      wc->col_limit = outdegree[to_state];
      wc->from_state = to_state;
    }
    from_state = to_state;
    from_cycle = to_cycle;
  }

  if (REPLAY) {
    assert(p == replay_to_pos);
    assert(nn == nnodes);
  } else {
    assert(p == 0);
  }

  pos = p;
  nnodes = nn;
}

// Explicit template instantiations since template method definitions are not in
// the `.h` file.

// regular versions
template void Worker::iterative_gen_loops_normal<true, false>();
template void Worker::iterative_gen_loops_normal<false, false>();
template void Worker::iterative_gen_loops_normal_marking<false>();
template void Worker::iterative_gen_loops_super<true, false>();
template void Worker::iterative_gen_loops_super<false, false>();

// replay versions
template void Worker::iterative_gen_loops_normal<false, true>();
template void Worker::iterative_gen_loops_normal_marking<true>();
template void Worker::iterative_gen_loops_super<true, true>();
template void Worker::iterative_gen_loops_super<false, true>();

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Initialize the WorkCells in our workspace.
//
// Leaves `pos` pointing to the last beat with loaded data.
//
// If a work assignment cannot be loaded, throw a std::invalid_argument
// exception with a relevant error message.

void Worker::iterative_init_workspace() {
  WorkAssignment wa = get_work_assignment();
  wa.to_workspace(this, 0);

  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, 0);
  assert(wa == wa2);
}

// Determine whether we will be able to respond to a SPLIT_WORK request at our
// current point in iterative search.
//
// Also update the worker's values of `root_pos` and `root_throwval_options`.

bool Worker::iterative_can_split() {
  WorkAssignment wa;
  wa.from_workspace(this, 0);

  root_throwval_options = wa.root_throwval_options;

  assert(wa.root_pos >= root_pos);
  if (wa.root_pos != root_pos) {
    root_pos = wa.root_pos;
    notify_coordinator_update();
  }

  return wa.is_splittable();
}

// Update the workspace in case a SPLIT_WORK request changed our work
// assignment.

void Worker::iterative_update_after_split() {
  const auto pos_orig = pos;
  WorkAssignment wa = get_work_assignment();
  wa.to_workspace(this, 0);

  // splitting shouldn't change our depth in the search tree
  assert(pos == pos_orig);

  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, 0);
  assert(wa == wa2);
}

inline void Worker::iterative_handle_finished_pattern() {
  for (int i = 0; i <= pos; ++i) {
    pattern.at(i) = graph.outthrowval.at(beat.at(i).from_state)
                                     .at(beat.at(i).col);
  }
  pattern.at(pos + 1) = -1;
  report_pattern();
}

//------------------------------------------------------------------------------
// WorkSpace methods
//------------------------------------------------------------------------------

const Graph& Worker::get_graph() const {
  return graph;
}

void Worker::set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state) {
  assert(slot == 0);
  assert(index < beat.size());
  WorkCell& wc = beat.at(index);
  wc.col = col;
  wc.col_limit = col_limit;
  wc.from_state = from_state;
}

std::tuple<unsigned, unsigned, unsigned> Worker::get_cell(unsigned slot,
    unsigned index) const {
  assert(slot == 0);
  assert(index < beat.size());
  const WorkCell& wc = beat.at(index);
  return std::make_tuple(wc.col, wc.col_limit, wc.from_state);
}

void Worker::set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos) {
  assert(slot == 0);
  start_state = new_start_state;
  end_state = new_end_state;
  pos = new_pos;
}

std::tuple<unsigned, unsigned, int> Worker::get_info(unsigned slot) const {
  assert(slot == 0);
  return std::make_tuple(start_state, end_state, pos);
}
