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


//------------------------------------------------------------------------------
// NORMAL and NORMAL_MARKING modes
//------------------------------------------------------------------------------

// This is a non-recursive version of gen_loops_normal() and
// gen_loops_normal_marking(), with identical interface and behavior.
//
// It is generally faster than the recursive versions and also avoids potential
// stack overflow on deeper searches.
//
// Template parameters:
// - `MARKING` specifies whether to apply the excluded states marking algorithm
// - `REPORT` specifies whether the patterns found are reported to the
//    coordinator (true), or merely counted (false)
// - `REPLAY` specifies whether we want to stop execution at a particular value
//    of `pos`; this is used for initialization

template<bool MARKING, bool REPORT, bool REPLAY>
void Worker::iterative_gen_loops_normal()
{
  if constexpr (!REPLAY) {
    // initializing the working variables is a two-step process, starting with
    // setting up the workspace based on our current position in the search
    // tree, followed by a replay pass through the algorithm to initialize all
    // other working variables
    iterative_init_workspace();

    // replay back through the algorithm up to and including position `pos`.
    // this sets up variables like used[], etc.
    if (pos == -1) {
      // new search for this `start_state` value; no need to replay
      pos = 0;
    } else {
      const auto pos_orig = pos;
      replay_to_pos = pos;
      pos = 0;
      iterative_gen_loops_normal<MARKING, false, true>();
      (void)pos_orig;
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

  // marking-related items
  std::vector<unsigned*> es_tail_row;
  unsigned** ds_bystate = nullptr;
  if constexpr (MARKING) {
    es_tail_row.resize(graph.numstates + 1, nullptr);
    for (size_t i = 0; i <= graph.numstates; ++i) {
      es_tail_row.at(i) = excludestates_tail.at(i).data();
    }
    ds_bystate = deadstates_bystate.data();
  }
  unsigned** const es_tail = es_tail_row.data();

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
  bool mark_any_link_throw = true;

  while (true) {
    if constexpr (REPLAY) {
      if (p == replay_to_pos) {
        break;
      }
    }

    if (wc->col == wc->col_limit) {
      // beat is finished, backtrack
      if constexpr (REPLAY) {
        assert(false);
      }
      unsigned* ds = nullptr;
      if constexpr (MARKING) {
        // equivalence of two ways of determining whether to unmark a throw
        assert( (wc->col > 1) == (wc->excludes_tail != nullptr) );
        ds = ds_bystate[from_state];
        if (wc->col > 1) {
          unmark(u, wc->excludes_tail, ds);
          /*
          unsigned* es = es_tail[from_state];
          unmark(u, es, ds);
          wc->excludes_tail = nullptr;
          */
        }
      }
      assert(from_state == st_state || u[from_state] == 1);
      u[from_state] = 0;
      ++nn;

      if (p == 0) {
        break;
      }
      --p;
      --wc;
      if constexpr (MARKING) {
        // equivalence of two ways of determining whether to unmark the head
        assert( (wc->col != 0) == (wc->excludes_head != nullptr) );
        if (wc->col != 0) {
          unmark(u, wc->excludes_head, ds);
          /*
          unsigned* es = excludestates_head[from_state].data();
          unmark(u, es, ds);
          wc->excludes_head = nullptr;
          */
        }
        mark_any_link_throw = false;
      }
      from_state = wc->from_state;
      ++wc->col;
      continue;
    }

    if constexpr (MARKING) {
      // equivalence of two ways of determining whether we need to do link throw
      // marking (note wc->col == 0 always corresponds to a shift throw)
      assert( (wc->col == 1 || (mark_any_link_throw && wc->col != 0)) ==
              (wc->excludes_tail == nullptr && wc->col != 0) );

      if (wc->col == 1 || (mark_any_link_throw && wc->col != 0)) {
        // First link throw at this position; mark states on the `from_state`
        // shift cycle that are excluded by a link throw. Only need to do this
        // once since the excluded states are independent of link throw value.
        if constexpr (!REPLAY) {
          mark_any_link_throw = false;  // switch to marking only when col == 1
        }

        unsigned* es = es_tail[from_state];
        wc->excludes_tail = es;  // save for backtracking

        if (!mark(u, es, ds_bystate[from_state])) {
          // not valid, bail to previous beat
          if constexpr (REPLAY) {
            assert(false);
          }
          wc->col = wc->col_limit;
          continue;
        }
      }
    }

    const unsigned to_state = outmatrix[from_state][wc->col];

    if (to_state == st_state) {
      // found a pattern
      if constexpr (REPLAY) {
        assert(false);
      }
      if constexpr (REPORT) {
        if (p + 1 >= n_min && !config.countflag) {
          pos = p;
          iterative_handle_finished_pattern();
        }
      }
      ++c[p + 1];
      ++wc->col;
      continue;
    }

    if (u[to_state]) {
      if constexpr (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    if (p + 1 == nmax) {
      if constexpr (REPLAY) {
        assert(false);
      }
      ++wc->col;
      continue;
    }

    if constexpr (MARKING) {
      if (wc->col != 0) {  // link throw
        // mark states excluded by the head
        unsigned* es = excludestates_head[to_state].data();
        unsigned* const ds = ds_bystate[to_state];
        wc->excludes_head = es;

        if (!mark(u, es, ds)) {
          // couldn't advance to next beat
          if constexpr (REPLAY) {
            assert(false);
          }
          unmark(u, wc->excludes_head, ds);
          ++wc->col;
          continue;
        }
      }
    }

    // invariant: check the inbox when we know the current throw is valid, and
    // just before moving to the next beat
    if constexpr (!REPLAY) {
      if (++steps >= steps_limit) {
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
    }

    // advance to next beat
    assert(u[to_state] == 0);
    u[to_state] = 1;

    ++p;
    ++wc;
    if constexpr (REPLAY) {
      assert(wc->from_state == to_state);
    } else {
      wc->col = 0;
      wc->col_limit = outdegree[to_state];
      wc->from_state = to_state;
    }
    wc->excludes_tail = nullptr;
    wc->excludes_head = nullptr;
    from_state = to_state;
    if constexpr (MARKING && !REPLAY) {
      mark_any_link_throw = false;
    }
  }

  if constexpr (REPLAY) {
    assert(p == replay_to_pos);
    assert(nn == nnodes);
  } else {
    assert(p == 0);
  }

  pos = p;
  nnodes = nn;
}

// Helpers for NORMAL_MARKING mode
//
// See comments for the analagous functions in GenLoopsRecursive.cc

inline bool Worker::mark(int* const& u, unsigned*& es, unsigned* const& ds)
{
  bool valid = true;
  for (unsigned statenum; (statenum = *es); ++es) {
    if ((u[statenum] ^= 1) && ++*ds > 1 &&
        --max_possible < static_cast<int>(n_min)) {
      valid = false;
    }
  }
  return valid;
}

inline void Worker::unmark(int* const& u, unsigned*& es, unsigned* const& ds)
{
  if (es) {
    for (unsigned statenum; (statenum = *es); ++es) {
      if (!(u[statenum] ^= 1) && --*ds > 0) {
        ++max_possible;
      }
    }
    es = nullptr;
  }
}

//------------------------------------------------------------------------------
// SUPER and SUPER0 modes
//------------------------------------------------------------------------------

// Non-recursive version of gen_loops_super() and gen_loops_super0().
//
// Template parameter `SUPER0` specifies whether no shift throws are allowed.
// When `SUPER0` == true then certain optimizations can be applied.

template<bool SUPER0, bool REPLAY>
void Worker::iterative_gen_loops_super()
{
  if constexpr (!REPLAY) {
    iterative_init_workspace();
    if (pos == -1) {
      pos = 0;
    } else {
      const auto pos_orig = pos;
      replay_to_pos = pos;
      pos = 0;
      iterative_gen_loops_super<SUPER0, true>();
      (void)pos_orig;
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
  int* const iec = isexitcycle.data();

  WorkCell* wc = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = wc->from_state;
  unsigned from_cycle = cyclenum[from_state];

  while (true) {
    if constexpr (REPLAY) {
      if (p == replay_to_pos) {
        break;
      }
    }

    if (wc->col == wc->col_limit) {
      // beat is finished, go back to previous one
      if constexpr (REPLAY) {
        assert(false);
      }
      if constexpr (!SUPER0) {
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
        if (iec[to_cycle]) {
          ++exitcyclesleft;
        }
      }
      ++wc->col;
      continue;
    }

    const unsigned to_state = outmatrix[from_state][wc->col];

    if constexpr (SUPER0) {
      if (to_state < start_state) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }
    }

    if constexpr (!SUPER0) {
      if (u[to_state]) {
        if (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }
    }

    const unsigned to_cycle = cyclenum[to_state];

    if (SUPER0 || (to_cycle != from_cycle)) {  // link throw
      if (to_state == start_state) {
        if constexpr (REPLAY) {
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
        if constexpr (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if ((SUPER0 || shiftcount == config.shiftlimit) && exitcyclesleft == 0) {
        if constexpr (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if (p + 1 == nmax) {
        if constexpr (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if constexpr (!REPLAY) {
        if (++steps >= steps_per_inbox_check) {
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
      }

      if constexpr (!SUPER0) {
        u[to_state] = 1;
      }
      cu[to_cycle] = true;
      if (iec[to_cycle]) {
        --exitcyclesleft;
      }
    } else {  // shift throw
      if (shiftcount == config.shiftlimit) {
        if constexpr (REPLAY) {
          assert(false);
        }
        ++wc->col;
        continue;
      }

      if (to_state == start_state) {
        if constexpr (REPLAY) {
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
        if constexpr (REPLAY) {
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
    if constexpr (REPLAY) {
      assert(wc->from_state == to_state);
    } else {
      wc->col = 0;
      wc->col_limit = outdegree[to_state];
      wc->from_state = to_state;
    }
    from_state = to_state;
    from_cycle = to_cycle;
  }

  if constexpr (REPLAY) {
    assert(p == replay_to_pos);
    assert(nn == nnodes);
  } else {
    assert(p == 0);
  }

  pos = p;
  nnodes = nn;
}

//------------------------------------------------------------------------------
// Template instantiations
//------------------------------------------------------------------------------

// Explicit template instantiations since template method definitions are not in
// the `.h` file.

// regular versions
template void Worker::iterative_gen_loops_normal<false, false, false>();
template void Worker::iterative_gen_loops_normal<false, true, false>();
template void Worker::iterative_gen_loops_normal<true, true, false>();
template void Worker::iterative_gen_loops_super<true, false>();
template void Worker::iterative_gen_loops_super<false, false>();

// replay versions
template void Worker::iterative_gen_loops_normal<false, false, true>();
template void Worker::iterative_gen_loops_normal<true, false, true>();
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

void Worker::iterative_init_workspace()
{
  WorkAssignment wa = get_work_assignment();
  wa.to_workspace(this, 0);

#ifndef NDEBUG
  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, 0);
  assert(wa == wa2);
#endif
}

// Determine whether we will be able to respond to a SPLIT_WORK request at our
// current point in iterative search.
//
// Also update the worker's values of `root_pos` and `root_throwval_options`.

bool Worker::iterative_can_split()
{
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

void Worker::iterative_update_after_split()
{
  const auto pos_orig = pos;
  WorkAssignment wa = get_work_assignment();
  wa.to_workspace(this, 0);

  // splitting shouldn't change our depth in the search tree
  (void)pos_orig;
  assert(pos == pos_orig);

#ifndef NDEBUG
  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, 0);
  assert(wa == wa2);
#endif
}

inline void Worker::iterative_handle_finished_pattern()
{
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

const Graph& Worker::get_graph() const
{
  return graph;
}

void Worker::set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state)
{
  (void)slot;
  assert(slot == 0);
  assert(index < beat.size());
  WorkCell& wc = beat.at(index);
  wc.col = col;
  wc.col_limit = col_limit;
  wc.from_state = from_state;
}

std::tuple<unsigned, unsigned, unsigned> Worker::get_cell(unsigned slot,
    unsigned index) const
{
  (void)slot;
  assert(slot == 0);
  assert(index < beat.size());
  const WorkCell& wc = beat.at(index);
  return std::make_tuple(wc.col, wc.col_limit, wc.from_state);
}

void Worker::set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos)
{
  (void)slot;
  assert(slot == 0);
  start_state = new_start_state;
  end_state = new_end_state;
  pos = new_pos;
}

std::tuple<unsigned, unsigned, int> Worker::get_info(unsigned slot) const
{
  (void)slot;
  assert(slot == 0);
  return std::make_tuple(start_state, end_state, pos);
}
