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
void Worker::iterative_gen_loops_normal(unsigned replay_to_pos) {
  if (!REPLAY) {
    iterative_init_workspace();

    // replay back through the algorithm up to and including position `pos`.
    // this sets up variables like used[], etc.
    const unsigned pos_orig = pos;
    pos = 0;
    iterative_gen_loops_normal<false, true>(pos_orig);
    assert(pos == pos_orig);

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

  SearchState* ss = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = ss->from_state;

  // main search loop
  while (true) {
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (ss->col == ss->col_limit) {
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
      --ss;
      from_state = ss->from_state;
      ++ss->col;
      continue;
    }

    const unsigned to_state = outmatrix[from_state][ss->col];

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
      ++ss->col;
      continue;
    }

    if (u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
      continue;
    }

    if (p + 1 == nmax) {
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
      continue;
    }

    if (!REPLAY && ++steps >= steps_limit) {
      // check the inbox for incoming messages, after doing some prep work
      // to ensure we can respond to any type of message we might receive
      //
      // this code is rarely evaluated so it is not performance-critical
      steps = 0;

      pos = p;
      if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
        for (size_t i = 0; i <= pos; ++i) {
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
    ++ss;
    if (REPLAY) {
      assert(ss->from_state == to_state);
    } else {
      ss->col = 0;
      ss->col_limit = outdegree[to_state];
      ss->from_state = to_state;
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
void Worker::iterative_gen_loops_normal_marking(unsigned replay_to_pos) {
  if (!REPLAY) {
    iterative_init_workspace();
    const unsigned pos_orig = pos;
    pos = 0;
    iterative_gen_loops_normal_marking<true>(pos_orig);
    assert(pos == pos_orig);
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

  SearchState* ss = &beat.at(pos);

  while (true) {
    // begin with any necessary cleanup from previous marking operations
    if (ss->to_state != 0) {
      u[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking1:
    if (ss->excludes_catch) {
      unsigned* const ds = ss->deadstates_catch;
      unsigned* es = ss->excludes_catch;
      for (unsigned statenum; (statenum = *es); ++es) {
        if (--u[statenum] == 0 && --*ds > 0) {
          ++max_possible;
        }
      }
      ss->excludes_catch = nullptr;
    }

    skip_unmarking2:
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (ss->col == ss->col_limit) {
      if (REPLAY) {
        assert(false);
      }
      if (ss->excludes_throw) {
        unsigned* const ds = ss->deadstates_throw;
        unsigned* es = ss->excludes_throw;
        for (unsigned statenum; (statenum = *es); ++es) {
          if (--u[statenum] == 0 && --*ds > 0) {
            ++max_possible;
          }
        }
      }

      ++nn;
      if (p == 0) {
        break;
      }
      --p;
      --ss;
      ++ss->col;
      continue;
    }

    const unsigned to_state = outmatrix[ss->from_state][ss->col];
    // const unsigned to_state = ss->outmatrix[ss->col];

    if (to_state == st_state) {
      if (REPLAY) {
        assert(false);
      }
      if (p + 1 >= n_min && !config.countflag) {
        pos = p;
        iterative_handle_finished_pattern();
      }
      ++count[p + 1];
      ++ss->col;
      goto skip_unmarking2;
    }

    if (u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
      goto skip_unmarking2;
    }

    if (p + 1 == nmax) {
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
      goto skip_unmarking2;
    }

    const unsigned throwval = outthrowval[ss->from_state][ss->col];

    if (throwval != 0 && throwval != graph.h) {  // link throw
      if (ss->excludes_throw == nullptr) {
        // mark states excluded by link throw; only need to do this once since
        // the link throws all come at the end of each row in `outmatrix`
        bool valid1 = true;
        unsigned* const ds = ds_bystate[ss->from_state];
        unsigned* es = es_throw[ss->from_state];
        ss->excludes_throw = es;  // save to clean up later
        ss->deadstates_throw = ds;

        for (unsigned statenum; (statenum = *es); ++es) {
          if (++u[statenum] == 1 && ++*ds > 1 &&
              --max_possible < static_cast<int>(n_min)) {
            valid1 = false;
          }
        }

        if (!valid1) {
          if (REPLAY) {
            assert(false);
          }
          // undo marking operation and bail to previous beat
          es = ss->excludes_throw;
          for (unsigned statenum; (statenum = *es); ++es) {
            if (--u[statenum] == 0 && --*ds > 0) {
              ++max_possible;
            }
          }

          ++nn;
          if (p == 0) {
            break;
          }
          --p;
          --ss;
          ++ss->col;
          continue;
        }
      }

      // account for states excluded by link catch
      bool valid2 = true;
      unsigned* const ds = ds_bystate[to_state];
      unsigned* es = graph.excludestates_catch[to_state].data();
      ss->excludes_catch = es;
      ss->deadstates_catch = ds;

      for (unsigned statenum; (statenum = *es); ++es) {
        if (++u[statenum] == 1 && ++*ds > 1 &&
            --max_possible < static_cast<int>(n_min)) {
          valid2 = false;
        }
      }

      if (valid2) {
        // advance to next beat
        u[to_state] = 1;
        ss->to_state = to_state;
        ++p;
        ++ss;
        if (REPLAY) {
          assert(ss->from_state == to_state);
        } else {
          ss->col = 0;
          ss->col_limit = outdegree[to_state];
          ss->from_state = to_state;
        }
        ss->to_state = 0;
        // ss->outmatrix = outmatrix[to_state];
        ss->excludes_throw = nullptr;
        ss->excludes_catch = nullptr;
        goto skip_unmarking2;
      }

      // couldn't advance to next beat, so go to next column in this one
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
      goto skip_unmarking1;
    } else {  // shift throw
      if (!REPLAY && ++steps >= steps_limit) {
        steps = 0;

        pos = p;
        if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
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
      ss->to_state = to_state;
      ++p;
      ++ss;
      if (REPLAY) {
        assert(ss->from_state == to_state);
      } else {
        ss->col = 0;
        ss->col_limit = outdegree[to_state];
        ss->from_state = to_state;
      }
      ss->to_state = 0;
      // ss->outmatrix = outmatrix[to_state];
      ss->excludes_throw = nullptr;
      ss->excludes_catch = nullptr;
      goto skip_unmarking2;
    }
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

// Non-recursive version of gen_loops_super() and gen_loops_super0().
//
// Template parameter `SUPER0` specifies whether no shift throws are allowed.
// When `SUPER0` == true then certain optimizations can be applied.

template<bool SUPER0, bool REPLAY>
void Worker::iterative_gen_loops_super(unsigned replay_to_pos) {
  if (!REPLAY) {
    iterative_init_workspace();
    const unsigned pos_orig = pos;
    pos = 0;
    iterative_gen_loops_super<SUPER0, true>(pos_orig);
    assert(pos == pos_orig);
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

  SearchState* ss = &beat.at(pos);

  // register-based state variables during search
  unsigned from_state = ss->from_state;
  unsigned from_cycle = cyclenum[from_state];

  while (true) {
    if (REPLAY && p == replay_to_pos) {
      break;
    }

    if (ss->col == ss->col_limit) {
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
      --ss;
      const unsigned to_cycle = from_cycle;
      from_state = ss->from_state;
      from_cycle = cyclenum[from_state];
      if (from_cycle == to_cycle) {  // unwinding a shift throw
        --shiftcount;
      } else {  // link throw
        cu[to_cycle] = false;
        if (isexitcycle[to_cycle]) {
          ++exitcyclesleft;
        }
      }
      ++ss->col;
      continue;
    }

    const unsigned to_state = outmatrix[from_state][ss->col];

    if (!SUPER0 && u[to_state]) {
      if (REPLAY) {
        assert(false);
      }
      ++ss->col;
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
        ++ss->col;
        continue;
      }

      if (cu[to_cycle]) {
        if (REPLAY) {
          assert(false);
        }
        ++ss->col;
        continue;
      }

      if ((SUPER0 || shiftcount == config.shiftlimit) && exitcyclesleft == 0) {
        if (REPLAY) {
          assert(false);
        }
        ++ss->col;
        continue;
      }

      if (p + 1 == nmax) {
        if (REPLAY) {
          assert(false);
        }
        ++ss->col;
        continue;
      }

      if (!REPLAY && ++steps >= steps_per_inbox_check) {
        steps = 0;

        pos = p;
        if (iterative_calc_rootpos_and_options() && iterative_can_split()) {
          for (size_t i = 0; i <= pos; ++i) {
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
        ++ss->col;
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
        ++ss->col;
        continue;
      }

      if (p + 1 == nmax) {
        if (REPLAY) {
          assert(false);
        }
        ++ss->col;
        continue;
      }

      u[to_state] = 1;
      ++shiftcount;
    }

    // advance to next beat
    ++p;
    ++ss;
    if (REPLAY) {
      assert(ss->from_state == to_state);
    } else {
      ss->col = 0;
      ss->col_limit = outdegree[to_state];
      ss->from_state = to_state;
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

// Explicit template instantiations since template method definition is not in
// the `.h` file.

// regular versions
template void Worker::iterative_gen_loops_normal<true, false>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_normal<false, false>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_normal_marking<false>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_super<true, false>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_super<false, false>(
    unsigned replay_to_pos);

// replay versions
template void Worker::iterative_gen_loops_normal<false, true>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_normal_marking<true>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_super<true, true>(
    unsigned replay_to_pos);
template void Worker::iterative_gen_loops_super<false, true>(
    unsigned replay_to_pos);

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Set up the SearchState workcells with initial values for `ss.col`,
// `ss.col_limit`, and `ss.from_state` based on the loaded work assignment.
//
// Leaves `pos` pointing to the last beat with loaded data.
//
// Returns true on success, false on failure.

bool Worker::iterative_init_workspace() {
  if (!loading_work) {
    pos = 0;
    SearchState& ss = beat.at(0);
    ss.col = 0;
    ss.col_limit = graph.outdegree.at(start_state);
    ss.from_state = start_state;
    return true;
  }

  // When loading from a work assignment, load_work_assignment() will have
  // set up `pattern`, `root_pos`, and `root_throwval_options`

  loading_work = false;
  unsigned from_state = start_state;

  for (size_t i = 0; pattern.at(i) != -1; ++i) {
    pos = static_cast<unsigned>(i);
    SearchState& ss = beat.at(i);
    ss.from_state = from_state;
    ss.col_limit = graph.outdegree.at(ss.from_state);

    const auto tv = static_cast<unsigned>(pattern.at(i));

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
        if (j != 0) {
          std::cerr << ',';
        }
        std::cerr << pattern.at(j);
      }
      std::cerr << '\n';
    }
    assert(ss.col < ss.col_limit);

    if (pos < root_pos) {
      ss.col_limit = ss.col + 1;
    }

    from_state = graph.outmatrix.at(ss.from_state).at(ss.col);;
  }

  if (pattern.at(0) == -1 || pos < root_pos) {
    // we're loading a work assignment that is either:
    // (a) brand new (no pattern prefix), or
    // (b) stolen from another worker (pos = root_pos - 1)
    //
    // in either case we didn't initialize the workcell at `root_pos` in the
    // loop above, so do it here

    assert(pattern.at(0) == -1 || pos + 1 == root_pos);

    SearchState& rss = beat.at(root_pos);
    rss.from_state = from_state;

    // Set `col` at `root_pos`. It's equal to the lowest index of the throws
    // in `root_throwval_options`. This works because the way we steal work
    // ensures that the unexplored throw options in `root_throwval_options`
    // have contiguous indices up to and including `col_limit` - 1.
    rss.col = -1;
    for (size_t i = 0; i < graph.outdegree.at(rss.from_state); ++i) {
      const unsigned throwval = graph.outthrowval.at(rss.from_state).at(i);
      if (std::find(root_throwval_options.cbegin(),
          root_throwval_options.cend(), throwval)
          != root_throwval_options.cend()) {
        rss.col = std::min(rss.col, static_cast<unsigned>(i));
      }
    }

    pos = root_pos;
  }

  // set `col_limit` at `root_pos`
  SearchState& rss = beat.at(root_pos);
  rss.col_limit = 0;
  for (size_t i = 0; i < graph.outdegree.at(rss.from_state); ++i) {
    const unsigned throwval = graph.outthrowval.at(rss.from_state).at(i);
    if (std::find(root_throwval_options.cbegin(), root_throwval_options.cend(),
        throwval) != root_throwval_options.cend()) {
      rss.col_limit = std::max(rss.col_limit, static_cast<unsigned>(i + 1));
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
  unsigned new_root_pos = 0;
  for (; new_root_pos < pos; ++new_root_pos) {
    const SearchState& ss = beat.at(new_root_pos);
    if (ss.col < ss.col_limit - 1)
      break;
  }
  if (new_root_pos == pos) {
    return false;
  }

  assert(new_root_pos >= root_pos);
  if (new_root_pos != root_pos) {
    root_pos = new_root_pos;
    notify_coordinator_update();
  }

  root_throwval_options.clear();
  const SearchState& ss = beat.at(new_root_pos);
  for (size_t col = ss.col + 1; col < ss.col_limit; ++col) {
    root_throwval_options.push_back(
        graph.outthrowval.at(ss.from_state).at(col));
  }
  return true;
}

// Determine whether we will be able to respond to a SPLIT_WORK request at our
// current point in iterative search.
//
// Needs an updated value of `root_pos`.

bool Worker::iterative_can_split() {
  for (size_t i = root_pos + 1; i <= pos; ++i) {
    const SearchState& ss = beat.at(i);
    if (ss.col < ss.col_limit - 1) {
      return true;
    }
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

  // ensure no further iteration on beats prior to `root_pos`
  for (size_t i = 0; i < root_pos; ++i) {
    SearchState& ss = beat.at(i);
    ss.col_limit = ss.col + 1;
  }

  // ensure we don't iterate over the throw options at `root_pos` that we just
  // gave away
  SearchState& ss = beat.at(root_pos);
  unsigned new_col_limit = ss.col + 1;
  for (size_t i = ss.col + 1; i < graph.outdegree.at(ss.from_state); ++i) {
    const unsigned throwval = graph.outthrowval.at(ss.from_state).at(i);
    if (std::find(root_throwval_options.cbegin(), root_throwval_options.cend(),
        throwval) != root_throwval_options.cend()) {
      new_col_limit = static_cast<unsigned>(i + 1);
    }
  }
  ss.col_limit = new_col_limit;
}

inline void Worker::iterative_handle_finished_pattern() {
  for (size_t i = 0; i <= pos; ++i) {
    pattern.at(i) = graph.outthrowval.at(beat.at(i).from_state)
                                      .at(beat.at(i).col);
  }
  pattern.at(pos + 1) = -1;
  report_pattern();
}
