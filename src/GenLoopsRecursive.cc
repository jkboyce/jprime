//
// GenLoopsRecursive.cc
//
// Core graph search routines, implemented as recursive functions. These
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


// Try all allowed throw values at the current pattern position `pos`,
// recursively continuing until a pattern is found or `l_max` is exceeded.
//
// This version is for NORMAL mode.

void Worker::gen_loops_normal() {
  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    if (used[to] != 0)
      continue;
    pattern[pos] = graph.outthrowval[from][col];

    if (to == start_state) {
      handle_finished_pattern();
      continue;
    }

    if (pos + 1 == l_max)
      continue;

    // see if it's time to check the inbox
    if (++steps_taken >= steps_per_inbox_check && pos > root_pos
          && col < limit - 1) {
      // the restrictions on when we enter here are in case we get a message
      // to hand off work to another worker; see split_work_assignment()

      // terminate the pattern at the current position in case we get a
      // STOP_WORKER message and need to unwind back to run()
      pattern[pos + 1] = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    ++used[to];
    ++pos;
    const int old_from = from;
    from = to;
    gen_loops_normal();
    from = old_from;
    --pos;
    --used[to];

    // only a single allowed throw value for `pos` < `root_pos`
    if (pos < root_pos)
      break;
  }

  ++nnodes;
}

// As above, but for long searches close to `l_bound`.
//
// This version marks off states that are made unreachable by link throws
// between shift cycles. We cut the search short when we determine we can't
// generate a pattern of length `l_min` or longer from our current position.

void Worker::gen_loops_normal_marking() {
  bool did_mark_for_throw = false;
  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    if (used[to] != 0)
      continue;

    const int throwval = graph.outthrowval[from][col];
    if (to == start_state) {
      pattern[pos] = throwval;
      handle_finished_pattern();
      continue;
    }

    if (pos + 1 == l_max)
      continue;

    if (throwval != 0 && throwval != graph.h) {
      // link throws make certain nearby states unreachable
      if (!did_mark_for_throw) {
        if (!mark_unreachable_states_throw()) {
          // bail since all additional `col` values will also be link throws
          unmark_unreachable_states_throw();
          ++nnodes;
          return;
        }
        did_mark_for_throw = true;
      }

      if (mark_unreachable_states_catch(to)) {
        // we need to go deeper
        pattern[pos] = throwval;
        ++used[to];
        ++pos;
        const int old_from = from;
        from = to;
        gen_loops_normal_marking();
        from = old_from;
        --pos;
        --used[to];
      }
      unmark_unreachable_states_catch(to);
    } else {
      pattern[pos] = throwval;

      if (++steps_taken >= steps_per_inbox_check && pos > root_pos
            && col < limit - 1) {
        pattern[pos + 1] = -1;
        process_inbox_running();
        steps_taken = 0;
      }

      ++used[to];
      ++pos;
      const int old_from = from;
      from = to;
      gen_loops_normal_marking();
      from = old_from;
      --pos;
      --used[to];
    }

    // only a single allowed throw value for `pos` < `root_pos`
    if (pos < root_pos)
      break;
  }

  if (did_mark_for_throw)
    unmark_unreachable_states_throw();
  ++nnodes;
}

// As above, but for SUPER mode.
//
// Since a superprime pattern can only visit a single state in each shift cycle,
// this is the fastest version because so many states are excluded by each
// throw to a new shift cycle.

void Worker::gen_loops_super() {
  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];
  const int* ov = graph.outthrowval[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos && !mark_off_rootpos_option(ov[col], to))
      continue;
    if (used[to] != 0)
      continue;

    const int throwval = ov[col];
    const bool linkthrow = (throwval != 0 && throwval != graph.h);

    if (linkthrow) {
      // going to a shift cycle that's already been visited?
      const int to_cycle = graph.cyclenum[to];
      if (cycleused[to_cycle])
        continue;

      pattern[pos] = throwval;
      if (to == start_state) {
        handle_finished_pattern();
      } else if (pos + 1 == l_max) {
        continue;
      } else {
        if (++steps_taken >= steps_per_inbox_check && pos > root_pos
              && col < limit - 1) {
          pattern[pos + 1] = -1;
          process_inbox_running();
          steps_taken = 0;
        }

        cycleused[to_cycle] = true;
        ++used[to];
        ++pos;
        const int old_from = from;
        from = to;
        gen_loops_super();
        from = old_from;
        --pos;
        --used[to];
        cycleused[to_cycle] = false;
      }
    } else {
      // check for shift throw limits
      if (shiftcount == config.shiftlimit)
        continue;

      pattern[pos] = throwval;
      if (to == start_state) {
        handle_finished_pattern();
      } else if (pos + 1 == l_max) {
        continue;
      } else {
        ++shiftcount;
        ++used[to];
        ++pos;
        const int old_from = from;
        from = to;
        gen_loops_super();
        from = old_from;
        --pos;
        --used[to];
        --shiftcount;
      }
    }

    if (pos < root_pos)
      break;
  }

  ++nnodes;
}

// A specialization of gen_loops_super() for the case `shiftthrows` == 0.
//
// This version tracks the specific "exit cycles" that can get back to the
// start state with a single throw. If those exit cycles are all used and the
// pattern isn't done, we terminate the search early.

void Worker::gen_loops_super0() {
  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    const int to_cycle = graph.cyclenum[to];
    if (cycleused[to_cycle])
      continue;

    pattern[pos] = graph.outthrowval[from][col];
    if (to == start_state) {
      handle_finished_pattern();
    } else if (pos + 1 == l_max) {
      continue;
    } else if (exitcyclesleft == 0) {
      continue;
    } else {
      if (++steps_taken >= steps_per_inbox_check && pos > root_pos
            && col < limit - 1) {
        pattern[pos + 1] = -1;
        process_inbox_running();
        steps_taken = 0;
      }

      const int old_exitcyclesleft = exitcyclesleft;
      if (graph.isexitcycle[to_cycle])
        --exitcyclesleft;
      cycleused[to_cycle] = true;
      ++pos;
      const int old_from = from;
      from = to;
      gen_loops_super0();
      from = old_from;
      --pos;
      cycleused[to_cycle] = false;
      exitcyclesleft = old_exitcyclesleft;
    }

    if (pos < root_pos)
      break;
  }

  ++nnodes;
}

// Return the column number in the `outmatrix[from]` row vector that
// corresponds to the throw value at position `pos` in the pattern. This allows
// us to resume where we left off when loading from a work assignment.

int Worker::load_one_throw() {
  if (pattern[pos] == -1) {
    loading_work = false;
    return 0;
  }
  if (pos + 1 == l_max) {
    loading_work = false;
  }

  for (int col = 0; col < graph.outdegree[from]; ++col) {
    if (graph.outthrowval[from][col] == pattern[pos])
      return col;
  }

  // diagnostic information if there's a problem
  std::ostringstream buffer;
  for (int i = 0; i <= pos; ++i)
    print_throw(buffer, pattern[i]);
  std::cerr << "worker: " << worker_id << '\n'
            << "pos: " << pos << '\n'
            << "root_pos: " << root_pos << '\n'
            << "from: " << from << '\n'
            << "state[from]: " << graph.state[from] << '\n'
            << "start_state: " << start_state << '\n'
            << "pattern: " << buffer.str() << '\n'
            << "outthrowval[from][]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.outthrowval[from][i] << ", ";
  std::cerr << "\noutmatrix[from][]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.outmatrix[from][i] << ", ";
  std::cerr << "\nstate[outmatrix[from][]]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.state[graph.outmatrix[from][i]] << ", ";
  std::cerr << '\n';
  std::exit(EXIT_FAILURE);
  return 0;
}

// Determine the set of throw options available at position `root_pos` in
// the pattern. This list of options is maintained in case we get a request
// to split work.

void Worker::build_rootpos_throw_options(int rootpos_from_state,
      int start_column) {
  root_throwval_options.clear();
  for (int col = start_column; col < graph.outdegree[rootpos_from_state]; ++col)
    root_throwval_options.push_back(graph.outthrowval[rootpos_from_state][col]);

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " options at root_pos " << root_pos
           << ": [";
    for (int v : root_throwval_options)
      print_throw(buffer, v);
    buffer << "]";
    message_coordinator_status(buffer.str());
  }
}

// Mark off `throwval` from our set of allowed throw options at position
// `root_pos` in the pattern.
//
// If this exhausts the set of allowed options, then advance `root_pos` by one
// and generate a new set of options. As an invariant we never allow
// `root_throwval_options` to be empty, in case we get a request to split work.
//
// Returns true if `throwval` is an allowed choice at position `root_pos`,
// false otherwise.

bool Worker::mark_off_rootpos_option(int throwval, int to_state) {
  bool found = false;
  int remaining = 0;
  std::list<int>::iterator iter = root_throwval_options.begin();
  std::list<int>::iterator end = root_throwval_options.end();

  while (iter != end) {
    // housekeeping: has this root_pos option been pruned from the graph?
    bool pruned = true;
    for (int i = 0; i < graph.outdegree[from]; ++i) {
      if (graph.outthrowval[from][i] == *iter) {
        pruned = false;
        break;
      }
    }

    if (pruned && config.verboseflag) {
      std::ostringstream buffer;
      buffer << "worker " << worker_id << " option ";
      print_throw(buffer, throwval);
      buffer << " at root_pos " << root_pos << " was pruned; removing";
      message_coordinator_status(buffer.str());
    }

    if (!pruned && *iter == throwval) {
      found = true;

      if (config.verboseflag) {
        std::ostringstream buffer;
        buffer << "worker " << worker_id << " starting option ";
        print_throw(buffer, throwval);
        buffer << " at root_pos " << root_pos;
        message_coordinator_status(buffer.str());
      }
    }

    if (pruned || *iter == throwval) {
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
      ++remaining;
    }
  }

  if (remaining == 0) {
    ++root_pos;
    notify_coordinator_rootpos();
    build_rootpos_throw_options(to_state, 0);
  }

  return (found || loading_work);
}

// Mark all of the states as used that are excluded by a throw from state
// `from` to state `to_state`.
//
// Returns false if the number of newly-excluded states implies that we can't
// finish a pattern of at least length `l_current` from our current position.
// Returns true otherwise.

inline bool Worker::mark_unreachable_states_throw() {
  bool valid = true;
  int* const ds = deadstates_bystate[from];
  int* es = graph.excludestates_throw[from];
  int statenum = 0;

  while ((statenum = *es++)) {
    if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min)
      valid = false;
  }
  return valid;
}

inline bool Worker::mark_unreachable_states_catch(int to_state) {
  bool valid = true;
  int* const ds = deadstates_bystate[to_state];
  int* es = graph.excludestates_catch[to_state];
  int statenum = 0;

  while ((statenum = *es++)) {
    if (++used[statenum] == 1 && ++*ds > 1 && --max_possible < l_min)
      valid = false;
  }

  return valid;
}

// Reverse the marking operations above, so we can backtrack.

inline void Worker::unmark_unreachable_states_throw() {
  int* const ds = deadstates_bystate[from];
  int* es = graph.excludestates_throw[from];
  int statenum = 0;

  while ((statenum = *es++)) {
    if (--used[statenum] == 0 && --*ds > 0)
      ++max_possible;
  }
}

inline void Worker::unmark_unreachable_states_catch(int to_state) {
  int* const ds = deadstates_bystate[to_state];
  int* es = graph.excludestates_catch[to_state];
  int statenum = 0;

  while ((statenum = *es++)) {
    if (--used[statenum] == 0 && --*ds > 0)
      ++max_possible;
  }
}

inline void Worker::handle_finished_pattern() {
  ++count[pos + 1];

  if ((pos + 1) >= l_min && !config.countflag) {
    report_pattern();
  }
}
