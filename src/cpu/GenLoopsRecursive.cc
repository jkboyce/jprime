//
// GenLoopsRecursive.cc
//
// Core graph search routines, implemented as recursive functions. These
// routines are by far the most performance-critical portions of jprime.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"

#include <iostream>
#include <sstream>
#include <format>
#include <cassert>


//------------------------------------------------------------------------------
// NORMAL mode
//------------------------------------------------------------------------------

// Try all allowed throw values at the current pattern position `pos`,
// recursively continuing until a pattern is found or `n_max` is exceeded.
//
// This version is for NORMAL mode.
//
// Note this has a recursion depth of `n_max`.

void Worker::gen_loops_normal()
{
  unsigned col = (loading_work ? load_one_throw() : 0);
  const unsigned limit = graph.outdegree[from];
  const unsigned* om = graph.outmatrix[from].data();
  const unsigned old_from = from;

  for (; col < limit; ++col) {
    const unsigned to = om[col];
    if (pos == static_cast<int>(root_pos) &&
        !mark_off_rootpos_option(graph.outthrowval.at(from).at(col), to)) {
      continue;
    }
    if (used[to] != 0) {
      continue;
    }
    pattern[pos] = static_cast<int>(graph.outthrowval[from][col]);

    if (to == start_state) {
      handle_finished_pattern();
      continue;
    }

    if (pos + 1 == static_cast<int>(n_max)) {
      continue;
    }

    // see if it's time to check the inbox
    if (++steps_taken >= steps_per_inbox_check &&
          pos > static_cast<int>(root_pos) && col < limit - 1) {
      // the restrictions on when we enter here are in case we get a message
      // to hand off work to another worker; see split_work_assignment()

      // terminate the pattern at the current position in case we get a
      // STOP_WORKER message and need to unwind back to run()
      pattern.at(pos + 1) = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    // we need to go deeper
    ++used[to];
    ++pos;
    from = to;
    gen_loops_normal();
    from = old_from;
    --pos;
    --used[to];

    // only a single allowed throw value for `pos` < `root_pos`
    if (pos < static_cast<int>(root_pos)) {
      break;
    }
  }

  ++nnodes;
}

//------------------------------------------------------------------------------
// NORMAL_MARKING mode
//------------------------------------------------------------------------------

// As above, but for pattern periods `n` close to `n_bound`.
//
// This version marks off states that are made unreachable by link throws
// between shift cycles. We cut the search short when we determine we can't
// generate a pattern of period `n_min` or larger from our current position.
//
// Note this has a recursion depth of `n_max`.

void Worker::gen_loops_normal_marking()
{
  bool did_mark_for_tail = false;
  unsigned col = (loading_work ? load_one_throw() : 0);
  const unsigned limit = graph.outdegree[from];
  const unsigned* om = graph.outmatrix[from].data();
  const unsigned old_from = from;

  for (; col < limit; ++col) {
    const unsigned to = om[col];
    if (pos == static_cast<int>(root_pos) &&
        !mark_off_rootpos_option(graph.outthrowval.at(from).at(col), to)) {
      continue;
    }
    if (used[to] != 0) {
      continue;
    }

    const unsigned throwval = graph.outthrowval[from][col];

    if (throwval != 0 && throwval != graph.h && !did_mark_for_tail) {
      // all larger `col` values will also be link throws, so we only need to
      // mark the "from" shift cycle once (marking is independent of link
      // throw value)
      if (!mark_unreachable_states_tail()) {
        unmark_unreachable_states_tail();
        ++nnodes;
        return;
      }
      did_mark_for_tail = true;
    }

    if (to == start_state) {
      pattern[pos] = static_cast<int>(throwval);
      handle_finished_pattern();
      continue;
    }

    if (pos + 1 == static_cast<int>(n_max)) {
      continue;
    }

    if (throwval != 0 && throwval != graph.h) {
      if (mark_unreachable_states_head(to)) {
        pattern[pos] = static_cast<int>(throwval);
        ++used[to];
        ++pos;
        from = to;
        gen_loops_normal_marking();
        from = old_from;
        --pos;
        --used[to];
      }
      unmark_unreachable_states_head(to);
    } else {
      pattern[pos] = static_cast<int>(throwval);

      if (++steps_taken >= steps_per_inbox_check &&
            pos > static_cast<int>(root_pos) && col < limit - 1) {
        pattern.at(pos + 1) = -1;
        process_inbox_running();
        steps_taken = 0;
      }

      ++used[to];
      ++pos;
      from = to;
      gen_loops_normal_marking();
      from = old_from;
      --pos;
      --used[to];
    }

    if (pos < static_cast<int>(root_pos)) {
      break;
    }
  }

  if (did_mark_for_tail) {
    unmark_unreachable_states_tail();
  }
  ++nnodes;
}

//------------------------------------------------------------------------------
// SUPER mode
//------------------------------------------------------------------------------

// As above, but for SUPER mode.
//
// Since a superprime pattern can never revisit a shift cycle, this is the
// fastest version because so many states are excluded by each throw to a new
// shift cycle.
//
// We also track the specific "exit cycles" that can get back to the start state
// with a single throw. If those exit cycles are all used and the pattern isn't
// done, we terminate the search early.
//
// Note this has a recursion depth of `n_max`.

void Worker::gen_loops_super()
{
  unsigned col = (loading_work ? load_one_throw() : 0);
  const unsigned limit = graph.outdegree[from];
  const unsigned* const om = graph.outmatrix[from].data();
  const unsigned* const ov = graph.outthrowval[from].data();
  const unsigned old_from = from;

  for (; col < limit; ++col) {
    const unsigned to = om[col];
    if (pos == static_cast<int>(root_pos) &&
        !mark_off_rootpos_option(ov[col], to)) {
      continue;
    }
    if (used[to] != 0) {
      continue;
    }

    const unsigned throwval = ov[col];
    const bool linkthrow = (throwval != 0 && throwval != graph.h);

    if (linkthrow) {
      pattern[pos] = static_cast<int>(throwval);
      if (to == start_state) {
        handle_finished_pattern();
        continue;
      }

      // going to a shift cycle that's already been visited?
      const unsigned to_cycle = graph.cyclenum[to];
      if (cycleused[to_cycle] != 0) {
        continue;
      }
      if (shiftcount == config.shiftlimit && exitcyclesleft == 0) {
        continue;
      }
      if (pos + 1 == static_cast<int>(n_max)) {
        continue;
      }

      if (++steps_taken >= steps_per_inbox_check &&
            pos > static_cast<int>(root_pos) && col < limit - 1) {
        pattern.at(pos + 1) = -1;
        process_inbox_running();
        steps_taken = 0;
      }

      const auto old_exitcyclesleft = exitcyclesleft;
      if (isexitcycle[to_cycle] != 0) {
        --exitcyclesleft;
      }
      cycleused[to_cycle] = 1;
      ++used[to];
      ++pos;
      from = to;
      gen_loops_super();
      from = old_from;
      --pos;
      --used[to];
      cycleused[to_cycle] = 0;
      exitcyclesleft = old_exitcyclesleft;
    } else {
      // check for shift throw limits
      if (shiftcount == config.shiftlimit) {
        continue;
      }

      pattern[pos] = static_cast<int>(throwval);
      if (to == start_state) {
        if (static_cast<int>(shiftcount) < pos) {
          // don't allow all shift throws
          handle_finished_pattern();
        }
      } else if (pos + 1 == static_cast<int>(n_max)) {
        continue;
      } else {
        ++shiftcount;
        ++used[to];
        ++pos;
        from = to;
        gen_loops_super();
        from = old_from;
        --pos;
        --used[to];
        --shiftcount;
      }
    }

    if (pos < static_cast<int>(root_pos)) {
      break;
    }
  }

  ++nnodes;
}

//------------------------------------------------------------------------------
// SUPER0 mode
//------------------------------------------------------------------------------

// A specialization of gen_loops_super() for the case `shiftlimit` == 0.
//
// Note this has a recursion depth of `n_max`.

void Worker::gen_loops_super0()
{
  unsigned col = (loading_work ? load_one_throw() : 0);
  const unsigned limit = graph.outdegree[from];
  const unsigned* const om = graph.outmatrix[from].data();
  const unsigned old_from = from;

  for (; col < limit; ++col) {
    const unsigned to = om[col];
    if (pos == static_cast<int>(root_pos) &&
        !mark_off_rootpos_option(graph.outthrowval.at(from).at(col), to)) {
      continue;
    }
    if (to < start_state) {
      continue;
    }

    pattern[pos] = static_cast<int>(graph.outthrowval[from][col]);
    if (to == start_state) {
      handle_finished_pattern();
      continue;
    }

    const unsigned to_cycle = graph.cyclenum[to];
    if (cycleused[to_cycle] != 0) {
      continue;
    }
    if (exitcyclesleft == 0) {
      continue;
    }
    if (pos + 1 == static_cast<int>(n_max)) {
      continue;
    }

    if (++steps_taken >= steps_per_inbox_check &&
          pos > static_cast<int>(root_pos) && col < limit - 1) {
      pattern.at(pos + 1) = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    const auto old_exitcyclesleft = exitcyclesleft;
    if (isexitcycle[to_cycle] != 0) {
      --exitcyclesleft;
    }
    cycleused[to_cycle] = 1;
    ++pos;
    from = to;
    gen_loops_super0();
    from = old_from;
    --pos;
    cycleused[to_cycle] = 0;
    exitcyclesleft = old_exitcyclesleft;

    if (pos < static_cast<int>(root_pos)) {
      break;
    }
  }

  ++nnodes;
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Return the column number in the `outmatrix[from]` row vector that corresponds
// to the throw value at position `pos` in the pattern. This allows us to resume
// where we left off when loading from a work assignment.

unsigned Worker::load_one_throw()
{
  if (pattern.at(pos) == -1) {
    loading_work = false;
    return 0;
  }
  if (pos + 1 == static_cast<int>(n_max)) {
    loading_work = false;
  }

  for (unsigned col = 0; col < graph.outdegree.at(from); ++col) {
    if (graph.outthrowval.at(from).at(col) ==
          static_cast<unsigned>(pattern.at(pos))) {
      return col;
    }
  }

  // diagnostic information if there's a problem
  std::ostringstream buffer;
  for (int i = 0; i <= pos; ++i) {
    if (i != 0) {
      buffer << ',';
    }
    buffer << pattern.at(i);
  }
  std::cerr << "worker: " << worker_id << '\n'
            << "pos: " << pos << '\n'
            << "root_pos: " << root_pos << '\n'
            << "from: " << from << '\n'
            << "state[from]: " << graph.state.at(from) << '\n'
            << "start_state: " << start_state << '\n'
            << "pattern: " << buffer.str() << '\n'
            << "outthrowval[from][]: ";
  for (size_t i = 0; i < graph.outdegree.at(from); ++i) {
    std::cerr << graph.outthrowval.at(from).at(i) << ", ";
  }
  std::cerr << "\noutmatrix[from][]: ";
  for (size_t i = 0; i < graph.outdegree.at(from); ++i) {
    std::cerr << graph.outmatrix.at(from).at(i) << ", ";
  }
  std::cerr << "\nstate[outmatrix[from][]]: ";
  for (size_t i = 0; i < graph.outdegree.at(from); ++i) {
    std::cerr << graph.state.at(graph.outmatrix.at(from).at(i)) << ", ";
  }
  std::cerr << '\n';
  std::exit(EXIT_FAILURE);
  return 0;
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

bool Worker::mark_off_rootpos_option(unsigned throwval, unsigned to_state)
{
  bool found = false;
  unsigned remaining = 0;

  // if there are no options when we get here, then we must be loading an
  // UNSPLITTABLE work assignment. The empty root_throwval_options signifies
  // "all possible values" at root_pos.
  if (root_throwval_options.empty()) {
    build_rootpos_throw_options(from, 0);
  }

  auto iter = root_throwval_options.begin();
  const auto end = root_throwval_options.end();

  while (iter != end) {
    // housekeeping: has this root_pos option been pruned from the graph?
    bool pruned = true;
    for (size_t i = 0; i < graph.outdegree.at(from); ++i) {
      if (graph.outthrowval.at(from).at(i) == *iter) {
        pruned = false;
        break;
      }
    }

    if (pruned && config.verboseflag) {
      const auto text = std::format(
          "worker {} option {} at root_pos {} was pruned; removing",
          worker_id, throwval, root_pos);
      message_coordinator_text(text);
    }

    if (!pruned && *iter == throwval) {
      found = true;

      if (config.verboseflag) {
        const auto text = std::format(
            "worker {} starting option {} at root_pos {}",
            worker_id, throwval, root_pos);
        message_coordinator_text(text);
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
    notify_coordinator_update();
    build_rootpos_throw_options(to_state, 0);
  }

  return (found || loading_work);
}

// Enumerate the set of throw options available at position `root_pos` in the
// pattern. This list of options is maintained in case we get a request to split
// work.

void Worker::build_rootpos_throw_options(unsigned from_state,
    unsigned start_column)
{
  root_throwval_options.clear();
  for (unsigned col = start_column; col < graph.outdegree.at(from_state);
      ++col) {
    root_throwval_options.push_back(graph.outthrowval.at(from_state).at(col));
  }

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << std::format("worker {} options at root_pos {}: [", worker_id,
        root_pos);
    for (const auto v : root_throwval_options) {
      if (v != root_throwval_options.front()) {
        buffer << ',';
      }
      buffer << v;
    }
    buffer << ']';
    message_coordinator_text(buffer.str());
  }
}

// Mark states that are excluded by a link throw from state `from`. These
// consist of the states downstream from `from` in its shift cycle that end in
// 'x'. In graph (b,h) those states can only be entered with shift throw(s) `h`
// from state `from`.
//
// Also, update working variables `deadstates` and `max_possible` to reflect the
// number of excluded states on the `from` shift cycle.
//
// Returns false if `max_possible` falls below `n_min`, indicating we should
// backtrack from the current position. Returns true otherwise.
// -----------------------------------------------------------------------------
//
// Implementation notes
//
// NOTE 1: In some cases it is possible for a state to be excluded twice during
// the construction of a pattern: Once via the head of a link throw, and again
// via the tail of a different link throw. Because of this we exclude states by
// flipping the used[] variable, rather than setting used=1. This (a) makes the
// marking process reversible, and (b) avoids double-counting the changes to
// `max_possible`. Any state that gets marked twice, back to used=0, does not
// affect the search because it cannot be accessed by a future link throw; such
// states end with 'x' and are only reachable with a shift throw `h`.
//
// INITIALIZATION: Prior to the start of gen_loops(), we initialize the
// algorithm by setting used=1 for any state that is a priori unusable. This
// consists of states < `start_state`, plus any states in their
// excludestates_tail[] and excludestates_head[] arrays. We then set
// deadstates[] and `max_possible` to be consistent with used[]. See
// Worker::initialize_working_variables().
//
// NOTE 2: For the states that are marked used=1 by this initialization
// procedure, some can be flipped back to used=0 during subsequent marking
// operations. However, a state that was excluded as part of an
// excludestates_head list for some state < `start_state` can only be flipped
// by the excludestates_tail list of a subsequent throw – and vice versa. This
// is because of how we've ordered the states: The excludestates[] arrays are
// monotonically increasing, so if state S is excluded by, say, the
// excludestates_tail[] list for some state < `start_state`, then all
// intermediate states have been marked used=1 as well, so none of the throws
// that could flip S via a tail will be encountered during pattern
// construction. A similar argument applies to states excluded by
// excludestates_head[]. An implication is that no state can be flipped twice,
// back to used=1.
//
// NOTE 3: The unusable states that are flipped back to used=0 (per NOTE 2)
// have a special form because they are in both an excludestates_head[] list
// and an excludestates_tail[] list: They start with `-` and end with `x`.
// As discussed in NOTE 1, such states ending in `x` do not affect subsequent
// pattern construction.

inline bool Worker::mark_unreachable_states_tail()
{
  bool valid = true;
  unsigned* const ds = deadstates_bystate[from];
  unsigned* es = excludestates_tail[from].data();
  unsigned statenum = 0;

  while ((statenum = *es++) != 0) {
    // assert that if we're flipping one of the unusable states, that it begins
    // with `-` and ends with `x`
    assert(start_state <= graph.max_startstate_usable.at(statenum) ||
        (graph.state.at(statenum).slot(0) == 0 &&
        graph.state.at(statenum).slot(graph.h - 1) == 1));

    const auto new_used = (used[statenum] ^= 1);
    if (new_used != 0 && ++*ds > 1 &&
        --max_possible < static_cast<int>(n_min)) {
      valid = false;
    }
  }
  return valid;
}

// Mark states that are excluded by a link throw onto state `to_state`. These
// consist of the states upstream from `to_state` in its shift cycle that begin
// with '-'. In graph (b,h) those states can only be exited with shift throw(s)
// 0 into state `to_state`.
//
// This updates working variables `deadstates` and `max_possible` to reflect the
// number of excluded states on the `to_state` shift cycle.
//
// Returns false if `max_possible` falls below `n_min`, indicating we should
// backtrack from the current position. Returns true otherwise.

inline bool Worker::mark_unreachable_states_head(unsigned to_state)
{
  bool valid = true;
  unsigned* const ds = deadstates_bystate[to_state];
  unsigned* es = excludestates_head[to_state].data();
  unsigned statenum = 0;

  while ((statenum = *es++) != 0) {
    assert(start_state <= graph.max_startstate_usable.at(statenum) ||
        (graph.state.at(statenum).slot(0) == 0 &&
        graph.state.at(statenum).slot(graph.h - 1) == 1));

    const auto new_used = (used[statenum] ^= 1);
    if (new_used != 0 && ++*ds > 1 &&
        --max_possible < static_cast<int>(n_min)) {
      valid = false;
    }
  }

  return valid;
}

// Reverse the marking operations above, so we can backtrack.

inline void Worker::unmark_unreachable_states_tail()
{
  unsigned* const ds = deadstates_bystate[from];
  unsigned* es = excludestates_tail[from].data();
  unsigned statenum = 0;

  while ((statenum = *es++) != 0) {
    const auto new_used = (used[statenum] ^= 1);
    if (new_used == 0 && --*ds > 0) {
      ++max_possible;
    }
  }
}

inline void Worker::unmark_unreachable_states_head(unsigned to_state)
{
  unsigned* const ds = deadstates_bystate[to_state];
  unsigned* es = excludestates_head[to_state].data();
  unsigned statenum = 0;

  while ((statenum = *es++) != 0) {
    const auto new_used = (used[statenum] ^= 1);
    if (new_used == 0 && --*ds > 0) {
      ++max_possible;
    }
  }
}

inline void Worker::handle_finished_pattern()
{
  ++count[pos + 1];

  if ((pos + 1) >= static_cast<int>(n_min) && !config.countflag) {
    pattern.at(pos + 1) = -1;
    report_pattern();
  }
}
