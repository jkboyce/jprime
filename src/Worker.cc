//
// Worker.cc
//
// Worker thread that executes work assignments given to it by the
// Coordinator thread.
//
// The overall computation is depth first search on multiple worker threads,
// with a work stealing scheme to balance work among the threads. Each worker
// communicates only with the coordinator thread, via a set of message types.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"
#include "Coordinator.h"
#include "Messages.h"
#include "Graph.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <ctime>


Worker::Worker(const SearchConfig& config, Coordinator& coord, int id)
    : config(config),
      coordinator(coord),
      worker_id(id),
      graph(config.n, config.h, config.xarray,
        (config.mode != RunMode::SUPER_SEARCH),
        (config.mode == RunMode::SUPER_SEARCH && config.shiftlimit == 0 &&
         config.groundmode == 1)) {
  l_current = config.l;
  maxlength = (config.mode == RunMode::SUPER_SEARCH)
      ? (graph.numcycles + config.shiftlimit)
      : (graph.numstates - graph.numcycles);
  if (config.l > maxlength) {
    std::cerr << "No patterns longer than " << maxlength << " are possible"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  allocate_arrays();
}

Worker::~Worker() {
  delete_arrays();
}

// Allocate all arrays used by the worker and initialize to default values.

void Worker::allocate_arrays() {
  pattern = new int[graph.numstates + 1];
  used = new int[graph.numstates + 1];
  cycleused = new bool[graph.numstates + 1];
  deadstates = new int[graph.numstates + 1];

  for (int i = 0; i <= graph.numstates; ++i) {
    pattern[i] = -1;
    used[i] = 0;
    cycleused[i] = false;
    deadstates[i] = 0;
  }
}

void Worker::delete_arrays() {
  delete[] pattern;
  delete[] used;
  delete[] cycleused;
  delete[] deadstates;
  pattern = nullptr;
  used = nullptr;
  cycleused = nullptr;
  deadstates = nullptr;
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the main run loop for the worker, which waits until it receives an
// assignment from the coordinator.

void Worker::run() {
  while (true) {
    bool new_assignment = false;
    bool stop_worker = false;

    inbox_lock.lock();
    if (!inbox.empty()) {
      MessageC2W msg = inbox.front();
      inbox.pop();

      if (msg.type == messages_C2W::DO_WORK) {
        load_work_assignment(msg.assignment);
        l_current = std::max(l_current, msg.l_current);
        new_assignment = true;
      } else if (msg.type == messages_C2W::UPDATE_METADATA) {
        // ignore in idle state
      } else if (msg.type == messages_C2W::SPLIT_WORK) {
        // leave in the inbox for when we get work
        inbox.push(msg);
      } else if (msg.type == messages_C2W::STOP_WORKER) {
        stop_worker = true;
      } else
        assert(false);
    }
    inbox_lock.unlock();

    if (stop_worker)
      break;
    if (!new_assignment)
      continue;

    // get timestamp so we can report working time to coordinator
    timespec start_ts;
    timespec_get(&start_ts, TIME_UTC);

    // complete the new work assignment
    try {
      gen_patterns();
      record_elapsed_time(start_ts);
    } catch (const JprimeStopException& jpse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the coordinator
      record_elapsed_time(start_ts);
      send_work_to_coordinator(get_work_assignment());
      break;
    }

    // empty the inbox
    inbox_lock.lock();
    inbox = std::queue<MessageC2W>();
    inbox_lock.unlock();

    notify_coordinator_idle();
  }
}

//------------------------------------------------------------------------------
// Handle interactions with the Coordinator thread
//------------------------------------------------------------------------------

void Worker::message_coordinator(const MessageW2C& msg) const {
  coordinator.inbox_lock.lock();
  coordinator.inbox.push(msg);
  coordinator.inbox_lock.unlock();
}

// Handle incoming messages from the coordinator that have queued while the
// worker is running.

void Worker::process_inbox_running() {
  if (calibrations_remaining > 0)
    calibrate_inbox_check();

  bool stopping_work = false;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageC2W msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_C2W::DO_WORK) {
      assert(false);
    } else if (msg.type == messages_C2W::UPDATE_METADATA) {
      l_current = std::max(l_current, msg.l_current);
    } else if (msg.type == messages_C2W::SPLIT_WORK) {
      process_split_work_request(msg);
    } else if (msg.type == messages_C2W::STOP_WORKER) {
      stopping_work = true;
    }
  }
  inbox_lock.unlock();

  if (stopping_work) {
    // unwind back to Worker::run()
    throw JprimeStopException();
  }
}

// Get a finishing timestamp and record elapsed-time statistics to report to
// the coordinator later on.

void Worker::record_elapsed_time(const timespec& start_ts) {
  timespec end_ts;
  timespec_get(&end_ts, TIME_UTC);
  double runtime =
      static_cast<double>(end_ts.tv_sec - start_ts.tv_sec) +
      1.0e-9 * (end_ts.tv_nsec - start_ts.tv_nsec);
  secs_working += runtime;
}

void Worker::calibrate_inbox_check() {
  if (calibrations_remaining == calibrations_initial) {
    timespec_get(&last_ts, TIME_UTC);
    --calibrations_remaining;
    return;
  }

  timespec current_ts;
  timespec_get(&current_ts, TIME_UTC);
  double time_spent =
      static_cast<double>(current_ts.tv_sec - last_ts.tv_sec) +
      1.0e-9 * (current_ts.tv_nsec - last_ts.tv_nsec);
  last_ts = current_ts;
  --calibrations_remaining;

  steps_per_inbox_check =
      static_cast<int>(static_cast<double>(steps_per_inbox_check) *
      secs_per_inbox_check_target / time_spent);
}

void Worker::send_work_to_coordinator(const WorkAssignment& wa) {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_WORK;
  msg.worker_id = worker_id;
  msg.assignment = wa;
  msg.ntotal = ntotal;
  msg.nnodes = nnodes;
  msg.numstates = graph.numstates;
  msg.numcycles = graph.numcycles;
  msg.numshortcycles = graph.numshortcycles;
  msg.maxlength = maxlength;
  msg.secs_working = secs_working;
  ntotal = 0;
  nnodes = 0;
  secs_working = 0;
  message_coordinator(msg);
}

void Worker::process_split_work_request(const MessageC2W& msg) {
  send_work_to_coordinator(split_work_assignment(msg.split_alg));

  if (config.verboseflag) {
    std::ostringstream sstr;
    sstr << "worker " << worker_id
         << " remaining work after split:" << std::endl
         << "  " << get_work_assignment();
    MessageW2C msg2;
    msg2.type = messages_W2C::WORKER_STATUS;
    msg2.worker_id = worker_id;
    msg2.meta = sstr.str();
    message_coordinator(msg2);
  }
}

void Worker::load_work_assignment(const WorkAssignment& wa) {
  loading_work = true;

  start_state = wa.start_state;
  end_state = wa.end_state;
  if (start_state == -1)
    start_state = (config.groundmode == 2) ? 2 : 1;
  if (end_state == -1)
    end_state = (config.groundmode == 1) ? 1 : graph.numstates;

  longest_found = 0;
  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  if (wa.start_state == -1 || wa.end_state == -1) {
    // assignment came from the coordinator which doesn't know how to correctly
    // set the throw options, so do that here
    build_rootpos_throw_options(start_state, 0);
  }
  assert(root_throwval_options.size() > 0);
  assert(pos == 0);

  for (int i = 0; i <= graph.numstates; ++i) {
    pattern[i] = (i < static_cast<int>(wa.partial_pattern.size())) ?
        wa.partial_pattern[i] : -1;
  }
}

// Return the work assignment corresponding to the current state of the worker.
// Note this is distinct from split_work_assignment(), which splits off a
// portion of the assignment to give back to the coordinator.

WorkAssignment Worker::get_work_assignment() const {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = end_state;
  wa.root_pos = root_pos;
  wa.root_throwval_options = root_throwval_options;
  for (int i = 0; i <= graph.numstates; ++i) {
    if (pattern[i] == -1)
      break;
    wa.partial_pattern.push_back(pattern[i]);
  }
  return wa;
}

// Notify the coordinator that the worker is idle and ready for another work
// assignment.

void Worker::notify_coordinator_idle() {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_IDLE;
  msg.worker_id = worker_id;
  msg.ntotal = ntotal;
  msg.nnodes = nnodes;
  msg.numstates = graph.numstates;
  msg.numcycles = graph.numcycles;
  msg.numshortcycles = graph.numshortcycles;
  msg.maxlength = maxlength;
  msg.secs_working = secs_working;
  ntotal = 0;
  nnodes = 0;
  secs_working = 0;
  message_coordinator(msg);
}

// Notify the coordinator of certain changes in the status of the search. The
// coordinator may use this information to determine which worker to steal work
// from when another worker goes idle.

void Worker::notify_coordinator_rootpos() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.root_pos = root_pos;
  message_coordinator(msg);
}

void Worker::notify_coordinator_longest() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.longest_found = longest_found;
  message_coordinator(msg);
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

// Return a work assignment that corresponds to a portion of the worker's
// current work assignment, for handing off to another idle worker. There is no
// single way to do this so we implement a number of strategies and measure
// performance.

WorkAssignment Worker::split_work_assignment(int split_alg) {
  switch (split_alg) {
    case 1:
      return split_work_assignment_takeall();
      break;
    case 2:
      return split_work_assignment_takehalf();
      break;
    default:
      assert(false);
  }
}

WorkAssignment Worker::split_work_assignment_takeall() {
  // strategy: take all of the throw options at root_pos
  return split_work_assignment_takefraction(1, false);
}

WorkAssignment Worker::split_work_assignment_takehalf() {
  // strategy: take half of the throw options at root_pos
  return split_work_assignment_takefraction(0.5, false);
}

WorkAssignment Worker::split_work_assignment_takefraction(double f,
      bool take_front) {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  for (int i = 0; i < root_pos; ++i)
    wa.partial_pattern.push_back(pattern[i]);

  // ensure the throw value at `root_pos` isn't on the list of throw options
  std::list<int>::iterator iter = root_throwval_options.begin();
  std::list<int>::iterator end = root_throwval_options.end();
  while (iter != end) {
    if (*iter == pattern[root_pos])
      iter = root_throwval_options.erase(iter);
    else
      ++iter;
  }
  assert(root_throwval_options.size() > 0);

  typedef std::list<int>::size_type li_size_t;
  li_size_t take_count =
      static_cast<li_size_t>(0.51 + f * root_throwval_options.size());
  take_count = std::min(
      std::max(static_cast<li_size_t>(1), take_count),
      root_throwval_options.size());

  li_size_t take_begin_idx = (take_front ? 0
        : root_throwval_options.size() - take_count);
  li_size_t take_end_idx = take_begin_idx + take_count;

  iter = root_throwval_options.begin();
  end = root_throwval_options.end();
  li_size_t index = 0;
  while (iter != end) {
    if (index >= take_begin_idx && index < take_end_idx) {
      wa.root_throwval_options.push_back(*iter);
      iter = root_throwval_options.erase(iter);
    } else
      ++iter;
    ++index;
  }

  if (root_throwval_options.size() == 0) {
    // Gave away all our throw options at this `root_pos`.
    //
    // We need to find the shallowest depth `new_root_pos` where there are
    // unexplored throw options. We have no more options at the current
    // root_pos, so new_root_pos > root_pos.
    //
    // We're also at a point in the search where we know there are unexplored
    // options remaining at the current value of `pos` (by virtue of how we got
    // here), and that pos > root_pos.
    //
    // So we know there must be a value of `new_root_pos` with the properties we
    // need, in the range root_pos < new_root_pos <= pos.

    int from_state = start_state;
    int new_root_pos = -1;
    int col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (int pos2 = 0; pos2 <= pos; ++pos2) {
      const int throwval = pattern[pos2];
      for (col = 0; col < graph.outdegree[from_state]; ++col) {
        if (throwval == graph.outthrowval[from_state][col])
          break;
      }
      // diagnostics if there's a problem
      if (col == graph.outdegree[from_state]) {
        std::cerr << "pos2 = " << pos2
                  << ", from_state = " << from_state
                  << ", start_state = " << start_state
                  << ", root_pos = " << root_pos
                  << ", col = " << col
                  << ", throwval = " << throwval
                  << std::endl;
      }
      assert(col != graph.outdegree[from_state]);

      if (pos2 > root_pos && col < graph.outdegree[from_state] - 1) {
        new_root_pos = pos2;
        break;
      }

      from_state = graph.outmatrix[from_state][col];
    }
    assert(new_root_pos != -1);
    root_pos = new_root_pos;
    notify_coordinator_rootpos();
    build_rootpos_throw_options(from_state, col + 1);
    assert(root_throwval_options.size() > 0);
  }

  return wa;
}

//------------------------------------------------------------------------------
// Search the juggling graph for patterns
//------------------------------------------------------------------------------

// Find all patterns within a range of `start_state` values.
//
// We enforce that a prime pattern has no state numbers smaller than the state
// it starts with, which ensures each pattern is generated exactly once.

void Worker::gen_patterns() {
  for (; start_state <= end_state; ++start_state) {
    // reset working variables
    pos = 0;
    from = start_state;
    firstblocklength = -1; // -1 signals unknown
    skipcount = 0;
    shiftcount = 0;
    blocklength = 0;
    max_possible = maxlength;
    exitcyclesleft = 0;
    for (int i = 0; i <= graph.numstates; ++i) {
      used[i] = 0;
      cycleused[i] = false;
      deadstates[i] = 0;
      if (graph.isexitcycle[i])
        ++exitcyclesleft;
      if (loading_work && pattern[i] != -1)
        --nnodes;  // avoid double-counting nodes when loading from a save
    }

    if (config.verboseflag) {
      std::ostringstream sstr;
      sstr << "worker " << worker_id
           << " starting at state " << graph.state_string(start_state)
           << " (" << start_state << ")";
      MessageW2C msg;
      msg.type = messages_W2C::WORKER_STATUS;
      msg.worker_id = worker_id;
      msg.meta = sstr.str();
      message_coordinator(msg);
    }

    // account for forbidden states and check if no way to make a pattern of the
    // target length
    if (config.mode != RunMode::SUPER_SEARCH) {
      for (int i = 1; i < start_state; ++i)
        mark_forbidden_state(i);
      if ((config.longestflag || config.exactflag) && max_possible < l_current)
        break;
    }

    if (!loading_work) {
      // reset `root_pos` and throw options there
      root_pos = 0;
      notify_coordinator_rootpos();
      build_rootpos_throw_options(start_state, 0);
      if (root_throwval_options.size() == 0)
        continue;
    }
    longest_found = 0;
    notify_coordinator_longest();

    std::vector<int> used_start(used, used + graph.numstates + 1);
    switch (config.mode) {
      case RunMode::NORMAL_SEARCH:
        gen_loops_normal();
        break;
      case RunMode::BLOCK_SEARCH:
        gen_loops_block();
        break;
      case RunMode::SUPER_SEARCH:
        if (config.shiftlimit == 0 && start_state == 1 && !config.exactflag)
          gen_loops_super0g();
        else
          gen_loops_super();
        break;
      default:
        assert(false);
        break;
    }
    std::vector<int> used_finish(used, used + graph.numstates + 1);
    assert(used_start == used_finish);
  }
}

// Try all allowed throw values at the current pattern position `pos`,
// recursively continuing until (a) a pattern is found, or (b) we determine
// that we can't generate a path of length `l` or longer from our current
// position.
//
// This version is for NORMAL mode.

void Worker::gen_loops_normal() {
  if (config.exactflag && pos >= l_current)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    const int throwval = graph.outthrowval[from][col];
    if (to == start_state) {
      pattern[pos] = throwval;
      handle_finished_pattern();
      continue;
    }

    bool valid = true;
    if (throwval != 0 && throwval != graph.h)
      valid = mark_unreachable_states(to);

    if (valid) {
      // we need to go deeper
      pattern[pos] = throwval;

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
    }

    // undo changes made above so we can backtrack
    if (throwval != 0 && throwval != graph.h)
      unmark_unreachable_states(to);

    // only a single allowed throw value for `pos` < `root_pos`
    if (pos < root_pos)
      break;
  }
}

// As above, but for BLOCK mode.
//
// Here there is additional structure we impose on the form of the pattern,
// which makes the search generally faster than NORMAL mode.

void Worker::gen_loops_block() {
  if (config.exactflag && pos >= l_current)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    const int throwval = graph.outthrowval[from][col];
    const bool linkthrow = (throwval != 0 && throwval != graph.h);
    const int old_blocklength = blocklength;
    const int old_skipcount = skipcount;
    const int old_firstblocklength = firstblocklength;

    // handle checks for link throws and skips
    if (linkthrow) {
      if (firstblocklength >= 0) {
        if (blocklength != (graph.h - 2)) {
          // got a skip
          if (skipcount == config.skiplimit)
            continue;
          else
            ++skipcount;
        }
      } else {
        // first link throw encountered
        firstblocklength = pos;
      }

      blocklength = 0;
    } else
      ++blocklength;

    bool valid = true;
    if (to == start_state) {
      if (skipcount == config.skiplimit
            && (blocklength + firstblocklength) != (graph.h - 2))
        valid = false;

      if (valid) {
        pattern[pos] = throwval;
        handle_finished_pattern();
      }
    } else if (valid) {
      if (linkthrow)
        valid = mark_unreachable_states(to);

      if (valid) {
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
        gen_loops_block();
        from = old_from;
        --pos;
        --used[to];
      }

      if (linkthrow)
        unmark_unreachable_states(to);
    }

    // undo changes so we can backtrack
    blocklength = old_blocklength;
    skipcount = old_skipcount;
    firstblocklength = old_firstblocklength;

    if (pos < root_pos)
      break;
  }
}

// As above, but for SUPER mode.
//
// Since a superprime pattern can only visit a single state in each shift cycle,
// this is the fastest version because so many states are excluded by each
// throw to a new shift cycle.

void Worker::gen_loops_super() {
  if (config.exactflag && pos >= l_current)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos &&
        !mark_off_rootpos_option(graph.outthrowval[from][col], to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    const int throwval = graph.outthrowval[from][col];
    const bool linkthrow = (throwval != 0 && throwval != graph.h);
    const int to_cycle = graph.cyclenum[to];
    // going to a shift cycle that's already been visited?
    if (linkthrow && cycleused[to_cycle])
      continue;

    const int old_shiftcount = shiftcount;
    // check for shift throw limits
    if (!linkthrow) {
      if (shiftcount == config.shiftlimit)
        continue;
      else
        ++shiftcount;
    }

    pattern[pos] = throwval;
    if (to == start_state) {
      handle_finished_pattern();
    } else {
      if (++steps_taken >= steps_per_inbox_check && pos > root_pos
            && col < limit - 1) {
        pattern[pos + 1] = -1;
        process_inbox_running();
        steps_taken = 0;
      }

      if (linkthrow)
        cycleused[to_cycle] = true;
      ++used[to];
      ++pos;
      const int old_from = from;
      from = to;
      gen_loops_super();
      from = old_from;
      --pos;
      --used[to];
      if (linkthrow)
        cycleused[to_cycle] = false;
    }

    shiftcount = old_shiftcount;

    if (pos < root_pos)
      break;
  }
}

// A specialization of gen_loops_super() for the case where `shiftthrows` == 0,
// we're searching for ground state patterns, and `exactflag` is false.
//
// This version tracks the specific "exit cycles" that can get back to the
// ground state with a single throw. If those exit cycles are all used and the
// pattern isn't done, we terminate the search early.

void Worker::gen_loops_super0g() {
  ++nnodes;

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
    if (to == 1) {
      handle_finished_pattern();
    } else {
      if (exitcyclesleft == 0)
        continue;

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
      gen_loops_super0g();
      from = old_from;
      --pos;
      cycleused[to_cycle] = false;
      exitcyclesleft = old_exitcyclesleft;
    }

    if (pos < root_pos)
      break;
  }
}

// Return the column number in the `outmatrix[from]` row vector that
// corresponds to the throw value at position `pos` in the pattern. This allows
// us to resume where we left off when loading from a work assignment.

int Worker::load_one_throw() {
  if (pattern[pos] == -1) {
    loading_work = false;
    return 0;
  }

  for (int col = 0; col < graph.outdegree[from]; ++col) {
    if (graph.outthrowval[from][col] == pattern[pos])
      return col;
  }

  // diagnostic information if there's a problem
  std::ostringstream buffer;
  for (int i = 0; i <= pos; ++i)
    print_throw(buffer, pattern[i]);
  std::cerr << "worker: " << worker_id << std::endl
            << "pos: " << pos << std::endl
            << "root_pos: " << root_pos << std::endl
            << "from: " << from << std::endl
            << "state[from]: " << graph.state[from] << std::endl
            << "start_state: " << start_state << std::endl
            << "pattern: " << buffer.str() << std::endl
            << "outthrowval[from][]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.outthrowval[from][i] << ", ";
  std::cerr << std::endl << "outmatrix[from][]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.outmatrix[from][i] << ", ";
  std::cerr << std::endl << "state[outmatrix[from][]]: ";
  for (int i = 0; i < graph.maxoutdegree; ++i)
    std::cerr << graph.state[graph.outmatrix[from][i]] << ", ";
  std::cerr << std::endl;
  assert(false);
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
    MessageW2C msg;
    msg.type = messages_W2C::WORKER_STATUS;
    msg.worker_id = worker_id;
    msg.meta = buffer.str();
    message_coordinator(msg);
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
      MessageW2C msg;
      msg.type = messages_W2C::WORKER_STATUS;
      msg.worker_id = worker_id;
      msg.meta = buffer.str();
      message_coordinator(msg);
    }

    if (!pruned && *iter == throwval) {
      found = true;

      if (config.verboseflag) {
        std::ostringstream buffer;
        buffer << "worker " << worker_id << " starting option ";
        print_throw(buffer, throwval);
        buffer << " at root_pos " << root_pos;
        MessageW2C msg;
        msg.type = messages_W2C::WORKER_STATUS;
        msg.worker_id = worker_id;
        msg.meta = buffer.str();
        message_coordinator(msg);
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

// Mark a state as forbidden in the current search, and update `used`,
// `deadstates`, and `max_possible` accordingly.
//
// This is called when `start_state` > 1 and we know at the outset that none
// of the lower-numbered states can be visited.

void Worker::mark_forbidden_state(int s) {
  if (used[s])
    return;

  used[s] = 1;
  const int cnum = graph.cyclenum[s];
  if (++deadstates[cnum] > 1)
    --max_possible;

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id
           << " forbidden state " << graph.state_string(s)
           << " (" << s
           << ") : max_possible = " << max_possible;
    MessageW2C msg;
    msg.type = messages_W2C::WORKER_STATUS;
    msg.worker_id = worker_id;
    msg.meta = buffer.str();
    message_coordinator(msg);
  }

  // if state starts with 'x', downcycle state is forbidden
  if (graph.state[s] & 1L)
    mark_forbidden_state(graph.cyclepartner[s][config.h - 2]);

  // if state ends with '-', upcycle state is forbidden
  if ((graph.state[s] & graph.highestbit) == 0)
    mark_forbidden_state(graph.cyclepartner[s][0]);
}

// Mark all of the states as used that are excluded by a throw from state
// `from` to state `to_state`.
//
// Returns false if the number of newly-excluded states implies that we can't
// finish a pattern of at least length `l_current` from our current position.
// Returns true otherwise.

bool inline Worker::mark_unreachable_states(int to_state) {
  bool valid = true;

  // 1. kill states downstream in `from` shift cycle that end in 'x'
  int j = graph.h - 2;
  std::uint64_t tempstate = graph.state[from];
  int cnum = graph.cyclenum[from];

  do {
    if (used[graph.cyclepartner[from][j]]++ == 0 && deadstates[cnum]++ >= 1
          && --max_possible < l_current)
      valid = false;
    --j;
    tempstate >>= 1;
  } while (tempstate & 1L);

  // 2. kill states upstream in 'to' shift cycle that start with '-'
  j = 0;
  tempstate = graph.state[to_state];
  cnum = graph.cyclenum[to_state];

  do {
    if (used[graph.cyclepartner[to_state][j]]++ == 0 && deadstates[cnum]++ >= 1
          && --max_possible < l_current)
      valid = false;
    ++j;
    tempstate = (tempstate << 1) & graph.allbits;
  } while ((tempstate & graph.highestbit) == 0);

  return valid;
}

// Reverse the marking operation above, so we can backtrack.

void inline Worker::unmark_unreachable_states(int to_state) {
  int j = graph.h - 2;
  std::uint64_t tempstate = graph.state[from];
  int cnum = graph.cyclenum[from];

  do {
    if (--used[graph.cyclepartner[from][j]] == 0 && --deadstates[cnum] >= 1)
      ++max_possible;
    --j;
    tempstate >>= 1;
  } while (tempstate & 1L);

  j = 0;
  tempstate = graph.state[to_state];
  cnum = graph.cyclenum[to_state];

  do {
    if (--used[graph.cyclepartner[to_state][j]] == 0 && --deadstates[cnum] >= 1)
      ++max_possible;
    ++j;
    tempstate = (tempstate << 1) & graph.allbits;
  } while ((tempstate & graph.highestbit) == 0);
}

void Worker::handle_finished_pattern() {
  ++ntotal;

  if ((pos + 1) >= l_current) {
    if (config.longestflag)
      l_current = pos + 1;
    report_pattern();
  }

  if ((pos + 1) > longest_found) {
    longest_found = pos + 1;
    notify_coordinator_longest();
  }
}

//------------------------------------------------------------------------------
// Output a pattern during run
//------------------------------------------------------------------------------

// Send a message to the coordinator with the completed pattern. Note that all
// console output is done by the coordinator, not the worker threads.

void Worker::report_pattern() const {
  std::ostringstream buffer;

  if (config.groundmode != 1) {
    if (start_state == 1)
      buffer << "  ";
    else
      buffer << "* ";
  }

  buffer << get_pattern();

  if (start_state != 1)
    buffer << " *";

  if (config.invertflag) {
    const std::string inverse = get_inverse();
    if (inverse.length() > 0) {
      if (config.groundmode != 1 && start_state == 1)
        buffer << "  ";
      buffer << " : " << inverse;
    }
  }

  MessageW2C msg;
  msg.type = messages_W2C::SEARCH_RESULT;
  msg.worker_id = worker_id;
  msg.pattern = buffer.str();
  msg.length = pos + 1;
  message_coordinator(msg);
}

// Output a single throw to a string buffer.

void Worker::print_throw(std::ostringstream& buffer, int val) const {
  if (!config.noplusminusflag && val == 0) {
    buffer << '-';
    return;
  } else if (!config.noplusminusflag && val == graph.h) {
    buffer << '+';
    return;
  }

  if (val < 10)
    buffer << static_cast<char>(val + '0');
  else
    buffer << static_cast<char>(val - 10 + 'a');
}

// Return the current pattern as a string.

std::string Worker::get_pattern() const {
  std::ostringstream buffer;

  for (int i = 0; i <= pos; ++i) {
    const int throwval = (config.dualflag ? (graph.h - pattern[pos - i])
        : pattern[i]);
    print_throw(buffer, throwval);
  }

  return buffer.str();
}

// Return the inverse of the current pattern as a string. If the pattern has
// no inverse, return an empty string.

std::string Worker::get_inverse() const {
  std::ostringstream buffer;
  std::vector<int> patternstate(graph.numstates + 1);
  std::vector<bool> stateused(graph.numstates + 1, false);
  std::vector<bool> cycleused(graph.numstates + 1, false);

  // step 1. build a vector of state numbers traversed by the pattern, and
  // determine if an inverse exists.
  //
  // a pattern has an inverse if and only if:
  // - it visits more than one shift cycle on the state graph, and
  // - it never revisits a shift cycle, and
  // - it never does a link throw (0 < t < h) within a single cycle

  int state_current = start_state;
  int cycle_current = graph.cyclenum[start_state];
  bool cycle_multiple = false;

  for (int i = 0; i <= pos; ++i) {
    patternstate[i] = state_current;
    stateused[state_current] = true;

    const int state_next = graph.advance_state(state_current, pattern[i]);
    assert(state_next > 0);
    const int cycle_next = graph.cyclenum[state_next];

    if (cycle_next != cycle_current) {
      // mark a shift cycle as used only when we transition off it
      if (cycleused[cycle_current]) {
        // revisited cycle number `cycle_current` --> no inverse
        return buffer.str();
      }
      cycleused[cycle_current] = true;
      cycle_multiple = true;
    } else if (pattern[i] != 0 && pattern[i] != graph.h) {
      // link throw within a single cycle --> no inverse
      return buffer.str();
    }

    state_current = state_next;
    cycle_current = cycle_next;
  }
  patternstate[pos + 1] = start_state;

  if (!cycle_multiple) {
    // never left starting shift cycle --> no inverse
    return buffer.str();
  }

  // step 2. Find the inverse pattern
  //
  // iterate through the link throws in the pattern to build up a list of
  // states and throws for the inverse

  std::vector<int> inversepattern(graph.numstates + 1);
  std::vector<int> inversestate(graph.numstates + 1);
  int inverse_start_state = -1;
  int inverse_pos = -1;

  for (int i = 0; i <= pos; ++i) {
    if (graph.cyclenum[patternstate[i]] == graph.cyclenum[patternstate[i + 1]])
      continue;

    if (inverse_start_state == -1) {
      // the inverse pattern starts at the (reversed version of) the next state
      // 'downstream' from `patternstate[i]`
      inverse_start_state = inversestate[0] =
          graph.reverse_state(graph.downstream_state(patternstate[i]));
    }

    ++inverse_pos;
    const int inversethrow = graph.h - pattern[i];
    inversepattern[inverse_pos] = inversethrow;
    inversestate[inverse_pos + 1] =
        graph.advance_state(inversestate[inverse_pos], inversethrow);

    if (inversestate[inverse_pos + 1] < 0) {
      std::cerr << "bad state advance: going from state "
                << inversestate[inverse_pos] << std::endl;
      std::cerr << "   (" << graph.state_string(inversestate[inverse_pos])
                << ")" << std::endl;
      std::cerr << "   using throw " << inversethrow << std::endl;
      std::cerr << "----------------" << std::endl;
      std::cerr << "orig. pattern = " << get_pattern() << std::endl;
      for (int j = 0; j <= pos; ++j) {
        std::cerr << graph.state_string(patternstate[j]) << "   "
                 << pattern[j] << std::endl;
      }
      std::cerr << "   orig. pattern position = " << i << std::endl;
      std::cerr << "----------------" << std::endl;
      std::cerr << "inverse pattern: " << std::endl;
      for (int j = 0; j <= inverse_pos; ++j) {
        std::cerr << graph.state_string(inversestate[j]) << "   "
                 << inversepattern[j] << std::endl;
      }
      std::exit(EXIT_FAILURE);
    }
    assert(inversestate[inverse_pos + 1] > 0);

    // the inverse pattern advances along the shift cycle until it gets
    // to a state that is used by the original pattern

    while (true) {
      int trial_state = graph.downstream_state(inversestate[inverse_pos + 1]);

      if (stateused[graph.reverse_state(trial_state)])
        break;
      else {
        ++inverse_pos;
        inversepattern[inverse_pos] =
            (graph.state[trial_state] & graph.highestbit) ? graph.h : 0;
        inversestate[inverse_pos + 1] = trial_state;
      }
    }

    if (inversestate[inverse_pos + 1] == inverse_start_state)
      break;
  }
  assert(inverse_pos > 0);
  assert(inverse_start_state > 0);
  assert(inversestate[inverse_pos + 1] == inverse_start_state);

  // step 3. Output the inverse pattern
  //
  // By convention we output all patterns starting with the smallest state.

  int min_state = inversestate[0];
  int min_index = 0;
  for (int i = 1; i <= inverse_pos; ++i) {
    if (inversestate[i] < min_state) {
      min_state = inversestate[i];
      min_index = i;
    }
  }
  if (config.dualflag)
    min_index = inverse_pos - min_index + 1;

  for (int i = 0; i <= inverse_pos; ++i) {
    int j = (i + min_index) % (inverse_pos + 1);
    int throwval = (config.dualflag ?
        (graph.h - inversepattern[inverse_pos - j]) : inversepattern[j]);
    print_throw(buffer, throwval);
  }

  return buffer.str();
}
