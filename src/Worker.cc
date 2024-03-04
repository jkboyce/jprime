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
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
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
#include <chrono>
#include <cassert>


Worker::Worker(const SearchConfig& config, Coordinator& coord, int id)
    : config(config),
      coordinator(coord),
      worker_id(id),
      graph(config.n, config.h, config.xarray,
        config.mode != RunMode::SUPER_SEARCH) {
  l_min = config.l_min;
  l_bound = (config.mode == RunMode::SUPER_SEARCH)
      ? graph.superprime_length_bound() + config.shiftlimit
      : graph.prime_length_bound();
  if (l_min > l_bound) {
    std::cerr << "No patterns longer than " << l_bound << " are possible\n";
    std::exit(EXIT_FAILURE);
  }
  l_max = (config.l_max > 0 ? config.l_max : l_bound);
  count.resize(l_max + 1, 0);
  allocate_arrays();
}

Worker::~Worker() {
  delete_arrays();
}

// Allocate all arrays used by the worker and initialize to default values.

void Worker::allocate_arrays() {
  beat = new SearchState[graph.numstates + 1];
  pattern = new int[graph.numstates + 1];
  used = new int[graph.numstates + 1];
  cycleused = new bool[graph.numstates + 1];
  deadstates = new int[graph.numstates + 1];
  deadstates_bystate = new int*[graph.numstates + 1];

  for (size_t i = 0; i <= graph.numstates; ++i) {
    beat[i] = {};
    pattern[i] = -1;
    used[i] = 0;
    cycleused[i] = false;
    deadstates[i] = 0;
    deadstates_bystate[i] = nullptr;
  }
}

void Worker::delete_arrays() {
  delete[] beat;
  delete[] pattern;
  delete[] used;
  delete[] cycleused;
  delete[] deadstates;
  delete[] deadstates_bystate;
  beat = nullptr;
  pattern = nullptr;
  used = nullptr;
  cycleused = nullptr;
  deadstates = nullptr;
  deadstates_bystate = nullptr;
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
        new_assignment = true;
      } else if (msg.type == messages_C2W::SPLIT_WORK) {
        // ignore in idle state
      } else if (msg.type == messages_C2W::SEND_STATS) {
        send_stats_to_coordinator();
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
    const auto start = std::chrono::high_resolution_clock::now();

    // complete the new work assignment
    try {
      gen_patterns();
      record_elapsed_time(start);
    } catch (const JprimeStopException& jpse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the coordinator
      (void)jpse;
      record_elapsed_time(start);
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

void Worker::message_coordinator(MessageW2C& msg) const {
  msg.worker_id = worker_id;
  coordinator.inbox_lock.lock();
  coordinator.inbox.push(msg);
  coordinator.inbox_lock.unlock();
}

void Worker::message_coordinator_status(const std::string& str) const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.meta = str;
  message_coordinator(msg);
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
    } else if (msg.type == messages_C2W::SPLIT_WORK) {
      process_split_work_request(msg);
    } else if (msg.type == messages_C2W::SEND_STATS) {
      send_stats_to_coordinator();
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

void Worker::record_elapsed_time(const
    std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double runtime = diff.count();
  secs_working += runtime;
}

void Worker::calibrate_inbox_check() {
  if (calibrations_remaining == calibrations_initial) {
    last_ts = std::chrono::high_resolution_clock::now();
    --calibrations_remaining;
    return;
  }

  const auto current_ts = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = current_ts - last_ts;
  double time_spent = diff.count();
  last_ts = current_ts;
  --calibrations_remaining;

  steps_per_inbox_check =
      static_cast<int>(static_cast<double>(steps_per_inbox_check) *
      secs_per_inbox_check_target / time_spent);
}

void Worker::process_split_work_request(const MessageC2W& msg) {
  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " splitting work...";
    message_coordinator_status(buffer.str());
  }

  send_work_to_coordinator(split_work_assignment(msg.split_alg));

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " remaining work after split:\n"
           << "  " << get_work_assignment();
    message_coordinator_status(buffer.str());
  }
}

void Worker::send_work_to_coordinator(const WorkAssignment& wa) {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_WORK;
  msg.assignment = wa;
  add_data_to_message(msg);
  message_coordinator(msg);
}

void Worker::send_stats_to_coordinator() {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_STATS;
  add_data_to_message(msg);
  msg.running = running;

  if (running) {
    // include a snapshot of the current state of the search
    msg.start_state = start_state;
    msg.worker_throw.resize(pos + 1);
    msg.worker_optionsleft.resize(pos + 1);

    int tempfrom = start_state;
    for (int i = 0; i <= pos; ++i) {
      msg.worker_throw[i] = pattern[i];

      for (int col = 0; col < graph.outdegree[tempfrom]; ++col) {
        if (graph.outthrowval[tempfrom][col] == pattern[i]) {
          if (i < root_pos)
            msg.worker_optionsleft[i] = 0;
          else if (i == root_pos)
            msg.worker_optionsleft[i] = root_throwval_options.size();
          else
            msg.worker_optionsleft[i] = graph.outdegree[tempfrom] - col - 1;
          tempfrom = graph.outmatrix[tempfrom][col];
          break;
        }
      }
    }
  }

  message_coordinator(msg);
}

void Worker::add_data_to_message(MessageW2C& msg) {
  std::uint64_t ntotal = 0;

  for (size_t i = 1; i < count.size(); ++i) {
    if (count[i] > 0) {
      ntotal += count[i];
      longest_found = std::max<int>(longest_found, i);
    }
  }

  msg.ntotal = ntotal;
  msg.count = count;
  msg.nnodes = nnodes;
  msg.numstates = graph.numstates;
  msg.numcycles = graph.numcycles;
  msg.numshortcycles = graph.numshortcycles;
  msg.l_bound = l_bound;
  msg.secs_working = secs_working;
  msg.longest_found = longest_found;

  count.assign(count.size(), 0);
  nnodes = 0;
  secs_working = 0;
}

void Worker::load_work_assignment(const WorkAssignment& wa) {
  assert(!running);
  loading_work = true;

  start_state = wa.start_state;
  end_state = wa.end_state;
  if (start_state == -1)
    start_state = (config.groundmode == 2) ? 2 : 1;
  if (end_state == -1)
    end_state = (config.groundmode == 1) ? 1 : graph.numstates;

  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  if (wa.start_state == -1 || wa.end_state == -1) {
    // assignment came from the coordinator which doesn't know how to correctly
    // set the throw options, so do that here
    build_rootpos_throw_options(start_state, 0);
  }
  assert(root_throwval_options.size() > 0);
  if (pos != 0) {
    std::cerr << "\nworker " << worker_id
              << ", pos: " << pos << '\n';
  }
  assert(pos == 0);

  for (size_t i = 0; i <= graph.numstates; ++i) {
    pattern[i] = (i < wa.partial_pattern.size()) ? wa.partial_pattern[i] : -1;
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
  for (size_t i = 0; i <= graph.numstates; ++i) {
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
  add_data_to_message(msg);
  message_coordinator(msg);
  running = false;
}

// Notify the coordinator of certain changes in the status of the search. The
// coordinator may use this information to determine which worker to steal work
// from when another worker goes idle.

void Worker::notify_coordinator_rootpos() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.root_pos = root_pos;
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
      return split_work_assignment_takeall();
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
  for (size_t i = 0; i < root_pos; ++i)
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
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
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
                  << '\n';
      }
      assert(col != graph.outdegree[from_state]);

      if (pos2 > root_pos && col < graph.outdegree[from_state] - 1) {
        new_root_pos = static_cast<int>(pos2);
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
  running = true;

  for (; start_state <= end_state; ++start_state) {
    // reset working variables
    pos = 0;
    from = start_state;
    shiftcount = 0;
    exitcyclesleft = 0;
    for (size_t i = 0; i <= graph.numstates; ++i) {
      used[i] = 0;
      cycleused[i] = false;
      deadstates[i] = 0;
    }

    // build the graph and initialize `deadstates`, `max_possible`, and
    // `exitcyclesleft`
    set_active_states();
    graph.build_graph();
    if (!graph.state_active[start_state])
      continue;
    for (size_t i = 0; i <= graph.numstates; ++i) {
      deadstates_bystate[i] = deadstates + graph.cyclenum[i];
      if (i > 0 && !graph.state_active[i])
        ++deadstates[graph.cyclenum[i]];
      if (graph.isexitcycle[i])
        ++exitcyclesleft;
    }
    max_possible = (config.mode == RunMode::SUPER_SEARCH)
        ? graph.superprime_length_bound() + config.shiftlimit
        : graph.prime_length_bound();

    if (config.verboseflag) {
      int inactive = std::count(graph.state_active.begin() + 1,
          graph.state_active.end(), false);
      std::ostringstream buffer;
      buffer << "worker " << worker_id
             << " starting at state " << graph.state_string(start_state)
             << " (" << start_state << ")\n";
      buffer << "worker " << worker_id
             << " deactivated " << inactive << " of " << graph.numstates
             << " states, max_possible = " << max_possible;
      message_coordinator_status(buffer.str());
    }
    if (max_possible < l_min)
      break;

    // when loading work, `root_pos` and `root_throwval_options` are given by
    // the work assignment, otherwise initialize here
    if (!loading_work) {
      root_pos = 0;
      notify_coordinator_rootpos();
      build_rootpos_throw_options(start_state, 0);
      if (root_throwval_options.size() == 0)
        continue;
    }
    longest_found = 0;

    std::vector<int> used_start(used, used + graph.numstates + 1);
    switch (config.mode) {
      case RunMode::NORMAL_SEARCH:
        gen_loops_normal_iterative();
        break;
      case RunMode::SUPER_SEARCH:
        if (config.shiftlimit == 0)
          gen_loops_super0();
        else
          gen_loops_super();
        break;
      default:
        assert(false);
        break;
    }
    std::vector<int> used_finish(used, used + graph.numstates + 1);
    assert(used_start == used_finish);
    assert(pos == 0);
  }
  assert(pos == 0);
}

// Set which states are active for the upcoming search.
//
// Note that Graph::build_graph() may deactivate additional states based on
// reachability.

void Worker::set_active_states() {
  for (size_t i = 0; i <= graph.numstates; ++i) {
    graph.state_active[i] = (i >= start_state);
  }

  if (config.mode == RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= graph.numstates; ++i) {
      // number of consecutive '-'s at the start of the state, plus number of
      // consecutive 'x's at the end of the state, cannot exceed `shiftlimit`
      int start0s = 0;
      while (start0s < graph.h && graph.state[i].slot[start0s] == 0)
        ++start0s;
      int end1s = 0;
      while (end1s < graph.h && graph.state[i].slot[graph.h - end1s - 1] != 0)
        ++end1s;
      if (start0s + end1s > config.shiftlimit) {
        graph.state_active[i] = false;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Core search routines (recursive versions)
//------------------------------------------------------------------------------

// Try all allowed throw values at the current pattern position `pos`,
// recursively continuing until (a) a pattern is found, or (b) we determine
// that we can't generate a path of length `l` or longer from our current
// position.
//
// These routines are by far the most performance-critical portions of jprime.
//
// This version is for NORMAL mode.

void Worker::gen_loops_normal() {
  if (pos >= l_max)
    return;

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

    if (throwval != 0 && throwval != graph.h) {
      // link throws make certain nearby states unreachable
      if (!did_mark_for_throw) {
        if (!mark_unreachable_states_throw()) {
          // bail since all additional `col` values will also be link throws
          unmark_unreachable_states_throw();
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
        gen_loops_normal();
        from = old_from;
        --pos;
        --used[to];
      }
      unmark_unreachable_states_catch(to);
    } else {
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
  if (pos >= l_max)
    return;

  int col = (loading_work ? load_one_throw() : 0);
  const int limit = graph.outdegree[from];
  const int* om = graph.outmatrix[from];
  const int* ov = graph.outthrowval[from];

  for (; col < limit; ++col) {
    const int to = om[col];
    if (pos == root_pos && !mark_off_rootpos_option(ov[col], to))
      continue;
    if (to < start_state)
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
  if (pos >= l_max)
    return;

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
  assert(false);
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

  if (!config.countflag && (pos + 1) >= l_min) {
    report_pattern();
  }
}

//------------------------------------------------------------------------------
// Core search routines (iterative versions)
//------------------------------------------------------------------------------

// This is a non-recursive version of get_loops_normal(), with identical
// interface and behavior.
//
// It is slightly faster than the recursive version and also avoids potential
// stack overflow on deeper searches.

void Worker::gen_loops_normal_iterative() {
  if (!iterative_init_workspace()) {
    // initialization fails when we're loading a work assignment that is no
    // longer valid, e.g. `l_current` has increased since the work was taken
    // from another worker
    for (size_t i = 1; i <= graph.numstates; ++i)
      used[i] = 0;
    return;
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

    if (used[to_state] || to_state < start_state || pos + 1 == l_max) {
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

// Set up the SearchState array with initial values.
//
// Leaves `pos` pointing to the last beat with loaded data, ready for the
// iterative algorithm to resume.
//
// Returns true on success, false on failure.

bool Worker::iterative_init_workspace() {
  beat[1].col = 0;
  beat[1].col_limit = graph.outdegree[start_state];
  beat[1].from_state = start_state;
  beat[1].to_state = 0;
  beat[1].outmatrix = graph.outmatrix[start_state];
  beat[1].excludes_throw = nullptr;
  beat[1].excludes_catch = nullptr;
  pos = 0;

  if (!loading_work)
    return true;

  // When loading from a work assignment, load_work_assignment() will have
  // set up `pattern`, `root_pos`, and `root_throwval_options`

  loading_work = false;
  pos = -1;
  int last_from_state = start_state;

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

    // mark next state as used
    assert(used[ss.to_state] == 0);
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

  // set up current beat
  SearchState& ss = beat[pos + 1];
  used[ss.to_state] = 0;
  ss.to_state = 0;
  assert(ss.excludes_throw == nullptr);
  assert(ss.excludes_catch == nullptr);

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

  if (!config.countflag && (pos + 1) >= l_min) {
    for (size_t i = 0; i <= pos; ++i) {
      pattern[i] = graph.outthrowval[beat[i + 1].from_state][beat[i + 1].col];
    }
    report_pattern();
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
                << inversestate[inverse_pos] << '\n';
      std::cerr << "   (" << graph.state_string(inversestate[inverse_pos])
                << ")\n";
      std::cerr << "   using throw " << inversethrow << '\n';
      std::cerr << "----------------" << '\n';
      std::cerr << "orig. pattern = " << get_pattern() << '\n';
      for (int j = 0; j <= pos; ++j) {
        std::cerr << graph.state_string(patternstate[j]) << "   "
                 << pattern[j] << '\n';
      }
      std::cerr << "   orig. pattern position = " << i << '\n';
      std::cerr << "----------------\n";
      std::cerr << "inverse pattern:\n";
      for (int j = 0; j <= inverse_pos; ++j) {
        std::cerr << graph.state_string(inversestate[j]) << "   "
                 << inversepattern[j] << '\n';
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
            graph.state[trial_state].slot[graph.h - 1] ? graph.h : 0;
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
