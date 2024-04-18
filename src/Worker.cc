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
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <cassert>


Worker::Worker(const SearchConfig& config, Coordinator& coord, int id,
    unsigned int l_max)
    : config(config),
      coordinator(coord),
      worker_id(id),
      graph(config.n, config.h, config.xarray,
        config.mode != RunMode::SUPER_SEARCH,
        config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH ? config.l_min : 0),
      l_min(config.l_min),
      l_max(l_max) {
  beat.resize(graph.numstates + 1);
  pattern.assign(graph.numstates + 1, -1);
  used.assign(graph.numstates + 1, 0);
  cycleused.assign(graph.numstates + 1, 0);
  deadstates.assign(graph.numstates + 1, 0);
  deadstates_bystate.assign(graph.numstates + 1, nullptr);
  count.assign(l_max + 1, 0);
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

// Deliver a message to the coordinator's inbox.

void Worker::message_coordinator(MessageW2C& msg) const {
  msg.worker_id = worker_id;
  coordinator.inbox_lock.lock();
  coordinator.inbox.push(msg);
  coordinator.inbox_lock.unlock();
}

// Deliver an informational text message to the coordinator's inbox.

void Worker::message_coordinator_text(const std::string& str) const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_UPDATE;
  msg.meta = str;
  message_coordinator(msg);
}

// Handle incoming messages from the coordinator that have queued while the
// worker is running.

void Worker::process_inbox_running() {
  if (calibrations_remaining > 0) {
    calibrate_inbox_check();
  }

  bool stopping_work = false;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageC2W msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_C2W::DO_WORK) {
      assert(false);
    } else if (msg.type == messages_C2W::SPLIT_WORK) {
      process_split_work_request();
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

// Get a finishing timestamp and record elapsed-time statistics to report to the
// coordinator later on.

void Worker::record_elapsed_time(const
    std::chrono::time_point<std::chrono::high_resolution_clock>& start) {
  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double runtime = diff.count();
  secs_working += runtime;
}

// Calibrate the number of steps to take in the gen_loops() functions to get the
// desired frequency of inbox checks.

void Worker::calibrate_inbox_check() {
  if (calibrations_remaining == CALIBRATIONS_INITIAL) {
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
      SECS_PER_INBOX_CHECK_TARGET / time_spent);
}

// Respond to the coordinator's request to split the current work assignment.

void Worker::process_split_work_request() {
  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " splitting work...";
    message_coordinator_text(buffer.str());
  }

  WorkAssignment wa = split_work_assignment(config.split_alg);
  send_work_to_coordinator(wa);

  // Avoid double counting nodes: Each of the "prefix" nodes up to and
  // including `root_pos` will be reported twice to the Coordinator: by this
  // worker, and the worker that does the job we just split off and returned.
  if (wa.start_state == start_state) {
    nnodes -= (wa.root_pos + 1);
  }

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " remaining work after split:\n"
           << "  " << get_work_assignment();
    message_coordinator_text(buffer.str());
  }
}

// Send a work assignment to the coordinator.

void Worker::send_work_to_coordinator(const WorkAssignment& wa) {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_WORK;
  msg.assignment = wa;
  add_data_to_message(msg);
  message_coordinator(msg);
}

// Respond to the coordinator's request to send back search statistics for the
// live status display.

void Worker::send_stats_to_coordinator() {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_STATS;
  add_data_to_message(msg);
  msg.running = running;

  if (!running) {
    message_coordinator(msg);
    return;
  }

  // add a snapshot of the current state of the search
  msg.worker_throw.assign(pos + 1, 0);
  msg.worker_options_left.assign(pos + 1, 0);
  msg.worker_deadstates_extra.assign(pos + 1, 0);

  unsigned int tempfrom = start_state;
  std::vector<bool> u(graph.numstates + 1, false);
  std::vector<unsigned int> ds(graph.numcycles, 0);

  for (size_t i = 0; i <= pos; ++i) {
    assert(pattern[i] >= 0);
    msg.worker_throw.at(i) = pattern[i];

    for (unsigned int col = 0; col < graph.outdegree[tempfrom]; ++col) {
      const unsigned int throwval = graph.outthrowval[tempfrom][col];
      if (throwval != static_cast<unsigned int>(pattern[i]))
        continue;

      const unsigned int tempto = graph.outmatrix[tempfrom][col];
      assert(tempto > 0);
      assert(!u.at(tempto));

      // options remaining at current position
      if (i < root_pos) {
        msg.worker_options_left.at(i) = 0;
      } else if (i == root_pos) {
        msg.worker_options_left.at(i) = root_throwval_options.size();
      } else {
        msg.worker_options_left.at(i) = graph.outdegree[tempfrom] - col - 1;
      }

      // number of deadstates induced by current link throw, above the
      // one-per-shift cycle baseline
      if (throwval != 0 && throwval != graph.h) {
        // throw
        for (size_t j = 0; true; ++j) {
          unsigned int es = graph.excludestates_throw[tempfrom][j];
          if (es == 0)
            break;
          if (!u.at(es) && ++ds.at(graph.cyclenum[tempfrom]) > 1) {
            ++msg.worker_deadstates_extra.at(i);
          }
          u.at(es) = true;
        }

        // catch
        for (size_t j = 0; true; ++j) {
          unsigned int es = graph.excludestates_catch[tempto][j];
          if (es == 0)
            break;
          if (!u.at(es) && ++ds.at(graph.cyclenum[tempto]) > 1) {
            ++msg.worker_deadstates_extra.at(i);
          }
          u.at(es) = true;
        }
      }

      u.at(tempto) = true;
      tempfrom = tempto;
      break;
    }
  }

  message_coordinator(msg);
}

// Add certain data items to a message for the coordinator. Several message
// types include these common items.

void Worker::add_data_to_message(MessageW2C& msg) {
  msg.count = count;
  msg.nnodes = nnodes;
  msg.secs_working = secs_working;

  count.assign(count.size(), 0);
  nnodes = 0;
  secs_working = 0;
}

// Respond to the coordinator's request to do the given work assignment.

void Worker::load_work_assignment(const WorkAssignment& wa) {
  assert(!running);
  loading_work = true;

  start_state = wa.start_state;
  end_state = wa.end_state;
  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;

  for (size_t i = 0; i <= graph.numstates; ++i) {
    pattern[i] = (i < wa.partial_pattern.size() ? wa.partial_pattern.at(i)
        : -1);
  }

  if (start_state == 0) {
    start_state = (config.groundmode == GroundMode::EXCITED_SEARCH ? 2 : 1);
  }
  if (end_state == 0) {
    end_state = (config.groundmode == GroundMode::GROUND_SEARCH ? 1 :
        graph.numstates);
  }
}

// Return the work assignment corresponding to the current state of the worker.
// Note this is distinct from split_work_assignment(), which splits off a
// portion of the assignment.

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

void Worker::notify_coordinator_update() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_UPDATE;
  msg.root_pos = root_pos;
  msg.start_state = start_state;
  msg.end_state = end_state;
  message_coordinator(msg);
}

// Determine the set of throw options available at position `root_pos` in
// the pattern. This list of options is maintained in case we get a request
// to split work.

void Worker::build_rootpos_throw_options(unsigned int from_state,
    unsigned int start_column) {
  root_throwval_options.clear();
  for (unsigned int col = start_column; col < graph.outdegree[from_state];
      ++col) {
    root_throwval_options.push_back(graph.outthrowval[from_state][col]);
  }

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " options at root_pos " << root_pos
           << ": [";
    for (unsigned int v : root_throwval_options) {
      if (config.throwdigits > 1 && v != root_throwval_options.front())
        buffer << ',';
      print_throw(buffer, v);
    }
    buffer << "]";
    message_coordinator_text(buffer.str());
  }
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

// Return a work assignment that corresponds to a portion of the worker's
// current work assignment, for handing off to another idle worker.

WorkAssignment Worker::split_work_assignment(unsigned int split_alg) {
  if (end_state > start_state) {
    return split_work_assignment_takestartstates();
  }

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

// Return a work assignment that corresponds to giving away approximately half
// of the unexplored `start_state` values in the current assignment.

WorkAssignment Worker::split_work_assignment_takestartstates() {
  unsigned int takenum = (end_state - start_state + 1) / 2;
  assert(takenum > 0);
  assert(end_state >= start_state + takenum);

  WorkAssignment wa;
  wa.start_state = end_state - takenum + 1;
  wa.end_state = end_state;
  wa.root_pos = 0;

  end_state = end_state - takenum;
  notify_coordinator_update();
  return wa;
}

// Return a work assignment that gives away all of the unexplored throw options
// at root_pos.

WorkAssignment Worker::split_work_assignment_takeall() {
  return split_work_assignment_takefraction(1, false);
}

// Return a work assignment that gives away approximately half of the unexplored
// throw options at root_pos.

WorkAssignment Worker::split_work_assignment_takehalf() {
  return split_work_assignment_takefraction(0.5, false);
}

// Return a work assignment that gives away approximately the target fraction of
// the unexplored throw options at root_pos.

WorkAssignment Worker::split_work_assignment_takefraction(double f,
      bool take_front) {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  for (size_t i = 0; i < root_pos; ++i) {
    wa.partial_pattern.push_back(pattern[i]);
  }

  // ensure the throw value at `root_pos` isn't on the list of throw options
  std::list<unsigned int>::iterator iter = root_throwval_options.begin();
  std::list<unsigned int>::iterator end = root_throwval_options.end();
  while (iter != end) {
    if (pattern[root_pos] >= 0 &&
        *iter == static_cast<unsigned int>(pattern[root_pos])) {
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
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
    } else {
      ++iter;
    }
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

    unsigned int from_state = start_state;
    int new_root_pos = -1;
    unsigned int col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const unsigned int throwval = static_cast<unsigned int>(pattern[pos2]);
      for (col = 0; col < graph.outdegree[from_state]; ++col) {
        if (throwval == graph.outthrowval[from_state][col]) {
          break;
        }
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
    root_pos = static_cast<unsigned int>(new_root_pos);
    notify_coordinator_update();
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

  // reset the graph so build_graph() below does a full recalc
  for (size_t i = 1; i <= graph.numstates; ++i) {
    graph.state_active.at(i) = true;
    graph.outdegree[i] = 0;
  }

  for (; start_state <= end_state; ++start_state) {
    if (!graph.state_active.at(start_state)) {
      loading_work = false;
      continue;
    }

    set_inactive_states();
    graph.build_graph();
    initialize_working_variables();

    if (config.verboseflag) {
      int num_inactive = std::count(graph.state_active.begin() + 1,
          graph.state_active.end(), false);
      std::ostringstream buffer;
      buffer << "worker " << worker_id
             << " starting at state " << graph.state_string(start_state)
             << " (" << start_state << ")\n";
      buffer << "worker " << worker_id
             << " deactivated " << num_inactive << " of " << graph.numstates
             << " states, max_possible = " << max_possible;
      message_coordinator_text(buffer.str());
    }

    if (max_possible < static_cast<int>(l_min)) {
      // larger values of `start_state` will have `max_possible` values that are
      // the same or smaller
      break;
    }
    if (!graph.state_active.at(start_state)) {
      loading_work = false;
      continue;
    }
    if (!loading_work || root_throwval_options.size() == 0) {
      // when loading work, `root_pos` (and usually `root_throwval_options`) are
      // given by the work assignment, otherwise initialize here
      root_pos = 0;
      build_rootpos_throw_options(start_state, 0);
    }
    if (root_throwval_options.size() == 0) {
      loading_work = false;
      continue;
    }
    notify_coordinator_update();

    // RELEASE THE KRAKEN

    std::vector<int> used_start(used);
    switch (config.mode) {
      case RunMode::NORMAL_SEARCH:
        if (config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH) {
          if (config.countflag) {
            iterative_gen_loops_normal_counting();
          } else {
            iterative_gen_loops_normal();
          }
        } else {
          graph.find_exclude_states();
          iterative_gen_loops_normal_marking();
        }
        break;
      case RunMode::SUPER_SEARCH:
        if (config.shiftlimit == 0) {
          iterative_gen_loops_super0();
        } else {
          iterative_gen_loops_super();
        }
        break;
      default:
        assert(false);
        break;
    }
    assert(used == used_start);
    assert(pos == 0);
  }
  assert(pos == 0);
}

// Set which states are inactive for the upcoming search.
//
// Note that Graph::build_graph() may deactivate additional states based on
// reachability. This routine should never mark states as active!

void Worker::set_inactive_states() {
  for (size_t i = 0; i < start_state; ++i)
    graph.state_active.at(i) = false;

  if (config.mode == RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= graph.numstates; ++i) {
      // number of consecutive '-'s at the start of the state, plus number of
      // consecutive 'x's at the end of the state, cannot exceed `shiftlimit`
      unsigned int start0s = 0;
      while (start0s < graph.h && graph.state.at(i).slot.at(start0s) == 0)
        ++start0s;
      unsigned int end1s = 0;
      while (end1s < graph.h && graph.state.at(i).slot.at(graph.h - end1s - 1)
          != 0)
        ++end1s;
      if (start0s + end1s > config.shiftlimit) {
        graph.state_active.at(i) = false;
      }
    }
  }
}

// Initialize all working variables prior to gen_loops().

void Worker::initialize_working_variables() {
  pos = 0;
  from = start_state;
  shiftcount = 0;
  exitcyclesleft = 0;
  for (size_t i = 0; i <= graph.numstates; ++i) {
    used[i] = 0;
    cycleused[i] = false;
    deadstates[i] = 0;
    deadstates_bystate[i] = deadstates.data() + graph.cyclenum[i];
    if (graph.isexitcycle[i])
      ++exitcyclesleft;
  }

  for (size_t i = 1; i <= graph.numstates; ++i) {
    if (!graph.state_active.at(i)) {
      ++deadstates_bystate[i];
    }
  }

  max_possible = (config.mode == RunMode::SUPER_SEARCH)
      ? graph.superprime_length_bound() + config.shiftlimit
      : graph.prime_length_bound();
}

//------------------------------------------------------------------------------
// Output a pattern during run
//------------------------------------------------------------------------------

// Send a message to the coordinator with the completed pattern. Note that all
// console output is done by the coordinator, not the worker threads.

void Worker::report_pattern() const {
  std::ostringstream buffer;

  if (config.groundmode != GroundMode::GROUND_SEARCH) {
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
      if (config.groundmode != GroundMode::GROUND_SEARCH && start_state == 1)
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

// Return a character for a given integer throw value (0 = '0', 1 = '1',
// 10 = 'a', 11 = 'b', ...

char Worker::throw_char(int val) {
  if (val < 0 || val > 35) {
    return '?';
  } else if (val < 10) {
    return static_cast<char>(val + '0');
  } else {
    return static_cast<char>(val - 10 + 'a');
  }
}

// Output a single throw to a string buffer.

void Worker::print_throw(std::ostringstream& buffer, unsigned int val) const {
  if (!config.noplusminusflag && val == 0) {
    buffer << '-';
    return;
  } else if (!config.noplusminusflag && val == graph.h) {
    buffer << '+';
    return;
  }

  if (config.throwdigits == 1) {
    buffer << throw_char(val);
  } else {
    buffer << std::setw(config.throwdigits) << val;
  }
}

// Return the current pattern as a string.

std::string Worker::get_pattern() const {
  std::ostringstream buffer;

  for (size_t i = 0; i <= pos; ++i) {
    if (config.throwdigits > 1 && i != 0)
      buffer << ',';
    const unsigned int throwval = (config.dualflag
      ? graph.h - static_cast<unsigned int>(pattern[pos - i])
      : static_cast<unsigned int>(pattern[i]));
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

  // Step 1. build a vector of state numbers traversed by the pattern, and
  // determine if an inverse exists.
  //
  // a pattern has an inverse if and only if:
  // - it visits more than one shift cycle on the state graph, and
  // - it never revisits a shift cycle, and
  // - it never does a link throw (0 < t < h) within a single cycle

  unsigned int state_current = start_state;
  unsigned int cycle_current = graph.cyclenum[start_state];
  bool cycle_multiple = false;

  for (size_t i = 0; i <= pos; ++i) {
    patternstate.at(i) = state_current;
    stateused.at(state_current) = true;

    const int state_next = graph.advance_state(state_current, pattern[i]);
    assert(state_next > 0);
    const unsigned int cycle_next = graph.cyclenum[state_next];

    if (cycle_next != cycle_current) {
      // mark a shift cycle as used only when we transition off it
      if (cycleused.at(cycle_current)) {
        // revisited cycle number `cycle_current` --> no inverse
        return buffer.str();
      }
      cycleused.at(cycle_current) = true;
      cycle_multiple = true;
    } else if (pattern[i] != 0 &&
        static_cast<unsigned int>(pattern[i]) != graph.h) {
      // link throw within a single cycle --> no inverse
      return buffer.str();
    }

    state_current = state_next;
    cycle_current = cycle_next;
  }
  patternstate.at(pos + 1) = start_state;

  if (!cycle_multiple) {
    // never left starting shift cycle --> no inverse
    return buffer.str();
  }

  // Step 2. Find the inverse pattern.
  //
  // Iterate through the link throws in the pattern to build up a list of
  // states and throws for the inverse.
  //
  // The inverse may go through states that aren't in the graph so we can't
  // refer to them by state number.

  std::vector<unsigned int> inversepattern;
  std::vector<State> inversestate;

  for (size_t i = 0; i <= pos; ++i) {
    // continue until `pattern[i]` is a link throw
    if (graph.cyclenum[patternstate.at(i)] ==
        graph.cyclenum[patternstate.at(i + 1)]) {
      continue;
    }

    if (inversestate.size() == 0) {
      // the inverse pattern starts at the (reversed version of) the next state
      // 'downstream' from `patternstate[i]`
      inversestate.push_back(
          graph.state.at(patternstate.at(i)).downstream().reverse());
    }

    const unsigned int inversethrow = graph.h -
        static_cast<unsigned int>(pattern[i]);
    inversepattern.push_back(inversethrow);
    inversestate.push_back(
        inversestate.back().advance_with_throw(inversethrow));

    // advance the inverse pattern along the shift cycle until it gets to a
    // state whose reverse is used by the original pattern

    while (true) {
      State trial_state = inversestate.back().downstream();
      unsigned int trial_statenum = graph.get_statenum(trial_state.reverse());
      if (trial_statenum > 0 && stateused.at(trial_statenum))
        break;

      inversepattern.push_back(trial_state.slot.at(graph.h - 1) ? graph.h : 0);
      inversestate.push_back(trial_state);
    }

    if (inversestate.back() == inversestate.front())
      break;
  }
  assert(inversestate.size() > 0);
  assert(inversestate.back() == inversestate.front());

  // Step 3. Output the inverse pattern.
  //
  // By convention we output all patterns starting with the smallest state.

  size_t min_index = 0;
  for (size_t i = 1; i < inversestate.size(); ++i) {
    if (inversestate.at(i) < inversestate.at(min_index)) {
      min_index = i;
    }
  }

  const size_t inverselength = inversepattern.size();
  if (config.dualflag)
    min_index = inverselength - min_index;

  for (size_t i = 0; i < inverselength; ++i) {
    size_t j = (i + min_index) % inverselength;
    const unsigned int throwval = (config.dualflag
        ? graph.h - inversepattern.at(inverselength - j - 1)
        : inversepattern.at(j));
    if (config.throwdigits > 1 && i != 0)
      buffer << ',';
    print_throw(buffer, throwval);
  }

  return buffer.str();
}
