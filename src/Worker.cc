//
// Worker.cc
//
// Worker that executes work assignments given to it by the Coordinator.
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
#include "Pattern.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <cassert>


Worker::Worker(const SearchConfig& config, Coordinator& coord, unsigned int id,
    unsigned int l_max)
    : config(config), coordinator(coord), worker_id(id), l_min(config.l_min),
      l_max(l_max) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the main run loop for the worker, which waits until it receives an
// assignment from the coordinator.

void Worker::run() {
  initialize_graph();

  while (true) {
    bool new_assignment = false;
    bool stop_worker = false;

    {
      std::unique_lock<std::mutex> lck {inbox_lock};
      if (!inbox.empty()) {
        MessageC2W msg = inbox.front();
        inbox.pop();

        if (msg.type == MessageC2W::Type::DO_WORK) {
          load_work_assignment(msg.assignment);
          new_assignment = true;
        } else if (msg.type == MessageC2W::Type::SPLIT_WORK) {
          // ignore in idle state
        } else if (msg.type == MessageC2W::Type::SEND_STATS) {
          send_stats_to_coordinator();
        } else if (msg.type == MessageC2W::Type::STOP_WORKER) {
          stop_worker = true;
        } else
          assert(false);
      }
    }

    if (stop_worker)
      break;
    if (!new_assignment)
      continue;

    // get timestamp so we can report working time to coordinator
    const auto start = std::chrono::high_resolution_clock::now();

    // complete the new work assignment
    try {
      gen_patterns();
      record_elapsed_time_from(start);
    } catch (const JprimeStopException& jpse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the coordinator
      (void)jpse;
      record_elapsed_time_from(start);
      send_work_to_coordinator(get_work_assignment());
      break;
    }

    {
      // empty the inbox
      std::unique_lock<std::mutex> lck {inbox_lock};
      inbox = std::queue<MessageC2W>();
    }

    notify_coordinator_idle();
  }
}

// Initialize the juggling graph and associated arrays used during search.

void Worker::initialize_graph() {
  graph = {config.n, config.h, config.xarray,
    config.mode != RunMode::SUPER_SEARCH,
    config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH ? config.l_min : 0};

  beat.resize(graph.numstates + 1);
  pattern.assign(graph.numstates + 1, -1);
  used.assign(graph.numstates + 1, 0);
  cycleused.assign(graph.numstates + 1, 0);
  deadstates.assign(graph.numstates + 1, 0);
  deadstates_bystate.assign(graph.numstates + 1, nullptr);
  count.assign(l_max + 1, 0);
}

//------------------------------------------------------------------------------
// Handle interactions with the Coordinator
//------------------------------------------------------------------------------

// Deliver a message to the coordinator's inbox.

void Worker::message_coordinator(MessageW2C& msg) const {
  msg.worker_id = worker_id;
  std::unique_lock<std::mutex> lck {coordinator.inbox_lock};
  coordinator.inbox.push(msg);
}

// Deliver an informational text message to the coordinator's inbox.

void Worker::message_coordinator_text(const std::string& str) const {
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_UPDATE;
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

  {
    std::unique_lock<std::mutex> lck {inbox_lock};

    while (!inbox.empty()) {
      MessageC2W msg = inbox.front();
      inbox.pop();

      if (msg.type == MessageC2W::Type::DO_WORK) {
        assert(false);
      } else if (msg.type == MessageC2W::Type::SPLIT_WORK) {
        process_split_work_request();
      } else if (msg.type == MessageC2W::Type::SEND_STATS) {
        send_stats_to_coordinator();
      } else if (msg.type == MessageC2W::Type::STOP_WORKER) {
        stopping_work = true;
      }
    }
  }

  if (stopping_work) {
    // unwind back to Worker::run()
    throw JprimeStopException();
  }
}

// Get a finishing timestamp and record elapsed-time statistics to report to the
// coordinator later on.

void Worker::record_elapsed_time_from(const
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
  // including `root_pos` will be reported twice to the coordinator: by this
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
  msg.type = MessageW2C::Type::RETURN_WORK;
  msg.assignment = wa;
  add_data_to_message(msg);
  message_coordinator(msg);
}

// Respond to the coordinator's request to send back search statistics for the
// live status display.

void Worker::send_stats_to_coordinator() {
  MessageW2C msg;
  msg.type = MessageW2C::Type::RETURN_STATS;
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
    assert(pattern.at(i) >= 0);
    msg.worker_throw.at(i) = pattern.at(i);

    for (unsigned int col = 0; col < graph.outdegree.at(tempfrom); ++col) {
      const unsigned int throwval = graph.outthrowval.at(tempfrom).at(col);
      if (throwval != static_cast<unsigned int>(pattern.at(i)))
        continue;

      const unsigned int tempto = graph.outmatrix.at(tempfrom).at(col);
      assert(tempto > 0);
      assert(!u.at(tempto));

      // unexplored options remaining at position `i`
      if (i < root_pos) {
        msg.worker_options_left.at(i) = 0;
      } else if (i == root_pos) {
        msg.worker_options_left.at(i) =
            static_cast<unsigned int>(root_throwval_options.size());
      } else {
        msg.worker_options_left.at(i) = graph.outdegree.at(tempfrom) - col - 1;
      }

      // number of deadstates induced by a link throw, above the
      // one-per-shift cycle baseline
      if (throwval != 0 && throwval != graph.h) {
        // throw
        for (size_t j = 0; true; ++j) {
          unsigned int es = graph.excludestates_throw.at(tempfrom).at(j);
          if (es == 0)
            break;
          if (!u.at(es) && ++ds.at(graph.cyclenum.at(tempfrom)) > 1) {
            ++msg.worker_deadstates_extra.at(i);
          }
          u.at(es) = true;
        }

        // catch
        for (size_t j = 0; true; ++j) {
          unsigned int es = graph.excludestates_catch.at(tempto).at(j);
          if (es == 0)
            break;
          if (!u.at(es) && ++ds.at(graph.cyclenum.at(tempto)) > 1) {
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
    pattern.at(i) = (i < wa.partial_pattern.size() ? wa.partial_pattern.at(i)
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
    if (pattern.at(i) == -1)
      break;
    wa.partial_pattern.push_back(pattern.at(i));
  }
  return wa;
}

// Notify the coordinator that the worker is idle and ready for another work
// assignment.

void Worker::notify_coordinator_idle() {
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_IDLE;
  add_data_to_message(msg);
  message_coordinator(msg);
  running = false;
}

// Notify the coordinator of certain changes in the status of the search. The
// coordinator may use this information to determine which worker to steal work
// from when another worker goes idle.

void Worker::notify_coordinator_update() const {
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_UPDATE;
  msg.start_state = start_state;
  msg.end_state = end_state;
  msg.root_pos = root_pos;
  message_coordinator(msg);
}

// Enumerate the set of throw options available at position `root_pos` in the
// pattern. This list of options is maintained in case we get a request to split
// work.

void Worker::build_rootpos_throw_options(unsigned int from_state,
    unsigned int start_column) {
  root_throwval_options.clear();
  for (unsigned int col = start_column; col < graph.outdegree.at(from_state);
      ++col) {
    root_throwval_options.push_back(graph.outthrowval.at(from_state).at(col));
  }

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "worker " << worker_id << " options at root_pos " << root_pos
           << ": [";
    for (unsigned int v : root_throwval_options) {
      if (config.throwdigits > 1 && v != root_throwval_options.front())
        buffer << ',';
      Pattern::print_throw(buffer, v, config.throwdigits,
          config.noplusminusflag ? 0 : config.h);
    }
    buffer << "]";
    message_coordinator_text(buffer.str());
  }
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

// Return a work assignment that corresponds to a portion of the current work
// assignment, for handing off to another worker.

WorkAssignment Worker::split_work_assignment(unsigned int split_alg) {
  if (end_state > start_state) {
    return split_work_assignment_takestartstates();
  }

  switch (split_alg) {
    case 1:
      return split_work_assignment_takeall();
      break;
    default:
      return split_work_assignment_takehalf();
      break;
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

  end_state -= takenum;
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
    wa.partial_pattern.push_back(pattern.at(i));
  }

  // ensure the throw value at `root_pos` isn't on the list of throw options
  auto iter = root_throwval_options.begin();
  auto end = root_throwval_options.end();
  while (iter != end) {
    if (pattern.at(root_pos) >= 0 &&
        *iter == static_cast<unsigned int>(pattern.at(root_pos))) {
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
  }
  assert(root_throwval_options.size() > 0);

  // move `take_count` unexplored root_pos options to the new work assignment
  size_t take_count =
      static_cast<size_t>(0.51 + f * root_throwval_options.size());
  take_count = std::min(std::max(take_count, static_cast<size_t>(1)),
      root_throwval_options.size());

  size_t take_begin_idx = (take_front ?
        0 : root_throwval_options.size() - take_count);
  size_t take_end_idx = take_begin_idx + take_count;

  iter = root_throwval_options.begin();
  end = root_throwval_options.end();
  size_t index = 0;
  while (iter != end) {
    if (index >= take_begin_idx && index < take_end_idx) {
      wa.root_throwval_options.push_back(*iter);
      iter = root_throwval_options.erase(iter);
    } else {
      ++iter;
    }
    ++index;
  }

  // did we give away all our throw options at `root_pos`?
  if (root_throwval_options.size() == 0) {
    // Find the shallowest depth `new_root_pos` where there are unexplored throw
    // options. We have no more options at the current root_pos, so
    // new_root_pos > root_pos.
    //
    // We're also at a point in the search where we know there are unexplored
    // options remaining somewhere between `root_pos` and `pos`; see e.g.
    // Worker::iterative_can_split().
    //
    // So we know there must be a value of `new_root_pos` with the properties we
    // need in the range root_pos < new_root_pos <= pos.

    unsigned int from_state = start_state;
    unsigned int new_root_pos = -1;
    unsigned int col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (size_t pos2 = 0; pos2 <= pos; ++pos2) {
      const unsigned int throwval = static_cast<unsigned int>(pattern.at(pos2));
      for (col = 0; col < graph.outdegree.at(from_state); ++col) {
        if (throwval == graph.outthrowval.at(from_state).at(col)) {
          break;
        }
      }
      // diagnostics if there's a problem
      if (col == graph.outdegree.at(from_state)) {
        std::cerr << "pos2 = " << pos2
                  << ", from_state = " << from_state
                  << ", start_state = " << start_state
                  << ", root_pos = " << root_pos
                  << ", col = " << col
                  << ", throwval = " << throwval
                  << '\n';
      }
      assert(col != graph.outdegree.at(from_state));

      if (pos2 > root_pos && col < graph.outdegree.at(from_state) - 1) {
        new_root_pos = static_cast<unsigned int>(pos2);
        break;
      }

      from_state = graph.outmatrix.at(from_state).at(col);
    }
    assert(new_root_pos != -1u);
    root_pos = new_root_pos;
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
    graph.outdegree.at(i) = 0;
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
      int num_inactive = static_cast<int>(
          std::count(graph.state_active.begin() + 1,
          graph.state_active.end(), false));
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
      // the same or smaller, so we can exit the loop
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

    ////////////////////// RELEASE THE KRAKEN //////////////////////

    std::vector<int> used_start(used);
    switch (config.mode) {
      case RunMode::NORMAL_SEARCH:
        if (config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH) {
          // typically these searches are not close to `l_bound` so the
          // marking version of gen_loops() is not worth the overhead
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
  for (size_t i = 0; i < start_state; ++i) {
    graph.state_active.at(i) = false;
  }

  if (config.mode == RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= graph.numstates; ++i) {
      // number of consecutive '-'s at the start of the state, plus number of
      // consecutive 'x's at the end of the state, cannot exceed `shiftlimit`
      unsigned int start0s = 0;
      while (start0s < graph.h && graph.state.at(i).slot.at(start0s) == 0) {
        ++start0s;
      }
      unsigned int end1s = 0;
      while (end1s < graph.h &&
          graph.state.at(i).slot.at(graph.h - end1s - 1) != 0) {
        ++end1s;
      }
      if (start0s + end1s > config.shiftlimit) {
        graph.state_active.at(i) = false;
      }
    }
  }
}

// Initialize all working variables prior to gen_loops().
//
// Need to execute Graph::build_graph() before calling this method.

void Worker::initialize_working_variables() {
  pos = 0;
  from = start_state;
  shiftcount = 0;
  exitcyclesleft = 0;
  for (size_t i = 0; i <= graph.numstates; ++i) {
    used.at(i) = 0;
    cycleused.at(i) = false;
    deadstates.at(i) = 0;
    deadstates_bystate.at(i) = deadstates.data() + graph.cyclenum.at(i);
    if (graph.isexitcycle.at(i))
      ++exitcyclesleft;
  }

  for (size_t i = 1; i <= graph.numstates; ++i) {
    if (!graph.state_active.at(i)) {
      ++*deadstates_bystate.at(i);
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

  Pattern pat(pattern, config.h);
  const unsigned int display_h = config.noplusminusflag ? 0 : config.h;

  if (config.dualflag) {
    buffer << pat.dual().to_string(config.throwdigits, display_h);
  } else {
    buffer << pat.to_string(config.throwdigits, display_h);
  }

  if (start_state != 1)
    buffer << " *";

  if (config.invertflag) {
    Pattern inverse = pat.inverse(graph);
    if (inverse.is_valid()) {
      if (config.groundmode != GroundMode::GROUND_SEARCH && start_state == 1)
        buffer << "  ";
      if (config.dualflag) {
        buffer << " : " << inverse.dual().to_string(config.throwdigits,
            display_h);
      } else {
        buffer << " : " << inverse.to_string(config.throwdigits, display_h);
      }
    }
  }

  MessageW2C msg;
  msg.type = MessageW2C::Type::SEARCH_RESULT;
  msg.pattern = buffer.str();
  msg.length = pos + 1;
  message_coordinator(msg);
}
