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


Worker::Worker(const SearchConfig& config, Coordinator& coord, int id)
    : config(config),
      coordinator(coord),
      worker_id(id),
      graph(config.n, config.h, config.xarray,
        config.mode != RunMode::SUPER_SEARCH,
        config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH ? config.l_min : 0) {
  if (config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH) {
    l_bound = l_max = l_min = config.l_min;
  } else {
    l_min = config.l_min;
    l_bound = (config.mode == RunMode::SUPER_SEARCH)
        ? graph.superprime_length_bound() + config.shiftlimit
        : graph.prime_length_bound();
    l_max = (config.l_max > 0 ? config.l_max : l_bound);

    if (l_min > l_bound || l_max > l_bound) {
      std::cerr << "No patterns longer than " << l_bound << " are possible\n";
      std::exit(EXIT_FAILURE);
    }
  }

  count.assign(l_max + 1, 0);
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

  for (size_t i = 0; i <= static_cast<size_t>(graph.numstates); ++i) {
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

  WorkAssignment wa = split_work_assignment(msg.split_alg);
  send_work_to_coordinator(wa);

  // Avoid double counting nodes: Each of the "prefix" nodes up to and
  // including `root_pos` will be reported twice to the Coordinator: by this
  // worker, and the worker that does the job we just split off and returned.
  nnodes -= (wa.root_pos + 1);

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
    msg.end_state = end_state;
    msg.worker_throw.resize(pos + 1);
    msg.worker_optionsleft.resize(pos + 1);

    int tempfrom = start_state;
    for (int i = 0; i <= pos; ++i) {
      msg.worker_throw.at(i) = pattern[i];

      for (int col = 0; col < graph.outdegree[tempfrom]; ++col) {
        if (graph.outthrowval[tempfrom][col] == pattern[i]) {
          if (i < root_pos)
            msg.worker_optionsleft.at(i) = 0;
          else if (i == root_pos)
            msg.worker_optionsleft.at(i) = root_throwval_options.size();
          else
            msg.worker_optionsleft.at(i) = graph.outdegree[tempfrom] - col - 1;
          tempfrom = graph.outmatrix[tempfrom][col];
          break;
        }
      }
    }
  }

  message_coordinator(msg);
}

void Worker::add_data_to_message(MessageW2C& msg) {
  msg.count = count;
  msg.nnodes = nnodes;
  msg.secs_working = secs_working;
  msg.numstates = graph.numstates;
  msg.numcycles = graph.numcycles;
  msg.numshortcycles = graph.numshortcycles;
  msg.l_bound = l_bound;

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
    start_state = (config.groundmode == GroundMode::EXCITED_SEARCH ? 2 : 1);
  if (end_state == -1)
    end_state = (config.groundmode == GroundMode::GROUND_SEARCH ? 1 :
      graph.numstates);

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

  for (size_t i = 0; i <= static_cast<size_t>(graph.numstates); ++i) {
    pattern[i] = (i < wa.partial_pattern.size()) ? wa.partial_pattern.at(i)
        : -1;
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
  for (size_t i = 0; i <= static_cast<size_t>(graph.numstates); ++i) {
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
  for (size_t i = 0; i < static_cast<size_t>(root_pos); ++i)
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
    for (size_t pos2 = 0; pos2 <= static_cast<size_t>(pos); ++pos2) {
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

      if (pos2 > static_cast<size_t>(root_pos) &&
          col < graph.outdegree[from_state] - 1) {
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

  for (size_t i = 1; i <= static_cast<size_t>(graph.numstates); ++i)
    graph.state_active.at(i) = true;

  for (; start_state <= end_state; ++start_state) {
    if (!graph.state_active.at(start_state)) {
      loading_work = false;
      continue;
    }

    set_inactive_states();
    graph.build_graph();

    // reset working variables for search
    pos = 0;
    from = start_state;
    shiftcount = 0;
    exitcyclesleft = 0;
    for (size_t i = 0; i <= static_cast<size_t>(graph.numstates); ++i) {
      used[i] = 0;
      cycleused[i] = false;
      deadstates[i] = 0;
      deadstates_bystate[i] = deadstates + graph.cyclenum[i];
      if (graph.isexitcycle[i])
        ++exitcyclesleft;
    }

    for (size_t i = 1; i <= static_cast<size_t>(graph.numstates); ++i) {
      if (!graph.state_active.at(i)) {
        ++deadstates_bystate[i];
      }
    }
    if (config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH) {
      max_possible = l_bound;
    } else {
      max_possible = (config.mode == RunMode::SUPER_SEARCH)
          ? graph.superprime_length_bound() + config.shiftlimit
          : graph.prime_length_bound();
    }

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
      message_coordinator_status(buffer.str());
    }
    if (max_possible < l_min || config.infoflag)
      break;
    if (!graph.state_active.at(start_state)) {
      loading_work = false;
      continue;
    }

    if (!loading_work) {
      // when loading work, `root_pos` and `root_throwval_options` are given by
      // the work assignment, otherwise initialize here
      root_pos = 0;
      notify_coordinator_rootpos();
      build_rootpos_throw_options(start_state, 0);
      if (root_throwval_options.size() == 0)
        continue;
    }

    std::vector<int> used_start(used, used + graph.numstates + 1);
    switch (config.mode) {
      case RunMode::NORMAL_SEARCH:
        if (config.graphmode == GraphMode::SINGLE_PERIOD_GRAPH)
          gen_loops_normal();
        else
          iterative_gen_loops_normal_marking();
        break;
      case RunMode::SUPER_SEARCH:
        if (config.shiftlimit == 0)
          iterative_gen_loops_super0();
        else
          iterative_gen_loops_super();
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

// Set which states are inactive for the upcoming search.
//
// Note that Graph::build_graph() may deactivate additional states based on
// reachability. This routine should never mark states as active!

void Worker::set_inactive_states() {
  for (size_t i = 0; i < static_cast<size_t>(start_state); ++i)
    graph.state_active.at(i) = false;

  if (config.mode == RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= static_cast<size_t>(graph.numstates); ++i) {
      // number of consecutive '-'s at the start of the state, plus number of
      // consecutive 'x's at the end of the state, cannot exceed `shiftlimit`
      int start0s = 0;
      while (start0s < graph.h && graph.state.at(i).slot.at(start0s) == 0)
        ++start0s;
      int end1s = 0;
      while (end1s < graph.h && graph.state.at(i).slot.at(graph.h - end1s - 1)
          != 0)
        ++end1s;
      if (start0s + end1s > config.shiftlimit) {
        graph.state_active.at(i) = false;
      }
    }
  }
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
  if (val < 10)
    return static_cast<char>(val + '0');
  else
    return static_cast<char>(val - 10 + 'a');
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

  if (config.throwdigits == 1) {
    buffer << throw_char(val);
  } else {
    buffer << std::setw(config.throwdigits) << val;
  }
}

// Return the current pattern as a string.

std::string Worker::get_pattern() const {
  std::ostringstream buffer;

  for (int i = 0; i <= pos; ++i) {
    if (config.throwdigits > 1 && i != 0)
      buffer << ',';
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
