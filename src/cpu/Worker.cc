//
// Worker.cc
//
// Worker that executes work assignments given to it by the Coordinator.
//
// The overall computation is depth first search on multiple worker threads,
// with a work stealing scheme to balance work among the threads. Each worker
// communicates only with the coordinator thread, via a set of message types.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"
#include "CoordinatorCPU.h"
#include "Pattern.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <cassert>
#include <format>
#include <array>


Worker::Worker(const SearchConfig& config, CoordinatorCPU& coord, Graph& g,
    unsigned id, unsigned n_max)
    : config(config), coordinator(coord), worker_id(id), n_min(config.n_min),
      n_max(n_max), graph(g)
{}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the main run loop for the worker, which waits until it receives an
// assignment from the coordinator.

void Worker::run()
{
  init();

  while (true) {
    bool new_assignment = false;
    bool stop_worker = false;

    {
      std::unique_lock<std::mutex> lck(inbox_lock);
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
        } else {
          assert(false);
        }
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
      do_work_assignment();
      record_elapsed_time_from(start);
      notify_coordinator_idle();
    } catch (const JprimeStopException& jpse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the coordinator
      (void)jpse;
      record_elapsed_time_from(start);
      send_work_to_coordinator(get_work_assignment());
      break;
    }
  }
}

// Initialize the arrays used during search.

void Worker::init()
{
  beat.resize(graph.numstates + 1);
  pattern.assign(graph.numstates + 1, -1);
  used.assign(graph.numstates + 1, 0);
  cycleused.assign(graph.numcycles, 0);
  deadstates.assign(graph.numcycles, 0);
  deadstates_bystate.assign(graph.numstates + 1, nullptr);
  count.assign(n_max + 1, 0);

  if (coordinator.get_search_algorithm() ==
      Coordinator::SearchAlgorithm::NORMAL_MARKING) {
    std::tie(excludestates_throw, excludestates_catch) =
        graph.get_exclude_states();
  }
}

//------------------------------------------------------------------------------
// Handle interactions with the Coordinator
//------------------------------------------------------------------------------

// Deliver a message to the coordinator's inbox.

void Worker::message_coordinator(MessageW2C& msg) const
{
  msg.worker_id = worker_id;
  std::unique_lock<std::mutex> lck(coordinator.inbox_lock);
  coordinator.inbox.push(msg);
}

// Deliver an informational text message to the coordinator's inbox.

void Worker::message_coordinator_text(const std::string& str) const
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_UPDATE;
  msg.meta = str;
  message_coordinator(msg);
}

// Handle incoming messages from the coordinator that have queued while the
// worker is running.

void Worker::process_inbox_running()
{
  if (calibrations_remaining > 0) {
    calibrate_inbox_check();
  }

  bool stopping_work = false;

  {
    std::unique_lock<std::mutex> lck(inbox_lock);

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
    throw JprimeStopException();  // unwind back to Worker::run()
  }
}

// Get a finishing timestamp and record elapsed-time statistics to report to the
// coordinator later on.

void Worker::record_elapsed_time_from(const
    std::chrono::time_point<std::chrono::high_resolution_clock>& start)
{
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> diff = end - start;
  const double runtime = diff.count();
  secs_working += runtime;
}

// Calibrate the number of steps to take in the gen_loops() functions to get the
// desired frequency of inbox checks.

void Worker::calibrate_inbox_check()
{
  if (calibrations_remaining == CALIBRATIONS_INITIAL) {
    last_ts = std::chrono::high_resolution_clock::now();
    --calibrations_remaining;
    return;
  }

  const auto current_ts = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> diff = current_ts - last_ts;
  const double time_spent = diff.count();
  last_ts = current_ts;
  --calibrations_remaining;

  steps_per_inbox_check =
      static_cast<int>(static_cast<double>(steps_per_inbox_check) *
      SECS_PER_INBOX_CHECK_TARGET / time_spent);
}

// Respond to the coordinator's request to split the current work assignment.

void Worker::process_split_work_request()
{
  if (config.verboseflag) {
    message_coordinator_text(
        std::format("worker {} splitting work...", worker_id));
  }

  WorkAssignment wa = get_work_assignment();
  WorkAssignment wa2 = wa.split(graph, config.split_alg);
  start_state = wa.start_state;
  end_state = wa.end_state;
  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  notify_coordinator_update();
  send_work_to_coordinator(wa2);

  // Avoid double counting nodes: Each of the nodes in the partial path for the
  // new assignment will be reported twice to the coordinator: by this worker,
  // and by the worker that does the job we just split off and returned.
  if (wa2.start_state == wa.start_state) {
    nnodes -= wa2.partial_pattern.size();
  }

  if (config.verboseflag) {
    auto text = std::format("worker {} remaining work after split:\n  {}",
                worker_id, wa.to_string());
    message_coordinator_text(text);
  }
}

// Send a work assignment to the coordinator.

void Worker::send_work_to_coordinator(const WorkAssignment& wa)
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::RETURN_WORK;
  msg.assignment = wa;
  add_data_to_message(msg);
  message_coordinator(msg);
}

// Respond to the coordinator's request to send back search statistics for the
// live status display.

void Worker::send_stats_to_coordinator()
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::RETURN_STATS;
  msg.running = running;
  add_data_to_message(msg);

  if (!running) {
    message_coordinator(msg);
    return;
  }

  // add a snapshot of the current state of the search
  msg.worker_throw.assign(pos + 1, 0);
  msg.worker_options_left.assign(pos + 1, 0);
  msg.worker_deadstates_extra.assign(pos + 1, 0);

  unsigned from_state = start_state;
  std::vector<int> u(graph.numstates + 1, 0);
  std::vector<unsigned> ds(graph.numcycles, 0);

  if (coordinator.get_search_algorithm() ==
      Coordinator::SearchAlgorithm::NORMAL_MARKING) {
    for (size_t i = 1; i <= graph.numstates; ++i) {
      if (start_state > graph.max_startstate_usable.at(i)) {
        u.at(i) = 1;
      }
    }
  } else {
    for (size_t i = 0; i < start_state; ++i) {
      u.at(i) = 1;
    }
  }

  for (int i = 0; i <= pos; ++i) {
    assert(pattern.at(i) >= 0);
    msg.worker_throw.at(i) = pattern.at(i);

    bool found = false;
    for (unsigned col = 0; col < graph.outdegree.at(from_state); ++col) {
      const auto throwval = graph.outthrowval.at(from_state).at(col);
      if (throwval != static_cast<unsigned>(pattern.at(i)))
        continue;

      const auto to_state = graph.outmatrix.at(from_state).at(col);
      assert(to_state > 0);
      assert(!u.at(to_state));

      // unexplored options remaining at position `i`
      if (i < static_cast<int>(root_pos)) {
        msg.worker_options_left.at(i) = 0;
      } else if (i == static_cast<int>(root_pos)) {
        msg.worker_options_left.at(i) =
            static_cast<unsigned>(root_throwval_options.size());
      } else {
        msg.worker_options_left.at(i) =
            graph.outdegree.at(from_state) - col - 1;
      }

      // number of deadstates induced by a link throw, above the
      // one-per-shift cycle baseline
      if (!excludestates_throw.empty() && throwval != 0 &&
          throwval != graph.h) {
        // throw
        for (size_t j = 0; ; ++j) {
          auto es = excludestates_throw.at(from_state).at(j);
          if (es == 0)
            break;
          if ((u.at(es) ^= 1) && ++ds.at(graph.cyclenum.at(from_state)) > 1) {
            ++msg.worker_deadstates_extra.at(i);
          }
        }

        // catch
        for (size_t j = 0; ; ++j) {
          auto es = excludestates_catch.at(to_state).at(j);
          if (es == 0)
            break;
          if ((u.at(es) ^= 1) && ++ds.at(graph.cyclenum.at(to_state)) > 1) {
            ++msg.worker_deadstates_extra.at(i);
          }
        }
      }

      if (i < pos) {
        // for i == pos we haven't yet marked used = 1 in gen_loops()
        u.at(to_state) = 1;
      }
      from_state = to_state;
      found = true;
      break;
    }
    (void)found;
    assert(found);
  }

  if (config.mode != SearchConfig::RunMode::SUPER_SEARCH ||
      config.shiftlimit != 0) {
    // check that we're accounting for used states in the correct way above;
    // note that `used` isn't used in SUPER0 mode
    /*
    if (u != used) {
      std::cout << "worker " << worker_id << ":\n";
      for (unsigned i = 0; i <= pos; ++i) {
        std::cout << "pattern[" << i << "] = " << pattern.at(i) << '\n';
      }
      for (unsigned i = 0; i <= graph.numstates; ++i) {
        std::cout << "state = " << i << ", u = " << u.at(i) << ", used = "
                  << used.at(i)
                  << (u.at(i) != used.at(i) ? " ERROR" : "") << '\n';
      }
      for (unsigned st : statenum) {
        std::cout << st << ',';
      }
      std::cout << '\n';
    }*/
    assert(u == used);
  }

  message_coordinator(msg);
}

// Add certain data items to a message for the coordinator. Several message
// types include these common items.

void Worker::add_data_to_message(MessageW2C& msg)
{
  msg.count = count;
  msg.nnodes = nnodes;
  msg.secs_working = secs_working;

  count.assign(count.size(), 0);
  nnodes = 0;
  secs_working = 0;
}

// Respond to the coordinator's request to do the given work assignment.

void Worker::load_work_assignment(const WorkAssignment& wa)
{
  assert(!running);

  start_state = wa.start_state;
  end_state = wa.end_state;
  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;

  for (size_t i = 0; i <= graph.numstates; ++i) {
    pattern.at(i) =
      (i < wa.partial_pattern.size() ? wa.partial_pattern.at(i) : -1);
  }

  if (wa.get_type() == WorkAssignment::Type::STARTUP) {
    start_state = (config.groundmode ==
      SearchConfig::GroundMode::EXCITED_SEARCH ? 2 : 1);
    end_state = (config.groundmode ==
      SearchConfig::GroundMode::GROUND_SEARCH ? 1 : graph.numstates);
  }
}

// Return the work assignment corresponding to the current state of the worker.

WorkAssignment Worker::get_work_assignment() const
{
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = end_state;
  wa.root_pos = root_pos;
  wa.root_throwval_options = root_throwval_options;
  for (const auto v : pattern) {
    if (v == -1)
      break;
    wa.partial_pattern.push_back(v);
  }
  return wa;
}

// Notify the coordinator that the worker is idle and ready for another work
// assignment.

void Worker::notify_coordinator_idle()
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_IDLE;
  add_data_to_message(msg);
  message_coordinator(msg);
  running = false;
}

// Notify the coordinator of certain changes in the status of the search. The
// coordinator may use this information to determine which worker to steal work
// from when another worker goes idle.

void Worker::notify_coordinator_update() const
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::WORKER_UPDATE;
  msg.start_state = start_state;
  msg.end_state = end_state;
  msg.root_pos = root_pos;
  message_coordinator(msg);
}

//------------------------------------------------------------------------------
// Search the juggling graph for patterns
//------------------------------------------------------------------------------

// Complete the current work assignment.

void Worker::do_work_assignment()
{
  running = true;
  loading_work = true;

  while (start_state <= end_state) {
    max_possible = coordinator.get_max_length(start_state);

    if (max_possible != -1) {
      if (config.verboseflag) {
        unsigned num_inactive = 0;
        for (size_t i = 1; i <= graph.numstates; ++i) {
          if (start_state > graph.max_startstate_usable.at(i)) {
            ++num_inactive;
          }
        }
        const auto text = std::format(
            "worker {} starting at state {} ({})\n"
            "worker {} deactivated {} of {} states, max_possible = {}",
            worker_id, graph.state_string(start_state), start_state,
            worker_id, num_inactive, graph.numstates, max_possible);
        message_coordinator_text(text);
      }

      if (max_possible < static_cast<int>(n_min)) {
        // larger values of `start_state` will have `max_possible` values that
        // are the same or smaller, so we can exit the loop
        if (config.verboseflag) {
          const auto text = std::format(
              "worker {} terminating because max_possible ({}) < n_min ({})",
              worker_id, max_possible, n_min);
          message_coordinator_text(text);
        }
        break;
      }

      notify_coordinator_update();
      initialize_working_variables();
      gen_loops();
    }

    // reset for the next start_state
    loading_work = false;
    root_pos = 0;
    root_throwval_options.clear();
    pattern[0] = -1;
    ++start_state;
  }
}

// Find all patterns for the current value of `start_state`.

void Worker::gen_loops()
{
  // choose a CPU search algorithm to use
  unsigned algnum = -1u;

  switch (coordinator.get_search_algorithm()) {
    case Coordinator::SearchAlgorithm::NORMAL:
      if (config.countflag) {
        algnum = (config.recursiveflag ? 0 : 4);
      } else {
        algnum = (config.recursiveflag ? 0 : 5);
      }
      break;
    case Coordinator::SearchAlgorithm::NORMAL_MARKING:
      algnum = (config.recursiveflag ? 1 : 6);
      break;
    case Coordinator::SearchAlgorithm::SUPER:
      algnum = (config.recursiveflag ? 2 : 7);
      break;
    case Coordinator::SearchAlgorithm::SUPER0:
      algnum = (config.recursiveflag ? 3 : 8);
      break;
    default:
      assert(false);
  }

  if (config.verboseflag) {
    static constexpr std::array cpu_algs = {
      "gen_loops_normal()",
      "gen_loops_normal_marking()",
      "gen_loops_super()",
      "gen_loops_super0()",
      "iterative_gen_loops_normal<REPORT=false>()",
      "iterative_gen_loops_normal<REPORT=true>()",
      "iterative_gen_loops_normal_marking()",
      "iterative_gen_loops_super<SUPER0=false>()",
      "iterative_gen_loops_super<SUPER0=true>()",
    };
    const auto text = std::format("worker {} starting algorithm {}", worker_id,
        cpu_algs.at(algnum));
    message_coordinator_text(text);
  }

  ////////////////////// RELEASE THE KRAKEN //////////////////////

  const auto pos_start = pos;
  const auto max_possible_start = max_possible;
  std::vector<int> used_start(used);

  switch (algnum) {
    case 0:
      gen_loops_normal();
      break;
    case 1:
      gen_loops_normal_marking();
      break;
    case 2:
      gen_loops_super();
      break;
    case 3:
      gen_loops_super0();
      break;
    case 4:
      iterative_gen_loops_normal<false, false>();
      break;
    case 5:
      iterative_gen_loops_normal<true, false>();
      break;
    case 6:
      iterative_gen_loops_normal_marking<false>();
      break;
    case 7:
      iterative_gen_loops_super<false, false>();
      break;
    case 8:
      iterative_gen_loops_super<true, false>();
      break;
    default:
      assert(false);
  }

  (void)pos_start;
  (void)max_possible_start;
  assert(pos == pos_start);
  assert(max_possible == max_possible_start);
  assert(used == used_start);
}

// Initialize all working variables prior to gen_loops().

void Worker::initialize_working_variables()
{
  used.assign(graph.numstates + 1, 0);
  pos = 0;
  from = start_state;

  // initialize `used` and `deadstates`/'deadstates_bystate` (if needed)
  if (coordinator.get_search_algorithm() ==
      Coordinator::SearchAlgorithm::NORMAL_MARKING) {
    for (size_t i = 1; i < start_state; ++i) {
      used.at(i) = 1;
      for (auto s : excludestates_throw.at(i)) {
        if (s != 0) {
          used.at(s) = 1;
        }
      }
      for (auto s : excludestates_catch.at(i)) {
        if (s != 0) {
          used.at(s) = 1;
        }
      }
    }

    deadstates.assign(graph.numcycles, 0);
    deadstates_bystate.assign(graph.numstates + 1, nullptr);

    for (size_t i = 1; i <= graph.numstates; ++i) {
      // assert equivalence of two ways of determining unusable graph states
      assert ((used.at(i) == 1) ==
          (start_state > graph.max_startstate_usable.at(i)));

      deadstates_bystate.at(i) = deadstates.data() + graph.cyclenum.at(i);
      if (used.at(i) == 1) {
        *deadstates_bystate.at(i) += 1;
      }
    }

    // compare the method we use to calculate `max_possible` in the GPU to the
    // value returned by Coordinator::get_max_length()
    int max_possible_gpu = graph.numstates - graph.numcycles;
    for (size_t i = 0; i < graph.numcycles; ++i) {
      if (deadstates.at(i) > 1) {
        max_possible_gpu -= (deadstates.at(i) - 1);
      }
    }
    (void)max_possible_gpu;
    assert(max_possible_gpu == max_possible);
  } else {
    for (size_t i = 0; i < start_state; ++i) {
      used.at(i) = 1;
    }

    /*
    NOTE: Instead of the above we could use the initialization below, which
    decreases node count by ~1%. It adds complexity to the GPU code though.

    for (size_t i = 1; i <= graph.numstates; ++i) {
      if (start_state > graph.max_startstate_usable.at(i)) {
        used.at(i) = 1;
      }
    }
    */
  }

  if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    cycleused.assign(graph.numcycles, 0);
    shiftcount = 0;
    isexitcycle = graph.get_exit_cycles(start_state);
    exitcyclesleft = static_cast<unsigned>(std::count(isexitcycle.cbegin(),
        isexitcycle.cend(), 1));
  }
}

//------------------------------------------------------------------------------
// Output a pattern during run
//------------------------------------------------------------------------------

// Send a message to the coordinator with the completed pattern. Note that all
// console output is done by the coordinator, not the worker threads.

void Worker::report_pattern() const
{
  MessageW2C msg;
  msg.type = MessageW2C::Type::SEARCH_RESULT;
  msg.pattern = coordinator.pattern_output_format(pattern, start_state);
  msg.period = pos + 1;
  message_coordinator(msg);
}
