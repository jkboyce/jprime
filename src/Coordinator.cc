//
// Coordinator.cc
//
// Coordinator that manages the overall search.
//
// The computation is depth first search on multiple worker threads using work
// stealing to keep the workers busy. The business of the coordinator is to
// interact with the workers to distribute work, and also to manage output.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Coordinator.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cassert>
#include <format>
#include <stdexcept>


Coordinator::Coordinator(const SearchConfig& a, SearchContext& b)
    : config(a), context(b) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the calculation specified in `config`, storing results in `context`.
//
// Returns true on success, false on failure.

bool Coordinator::run() {
  try {
    calc_graph_size();
  } catch (const std::overflow_error& oe) {
    std::cout << "Overflow occurred computing graph size\n";
    return false;
  }
  if (!passes_prechecks())
    return false;

  // the search is a go and `l_bound` fits into an unsigned int
  l_max = (config.l_max > 0) ? config.l_max
      : static_cast<unsigned>(context.l_bound);

  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);
  start_workers();

  constexpr auto NANOSECS_WAIT = std::chrono::nanoseconds(
      static_cast<long>(NANOSECS_PER_INBOX_CHECK));
  const auto start = std::chrono::high_resolution_clock::now();

  while (true) {
    give_assignments();
    steal_work();
    collect_stats();
    process_inbox();

    if (Coordinator::stopping || (workers_idle.size() == config.num_threads
          && context.assignments.size() == 0)) {
      break;
    }

    std::this_thread::sleep_for(NANOSECS_WAIT);
  }

  stop_workers();
  process_inbox();  // running worker will have sent back a RETURN_WORK message

  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double runtime = diff.count();
  context.secs_elapsed += runtime;
  context.secs_available += runtime * config.num_threads;

  erase_status_output();
  if (config.verboseflag)
    std::cout << "Finished on: " << current_time_string();
  if (context.assignments.size() > 0)
    std::cout << "\nPARTIAL RESULTS:\n";
  print_search_description();
  print_results();
  return true;
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

// Deliver a message to a given worker's inbox.

void Coordinator::message_worker(const MessageC2W& msg,
    unsigned worker_id) const {
  std::unique_lock<std::mutex> lck {worker.at(worker_id)->inbox_lock};
  worker.at(worker_id)->inbox.push(msg);
}

// Send messages to all workers requesting a status update.

void Coordinator::collect_stats() {
  if (!config.statusflag || ++stats_counter < WAITS_PER_STATUS)
    return;

  stats_counter = 0;
  stats_received = 0;
  for (unsigned id = 0; id < config.num_threads; ++id) {
    MessageC2W msg;
    msg.type = MessageC2W::Type::SEND_STATS;
    message_worker(msg, id);
  }
}

// Give assignments to workers, while there are available assignments and idle
// workers to take them.

void Coordinator::give_assignments() {
  while (workers_idle.size() > 0 && context.assignments.size() > 0) {
    auto it = workers_idle.begin();
    unsigned id = *it;
    workers_idle.erase(it);
    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();

    MessageC2W msg;
    msg.type = MessageC2W::Type::DO_WORK;
    msg.assignment = wa;
    worker_startstate.at(id) = wa.start_state;
    worker_endstate.at(id) = wa.end_state;
    worker_rootpos.at(id) = wa.root_pos;
    message_worker(msg, id);

    if (config.statusflag) {
      worker_options_left_start.at(id).resize(0);
      worker_options_left_last.at(id).resize(0);
      worker_longest_start.at(id) = 0;
      worker_longest_last.at(id) = 0;
    }

    if (config.verboseflag) {
      erase_status_output();
      std::cout << std::format("worker {} given work ({} idle):\n  ", id,
                     workers_idle.size())
                << msg.assignment << std::endl;
      print_status_output();
    }
  }
}

// Receive and handle messages from the worker threads.

void Coordinator::process_inbox() {
  std::unique_lock<std::mutex> lck {inbox_lock};
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == MessageW2C::Type::SEARCH_RESULT) {
      process_search_result(msg);
    } else if (msg.type == MessageW2C::Type::WORKER_IDLE) {
      process_worker_idle(msg);
    } else if (msg.type == MessageW2C::Type::RETURN_WORK) {
      process_returned_work(msg);
    } else if (msg.type == MessageW2C::Type::RETURN_STATS) {
      process_returned_stats(msg);
    } else if (msg.type == MessageW2C::Type::WORKER_UPDATE) {
      process_worker_update(msg);
    } else {
      assert(false);
    }
  }
}

// Handle a pattern sent to us by a worker. We store it and optionally print it
// to the terminal.

void Coordinator::process_search_result(const MessageW2C& msg) {
  // workers only send patterns in the target length range
  context.patterns.push_back(msg.pattern);

  if (config.printflag) {
    print_pattern(msg);
  }
}

// Handle a notification that a worker is now idle.

void Coordinator::process_worker_idle(const MessageW2C& msg) {
  workers_idle.insert(msg.worker_id);
  record_data_from_message(msg);
  worker_rootpos.at(msg.worker_id) = 0;
  if (config.statusflag) {
    worker_longest_start.at(msg.worker_id) = 0;
    worker_longest_last.at(msg.worker_id) = 0;
  }

  if (config.verboseflag) {
    erase_status_output();
    std::cout << std::format("worker {} went idle ({} idle)", msg.worker_id,
                   workers_idle.size());
    if (workers_splitting.count(msg.worker_id) > 0) {
      std::cout << std::format(", removed from splitting queue ({} splitting)",
                     (workers_splitting.size() - 1));
    }
    std::cout << " on: " << current_time_string();
    print_status_output();
  }

  // If we have a SPLIT_WORK request out for the worker, it will be ignored.
  // Remove it from the list of workers we're expecting to return work.
  workers_splitting.erase(msg.worker_id);
}

// Handle a work assignment sent back from a worker.
//
// This happens in two contexts: (a) when the worker is responding to a
// SPLIT_WORK request, and (b) when the worker is notified to quit.

void Coordinator::process_returned_work(const MessageW2C& msg) {
  workers_splitting.erase(msg.worker_id);
  context.assignments.push_back(msg.assignment);
  record_data_from_message(msg);

  if (config.verboseflag) {
    erase_status_output();
    std::cout << std::format("worker {} returned work:\n  ", msg.worker_id)
              << msg.assignment << std::endl;
    print_status_output();
  }
}

// Handle a worker's response to a SEND_STATS message, for doing the live status
// tracker (`-status` option).
//
// We create a status string for each worker as their stats return, and once all
// workers have responded we print it.

void Coordinator::process_returned_stats(const MessageW2C& msg) {
  record_data_from_message(msg);
  if (!config.statusflag)
    return;

  worker_status.at(msg.worker_id) = make_worker_status(msg);
  if (++stats_received == config.num_threads) {
    erase_status_output();
    print_status_output();
  }
}

// Handle an update from a worker on the state of its search.
//
// There are two types of updates: (a) informational text updates, which are
// printed in `-verbose` mode, and (b) updates to `start_state`, `end_state`,
// and `root_pos`, which are used by the coordinator when it needs to select a
// worker to send a SPLIT_WORK request to.

void Coordinator::process_worker_update(const MessageW2C& msg) {
  if (msg.meta.size() > 0) {
    if (config.verboseflag) {
      erase_status_output();
      std::cout << msg.meta << '\n';
      print_status_output();
    }
    return;
  }

  bool startstate_changed = false;
  bool endstate_changed = false;
  bool rootpos_changed = false;

  if (msg.start_state != worker_startstate.at(msg.worker_id)) {
    startstate_changed = true;
    worker_startstate.at(msg.worker_id) = msg.start_state;

    if (config.statusflag) {
      // reset certain elements of the status display
      worker_options_left_start.at(msg.worker_id).resize(0);
      worker_options_left_last.at(msg.worker_id).resize(0);
      worker_longest_start.at(msg.worker_id) = 0;
      worker_longest_last.at(msg.worker_id) = 0;
    }
  }
  if (msg.end_state != worker_endstate.at(msg.worker_id)) {
    endstate_changed = true;
    worker_endstate.at(msg.worker_id) = msg.end_state;
  }
  if (msg.root_pos != worker_rootpos.at(msg.worker_id)) {
    rootpos_changed = true;
    worker_rootpos.at(msg.worker_id) = msg.root_pos;
  }

  if (config.verboseflag &&
      (startstate_changed || endstate_changed || rootpos_changed)) {
    erase_status_output();
    bool comma = false;
    std::cout << "worker " << msg.worker_id;
    if (startstate_changed) {
      std::cout << " new start_state " << msg.start_state;
      comma = true;
    }
    if (endstate_changed) {
      if (comma)
        std::cout << ',';
      std::cout << " new end_state " << msg.end_state;
      comma = true;
    }
    if (rootpos_changed) {
      if (comma)
        std::cout << ',';
      std::cout << " new root_pos " << msg.root_pos;
    }
    std::cout << " on: " << current_time_string();
    print_status_output();
  }
}

//------------------------------------------------------------------------------
// Steal work from one of the running workers
//------------------------------------------------------------------------------

// Identify a (not idle) worker to steal work from, and send it a SPLIT_WORK
// message.

void Coordinator::steal_work() {
  bool sent_split_request = false;

  while (workers_idle.size() > workers_splitting.size()) {
    // when all of the workers are either idle or queued for splitting, there
    // are no active workers to take work from
    if (workers_idle.size() + workers_splitting.size() == config.num_threads) {
      if (config.verboseflag && sent_split_request) {
        erase_status_output();
        std::cout << std::format("could not steal work ({} idle)",
                       workers_idle.size())
                  << std::endl;
        print_status_output();
      }
      break;
    }

    unsigned id = -1;
    switch (config.steal_alg) {
      case 1:
        id = find_stealing_target_mostremaining();
        break;
    }
    assert(id < config.num_threads);

    MessageC2W msg;
    msg.type = MessageC2W::Type::SPLIT_WORK;
    message_worker(msg, id);
    workers_splitting.insert(id);
    sent_split_request = true;

    if (config.verboseflag) {
      erase_status_output();
      std::cout << std::format(
                     "worker {} given work split request ({} splitting)", id,
                     workers_splitting.size())
                << std::endl;
      print_status_output();
    }
  }
}

// Return the id of the busy worker with the most remaining work.
//
// First look at most remaining `start_state` values, and if no workers have
// unexplored start states then find the lowest `root_pos` value.

unsigned Coordinator::find_stealing_target_mostremaining() const {
  int id_startstates = -1;
  int id_rootpos = -1;
  unsigned max_startstates_remaining = 0;
  unsigned min_rootpos = 0;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (is_worker_idle(id) || is_worker_splitting(id))
      continue;

    unsigned startstates_remaining =
        worker_endstate.at(id) - worker_startstate.at(id);
    if (startstates_remaining > 0 && (id_startstates == -1 ||
        max_startstates_remaining < startstates_remaining)) {
      max_startstates_remaining = startstates_remaining;
      id_startstates = id;
    }

    if (id_rootpos == -1 || min_rootpos > worker_rootpos.at(id)) {
      min_rootpos = worker_rootpos.at(id);
      id_rootpos = id;
    }
  }
  assert(id_startstates != -1 || id_rootpos != -1);
  return (id_startstates == -1 ? id_rootpos : id_startstates);
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// Determine as much as possible about the size of the computation before
// starting up the workers, saving results in `context`.
//
// Note these quantities may be very large so we use uint64s for all of them.
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message. We only check for overflow when
// calculating `full_numstates` since this quantity is the largest.

void Coordinator::calc_graph_size() {
  // size of the full graph
  context.full_numstates = Graph::combinations(config.h, config.b);
  context.full_numcycles = 0;
  context.full_numshortcycles = 0;
  unsigned max_cycle_period = 0;

  for (unsigned p = 1; p <= config.h; ++p) {
    const std::uint64_t cycles =
        Graph::shift_cycle_count(config.b, config.h, p);
    context.full_numcycles += cycles;
    if (p < config.h) {
      context.full_numshortcycles += cycles;
    }
    if (cycles > 0) {
      max_cycle_period = p;
    }
  }

  // longest patterns possible of the type selected
  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    // two possibilities: Stay on a single cycle, or use multiple cycles
    context.l_bound = std::max(static_cast<std::uint64_t>(max_cycle_period),
        context.full_numstates - context.full_numcycles);
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    context.l_bound = (context.full_numcycles > 1 ?
        context.full_numcycles + config.shiftlimit : 0);
  }

  // number of states that will be resident in memory if we build the graph
  if (config.graphmode == SearchConfig::GraphMode::FULL_GRAPH) {
    context.memory_numstates = context.full_numstates;
  } else if (config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH) {
    context.memory_numstates =
        Graph::ordered_partitions(config.b, config.h, config.l_min);
  }
}

// Perform checks before starting the workers.
//
// Returns true if the computation is cleared to proceed.

bool Coordinator::passes_prechecks() {
  bool do_search = true;

  if (config.infoflag) {
    print_search_description();
    do_search = false;
  }
  if (config.l_min > context.l_bound || config.l_max > context.l_bound) {
    std::cout << std::format("No patterns longer than {} are possible",
                   context.l_bound)
              << std::endl;
    do_search = false;
  }
  if (context.memory_numstates > MAX_STATES) {
    std::cout << std::format("Number of states {} exceeds limit of {}",
                   context.memory_numstates, MAX_STATES)
              << std::endl;
    do_search = false;
  }

  return do_search;
}

bool Coordinator::is_worker_idle(const unsigned id) const {
  return (workers_idle.count(id) != 0);
}

bool Coordinator::is_worker_splitting(const unsigned id) const {
  return (workers_splitting.count(id) != 0);
}

// Copy status data out of the worker message, into appropriate data structures
// in the coordinator.

void Coordinator::record_data_from_message(const MessageW2C& msg) {
  context.nnodes += msg.nnodes;
  context.secs_working += msg.secs_working;

  // pattern counts by length
  assert(msg.count.size() == l_max + 1);
  assert(context.count.size() <= msg.count.size());
  context.count.resize(msg.count.size(), 0);

  for (size_t i = 1; i < msg.count.size(); ++i) {
    context.count.at(i) += msg.count.at(i);
    context.ntotal += msg.count.at(i);
    if (i >= config.l_min && i <= l_max) {
      context.npatterns += msg.count.at(i);
    }
    if (config.statusflag && msg.count.at(i) > 0) {
      worker_longest_start.at(msg.worker_id) = std::max(
          worker_longest_start.at(msg.worker_id), static_cast<unsigned>(i));
      worker_longest_last.at(msg.worker_id) = std::max(
          worker_longest_last.at(msg.worker_id), static_cast<unsigned>(i));
    }
  }
}

// Start all of the worker threads into a ready state, and initialize data
// structures for tracking them.

void Coordinator::start_workers() {
  if (config.verboseflag) {
    std::cout << "Started on: " << current_time_string();
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (config.verboseflag) {
      std::cout << std::format("worker {} starting...", id) << std::endl;
    }

    worker.push_back(std::make_unique<Worker>(config, *this, id, l_max));
    worker_thread.push_back(
        std::make_unique<std::thread>(&Worker::run, worker.at(id).get()));
    worker_startstate.push_back(0);
    worker_endstate.push_back(0);
    worker_rootpos.push_back(0);
    if (config.statusflag) {
      worker_status.push_back("     ");
      worker_options_left_start.push_back({});
      worker_options_left_last.push_back({});
      worker_longest_start.push_back(0);
      worker_longest_last.push_back(0);
    }
    workers_idle.insert(id);
  }
}

// Stop all workers.

void Coordinator::stop_workers() {
  if (config.verboseflag)
    erase_status_output();

  for (unsigned id = 0; id < config.num_threads; ++id) {
    MessageC2W msg;
    msg.type = MessageC2W::Type::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << std::format("worker {} asked to stop", id) << std::endl;

    worker_thread.at(id)->join();
  }

  if (config.verboseflag)
    print_status_output();
}

// Use the distribution of patterns found so far to extrapolate the expected
// number of patterns at length `l_bound`. This may be a useful signal of the
// degree of search completion.
//
// The distribution of patterns by length is observed to closely follow a
// Gaussian (normal) shape, so we fit the logarithm to a parabola and use that
// to extrapolate.

double Coordinator::expected_patterns_at_maxlength() {
  size_t mode = 0;
  size_t max = 0;
  std::uint64_t modeval = 0;

  for (size_t i = 0; i < context.count.size(); ++i) {
    if (context.count.at(i) > modeval) {
      mode = i;
      modeval = context.count.at(i);
    }
    if (context.count.at(i) > 0) {
      max = i;
    }
  }

  // fit a parabola to the log of pattern count
  double s1 = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  double sy = 0, sxy = 0, sx2y = 0;
  size_t xstart = std::max(max - 10, mode);

  for (size_t i = xstart; i < context.count.size(); ++i) {
    if (context.count.at(i) < 5)
      continue;

    const double x = static_cast<double>(i);
    const double y = log(static_cast<double>(context.count.at(i)));
    s1 += 1;
    sx += x;
    sx2 += x * x;
    sx3 += x * x * x;
    sx4 += x * x * x * x;
    sy += y;
    sxy += x * y;
    sx2y += x * x * y;
  }

  // Solve this 3x3 linear system for A, B, C, the coefficients in the parabola
  // of best fit y = Ax^2 + Bx + C:
  //
  // | sx4  sx3  sx2  | | A |   | sx2y |
  // | sx3  sx2  sx   | | B | = | sxy  |
  // | sx2  sx   s1   | | C |   | sy   |

  // Find matrix inverse
  double det = sx4 * (sx2 * s1 - sx * sx) - sx3 * (sx3 * s1 - sx * sx2) +
      sx2 * (sx3 * sx - sx2 * sx2);
  double M11 = (sx2 * s1 - sx * sx) / det;
  double M12 = (sx2 * sx - sx3 * s1) / det;
  double M13 = (sx3 * sx - sx2 * sx2) / det;
  double M21 = M12;
  double M22 = (sx4 * s1 - sx2 * sx2) / det;
  double M23 = (sx2 * sx3 - sx4 * sx) / det;
  double M31 = M13;
  double M32 = M23;
  double M33 = (sx4 * sx2 - sx3 * sx3) / det;

  auto is_close = [](double a, double b) {
    double epsilon = 1e-3;
    return (b > a - epsilon && b < a + epsilon);
  };
  assert(is_close(M11 * sx4 + M12 * sx3 + M13 * sx2, 1));
  assert(is_close(M11 * sx3 + M12 * sx2 + M13 * sx, 0));
  assert(is_close(M11 * sx2 + M12 * sx + M13 * s1, 0));
  assert(is_close(M21 * sx4 + M22 * sx3 + M23 * sx2, 0));
  assert(is_close(M21 * sx3 + M22 * sx2 + M23 * sx, 1));
  assert(is_close(M21 * sx2 + M22 * sx + M23 * s1, 0));
  assert(is_close(M31 * sx4 + M32 * sx3 + M33 * sx2, 0));
  assert(is_close(M31 * sx3 + M32 * sx2 + M33 * sx, 0));
  assert(is_close(M31 * sx2 + M32 * sx + M33 * s1, 1));

  double A = M11 * sx2y + M12 * sxy + M13 * sy;
  double B = M21 * sx2y + M22 * sxy + M23 * sy;
  double C = M31 * sx2y + M32 * sxy + M33 * sy;

  // evaluate the expected number of patterns found at x = l_bound
  double x = static_cast<double>(context.l_bound);
  double lny = A * x * x + B * x + C;
  return exp(lny);
}

// Static variable for indicating the user has interrupted execution.

volatile sig_atomic_t Coordinator::stopping = 0;

// Respond to a SIGINT (ctrl-c) interrupt during execution.

void Coordinator::signal_handler(int signum) {
  (void)signum;
  stopping = true;
}

//------------------------------------------------------------------------------
// Handle terminal output
//------------------------------------------------------------------------------

void Coordinator::print_pattern(const MessageW2C& msg) {
  erase_status_output();
  if (config.verboseflag) {
    std::cout << msg.worker_id << ": " << msg.pattern << std::endl;
  } else {
    std::cout << msg.pattern << std::endl;
  }
  print_status_output();
}

void Coordinator::print_search_description() const {
  std::cout << std::format("objects: {}, max throw: {}\n",
      (config.dualflag ? config.h - config.b : config.b), config.h);

  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    std::cout << "prime ";
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    std::cout << "superprime ";
    if (config.shiftlimit == 1) {
      std::cout << "(+1 shift) ";
    } else {
      std::cout << std::format("(+{} shifts) ", config.shiftlimit);
    }
  }
  std::cout << "search for length: " << config.l_min;
  if (config.l_max != config.l_min) {
    if (config.l_max == 0) {
      std::cout << '-';
    } else {
      std::cout << '-' << config.l_max;
    }
  }
  std::cout << std::format(" (bound {})", context.l_bound);
  if (config.groundmode == SearchConfig::GroundMode::GROUND_SEARCH) {
    std::cout << ", ground state only\n";
  } else if (config.groundmode == SearchConfig::GroundMode::EXCITED_SEARCH) {
    std::cout << ", excited states only\n";
  } else {
    std::cout << '\n';
  }

  std::cout << std::format("graph: {} states, {} shift cycles, {} short cycles",
                 context.full_numstates, context.full_numcycles,
                 context.full_numshortcycles)
            << std::endl;

  if (config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH) {
    std::cout << std::format("period-{} subgraph: {} states", config.l_min,
                   context.memory_numstates)
              << std::endl;
  }
}

void Coordinator::print_results() const {
  std::cout << std::format("{} patterns in range ({} seen, {} nodes)\n",
                 context.npatterns, context.ntotal, context.nnodes);

  std::cout << std::format("runtime = {:.4f} sec ({:.1f}M nodes/sec",
                 context.secs_elapsed, static_cast<double>(context.nnodes) /
                 context.secs_elapsed / 1000000);
  if (config.num_threads > 1) {
    std::cout << std::format(", {:.1f} % util)\n",
                   (context.secs_working / context.secs_available) * 100);
  } else {
    std::cout << ")\n";
  }

  if (config.countflag || l_max > config.l_min) {
    std::cout << "\nPattern count by length:\n";
    for (unsigned i = config.l_min; i <= l_max; ++i) {
      std::cout << i << ", " << context.count.at(i) << '\n';
    }
  }
}

void Coordinator::erase_status_output() const {
  if (!config.statusflag || !stats_printed)
    return;
  for (unsigned i = 0; i < config.num_threads + 2; ++i) {
    std::cout << '\x1B' << "[1A"
              << '\x1B' << "[2K";
  }
}

void Coordinator::print_status_output() {
  if (!config.statusflag)
    return;

  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && l_max > 2 * STATUS_WIDTH);
  std::cout << "Status on: " << current_time_string();
  std::cout << " cur/ end  rp options remaining at position";
  if (compressed) {
    std::cout << " (compressed view)";
    for (int i = 47; i < STATUS_WIDTH; ++i) {
      std::cout << ' ';
    }
  } else {
    for (int i = 29; i < STATUS_WIDTH; ++i) {
      std::cout << ' ';
    }
  }
  std::cout << "    length\n";
  for (unsigned i = 0; i < config.num_threads; ++i) {
    std::cout << worker_status.at(i) << std::endl;
  }

  stats_printed = true;
}

std::string Coordinator::current_time_string() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_timet = std::chrono::system_clock::to_time_t(now);
  return std::ctime(&now_timet);
}

std::string Coordinator::make_worker_status(const MessageW2C& msg) {
  std::ostringstream buffer;

  if (!msg.running) {
    buffer << "   -/   -   - IDLE";
    for (int i = 1; i < STATUS_WIDTH; ++i) {
      buffer << ' ';
    }
    buffer << "-    -";
    return buffer.str();
  }

  const unsigned id = msg.worker_id;
  const unsigned root_pos = worker_rootpos.at(id);
  const std::vector<unsigned>& ops = msg.worker_options_left;
  const std::vector<unsigned>& ds_extra = msg.worker_deadstates_extra;
  std::vector<unsigned>& ops_start = worker_options_left_start.at(id);
  std::vector<unsigned>& ops_last = worker_options_left_last.at(id);

  buffer << std::setw(4) << std::min(worker_startstate.at(id), 9999u) << '/';
  buffer << std::setw(4) << std::min(worker_endstate.at(id), 9999u) << ' ';
  buffer << std::setw(3) << std::min(worker_rootpos.at(id), 999u) << ' ';

  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && l_max > 2 * STATUS_WIDTH);
  const bool show_deadstates =
      (config.mode == SearchConfig::RunMode::NORMAL_SEARCH &&
      config.graphmode == SearchConfig::GraphMode::FULL_GRAPH);
  const bool show_shifts = (config.mode == SearchConfig::RunMode::SUPER_SEARCH);

  unsigned printed = 0;
  bool hl_start = false;
  bool did_hl_start = false;
  bool hl_last = false;
  bool did_hl_last = false;
  bool hl_deadstate = false;
  bool hl_shift = false;

  assert(ops.size() == msg.worker_throw.size());
  assert(ops.size() == ds_extra.size());

  for (size_t i = 0; i < ops.size(); ++i) {
    const unsigned throwval = msg.worker_throw.at(i);

    if (!hl_start && !did_hl_start && i < ops_start.size() &&
        ops.at(i) != ops_start.at(i)) {
      hl_start = did_hl_start = true;
    }
    if (!hl_last && !did_hl_last && i < ops_last.size() &&
        ops.at(i) != ops_last.at(i)) {
      hl_last = did_hl_last = true;
    }
    if (show_deadstates && i < ds_extra.size() && ds_extra.at(i) > 0) {
      hl_deadstate = true;
    }
    if (show_shifts && i < msg.worker_throw.size() && (throwval == 0 ||
        throwval == config.h)) {
      hl_shift = true;
    }

    if (i < root_pos)
      continue;

    char ch = '\0';

    if (compressed) {
      if (i == root_pos) {
        ch = '0' + ops.at(i);
      } else if (throwval == 0 || throwval == config.h) {
        // skip
      } else {
        ch = '0' + ops.at(i);
      }
    } else {
      ch = '0' + ops.at(i);
    }

    if (ch == '\0')
      continue;

    // use ANSI terminal codes to do inverse, bolding, and color
    const bool escape = (hl_start || hl_last || hl_deadstate || hl_shift);
    if (escape) { buffer << '\x1B' << '['; }
    if (hl_start) { buffer << "7;"; }
    if (hl_last) { buffer << "1;"; }
    if (hl_deadstate || hl_shift) { buffer << "32"; }
    if (escape) { buffer << 'm'; }
    buffer << ch;
    if (escape) { buffer << '\x1B' << "[0m"; }
    hl_start = hl_last = hl_deadstate = hl_shift = false;

    if (++printed >= STATUS_WIDTH)
      break;
  }

  while (printed < STATUS_WIDTH) {
    buffer << ' ';
    ++printed;
  }

  buffer << std::setw(5) << worker_longest_last.at(id)
         << std::setw(5) << worker_longest_start.at(id);

  ops_last = ops;
  if (ops_start.size() == 0) {
    ops_start = ops;
  }
  worker_longest_last.at(id) = 0;

  return buffer.str();
}
