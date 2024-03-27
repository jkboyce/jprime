//
// Coordinator.cc
//
// Coordinator thread that manages the overall search.
//
// The computation is depth first search on multiple worker threads using work
// stealing to keep the workers busy. The business of the coordinator is to
// interact with the workers to distribute work, and also to manage output.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"
#include "Worker.h"
#include "Messages.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cassert>


Coordinator::Coordinator(const SearchConfig& a, SearchContext& b)
    : config(a), context(b) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run() {
  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  if (config.verboseflag)
    std::cout << "Started on: " << current_time_string();

  // start worker threads
  for (unsigned int id = 0; id < context.num_threads; ++id) {
    if (config.verboseflag) {
      std::cout << "worker " << id << " starting..." << std::endl;
    }
    worker.push_back(new Worker(config, *this, id));
    worker_thread.push_back(new std::thread(&Worker::run, worker.at(id)));
    worker_startstate.push_back(0);
    worker_endstate.push_back(0);
    worker_rootpos.push_back(0);
    if (config.statusflag) {
      worker_status.push_back("     ");
      worker_optionsleft_start.push_back({});
      worker_optionsleft_last.push_back({});
      worker_longest.push_back(0);
    }
    workers_idle.insert(id);
  }

  constexpr auto nanosecs_wait = std::chrono::nanoseconds(
      static_cast<long>(nanosecs_per_inbox_check));
  const auto start = std::chrono::high_resolution_clock::now();

  while (true) {
    collect_stats();
    give_assignments();
    steal_work();
    process_inbox();

    if ((workers_idle.size() == context.num_threads
          && context.assignments.size() == 0) || Coordinator::stopping) {
      stop_workers();
      // any worker that was running will have sent back a RETURN_WORK message
      process_inbox();
      break;
    }

    std::this_thread::sleep_for(nanosecs_wait);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  double runtime = diff.count();
  context.secs_elapsed += runtime;
  context.secs_available += runtime * context.num_threads;

  for (unsigned int id = 0; id < context.num_threads; ++id) {
    delete worker.at(id);
    delete worker_thread.at(id);
    worker.at(id) = nullptr;
    worker_thread.at(id) = nullptr;
  }

  erase_status_output();
  if (config.verboseflag)
    std::cout << "Finished on: " << current_time_string();
  if (context.assignments.size() > 0)
    std::cout << "\nPARTIAL RESULTS:\n";
  print_summary();
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

void Coordinator::message_worker(const MessageC2W& msg,
    unsigned int worker_id) const {
  worker.at(worker_id)->inbox_lock.lock();
  worker.at(worker_id)->inbox.push(msg);
  worker.at(worker_id)->inbox_lock.unlock();
}

void Coordinator::collect_stats() {
  if (!config.statusflag || ++stats_counter < waits_per_status)
    return;

  stats_counter = 0;
  stats_received = 0;
  for (unsigned int id = 0; id < context.num_threads; ++id) {
    MessageC2W msg;
    msg.type = messages_C2W::SEND_STATS;
    message_worker(msg, id);
  }
}

// Give assignments to idle workers, while there are available assignments and
// idle workers to take them.

void Coordinator::give_assignments() {
  while (workers_idle.size() > 0 && context.assignments.size() > 0) {
    auto it = workers_idle.begin();
    unsigned int id = *it;
    workers_idle.erase(it);
    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();

    MessageC2W msg;
    msg.type = messages_C2W::DO_WORK;
    msg.assignment = wa;
    worker_startstate.at(id) = wa.start_state;
    worker_endstate.at(id) = wa.end_state;
    worker_rootpos.at(id) = wa.root_pos;
    message_worker(msg, id);

    if (config.statusflag) {
      worker_optionsleft_start.at(id).resize(0);
      worker_optionsleft_last.at(id).resize(0);
      worker_longest.at(id) = 0;
    }

    if (config.verboseflag) {
      erase_status_output();
      std::cout << "worker " << id << " given work ("
                << workers_idle.size() << " idle):\n "
                << msg.assignment << std::endl;
      print_status_output();
    }
  }
}

// Receive and handle messages from the worker threads.

void Coordinator::process_inbox() {
  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_W2C::SEARCH_RESULT) {
      process_search_result(msg);
    } else if (msg.type == messages_W2C::WORKER_IDLE) {
      process_worker_idle(msg);
    } else if (msg.type == messages_W2C::RETURN_WORK) {
      process_returned_work(msg);
    } else if (msg.type == messages_W2C::RETURN_STATS) {
      process_returned_stats(msg);
    } else if (msg.type == messages_W2C::WORKER_UPDATE) {
      process_worker_update(msg);
    } else {
      assert(false);
    }
  }
  inbox_lock.unlock();
}

void Coordinator::process_search_result(const MessageW2C& msg) {
  // workers will only send patterns in the right length range
  context.patterns.push_back(msg.pattern);

  if (config.printflag)
    print_pattern(msg);
}

void Coordinator::process_worker_idle(const MessageW2C& msg) {
  workers_idle.insert(msg.worker_id);
  record_data_from_message(msg);
  worker_rootpos.at(msg.worker_id) = 0;
  if (config.statusflag)
    worker_longest.at(msg.worker_id) = 0;

  if (config.verboseflag) {
    erase_status_output();
    std::cout << "worker " << msg.worker_id << " went idle ("
              << workers_idle.size() << " idle)";
    if (workers_splitting.count(msg.worker_id) > 0) {
      std::cout << ", removed from splitting queue ("
                << (workers_splitting.size() - 1) << " splitting)";
    }
    std::cout << " on: " << current_time_string();
    print_status_output();
  }

  // If we have a SPLIT_WORK request out for the worker, it will be ignored.
  // Remove it from the list of workers we're expecting to return work.
  workers_splitting.erase(msg.worker_id);
}

void Coordinator::process_returned_work(const MessageW2C& msg) {
  workers_splitting.erase(msg.worker_id);
  context.assignments.push_back(msg.assignment);
  record_data_from_message(msg);

  if (config.verboseflag) {
    erase_status_output();
    std::cout << "worker " << msg.worker_id << " returned work:\n "
              << msg.assignment << std::endl;
    print_status_output();
  }
}

void Coordinator::process_returned_stats(const MessageW2C& msg) {
  record_data_from_message(msg);
  if (!config.statusflag)
    return;

  worker_status.at(msg.worker_id) = make_worker_status(msg);
  if (++stats_received == context.num_threads) {
    erase_status_output();
    print_status_output();
  }
}

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
      worker_optionsleft_start.at(msg.worker_id).resize(0);
      worker_optionsleft_last.at(msg.worker_id).resize(0);
      worker_longest.at(msg.worker_id) = 0;
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
    if (workers_idle.size() + workers_splitting.size() == context.num_threads) {
      if (config.verboseflag && sent_split_request) {
        erase_status_output();
        std::cout << "could not steal work (" << workers_idle.size()
                  << " idle)" << std::endl;
        print_status_output();
      }
      break;
    }

    unsigned int id = 0;
    switch (context.steal_alg) {
      case 1:
        id = find_stealing_target_mostremaining();
        break;
      default:
        assert(false);
    }
    assert(id < context.num_threads);

    MessageC2W msg;
    msg.type = messages_C2W::SPLIT_WORK;
    msg.split_alg = context.split_alg;
    message_worker(msg, id);
    workers_splitting.insert(id);
    sent_split_request = true;

    if (config.verboseflag) {
      erase_status_output();
      std::cout << "worker " << id << " given work split request ("
                << workers_splitting.size() << " splitting)" << std::endl;
      print_status_output();
    }
  }
}

// Return the id of the busy worker with the most remaining work.
//
// First look at most remaining `start_state` values, and if no workers have
// unexplored start states then find the lowest `root_pos` value.

unsigned int Coordinator::find_stealing_target_mostremaining() const {
  int id_startstates = -1;
  int id_rootpos = -1;
  unsigned int max_startstates_remaining = 0;
  unsigned int min_rootpos = 0;

  for (unsigned int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id) || is_worker_splitting(id))
      continue;

    unsigned int startstates_remaining = worker_endstate.at(id) -
        worker_startstate.at(id);
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

bool Coordinator::is_worker_idle(const unsigned int id) const {
  return (workers_idle.count(id) != 0);
}

bool Coordinator::is_worker_splitting(const unsigned int id) const {
  return (workers_splitting.count(id) != 0);
}

void Coordinator::record_data_from_message(const MessageW2C& msg) {
  context.nnodes += msg.nnodes;
  context.secs_working += msg.secs_working;
  context.numstates = msg.numstates;
  context.numcycles = msg.numcycles;
  context.numshortcycles = msg.numshortcycles;
  context.l_bound = std::max(context.l_bound, msg.l_bound);

  l_max = (config.l_max > 0 ? config.l_max : context.l_bound);
  assert(msg.count.size() == static_cast<size_t>(l_max + 1));
  assert(context.count.size() <= msg.count.size());
  context.count.resize(msg.count.size(), 0);

  for (size_t i = 1; i < msg.count.size(); ++i) {
    context.count.at(i) += msg.count.at(i);
    context.ntotal += msg.count.at(i);
    if (i >= config.l_min && i <= static_cast<size_t>(l_max)) {
      context.npatterns += msg.count.at(i);
    }
    if (config.statusflag && msg.count.at(i) > 0) {
      worker_longest.at(msg.worker_id) = std::max<int>(
          worker_longest.at(msg.worker_id), i);
    }
  }
}

void Coordinator::stop_workers() {
  if (config.verboseflag)
    erase_status_output();
  for (unsigned int id = 0; id < context.num_threads; ++id) {
    MessageC2W msg;
    msg.type = messages_C2W::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << "worker " << id << " asked to stop" << std::endl;
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
// Gaussian (bell) shape, so we fit to this shape.

double Coordinator::expected_patterns_at_maxlength() {
  size_t mode = 0;
  size_t max = 0;
  std::uint64_t modeval = 0;

  for (size_t i = 0; i < context.count.size(); ++i) {
    if (context.count.at(i) > modeval) {
      mode = i;
      modeval = context.count.at(i);
    }
    if (context.count.at(i) > 0)
      max = i;
  }

  // fit a parabola to the log of pattern count
  double s1 = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  double sy = 0, sxy = 0, sx2y = 0;
  size_t xstart = std::max<size_t>(max - 10, mode);

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

  // Solve this 3x3 linear system for A, B, C, the coefficients in the
  // parabola of best fit y = Ax^2 + Bx + C:
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

// static variable for indicating the user has interrupted execution
bool Coordinator::stopping = false;

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

void Coordinator::print_summary() const {
  std::cout << "objects: " << (config.dualflag ? config.h - config.n : config.n)
            << ", max throw: " << config.h << '\n';

  std::cout << "length: " << config.l_min;
  if (l_max > static_cast<int>(config.l_min)) {
    if (l_max == static_cast<int>(context.l_bound))
      std::cout << '-';
    else
      std::cout << '-' << l_max;
  }
  std::cout << " (bound " << context.l_bound << "), ";

  switch (config.mode) {
    case RunMode::NORMAL_SEARCH:
      std::cout << "normal mode";
      break;
    case RunMode::SUPER_SEARCH:
      std::cout << "super mode (" << config.shiftlimit << " shift limit)";
      break;
    default:
      break;
  }

  if (config.invertflag)
    std::cout << ", inverse output\n";
  else
    std::cout << '\n';

  if (config.groundmode == GroundMode::GROUND_SEARCH)
    std::cout << "ground state search" << std::endl;
  if (config.groundmode == GroundMode::EXCITED_SEARCH)
    std::cout << "excited state search" << std::endl;

  std::cout << "graph: " << context.numstates << " states";
  if (config.graphmode == GraphMode::FULL_GRAPH) {
    std::cout << ", " << context.numcycles << " shift cycles, "
              << context.numshortcycles << " short cycles\n";
  } else {
    std::cout << " (" << Graph::combinations(config.n, config.h)
              << " in full graph)\n";
  }

  if (!config.infoflag) {
    std::cout << context.npatterns << " patterns found ("
              << context.ntotal << " seen, "
              << context.nnodes << " nodes, "
              << std::fixed << std::setprecision(2)
              << (static_cast<double>(context.nnodes) / context.secs_elapsed /
                  1000000)
              << "M nodes/sec)\n";

    std::cout << "running time = "
              << std::fixed << std::setprecision(4)
              << context.secs_elapsed << " sec";
    if (context.num_threads > 1) {
      std::cout << " (worker util = " << std::setprecision(2)
                << ((context.secs_working / context.secs_available) * 100)
                << " %)";
    }
    std::cout << '\n';

    if (config.countflag || l_max > static_cast<int>(config.l_min)) {
      std::cout << "\nPattern count by length:\n";
      for (int i = config.l_min; i <= l_max; ++i)
        std::cout << i << ", " << context.count.at(i) << '\n';
    }
  }

  std::cout << "------------------------------------------------------------"
            << std::endl;
}

void Coordinator::erase_status_output() const {
  if (!config.statusflag || !stats_printed)
    return;
  for (unsigned int i = 0; i < context.num_threads + 2; ++i) {
    std::cout << '\x1B' << "[1A"
              << '\x1B' << "[2K";
  }
}

void Coordinator::print_status_output() {
  if (!config.statusflag)
    return;

  const bool compressed = (config.mode == RunMode::NORMAL_SEARCH &&
      l_max > status_width);
  std::cout << "Status on: " << current_time_string();
  std::cout << " cur/ end  rp options remaining at position";
  if (compressed) {
    std::cout << " (compressed view)";
    for (int i = 47; i < status_width; ++i)
      std::cout << ' ';
  } else {
    for (int i = 29; i < status_width; ++i)
      std::cout << ' ';
  }
  std::cout << "dist  len\n";
  for (unsigned int i = 0; i < context.num_threads; ++i)
    std::cout << worker_status.at(i) << std::endl;

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
    for (int i = 1; i < status_width; ++i)
      buffer << ' ';
    buffer << "-    -";
    return buffer.str();
  }

  const unsigned int id = msg.worker_id;
  const unsigned int root_pos = worker_rootpos.at(id);
  const std::vector<unsigned int>& ops = msg.worker_optionsleft;
  std::vector<unsigned int>& ops_start = worker_optionsleft_start.at(id);
  std::vector<unsigned int>& ops_last = worker_optionsleft_last.at(id);

  buffer << std::setw(4) << std::min(worker_startstate.at(id), 9999u) << '/';
  buffer << std::setw(4) << std::min(worker_endstate.at(id), 9999u) << ' ';
  buffer << std::setw(3) << std::min(worker_rootpos.at(id), 999u) << ' ';

  const bool compressed = (config.mode == RunMode::NORMAL_SEARCH &&
      l_max > status_width);

  int printed = 0;
  bool did_hl_start = false;
  bool hl_start = false;
  bool did_hl_last = false;
  bool hl_last = false;
  unsigned int rootpos_distance = 1000u;

  for (size_t i = 0; i < ops.size(); ++i) {
    if (!hl_start && !did_hl_start && i < ops_start.size() &&
        ops.at(i) != ops_start.at(i)) {
      hl_start = did_hl_start = true;
      rootpos_distance = (i > root_pos ? i - root_pos : 0);
    }
    if (!hl_last && !did_hl_last && i < ops_last.size() &&
        ops.at(i) != ops_last.at(i)) {
      hl_last = did_hl_last = true;
    }

    if (i < root_pos)
      continue;

    char ch = '\0';

    if (compressed) {
      if (i == root_pos) {
        ch = '0' + ops.at(i);
      } else if (msg.worker_throw.at(i) == 0) {
        // skip
      } else if (msg.worker_throw.at(i) == config.h) {
        // skip
      } else {
        ch = '0' + ops.at(i);
      }
    } else {
      ch = '0' + ops.at(i);
    }

    if (ch != '\0') {
      if (hl_start) {
        buffer << '\x1B' << "[7m" << ch << '\x1B' << "[27m";
        hl_start = hl_last = false;
      } else if (hl_last) {
        buffer << '\x1B' << "[1m" << ch << '\x1B' << "[22m";
        hl_last = false;
      } else {
        buffer << ch;
      }
      ++printed;
    }

    if (printed >= status_width)
      break;
  }

  while (printed < status_width) {
    buffer << ' ';
    ++printed;
  }

  if (rootpos_distance < 1000u)
    buffer << std::setw(4) << rootpos_distance;
  else
    buffer << " ---";
  buffer << std::setw(5) << worker_longest.at(id);

  ops_last = ops;
  if (ops_start.size() == 0) {
    ops_start = ops;
  }

  return buffer.str();
}
