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
  for (int id = 0; id < context.num_threads; ++id) {
    if (config.verboseflag) {
      std::cout << "worker " << id << " starting..." << std::endl;
    }
    worker.push_back(new Worker(config, *this, id));
    worker_thread.push_back(new std::thread(&Worker::run, worker.at(id)));
    worker_rootpos.push_back(0);
    worker_longest.push_back(0);
    if (config.statusflag) {
      worker_status.push_back("     ");
      worker_start_state.push_back(0);
      worker_optionsleft_start.push_back({});
      worker_optionsleft_last.push_back({});
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

    if ((static_cast<int>(workers_idle.size()) == context.num_threads
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

  for (int id = 0; id < context.num_threads; ++id) {
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

void Coordinator::message_worker(const MessageC2W& msg, int worker_id) const {
  worker.at(worker_id)->inbox_lock.lock();
  worker.at(worker_id)->inbox.push(msg);
  worker.at(worker_id)->inbox_lock.unlock();
}

void Coordinator::collect_stats() {
  if (!config.statusflag || --stats_counter > 0)
    return;

  stats_counter = waits_per_status;
  stats_received = 0;
  for (int id = 0; id < context.num_threads; ++id) {
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
    int id = *it;
    workers_idle.erase(it);
    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();

    MessageC2W msg;
    msg.type = messages_C2W::DO_WORK;
    msg.assignment = wa;
    workers_run_order.push_back(id);
    worker_rootpos.at(id) = wa.root_pos;
    worker_longest.at(id) = 0;
    if (config.statusflag) {
      worker_optionsleft_start.at(id).resize(0);
      worker_optionsleft_last.at(id).resize(0);
    }
    message_worker(msg, id);

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
    } else if (msg.type == messages_W2C::WORKER_STATUS) {
      process_worker_status(msg);
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
  remove_from_run_order(msg.worker_id);
  workers_idle.insert(msg.worker_id);
  record_data_from_message(msg);
  worker_rootpos.at(msg.worker_id) = 0;
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

  // put splittee at the back of the run order list
  remove_from_run_order(msg.worker_id);
  workers_run_order.push_back(msg.worker_id);

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

void Coordinator::process_worker_status(const MessageW2C& msg) {
  if (msg.meta.size() > 0 && config.verboseflag) {
    erase_status_output();
    std::cout << msg.meta << '\n';
    print_status_output();
  }

  if (msg.root_pos >= 0) {
    worker_rootpos.at(msg.worker_id) = msg.root_pos;

    if (config.verboseflag) {
      erase_status_output();
      std::cout << "worker " << msg.worker_id
                << " new root_pos " << msg.root_pos
                << " on: " << current_time_string();
      print_status_output();
    }
  }
}

//------------------------------------------------------------------------------
// Steal work from one of the running workers
//------------------------------------------------------------------------------

// Identify a (not idle) worker to steal work from, and send it a SPLIT_WORK
// message. There is no single preferred way to identify the best target, so we
// implement several approaches.

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

    int id = 0;
    switch (context.steal_alg) {
      case 1:
        id = find_stealing_target_longestpattern();
        break;
      case 2:
        id = find_stealing_target_lowestid();
        break;
      case 3:
        id = find_stealing_target_lowestrootpos();
        break;
      case 4:
        id = find_stealing_target_longestruntime();
        break;
      default:
        assert(false);
    }
    assert(id >= 0 && id < context.num_threads);

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

int Coordinator::find_stealing_target_longestpattern() const {
  // strategy: take work from the worker with the longest patterns found
  int id_max = -1;
  int longest_max = -1;
  for (int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id) || is_worker_splitting(id))
      continue;
    if (longest_max < worker_longest.at(id)) {
      longest_max = worker_rootpos.at(id);
      id_max = id;
    }
  }
  assert(id_max != -1);
  return id_max;
}

int Coordinator::find_stealing_target_lowestid() const {
  // strategy: take work from lowest-id worker that's busy
  for (int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id) || is_worker_splitting(id))
      continue;
    return id;
  }
  assert(false);
  return 0;
}

int Coordinator::find_stealing_target_lowestrootpos() const {
  // strategy: take work from the worker with the lowest root_pos
  int id_min = -1;
  int root_pos_min = -1;
  for (int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id) || is_worker_splitting(id))
      continue;
    if (root_pos_min == -1 || root_pos_min > worker_rootpos.at(id)) {
      root_pos_min = worker_rootpos.at(id);
      id_min = id;
    }
  }
  assert(id_min != -1);
  return id_min;
}

int Coordinator::find_stealing_target_longestruntime() const {
  // strategy: take work from the worker running the longest
  return workers_run_order.front();
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

bool Coordinator::is_worker_idle(const int id) const {
  return (workers_idle.count(id) != 0);
}

bool Coordinator::is_worker_splitting(const int id) const {
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
  assert(msg.count.size() == l_max + 1);
  assert(context.count.size() <= msg.count.size());
  context.count.resize(msg.count.size(), 0);

  for (int i = 1; i < msg.count.size(); ++i) {
    context.count.at(i) += msg.count.at(i);
    context.ntotal += msg.count.at(i);
    if (i >= config.l_min && i <= l_max) {
      context.npatterns += msg.count.at(i);
    }
    if (msg.count.at(i) > 0) {
      worker_longest.at(msg.worker_id) = std::max<int>(
          worker_longest.at(msg.worker_id), i);
    }
  }
}

void Coordinator::remove_from_run_order(const int id) {
  // remove worker from workers_run_order
  std::list<int>::iterator iter = workers_run_order.begin();
  std::list<int>::iterator end = workers_run_order.end();
  bool found = false;

  while (iter != end) {
    if (*iter == id) {
      found = true;
      iter = workers_run_order.erase(iter);
    } else
      ++iter;
  }
  assert(found);
}

void Coordinator::stop_workers() {
  if (config.verboseflag)
    erase_status_output();
  for (int id = 0; id < context.num_threads; ++id) {
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
  int mode = 0;
  int max = 0;
  std::uint64_t modeval = 0;

  for (int i = 0; i < context.count.size(); ++i) {
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
  int xstart = std::max<int>(max - 10, mode);

  for (int i = xstart; i < context.count.size(); ++i) {
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
  if (l_max > config.l_min) {
    if (l_max == context.l_bound)
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

    if (config.countflag || l_max > config.l_min) {
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
  for (int i = 0; i < context.num_threads + 2; ++i) {
    std::cout << '\x1B' << "[1A"
              << '\x1B' << "[2K";
  }
}

void Coordinator::print_status_output() {
  if (!config.statusflag)
    return;

  const bool compressed = (config.mode == RunMode::NORMAL_SEARCH &&
      l_max > status_width);
  if (compressed)
    std::cout << "Status (compressed display) on: " << current_time_string();
  else
    std::cout << "Status on: " << current_time_string();
  std::cout << "sta/end  rp options remaining at position";
  for (int i = 29; i < status_width; ++i)
    std::cout << ' ';
  std::cout << "dist  len\n";
  for (int i = 0; i < context.num_threads; ++i)
    std::cout << worker_status.at(i) << std::endl;

  stats_printed = true;
}

std::string Coordinator::current_time_string() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_timet = std::chrono::system_clock::to_time_t(now);
  return std::ctime(&now_timet);
}

std::string Coordinator::make_worker_status(const MessageW2C& msg) {
  if (!msg.running)
    return "    idle";

  std::ostringstream buffer;
  const int id = msg.worker_id;
  const std::vector<int>& options = msg.worker_optionsleft;

  buffer << std::setw(3) << std::min(msg.start_state, 999) << '/';
  buffer << std::setw(3) << std::min(msg.end_state, 999) << ' ';
  buffer << std::setw(3) << std::min(worker_rootpos.at(id), 999) << ' ';

  const bool compressed = (config.mode == RunMode::NORMAL_SEARCH &&
      l_max > status_width);

  int printed = 0;
  bool have_highlighted_start = false;
  bool highlight_start = false;
  bool have_highlighted_last = false;
  bool highlight_last = false;
  int rootpos_distance = 999;

  int skipped = l_max;
  for (int i = 0; i < context.num_threads; ++i)
    skipped = std::min(skipped, worker_rootpos.at(i));

  for (int i = skipped; i < options.size(); ++i) {
    if (!highlight_start && !have_highlighted_start &&
        i < worker_optionsleft_start.at(id).size() &&
        options.at(i) != worker_optionsleft_start.at(id).at(i)) {
      highlight_start = have_highlighted_start = true;
      rootpos_distance = i - worker_rootpos.at(id);
    }
    if (!highlight_last && !have_highlighted_last &&
        i < worker_optionsleft_last.at(id).size() &&
        options.at(i) != worker_optionsleft_last.at(id).at(i)) {
      highlight_last = have_highlighted_last = true;
    }

    char ch = '\0';

    if (compressed) {
      if (i < worker_rootpos.at(id)) {
        ch = '*';
      } else if (i == worker_rootpos.at(id)) {
        ch = '0' + options.at(i);
      } else if (msg.worker_throw.at(i) == 0) {
        // skip
      } else if (msg.worker_throw.at(i) == config.h) {
        // skip
      } else {
        ch = '0' + options.at(i);
      }
    } else {
      if (i < worker_rootpos.at(id)) {
        ch = '*';
      } else {
        ch = '0' + options.at(i);
      }
    }

    if (ch != '\0') {
      if (highlight_start) {
        buffer << '\x1B' << "[7m" << ch << '\x1B' << "[27m";
        highlight_start = highlight_last = false;
      } else if (highlight_last) {
        buffer << '\x1B' << "[1m" << ch << '\x1B' << "[22m";
        highlight_last = false;
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

  buffer << std::setw(4) << std::max(0, rootpos_distance);
  buffer << std::setw(5) << worker_longest.at(id);

  worker_optionsleft_last.at(id) = msg.worker_optionsleft;
  if (worker_optionsleft_start.at(id).size() == 0 ||
      worker_start_state.at(id) != msg.start_state) {
    worker_optionsleft_start.at(id) = msg.worker_optionsleft;
    worker_start_state.at(id) = msg.start_state;
  }

  return buffer.str();
}
