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
#include <thread>
#include <chrono>
#include <algorithm>
#include <csignal>
#include <cassert>


Coordinator::Coordinator(const SearchConfig& a, SearchContext& b)
    : config(a), context(b) {
  worker.reserve(context.num_threads);
  worker_thread.reserve(context.num_threads);
  worker_rootpos.reserve(context.num_threads);
  worker_longest.reserve(context.num_threads);
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run() {
  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  // start worker threads
  for (int id = 0; id < context.num_threads; ++id) {
    worker[id] = new Worker(config, *this, id);
    worker_thread[id] = new std::thread(&Worker::run, worker[id]);
    workers_idle.insert(id);
    worker_rootpos[id] = 0;
    worker_longest[id] = 0;
  }

  // check the inbox 10x more frequently than workers
  constexpr auto nanosecs_wait = std::chrono::nanoseconds(
      static_cast<long>(100000000 * Worker::secs_per_inbox_check_target));

  timespec start_ts, end_ts;
  (void)timespec_get(&start_ts, TIME_UTC);

  while (true) {
    give_assignments();
    steal_work();
    process_inbox();
    print_stats();

    if ((static_cast<int>(workers_idle.size()) == context.num_threads
          && context.assignments.size() == 0) || Coordinator::stopping) {
      stop_workers();
      // any worker that was running will have sent back a RETURN_WORK message
      process_inbox();
      break;
    }

    std::this_thread::sleep_for(nanosecs_wait);
  }

  (void)timespec_get(&end_ts, TIME_UTC);
  double runtime =
      static_cast<double>(end_ts.tv_sec - start_ts.tv_sec) +
      1.0e-9 * (end_ts.tv_nsec - start_ts.tv_nsec);
  context.secs_elapsed += runtime;
  context.secs_available += runtime * context.num_threads;

  for (int id = 0; id < context.num_threads; ++id) {
    delete worker[id];
    delete worker_thread[id];
    worker[id] = nullptr;
    worker_thread[id] = nullptr;
  }

  if (context.assignments.size() > 0)
    std::cout << "\nPARTIAL RESULTS:\n";
  print_summary();
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

void Coordinator::message_worker(const MessageC2W& msg, int worker_id) const {
  worker[worker_id]->inbox_lock.lock();
  worker[worker_id]->inbox.push(msg);
  worker[worker_id]->inbox_lock.unlock();
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
    msg.l_current = context.l_current;
    workers_run_order.push_back(id);
    worker_rootpos[id] = wa.root_pos;
    worker_longest[id] = 0;
    message_worker(msg, id);

    if (config.verboseflag) {
      std::cout << "worker " << id << " given work ("
                << workers_idle.size() << " idle):\n"
                << "  " << msg.assignment << std::endl;
    }
  }
}

// Receive and handle messages from the worker threads.

void Coordinator::process_inbox() {
  int new_longest_pattern_from_id = -1;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_W2C::SEARCH_RESULT) {
      new_longest_pattern_from_id = process_search_result(msg);
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

  if (new_longest_pattern_from_id >= 0)
    notify_metadata(new_longest_pattern_from_id);
}

int Coordinator::process_search_result(const MessageW2C& msg) {
  int new_longest_pattern_from_id = -1;

  if (!config.longestflag) {
    print_pattern(msg);
    ++context.npatterns;
  } else if (msg.length > context.l_current) {
    context.patterns.clear();
    print_pattern(msg);
    context.l_current = msg.length;
    context.npatterns = 1;
    new_longest_pattern_from_id = msg.worker_id;
  } else if (msg.length == context.l_current) {
    print_pattern(msg);
    ++context.npatterns;
  }
  // ignore patterns shorter than current length if `longestflag`==true

  return new_longest_pattern_from_id;
}

bool Coordinator::is_worker_idle(const int id) const {
  return (workers_idle.count(id) != 0);
}

bool Coordinator::is_worker_splitting(const int id) const {
  return (workers_splitting.count(id) != 0);
}

void Coordinator::process_worker_idle(const MessageW2C& msg) {
  remove_from_run_order(msg.worker_id);
  workers_idle.insert(msg.worker_id);

  collect_stats(msg);
  worker_rootpos[msg.worker_id] = 0;
  worker_longest[msg.worker_id] = 0;

  if (config.verboseflag) {
    std::cout << "worker " << msg.worker_id << " went idle ("
              << workers_idle.size() << " idle)";
    if (workers_splitting.count(msg.worker_id) > 0) {
      std::cout << ", removed from splitting queue ("
                << (workers_splitting.size() - 1) << " splitting)";
    }
    std::cout << std::endl;
  }

  // If we have a SPLIT_WORK request out for the worker, it will be ignored.
  // Remove it from the list of workers we're expecting to return work.
  workers_splitting.erase(msg.worker_id);
}

void Coordinator::process_returned_work(const MessageW2C& msg) {
  workers_splitting.erase(msg.worker_id);
  context.assignments.push_back(msg.assignment);
  collect_stats(msg);

  // put splittee at the back of the run order list
  remove_from_run_order(msg.worker_id);
  workers_run_order.push_back(msg.worker_id);

  if (config.verboseflag) {
    std::cout << "worker " << msg.worker_id << " returned work:" << std::endl
              << "  " << msg.assignment << std::endl;
  }
}

bool doubles_are_close(double a, double b) {
  double epsilon = 1e-3;
  return (b > a - epsilon && b < a + epsilon);
}

void Coordinator::process_returned_stats(const MessageW2C& msg) {
  collect_stats(msg);
  if (++stats_received < context.num_threads)
    return;

  int mode = 0;
  int max = 0;
  unsigned long modeval = 0L;

  for (int i = 0; i < count.size(); ++i) {
    if (count[i] > modeval) {
      mode = i;
      modeval = count[i];
    }
    if (count[i] > 0)
      max = i;
  }

  double s1 = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  double sy = 0, sxy = 0, sx2y = 0;
  int xstart = std::max<int>(max - 10, mode);

  for (int i = xstart; i < count.size(); ++i) {
    if (count[i] < 5)
      continue;

    const double x = static_cast<double>(i);
    const double y = log(static_cast<double>(count[i]));
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

  assert(doubles_are_close(M11 * sx4 + M12 * sx3 + M13 * sx2, 1));
  assert(doubles_are_close(M11 * sx3 + M12 * sx2 + M13 * sx, 0));
  assert(doubles_are_close(M11 * sx2 + M12 * sx + M13 * s1, 0));
  assert(doubles_are_close(M21 * sx4 + M22 * sx3 + M23 * sx2, 0));
  assert(doubles_are_close(M21 * sx3 + M22 * sx2 + M23 * sx, 1));
  assert(doubles_are_close(M21 * sx2 + M22 * sx + M23 * s1, 0));
  assert(doubles_are_close(M31 * sx4 + M32 * sx3 + M33 * sx2, 0));
  assert(doubles_are_close(M31 * sx3 + M32 * sx2 + M33 * sx, 0));
  assert(doubles_are_close(M31 * sx2 + M32 * sx + M33 * s1, 1));

  double A = M11 * sx2y + M12 * sxy + M13 * sy;
  double B = M21 * sx2y + M22 * sxy + M23 * sy;
  double C = M31 * sx2y + M32 * sxy + M33 * sy;

  // evaluate the expected number of patterns found at x = maxlength
  double x = static_cast<double>(context.maxlength);
  double lny = A * x * x + B * x + C;
  double completion = exp(lny);

  std::cout << "patterns: " << context.ntotal
            << " (mode " << mode
            << ", max " << max
            << ", lim " << context.maxlength << ")"
            << " completion " << completion << std::endl;
}

void Coordinator::collect_stats(const MessageW2C& msg) {
  context.ntotal += msg.ntotal;
  if (count.size() != msg.count.size())
    count.assign(msg.count.size(), 0L);
  for (int i = 0; i < msg.count.size(); ++i)
    count[i] += msg.count[i];
  context.nnodes += msg.nnodes;
  context.numstates = msg.numstates;
  context.numcycles = msg.numcycles;
  context.numshortcycles = msg.numshortcycles;
  context.maxlength = msg.maxlength;
  context.secs_working += msg.secs_working;
}

void Coordinator::process_worker_status(const MessageW2C& msg) {
  if (msg.meta.size() > 0 && config.verboseflag)
      std::cout << msg.meta << std::endl;

  if (msg.root_pos >= 0) {
    worker_rootpos[msg.worker_id] = msg.root_pos;

    if (config.verboseflag) {
      std::cout << "worker " << msg.worker_id
                << " new root_pos: " << msg.root_pos << std::endl;
    }
  }

  if (msg.longest_found >= 0) {
    worker_longest[msg.worker_id] = msg.longest_found;

    if (config.verboseflag) {
      std::cout << "worker " << msg.worker_id
                << " new longest_found: " << msg.longest_found << std::endl;
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

void Coordinator::notify_metadata(int skip_id) const {
  for (int id = 0; id < context.num_threads; ++id) {
    if (id == skip_id || is_worker_idle(id))
      continue;

    MessageC2W msg;
    msg.type = messages_C2W::UPDATE_METADATA;
    msg.l_current = context.l_current;
    message_worker(msg, id);

    if (config.verboseflag) {
      std::cout << "worker " << id << " notified of new length "
                << context.l_current << std::endl;
    }
  }
}

void Coordinator::stop_workers() const {
  for (int id = 0; id < context.num_threads; ++id) {
    MessageC2W msg;
    msg.type = messages_C2W::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << "worker " << id << " asked to stop" << std::endl;
    worker_thread[id]->join();
  }
}

void Coordinator::print_stats() {
  const int counts_until_print = waits_per_second * 60;
  if (++stats_counter < counts_until_print)
    return;

  stats_counter = 0;
  stats_received = 0;
  for (int id = 0; id < context.num_threads; ++id) {
    MessageC2W msg;
    msg.type = messages_C2W::SEND_STATS;
    message_worker(msg, id);
  }
}

// static variable for indicating the user has interrupted execution
bool Coordinator::stopping = false;

void Coordinator::signal_handler(int signum) {
  stopping = true;
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
        std::cout << "could not steal work (" << workers_idle.size()
                  << " idle)" << std::endl;
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
      std::cout << "worker " << id << " given work split request ("
                << workers_splitting.size() << " splitting)" << std::endl;
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
    if (longest_max < worker_longest[id]) {
      longest_max = worker_rootpos[id];
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
    if (root_pos_min == -1 || root_pos_min > worker_rootpos[id]) {
      root_pos_min = worker_rootpos[id];
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
// Handle output
//------------------------------------------------------------------------------

void Coordinator::print_pattern(const MessageW2C& msg) {
  if (config.printflag) {
    if (config.verboseflag)
      std::cout << msg.worker_id << ": " << msg.pattern << std::endl;
    else
      std::cout << msg.pattern << std::endl;
  }
  context.patterns.push_back(msg.pattern);
}

void Coordinator::print_summary() const {
  std::cout << "balls: " << (config.dualflag ? config.h - config.n : config.n)
            << ", max throw: " << config.h << '\n';

  std::cout << "graph: " << context.numstates << " states, "
            << context.numcycles << " shift cycles, "
            << context.numshortcycles << " short cycles\n";

  switch (config.mode) {
    case RunMode::BLOCK_SEARCH:
      std::cout << "block mode, " << config.skiplimit << " skips allowed\n";
      break;
    case RunMode::SUPER_SEARCH:
      std::cout << "super mode, " << config.shiftlimit << " shifts allowed";
      if (config.invertflag)
        std::cout << ", inverse output";
      std::cout << '\n';
      break;
    default:
      break;
  }

  if (config.longestflag) {
    std::cout << "pattern length: " << context.l_current
              << " throws (" << context.maxlength << " maximum)\n";
  }

  std::cout << context.npatterns << " patterns found ("
            << context.ntotal << " seen, "
            << context.nnodes << " nodes, "
            << std::fixed << std::setprecision(2)
            << (static_cast<double>(context.nnodes) / context.secs_elapsed /
                1000000)
            << "M nodes/sec)\n";

  if (config.groundmode == 1)
    std::cout << "ground state search\n";
  if (config.groundmode == 2)
    std::cout << "excited state search\n";

  std::cout << "running time = "
            << std::fixed << std::setprecision(4)
            << context.secs_elapsed << " sec";
  if (context.num_threads > 1) {
    std::cout << " (worker util = " << std::setprecision(2)
              << ((context.secs_working / context.secs_available) * 100)
              << " %)";
  }
  std::cout << std::endl;

  std::cout << std::endl << "FROM COORDINATOR:" << std::endl;

  for (int i = 1; i <= context.maxlength; ++i) {
    std::cout << i << ", " << count[i] << std::endl;
  }
}
