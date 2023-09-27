//
// Coordinator.cc
//
// Coordinator thread that manages the overall search.
//
// The computation is depth first search on multiple worker threads using work
// stealing to keep the workers busy. The business of the coordinator is to
// interact with the workers to distribute work, and also to manage output.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
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
    worker[id] = new Worker(config, this, id);
    worker_thread[id] = new std::thread(&Worker::run, worker[id]);
    workers_idle.push_back(id);
    worker_rootpos[id] = 0;
    worker_longest[id] = 0;
  }

  // check the inbox 10x more frequently than workers
  constexpr auto nanosecs_wait = std::chrono::nanoseconds(
      static_cast<long>(100000000 * Worker::secs_per_inbox_check_target));

  timespec start_ts, end_ts;
  timespec_get(&start_ts, TIME_UTC);

  while (true) {
    give_assignments();
    steal_work();
    process_inbox();

    if (((int)workers_idle.size() == context.num_threads
          && context.assignments.size() == 0) || Coordinator::stopping) {
      stop_workers();
      // any worker that was running will have sent back a RETURN_WORK message
      process_inbox();
      break;
    }

    std::this_thread::sleep_for(nanosecs_wait);
  }

  timespec_get(&end_ts, TIME_UTC);
  double runtime =
      (static_cast<double>(end_ts.tv_sec) + 1.0e-9 * end_ts.tv_nsec) -
      (static_cast<double>(start_ts.tv_sec) + 1.0e-9 * start_ts.tv_nsec);
  context.secs_elapsed += runtime;
  context.secs_available += runtime * context.num_threads;

  for (int id = 0; id < context.num_threads; ++id) {
    delete worker[id];
    delete worker_thread[id];
    worker[id] = nullptr;
    worker_thread[id] = nullptr;
  }

  if (context.assignments.size() > 0)
    std::cout << std::endl << "PARTIAL RESULTS:" << std::endl;
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
    int id = workers_idle.front();
    workers_idle.pop_front();
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
      std::cout << "gave work to worker " << id << ":" << std::endl
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
  return std::find(workers_idle.begin(), workers_idle.end(), id)
      != workers_idle.end();
}

void Coordinator::process_worker_idle(const MessageW2C& msg) {
  remove_from_run_order(msg.worker_id);
  workers_idle.push_back(msg.worker_id);

  // collect statistics reported by the worker
  context.ntotal += msg.ntotal;
  context.nnodes += msg.nnodes;
  context.numstates = msg.numstates;
  context.maxlength = msg.maxlength;
  context.secs_working += msg.secs_working;
  worker_rootpos[msg.worker_id] = 0;
  worker_longest[msg.worker_id] = 0;

  // If worker went idle before it could return a work assignment, the
  // SPLIT_WORK message will be held in the worker's inbox until it becomes
  // active again. In any case we don't want to block on it because that may
  // deadlock the program.
  if (msg.worker_id == waiting_for_work_from_id)
    waiting_for_work_from_id = -1;

  if (config.verboseflag)
    std::cout << "worker " << msg.worker_id << " went idle" << std::endl;
}

void Coordinator::process_returned_work(const MessageW2C& msg) {
  if (msg.worker_id == waiting_for_work_from_id)
    waiting_for_work_from_id = -1;
  context.assignments.push_back(msg.assignment);

  // put splittee at the back of the run order list
  remove_from_run_order(msg.worker_id);
  workers_run_order.push_back(msg.worker_id);

  context.ntotal += msg.ntotal;
  context.nnodes += msg.nnodes;
  context.numstates = msg.numstates;
  context.maxlength = msg.maxlength;
  context.secs_working += msg.secs_working;

  if (config.verboseflag) {
    std::cout << "worker " << msg.worker_id << " returned work:" << std::endl
              << "  " << msg.assignment << std::endl;
  }
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
  if (waiting_for_work_from_id != -1 || workers_idle.size() == 0)
    return;

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

  waiting_for_work_from_id = id;
  MessageC2W msg;
  msg.type = messages_C2W::SPLIT_WORK;
  msg.split_alg = context.split_alg;
  message_worker(msg, id);

  if (config.verboseflag)
    std::cout << "requested work from worker " << id << std::endl;
}

int Coordinator::find_stealing_target_longestpattern() const {
  // strategy: take work from the worker with the longest patterns found
  int id_max = -1;
  int longest_max = -1;
  for (int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id))
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
    if (is_worker_idle(id))
      continue;
    return id;
  }
  assert(false);
}

int Coordinator::find_stealing_target_lowestrootpos() const {
  // strategy: take work from the worker with the lowest root_pos
  int id_min = -1;
  int root_pos_min = -1;
  for (int id = 0; id < context.num_threads; ++id) {
    if (is_worker_idle(id))
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
            << ", max throw: " << config.h << std::endl;

  switch (config.mode) {
    case RunMode::BLOCK_SEARCH:
      std::cout << "block mode, " << config.skiplimit << " skips allowed"
                << std::endl;
      break;
    case RunMode::SUPER_SEARCH:
      std::cout << "super mode, " << config.shiftlimit << " shifts allowed";
      if (config.invertflag)
        std::cout << ", inverse output";
      std::cout << std::endl;
      break;
    default:
      break;
  }

  if (config.longestflag) {
    std::cout << "pattern length: " << context.l_current
              << " throws (" << context.maxlength << " maximum, "
              << context.numstates << " states)" << std::endl;
  }

  std::cout << context.npatterns << " patterns found (" << context.ntotal
            << " seen, " << context.nnodes << " nodes, "
            << std::fixed << std::setprecision(2)
            << (static_cast<double>(context.nnodes) / context.secs_elapsed /
                1000000)
            << "M nodes/sec)" << std::endl;

  if (config.groundmode == 1)
    std::cout << "ground state search" << std::endl;
  if (config.groundmode == 2)
    std::cout << "excited state search" << std::endl;

  std::cout << "running time = "
            << std::fixed << std::setprecision(4)
            << context.secs_elapsed << " sec";
  if (context.num_threads > 1) {
    std::cout << " (worker util = " << std::setprecision(2)
              << ((context.secs_working / context.secs_available) * 100)
              << " %)";
  }
  std::cout << std::endl;
}
