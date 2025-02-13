//
// CoordinatorCPU.cc
//
// Routines for executing the search on a CPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
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


// Run the search on the CPU, using one or more worker threads.

void Coordinator::run_cpu() {
  constexpr auto NANOSECS_WAIT = std::chrono::nanoseconds(
    static_cast<long>(NANOSECS_PER_INBOX_CHECK));
  start_workers();

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
  workers_splitting.clear();
  process_inbox();  // running worker will have sent back a RETURN_WORK message
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

// Deliver a message to a given worker's inbox.

void Coordinator::message_worker(const MessageC2W& msg,
    unsigned worker_id) const {
  std::unique_lock<std::mutex> lck(worker.at(worker_id)->inbox_lock);
  worker.at(worker_id)->inbox.push(msg);
}

// Give assignments to workers, while there are available assignments and idle
// workers to take them.

void Coordinator::give_assignments() {
  while (workers_idle.size() > 0 && context.assignments.size() > 0) {
    auto iter = workers_idle.begin();
    auto id = *iter;
    workers_idle.erase(iter);
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
      jpout << std::format("worker {} given work ({} idle):\n  ", id,
                 workers_idle.size())
            << msg.assignment << std::endl;
      print_status_output();
    }
  }
}

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
        jpout << std::format("could not steal work ({} idle)",
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
      jpout << std::format(
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

// Receive and handle messages from the worker threads.

void Coordinator::process_inbox() {
  std::unique_lock<std::mutex> lck(inbox_lock);
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
    jpout << std::format("worker {} went idle ({} idle)", msg.worker_id,
               workers_idle.size());
    if (workers_splitting.count(msg.worker_id) > 0) {
      jpout << std::format(", removed from splitting queue ({} splitting)",
                 (workers_splitting.size() - 1));
    }
    jpout << " on: " << current_time_string();
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
  if (workers_splitting.count(msg.worker_id) > 0) {
    ++context.splits_total;
    workers_splitting.erase(msg.worker_id);
  }
  context.assignments.push_back(msg.assignment);
  record_data_from_message(msg);

  if (config.verboseflag) {
    erase_status_output();
    jpout << std::format("worker {} returned work:\n  ", msg.worker_id)
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
      jpout << msg.meta << '\n';
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
    jpout << "worker " << msg.worker_id;
    if (startstate_changed) {
      jpout << " new start_state " << msg.start_state;
      comma = true;
    }
    if (endstate_changed) {
      if (comma) {
        jpout << ',';
      }
      jpout << " new end_state " << msg.end_state;
      comma = true;
    }
    if (rootpos_changed) {
      if (comma) {
        jpout << ',';
      }
      jpout << " new root_pos " << msg.root_pos;
    }
    jpout << " on: " << current_time_string();
    print_status_output();
  }
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// Start all of the worker threads into a ready state, and initialize data
// structures for tracking them.

void Coordinator::start_workers() {
  if (config.verboseflag) {
    jpout << "Started on: " << current_time_string();
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (config.verboseflag) {
      jpout << std::format("worker {} starting...", id) << std::endl;
    }

    worker.push_back(std::make_unique<Worker>(config, *this, id, n_max));
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
  if (config.verboseflag) {
    erase_status_output();
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    MessageC2W msg;
    msg.type = MessageC2W::Type::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag) {
      jpout << std::format("worker {} asked to stop", id) << std::endl;
    }

    worker_thread.at(id)->join();
  }

  if (config.verboseflag) {
    print_status_output();
  }
}

bool Coordinator::is_worker_idle(const unsigned id) const {
  return (workers_idle.count(id) != 0);
}

bool Coordinator::is_worker_splitting(const unsigned id) const {
  return (workers_splitting.count(id) != 0);
}

