//
// CoordinatorCPU.cc
//
// Coordinator that executes the search on one or more CPU threads.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "CoordinatorCPU.h"

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


CoordinatorCPU::CoordinatorCPU(SearchConfig& a, SearchContext& b,
    std::ostream& c)
    : Coordinator(a, b, c)
{}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void CoordinatorCPU::run_search()
{
  constexpr auto NANOSECS_WAIT = std::chrono::nanoseconds(
    static_cast<long>(NANOSECS_PER_INBOX_CHECK));
  last_status_time = std::chrono::high_resolution_clock::now();
  last_nnodes = context.nnodes;
  last_ntotal = context.ntotal;
  start_workers();

  while (true) {
    process_inbox();
    give_assignments();
    steal_work();
    collect_status();

    if (Coordinator::stopping || (workers_idle.size() == config.num_threads &&
          context.assignments.empty())) {
      break;
    }

    std::this_thread::sleep_for(NANOSECS_WAIT);
  }

  stop_workers();
  workers_splitting.clear();  // don't count returned work as splits
  process_inbox();  // running worker will have sent back a RETURN_WORK message
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

// Deliver a message to a given worker's inbox.

void CoordinatorCPU::message_worker(const MessageC2W& msg,
    unsigned worker_id) const
{
  std::unique_lock<std::mutex> lck(worker.at(worker_id)->inbox_lock);
  worker.at(worker_id)->inbox.push(msg);
}

// Give assignments to workers, while there are available assignments and idle
// workers to take them.

void CoordinatorCPU::give_assignments()
{
  while (!workers_idle.empty() && !context.assignments.empty()) {
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
      worker_options_left_current.at(id).resize(0);
      longest_by_worker_ever.at(id) = 0;
      longest_by_worker_current.at(id) = 0;
    }

    if (config.verboseflag) {
      std::ostringstream buffer;
      buffer << std::format("worker {} given work ({} idle):\n  ", id,
                 workers_idle.size())
             << msg.assignment;
      print_string(buffer.str());
    }
  }
}

// Identify a (not idle) worker to steal work from, and send it a SPLIT_WORK
// message.

void CoordinatorCPU::steal_work()
{
  bool sent_split_request = false;

  while (workers_idle.size() > workers_splitting.size()) {
    // when all of the workers are either idle or queued for splitting, there
    // are no active workers to take work from
    if (workers_idle.size() + workers_splitting.size() == config.num_threads) {
      if (config.verboseflag && sent_split_request) {
        print_string(std::format("could not steal work ({} idle)",
            workers_idle.size()));
      }
      break;
    }

    unsigned id = -1u;
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
      print_string(std::format(
        "worker {} given work split request ({} splitting)", id,
        workers_splitting.size()));
    }
  }
}

// Return the id of the busy worker with the most remaining work.
//
// First look at most remaining `start_state` values, and if no workers have
// unexplored start states then find the lowest `root_pos` value.

unsigned CoordinatorCPU::find_stealing_target_mostremaining() const
{
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

void CoordinatorCPU::collect_status()
{
  if (!config.statusflag)
    return;
  const auto now = std::chrono::high_resolution_clock::now();
  if (calc_duration_secs(last_status_time, now) < SECS_PER_STATUS)
    return;

  status_interval = calc_duration_secs(last_status_time, now);
  last_status_time = now;

  status_lines.resize(config.num_threads + 3);
  status_lines.at(0) = "Status on: " + current_time_string();
  std::ostringstream buffer;
  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && n_max > 3 * STATUS_WIDTH);
  buffer << " cur/ end  rp options remaining at position";
  if (compressed) {
    buffer << " (compressed view)";
    for (int i = 47; i < STATUS_WIDTH; ++i) {
      buffer << ' ';
    }
  } else {
    for (int i = 29; i < STATUS_WIDTH; ++i) {
      buffer << ' ';
    }
  }
  buffer << "    period";
  status_lines.at(1) = buffer.str();

  stats_received = 0;
  for (unsigned id = 0; id < config.num_threads; ++id) {
    status_lines.at(id + 2) = "   ?/   ?   ? ?";  // should never display
    MessageC2W msg;
    msg.type = MessageC2W::Type::SEND_STATS;
    message_worker(msg, id);
  }
}

// Receive and handle messages from the worker threads.

void CoordinatorCPU::process_inbox()
{
  std::unique_lock<std::mutex> lck(inbox_lock);
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == MessageW2C::Type::SEARCH_RESULT) {
      process_search_result(msg.pattern);
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

void CoordinatorCPU::process_worker_idle(const MessageW2C& msg)
{
  workers_idle.insert(msg.worker_id);
  record_data_from_message(msg);
  worker_rootpos.at(msg.worker_id) = 0;
  if (config.statusflag) {
    longest_by_worker_ever.at(msg.worker_id) = 0;
    longest_by_worker_current.at(msg.worker_id) = 0;
  }

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << std::format("worker {} went idle ({} idle)", msg.worker_id,
               workers_idle.size());
    if (workers_splitting.count(msg.worker_id) > 0) {
      buffer << std::format(", removed from splitting queue ({} splitting)",
                 (workers_splitting.size() - 1));
    }
    buffer << " on: " << current_time_string();
    print_string(buffer.str());
  }

  // If we have a SPLIT_WORK request out for the worker, it will be ignored.
  // Remove it from the list of workers we're expecting to return work.
  workers_splitting.erase(msg.worker_id);
}

// Handle a work assignment sent back from a worker.
//
// This happens in two contexts: (a) when the worker is responding to a
// SPLIT_WORK request, and (b) when the worker is notified to quit.

void CoordinatorCPU::process_returned_work(const MessageW2C& msg)
{
  if (workers_splitting.count(msg.worker_id) > 0) {
    ++context.splits_total;
    workers_splitting.erase(msg.worker_id);
  }
  context.assignments.push_back(msg.assignment);
  record_data_from_message(msg);

  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << std::format("worker {} returned work:\n  ", msg.worker_id)
           << msg.assignment;
    print_string(buffer.str());
  }
}

// Handle a worker's response to a SEND_STATS message, for doing the live status
// tracker (`-status` option).
//
// We create a status string for each worker as their stats return, and once all
// workers have responded we print it.

void CoordinatorCPU::process_returned_stats(const MessageW2C& msg)
{
  record_data_from_message(msg);
  if (!config.statusflag)
    return;

  status_lines.at(msg.worker_id + 2) = make_worker_status(msg);

  assert(stats_received < config.num_threads);
  if (++stats_received != config.num_threads)
    return;

  // add bottom line to status output
  auto format2 = [](double a) {
    if (a < 1 || a > 9999000000000) {
      return std::string("-----");
    }
    if (a < 99999) {
      auto result = std::format("{:5g}", a);
      return result.substr(0, 5);
    } else if (a < 1000000) {
      auto result = std::format("{:4g}", a / 1000);
      return result.substr(0, 4) + "K";
    } else if (a < 1000000000) {
      auto result = std::format("{:4g}", a / 1000000);
      return result.substr(0, 4) + "M";
    } else {
      auto result = std::format("{:4g}", a / 1000000000);
      return result.substr(0, 4) + "B";
    }
  };

  const double nodespersec =
      static_cast<double>(context.nnodes - last_nnodes) / status_interval;
  const double patspersec =
      static_cast<double>(context.ntotal - last_ntotal) / status_interval;
  last_nnodes = context.nnodes;
  last_ntotal = context.ntotal;

  status_lines.at(config.num_threads + 2) = std::format(
    "jobs:{:8}, nodes/s: {}, pats/s: {}, pats in range:{:19}",
    config.num_threads - workers_idle.size() + context.assignments.size(),
    format2(nodespersec),
    format2(patspersec),
    context.npatterns
  );

  erase_status_output();
  print_status_output();
}

// Handle an update from a worker on the state of its search.
//
// There are two types of updates: (a) informational text updates, which are
// printed in `-verbose` mode, and (b) updates to `start_state`, `end_state`,
// and `root_pos`, which are used by the coordinator when it needs to select a
// worker to send a SPLIT_WORK request to.

void CoordinatorCPU::process_worker_update(const MessageW2C& msg)
{
  if (!msg.meta.empty()) {
    if (config.verboseflag) {
      print_string(msg.meta);
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
      worker_options_left_current.at(msg.worker_id).resize(0);
      longest_by_worker_ever.at(msg.worker_id) = 0;
      longest_by_worker_current.at(msg.worker_id) = 0;
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
    bool comma = false;
    std::ostringstream buffer;
    buffer << "worker " << msg.worker_id;
    if (startstate_changed) {
      buffer << " new start_state " << msg.start_state;
      comma = true;
    }
    if (endstate_changed) {
      if (comma) {
        buffer << ',';
      }
      buffer << " new end_state " << msg.end_state;
      comma = true;
    }
    if (rootpos_changed) {
      if (comma) {
        buffer << ',';
      }
      buffer << " new root_pos " << msg.root_pos;
    }
    buffer << " on: " << current_time_string();
    print_string(buffer.str());
  }
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// Start all of the worker threads into a ready state, and initialize data
// structures for tracking them.

void CoordinatorCPU::start_workers()
{
  if (config.verboseflag) {
    std::ostringstream buffer;
    buffer << "Started on: " << current_time_string();
    print_string(buffer.str());
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (config.verboseflag) {
      print_string(std::format("worker {} starting...", id));
    }

    worker.push_back(std::make_unique<Worker>(config, *this, graph, id, n_max));
    worker_thread.push_back(
        std::make_unique<std::thread>(&Worker::run, worker.at(id).get()));
    worker_startstate.push_back(0);
    worker_endstate.push_back(0);
    worker_rootpos.push_back(0);
    if (config.statusflag) {
      worker_options_left_start.push_back({});
      worker_options_left_current.push_back({});
      longest_by_worker_ever.push_back(0);
      longest_by_worker_current.push_back(0);
    }
    workers_idle.insert(id);
  }
}

// Stop all workers.

void CoordinatorCPU::stop_workers()
{
  if (config.verboseflag) {
    erase_status_output();
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    MessageC2W msg;
    msg.type = MessageC2W::Type::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag) {
      print_string(std::format("worker {} asked to stop", id));
    }

    worker_thread.at(id)->join();
  }

  if (config.verboseflag) {
    print_status_output();
  }
}

bool CoordinatorCPU::is_worker_idle(const unsigned id) const
{
  return (workers_idle.count(id) != 0);
}

bool CoordinatorCPU::is_worker_splitting(const unsigned id) const
{
  return (workers_splitting.count(id) != 0);
}

//------------------------------------------------------------------------------
// Manage worker status
//------------------------------------------------------------------------------

// Copy status data out of the worker message, into appropriate data structures
// in the coordinator.

void CoordinatorCPU::record_data_from_message(const MessageW2C& msg)
{
  context.nnodes += msg.nnodes;
  context.secs_working += msg.secs_working;

  // pattern counts by period
  assert(msg.count.size() == n_max + 1);
  assert(context.count.size() == n_max + 1);

  for (size_t i = 1; i < msg.count.size(); ++i) {
    context.count.at(i) += msg.count.at(i);
    context.ntotal += msg.count.at(i);
    if (i >= config.n_min && i <= n_max) {
      context.npatterns += msg.count.at(i);
    }
    if (config.statusflag && msg.count.at(i) > 0) {
      longest_by_worker_ever.at(msg.worker_id) = std::max(
        longest_by_worker_ever.at(msg.worker_id), static_cast<unsigned>(i));
      longest_by_worker_current.at(msg.worker_id) = std::max(
        longest_by_worker_current.at(msg.worker_id), static_cast<unsigned>(i));
    }
  }
}

// Create a status display for a worker, showing its current state in the
// search.

std::string CoordinatorCPU::make_worker_status(const MessageW2C& msg)
{
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
  std::vector<unsigned>& ops_last = worker_options_left_current.at(id);

  buffer << std::setw(4) << std::min(worker_startstate.at(id), 9999u) << '/';
  buffer << std::setw(4) << std::min(worker_endstate.at(id), 9999u) << ' ';
  buffer << std::setw(3) << std::min(worker_rootpos.at(id), 999u) << ' ';

  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && n_max > 3 * STATUS_WIDTH);
  const bool show_deadstates =
      (config.mode == SearchConfig::RunMode::NORMAL_SEARCH &&
      config.graphmode == SearchConfig::GraphMode::FULL_GRAPH);
  const bool show_shifts = (config.mode == SearchConfig::RunMode::SUPER_SEARCH
      && config.shiftlimit != -1u);

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
        ch = '0' + static_cast<char>(ops.at(i));
      } else if (throwval == 0 || throwval == config.h) {
        // skip
      } else {
        ch = '0' + static_cast<char>(ops.at(i));
      }
    } else {
      ch = '0' + static_cast<char>(ops.at(i));
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

  buffer << std::setw(5) << longest_by_worker_current.at(id)
         << std::setw(5) << longest_by_worker_ever.at(id);

  ops_last = ops;
  if (ops_start.size() == 0) {
    ops_start = ops;
  }
  longest_by_worker_current.at(id) = 0;

  return buffer.str();
}
