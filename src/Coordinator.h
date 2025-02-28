//
// Coordinator.h
//
// Coordinator that manages the overall search.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATOR_H_
#define JPRIME_COORDINATOR_H_

#include "Messages.h"
#include "SearchConfig.h"
#include "SearchContext.h"
#include "Worker.h"

#include <queue>
#include <mutex>
#include <list>
#include <set>
#include <vector>
#include <string>
#include <thread>
#include <memory>
#include <csignal>


class Coordinator {
 public:
  Coordinator(const SearchConfig& config, SearchContext& context,
      std::ostream& jpout);
  Coordinator() = delete;

 public:
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

 private:
  const SearchConfig& config;
  SearchContext& context;
  std::ostream& jpout;  // all console output goes here
  unsigned n_max = 0;  // max pattern period to find

  // workers
  std::vector<std::unique_ptr<Worker>> worker;
  std::vector<std::unique_ptr<std::thread>> worker_thread;
  std::set<unsigned> workers_idle;
  std::set<unsigned> workers_splitting;
  std::vector<unsigned> worker_startstate;
  std::vector<unsigned> worker_endstate;
  std::vector<unsigned> worker_rootpos;

  static volatile sig_atomic_t stopping;
  static constexpr unsigned MAX_STATES = 1000000u;  // memory limit

  // check inbox 10x more often than workers do
  static constexpr double NANOSECS_PER_INBOX_CHECK =
      1e8 * Worker::SECS_PER_INBOX_CHECK_TARGET;

  // live status display
  static constexpr double SECS_PER_STATUS = 1;
  static constexpr int WAITS_PER_STATUS = static_cast<int>(1e9 *
      SECS_PER_STATUS / NANOSECS_PER_INBOX_CHECK);
  static constexpr int STATUS_WIDTH = 55;
  unsigned stats_counter = 0;
  unsigned stats_received = 0;
  bool stats_printed = false;
  std::vector<std::string> worker_status;
  std::vector<std::vector<unsigned>> worker_options_left_start;
  std::vector<std::vector<unsigned>> worker_options_left_last;
  std::vector<unsigned> worker_longest_start;
  std::vector<unsigned> worker_longest_last;

 public:
  bool run();

 private:
  void message_worker(const MessageC2W& msg, unsigned worker_id) const;
  void give_assignments();
  void process_inbox();
  void process_search_result(const MessageW2C& msg);
  void process_worker_idle(const MessageW2C& msg);
  void process_returned_work(const MessageW2C& msg);
  void process_returned_stats(const MessageW2C& msg);
  void process_worker_update(const MessageW2C& msg);
  void collect_stats();
  void steal_work();
  unsigned find_stealing_target_mostremaining() const;
  void calc_graph_size();
  bool passes_prechecks();
  bool is_worker_idle(const unsigned id) const;
  bool is_worker_splitting(const unsigned id) const;
  void record_data_from_message(const MessageW2C& msg);
  void start_workers();
  void stop_workers();
  double expected_patterns_at_maxperiod();
  static void signal_handler(int signum);
  void print_pattern(const MessageW2C& msg);
  void print_search_description() const;
  void print_results() const;
  void erase_status_output() const;
  void print_status_output();
  static std::string current_time_string();
  std::string make_worker_status(const MessageW2C& msg);
};

#endif
