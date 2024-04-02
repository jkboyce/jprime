//
// Coordinator.h
//
// Coordinator thread that manages the overall search.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
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


class Coordinator {
 public:
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

 private:
  const SearchConfig& config;
  SearchContext& context;
  unsigned int l_max = 0;
  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  std::set<unsigned int> workers_idle;
  std::set<unsigned int> workers_splitting;
  std::vector<unsigned int> worker_startstate;
  std::vector<unsigned int> worker_endstate;
  std::vector<unsigned int> worker_rootpos;
  static bool stopping;
  static constexpr std::uint64_t MAX_STATES = 200000u;

  // check inbox 10x more often than workers do
  static constexpr double NANOSECS_PER_INBOX_CHECK =
      1e8 * Worker::SECS_PER_INBOX_CHECK_TARGET;

  // live status display
  static constexpr double SECS_PER_STATUS = 1;
  static constexpr int WAITS_PER_STATUS = static_cast<int>(1e9 *
      SECS_PER_STATUS / NANOSECS_PER_INBOX_CHECK);
  static constexpr int STATUS_WIDTH = 56;
  unsigned int stats_counter = 0;
  unsigned int stats_received = 0;
  bool stats_printed = false;
  std::vector<std::string> worker_status;
  std::vector<std::vector<unsigned int>> worker_options_left_start;
  std::vector<std::vector<unsigned int>> worker_options_left_last;
  std::vector<unsigned int> worker_longest;

 public:
  Coordinator(const SearchConfig& config, SearchContext& context);
  void run();

 private:
  void message_worker(const MessageC2W& msg, unsigned int worker_id) const;
  void give_assignments();
  void process_inbox();
  void process_search_result(const MessageW2C& msg);
  void process_worker_idle(const MessageW2C& msg);
  void process_returned_work(const MessageW2C& msg);
  void process_returned_stats(const MessageW2C& msg);
  void process_worker_update(const MessageW2C& msg);
  void collect_stats();
  void steal_work();
  unsigned int find_stealing_target_mostremaining() const;
  bool passes_prechecks();
  void calc_graph_size();
  bool is_worker_idle(const unsigned int id) const;
  bool is_worker_splitting(const unsigned int id) const;
  void record_data_from_message(const MessageW2C& msg);
  void start_workers();
  void stop_workers();
  double expected_patterns_at_maxlength();
  static void signal_handler(int signum);
  void print_preamble() const;
  void print_pattern(const MessageW2C& msg);
  void print_summary() const;
  void erase_status_output() const;
  void print_status_output();
  static std::string current_time_string();
  std::string make_worker_status(const MessageW2C& msg);
};

#endif
