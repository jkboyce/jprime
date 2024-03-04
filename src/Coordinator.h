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
  int l_max = -1;
  SearchContext& context;
  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  std::set<int> workers_idle;
  std::set<int> workers_splitting;
  std::list<int> workers_run_order;
  std::vector<int> worker_rootpos;
  std::vector<int> worker_longest;
  std::vector<std::string> worker_status;
  static bool stopping;

  static constexpr double nanosecs_per_inbox_check = 100000000 *
      Worker::secs_per_inbox_check_target;

  static constexpr double secs_per_status = 1;
  static constexpr int waits_per_status = static_cast<int>(1e9 *
      secs_per_status / nanosecs_per_inbox_check);
  int stats_counter = 0;
  int stats_received = 0;
  bool stats_printed = false;
  std::vector<int> worker_start_state;
  std::vector<std::vector<int>> worker_optionsleft_start;
  std::vector<std::vector<int>> worker_optionsleft_last;

 public:
  Coordinator(const SearchConfig& config, SearchContext& context);
  void run();

 private:
  void message_worker(const MessageC2W& msg, int worker_id) const;
  void give_assignments();
  void process_inbox();
  void process_search_result(const MessageW2C& msg);
  void process_worker_idle(const MessageW2C& msg);
  void process_returned_work(const MessageW2C& msg);
  void process_returned_stats(const MessageW2C& msg);
  void process_worker_status(const MessageW2C& msg);
  void collect_stats();
  void steal_work();
  int find_stealing_target_longestpattern() const;
  int find_stealing_target_lowestid() const;
  int find_stealing_target_lowestrootpos() const;
  int find_stealing_target_longestruntime() const;
  bool is_worker_idle(const int id) const;
  bool is_worker_splitting(const int id) const;
  void record_data_from_message(const MessageW2C& msg);
  void remove_from_run_order(const int id);
  void stop_workers();
  double expected_patterns_at_maxlength();
  static void signal_handler(int signum);
  void print_pattern(const MessageW2C& msg);
  void print_summary() const;
  void erase_status_output() const;
  void print_status_output();
  std::string make_worker_status(const MessageW2C& msg);
};

#endif
