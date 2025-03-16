//
// CoordinatorCPU.h
//
// Coordinator that manages the search on a multicore CPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATORCPU_H_
#define JPRIME_COORDINATORCPU_H_

#include "Coordinator.h"
#include "Messages.h"
#include "SearchConfig.h"
#include "SearchContext.h"
#include "Worker.h"

#include <queue>
#include <mutex>
#include <set>
#include <vector>
#include <string>
#include <thread>


class CoordinatorCPU : public Coordinator {
 public:
  CoordinatorCPU(SearchConfig& config, SearchContext& context,
    std::ostream& jpout);

 public:
  // for communicating with workers
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;
    
 protected:
  // workers
  std::vector<std::unique_ptr<Worker>> worker;
  std::vector<std::unique_ptr<std::thread>> worker_thread;
  std::set<unsigned> workers_idle;
  std::set<unsigned> workers_splitting;
  std::vector<unsigned> worker_startstate;
  std::vector<unsigned> worker_endstate;
  std::vector<unsigned> worker_rootpos;

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
  std::vector<std::vector<unsigned>> worker_options_left_start;
  std::vector<std::vector<unsigned>> worker_options_left_last;
  std::vector<unsigned> worker_longest_start;
  std::vector<unsigned> worker_longest_last;
  std::chrono::time_point<std::chrono::system_clock> last_status_time;
  uint64_t last_nnodes = 0;
  uint64_t last_ntotal = 0;

 protected:
  virtual void run_search() override;

  // handle interactions with worker threads
  void message_worker(const MessageC2W& msg, unsigned worker_id) const;
  void give_assignments();
  void steal_work();
  unsigned find_stealing_target_mostremaining() const;
  void collect_status();
  void process_inbox();
  void process_worker_idle(const MessageW2C& msg);
  void process_returned_work(const MessageW2C& msg);
  void process_returned_stats(const MessageW2C& msg);
  void process_worker_update(const MessageW2C& msg);

  // helper functions
  void start_workers();
  void stop_workers();
  bool is_worker_idle(const unsigned id) const;
  bool is_worker_splitting(const unsigned id) const;

  // manage worker status
  void record_data_from_message(const MessageW2C& msg);
  std::string make_worker_status(const MessageW2C& msg);
};

#endif
