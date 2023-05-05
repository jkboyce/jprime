
#ifndef JDEEP_COORDINATOR_H
#define JDEEP_COORDINATOR_H

#include "Messages.hpp"
#include "SearchConfig.hpp"
#include "SearchContext.hpp"

#include <queue>
#include <mutex>
#include <list>
#include <vector>
#include <thread>

class Worker;

class Coordinator {
 public:
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

  Coordinator(const SearchConfig& config, SearchContext& context);
  void run();

 private:
  const SearchConfig& config;
  SearchContext& context;

  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  std::list<int> workers_idle;
  std::list<int> workers_run_order;
  std::vector<int> worker_rootpos;
  std::vector<int> worker_longest;
  int waiting_for_work_from_id = -1;
  static bool stopping;

  void message_worker(const MessageC2W& msg, int worker_id) const;
  void give_assignments();
  void steal_work();
  void process_inbox();
  int process_search_result(const MessageW2C& msg);
  bool is_worker_idle(const int id) const;
  void process_worker_idle(const MessageW2C& msg);
  void process_worker_returned_work(const MessageW2C& msg);
  void process_worker_status(const MessageW2C& msg);
  void remove_from_run_order(const int id);
  void notify_metadata(int skip_id) const;
  void stop_workers() const;
  static void signal_handler(int signum);
  int find_stealing_target_longestpattern() const;
  int find_stealing_target_lowestid() const;
  int find_stealing_target_lowestrootpos() const;
  int find_stealing_target_longestruntime() const;
  void print_pattern(const MessageW2C& msg);
  void print_trailer() const;
};

#endif
