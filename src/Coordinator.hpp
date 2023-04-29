
#ifndef JDEEP_COORDINATOR_H
#define JDEEP_COORDINATOR_H

#include "Messages.hpp"
#include "SearchConfig.hpp"
#include "SearchContext.hpp"

#include <queue>
#include <mutex>
#include <list>
#include <vector>

class Worker;

class Coordinator {
 public:
  static bool stopping;
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

  Coordinator(const SearchConfig& config, SearchContext& context);
  void run();

 private:
  const SearchConfig config;
  SearchContext& context;

  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  std::list<int> workers_idle;
  int waiting_for_work_from_id = -1;

  void message_worker(const MessageC2W& msg, int worker_id) const;
  void give_assignments();
  void steal_work();
  void process_inbox();
  int process_search_result(const MessageW2C& msg);
  void notify_metadata(int skip_id) const;
  void stop_workers() const;
  static void signal_handler(int signum);
  void print_pattern(const MessageW2C& msg);
  void print_trailer() const;
};

#endif
