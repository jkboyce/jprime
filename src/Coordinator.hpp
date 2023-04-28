
#ifndef JDEEP_COORDINATOR_H
#define JDEEP_COORDINATOR_H

#include "Messages.hpp"
#include "SearchConfig.hpp"

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

  Coordinator(const SearchConfig& config);
  void run(int threads, std::list<WorkAssignment>& assignments);

 private:
  int num_threads;
  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  // std::vector<bool> worker_running;
  std::list<int> workers_idle;
  int waiting_for_work_from_id = -1;

  const SearchConfig config;
  std::vector<std::string> patterns;
  int l_current = 0;
  unsigned long npatterns = 0L;
  unsigned long ntotal = 0L;
  int numstates = 0;
  int maxlength = 0;

  void message_worker(const MessageC2W& msg, int worker_id);
  void give_assignments(std::list<WorkAssignment>& assignments);
  void steal_assignment();
  void process_inbox(std::list<WorkAssignment>& assignments);
  void stop_workers();
  static void signal_handler(int signum);
  void print_pattern(const MessageW2C& msg);
  void print_trailer();
};

#endif