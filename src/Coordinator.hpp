
#ifndef JDEEP_COORDINATOR_H
#define JDEEP_COORDINATOR_H

#include <stdio.h>
#include <mutex>
#include <queue>

#include "Messages.hpp"

#define NORMAL_MODE     1
#define BLOCK_MODE      2
#define SUPER_MODE      3

struct SearchConfig {
  // number of objects
  int n = 0;
  // maximum throw value
  int h = 0;
  // (min) pattern length to find
  int l = 0;

  // search mode
  int mode = NORMAL_MODE;
  // ground state, excited state, or both
  int groundmode = 0;
  // print patterns to console
  bool printflag = true;
  // print inverses in super mode
  bool invertflag = false;
  // trim out states excluded by block throws
  bool trimflag = true;
  // search for the longest pattern(s)
  bool longestflag = true;
  // search for an exact pattern length
  bool exactflag = false;
  // find patterns in dual graph
  bool dualflag = false;
  // print search metadata
  bool verboseflag = false;
  // for block mode
  int skiplimit = 0;
  // for super mode
  int shiftlimit = 0;
  // throw values to exclude from search
  std::vector<bool> xarray;
};

class Worker;

class Coordinator {
 public:
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

  Coordinator(const SearchConfig& config);
  void run(int threads);

 private:
  int num_threads;
  std::vector<Worker*> worker;
  std::vector<std::thread*> worker_thread;
  std::vector<bool> worker_running;
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
  void give_first_assignments();
  void process_inbox();
  void print_pattern(const MessageW2C& msg);
  char print_throw(int val);
  void print_trailer();
};

#endif