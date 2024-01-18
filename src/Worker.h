//
// Worker.h
//
// Worker thread that executes work assignments given to it by the
// Coordinator thread.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKER_H_
#define JPRIME_WORKER_H_

#include "Coordinator.h"
#include "Messages.h"
#include "SearchConfig.h"
#include "Graph.h"

#include <queue>
#include <mutex>
#include <list>
#include <ctime>
#include <cstdint>


class Coordinator;

class Worker {
 public:
  std::queue<MessageC2W> inbox;
  std::mutex inbox_lock;
  static constexpr double secs_per_inbox_check_target = 0.001;

 private:
  // set during construction and do not change
  const SearchConfig config;
  Coordinator& coordinator;
  const int worker_id;
  const Graph graph;
  int maxlength = 0;

  // for loading and sharing work assignments
  int start_state = 1;
  int end_state = 1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  bool loading_work = false;

  // working variables for search
  int pos = 0;
  int from = 1;
  int firstblocklength = -1;
  int skipcount = 0;
  int shiftcount = 0;
  int blocklength = 0;
  int l_current = 0;  // minimum length to find
  int max_possible = 0;
  int* pattern;
  int* used;
  bool* cycleused;  // whether cycle has been visited, in SUPER mode
  int* deadstates;  // indexed by shift cycle number
  int exitcyclesleft = 0;

  // status data to report to Coordinator
  std::uint64_t ntotal = 0;
  std::uint64_t nnodes = 0;
  int longest_found = 0;
  double secs_working = 0;

  // for managing the frequency to check the inbox while running
  static constexpr int steps_per_inbox_check_initial = 50000;
  int steps_per_inbox_check = steps_per_inbox_check_initial;
  static constexpr int calibrations_initial = 10;
  int calibrations_remaining = calibrations_initial;
  int steps_taken = 0;
  timespec last_ts;

 public:
  Worker(const SearchConfig& config, Coordinator& coord, int id);
  Worker(const Worker&) =delete;
  Worker(Worker&&) =delete;
  Worker& operator=(const Worker&) =delete;
  Worker& operator=(Worker&&) =delete;
  ~Worker();
  void run();

 private:
  void allocate_arrays();
  void delete_arrays();
  void message_coordinator(const MessageW2C& msg) const;
  void process_inbox_running();
  void record_elapsed_time(const timespec& start);
  void calibrate_inbox_check();
  void send_work_to_coordinator(const WorkAssignment& wa);
  void process_split_work_request(const MessageC2W& msg);
  void load_work_assignment(const WorkAssignment& wa);
  WorkAssignment get_work_assignment() const;
  void notify_coordinator_idle();
  void notify_coordinator_rootpos() const;
  void notify_coordinator_longest() const;
  WorkAssignment split_work_assignment(int split_alg);
  WorkAssignment split_work_assignment_takeall();
  WorkAssignment split_work_assignment_takehalf();
  WorkAssignment split_work_assignment_takefraction(double f, bool take_front);
  void gen_patterns();
  void gen_loops_normal();
  void gen_loops_block();
  void gen_loops_super();
  void gen_loops_super0g();
  int load_one_throw();
  void build_rootpos_throw_options(int rootpos_from_state, int min_column);
  bool mark_off_rootpos_option(int throwval, int to_state);
  void mark_forbidden_state(int s);
  bool mark_unreachable_states_throw();
  bool mark_unreachable_states_catch(int to_state);
  void unmark_unreachable_states_throw();
  void unmark_unreachable_states_catch(int to_state);
  void handle_finished_pattern();
  void report_pattern() const;
  void print_throw(std::ostringstream& buffer, int val) const;
  std::string get_pattern() const;
  std::string get_inverse() const;
};

class JprimeStopException : public std::exception {
};

#endif
