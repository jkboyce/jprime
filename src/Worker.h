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

#include "Messages.h"
#include "SearchConfig.h"
#include "Graph.h"
#include "SearchState.h"

#include <queue>
#include <mutex>
#include <list>
#include <chrono>
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
  int l_min = 0;
  int l_max = 0;
  int l_bound = 0;

  // working variables for search
  Graph graph;
  int pos = 0;
  int from = 1;
  int shiftcount = 0;
  int max_possible = 0;
  int exitcyclesleft = 0;
  SearchState *beat;  // workspace for search
  int* pattern;
  int* used;
  bool* cycleused;  // whether cycle has been visited, in SUPER mode
  int* deadstates;  // indexed by shift cycle number
  int** deadstates_bystate;  // indexed by state number

  // for loading and sharing work assignments
  int start_state = 1;
  int end_state = 1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  bool loading_work = false;

  // status data to report to Coordinator
  std::uint64_t ntotal = 0;
  std::uint64_t nnodes = 0;
  int longest_found = 0;
  double secs_working = 0;
  std::vector<std::uint64_t> count;
  bool running = false;

  // for managing the frequency to check the inbox while running
  static constexpr int steps_per_inbox_check_initial = 50000;
  int steps_per_inbox_check = steps_per_inbox_check_initial;
  static constexpr int calibrations_initial = 10;
  int calibrations_remaining = calibrations_initial;
  int steps_taken = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_ts;

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
  void message_coordinator(MessageW2C& msg) const;
  void message_coordinator_status(const std::string& str) const;
  void process_inbox_running();
  void record_elapsed_time(const
    std::chrono::time_point<std::chrono::high_resolution_clock>& start);
  void calibrate_inbox_check();
  void process_split_work_request(const MessageC2W& msg);
  void send_work_to_coordinator(const WorkAssignment& wa);
  void send_stats_to_coordinator();
  void add_data_to_message(MessageW2C& msg);
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
  void set_active_states();
  void gen_loops_normal();
  void gen_loops_super();
  void gen_loops_super0();
  int load_one_throw();
  void build_rootpos_throw_options(int rootpos_from_state, int min_column);
  bool mark_off_rootpos_option(int throwval, int to_state);
  bool mark_unreachable_states_throw();
  bool mark_unreachable_states_catch(int to_state);
  void unmark_unreachable_states_throw();
  void unmark_unreachable_states_catch(int to_state);
  void handle_finished_pattern();
  void gen_loops_normal_iterative();
  bool iterative_init_workspace();
  void iterative_calc_rootpos_and_options();
  bool iterative_can_split();
  void iterative_update_after_split();
  void iterative_handle_finished_pattern();
  void report_pattern() const;
  void print_throw(std::ostringstream& buffer, int val) const;
  std::string get_pattern() const;
  std::string get_inverse() const;
};

class JprimeStopException : public std::exception {
};

#endif
