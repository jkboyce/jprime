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
  static constexpr double secs_per_inbox_check_target = 0.01;

 private:
  // set during construction and do not change
  const SearchConfig config;
  Coordinator& coordinator;
  const unsigned int worker_id;
  Graph graph;
  int l_min = 0;
  int l_max = 0;
  unsigned int l_bound = 0;

  // working variables for search
  int pos = 0;
  int from = 1;
  int shiftcount = 0;
  int exitcyclesleft = 0;
  int max_possible = 0;
  SearchState *beat;  // workspace for search
  int* pattern;
  int* used;
  bool* cycleused;  // whether cycle has been visited, in SUPER mode
  unsigned int* deadstates;  // indexed by shift cycle number
  unsigned int** deadstates_bystate;  // indexed by state number

  // for loading and sharing work assignments
  unsigned int start_state = 0;
  unsigned int end_state = 0;
  int root_pos = 0;
  std::list<unsigned int> root_throwval_options;
  bool loading_work = false;

  // status data to report to Coordinator
  std::vector<std::uint64_t> count;
  std::uint64_t nnodes = 0;
  double secs_working = 0;
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
  void message_coordinator_text(const std::string& str) const;
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
  void notify_coordinator_update() const;
  void build_rootpos_throw_options(unsigned int from_state,
      unsigned int min_column);
  WorkAssignment split_work_assignment(int split_alg);
  WorkAssignment split_work_assignment_takestartstates();
  WorkAssignment split_work_assignment_takeall();
  WorkAssignment split_work_assignment_takehalf();
  WorkAssignment split_work_assignment_takefraction(double f, bool take_front);
  void gen_patterns();
  void set_inactive_states();
  void report_pattern() const;
  static char throw_char(int val);
  void print_throw(std::ostringstream& buffer, int val) const;
  std::string get_pattern() const;
  std::string get_inverse() const;

  // core search routines (recursive)
  void gen_loops_normal();
  void gen_loops_normal_marking();
  void gen_loops_super();
  void gen_loops_super0();
  unsigned int load_one_throw();
  bool mark_off_rootpos_option(unsigned int throwval, unsigned int to_state);
  bool mark_unreachable_states_throw();
  bool mark_unreachable_states_catch(unsigned int to_state);
  void unmark_unreachable_states_throw();
  void unmark_unreachable_states_catch(unsigned int to_state);
  void handle_finished_pattern();

  // core search routines (iterative)
  void iterative_gen_loops_normal();
  void iterative_gen_loops_normal_marking();
  void iterative_gen_loops_super();
  void iterative_gen_loops_super0();
  bool iterative_init_workspace(bool marking);
  bool iterative_calc_rootpos_and_options();
  bool iterative_can_split();
  void iterative_update_after_split();
  void iterative_handle_finished_pattern();
};

class JprimeStopException : public std::exception {
};

#endif
