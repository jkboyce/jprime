//
// Worker.h
//
// Worker that executes work assignments given to it by the Coordinator.
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
#include <vector>
#include <chrono>
#include <cstdint>


class Coordinator;

class Worker {
 public:
  std::queue<MessageC2W> inbox;
  std::mutex inbox_lock;
  static constexpr double SECS_PER_INBOX_CHECK_TARGET = 0.01;

 private:
  // set during construction and do not change
  const SearchConfig config;
  Coordinator& coordinator;
  const unsigned int worker_id;
  Graph graph;
  const unsigned int l_min;
  const unsigned int l_max;

  // working variables for search
  unsigned int pos = 0;
  unsigned int from = 1;
  unsigned int shiftcount = 0;
  unsigned int exitcyclesleft = 0;
  int max_possible = 0;
  std::vector<SearchState> beat;  // workspace for iterative search
  std::vector<int> pattern;
  std::vector<int> used;
  std::vector<int> cycleused;  // whether cycle has been visited, in SUPER mode
  std::vector<unsigned int> deadstates;  // indexed by shift cycle number
  std::vector<unsigned int*> deadstates_bystate;  // indexed by state number

  // for loading and sharing work assignments
  unsigned int start_state = 0;
  unsigned int end_state = 0;
  unsigned int root_pos = 0;
  std::list<unsigned int> root_throwval_options;
  bool loading_work = false;

  // status data to report to Coordinator
  std::vector<std::uint64_t> count;
  std::uint64_t nnodes = 0;
  double secs_working = 0;
  bool running = false;

  // for managing the frequency to check the inbox while running
  static constexpr unsigned int STEPS_PER_INBOX_CHECK_INITIAL = 50000u;
  unsigned int steps_per_inbox_check = STEPS_PER_INBOX_CHECK_INITIAL;
  static constexpr unsigned int CALIBRATIONS_INITIAL = 10;
  unsigned int calibrations_remaining = CALIBRATIONS_INITIAL;
  unsigned int steps_taken = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_ts;

 public:
  Worker(const SearchConfig& config, Coordinator& coord, int id,
    unsigned int l_max);
  // Note that Worker contains a `std::mutex` so its default copy and move
  // constructors are deleted
  void run();

 private:
  void message_coordinator(MessageW2C& msg) const;
  void message_coordinator_text(const std::string& str) const;
  void process_inbox_running();
  void record_elapsed_time_from(const
    std::chrono::time_point<std::chrono::high_resolution_clock>& start);
  void calibrate_inbox_check();
  void process_split_work_request();
  void send_work_to_coordinator(const WorkAssignment& wa);
  void send_stats_to_coordinator();
  void add_data_to_message(MessageW2C& msg);
  void load_work_assignment(const WorkAssignment& wa);
  WorkAssignment get_work_assignment() const;
  void notify_coordinator_idle();
  void notify_coordinator_update() const;
  void build_rootpos_throw_options(unsigned int from_state,
      unsigned int min_column);
  WorkAssignment split_work_assignment(unsigned int split_alg);
  WorkAssignment split_work_assignment_takestartstates();
  WorkAssignment split_work_assignment_takeall();
  WorkAssignment split_work_assignment_takehalf();
  WorkAssignment split_work_assignment_takefraction(double f, bool take_front);
  void gen_patterns();
  void set_inactive_states();
  void initialize_working_variables();
  void report_pattern() const;
  static char throw_char(int val);
  void print_throw(std::ostringstream& buffer, unsigned int val) const;
  std::string get_pattern() const;
  std::string get_inverse() const;

  // core search routines (recursive versions)
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

  // core search routines (iterative versions; identical in function to above)
  void iterative_gen_loops_normal();
  void iterative_gen_loops_normal_counting();
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
