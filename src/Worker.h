//
// Worker.h
//
// Worker that executes work assignments given to it by the Coordinator.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
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


class CoordinatorCPU;

class Worker {
 public:
  Worker(const SearchConfig& config, CoordinatorCPU& coord, unsigned id,
      unsigned n_max);
  // Note that Worker contains a `std::mutex` so its default copy and move
  // constructors are deleted

 public:
  std::queue<MessageC2W> inbox;
  std::mutex inbox_lock;
  static constexpr double SECS_PER_INBOX_CHECK_TARGET = 0.01;

 private:
  // set during construction and do not change
  const SearchConfig config;
  CoordinatorCPU& coordinator;
  const unsigned worker_id;
  const unsigned n_min;  // minimum period to find
  const unsigned n_max;  // maximum period

  // working variables for search
  Graph graph;
  std::vector<SearchState> beat;  // workspace for iterative search
  std::vector<int> pattern;  // throw value at each position
  std::vector<int> used;  // whether a state has been visited
  std::vector<int> cycleused;  // whether cycle has been visited, in SUPER mode
  std::vector<unsigned> deadstates;  // indexed by shift cycle number
  std::vector<unsigned*> deadstates_bystate;  // indexed by state number
  unsigned pos = 0;  // current index in the pattern
  unsigned from = 1;  // current state number
  unsigned shiftcount = 0;
  unsigned exitcyclesleft = 0;
  int max_possible = 0;  // max period possible from the current position

  // for loading and sharing work assignments
  unsigned start_state = 0;
  unsigned end_state = 0;
  unsigned root_pos = 0;  // lowest `pos` with unexplored tree options
  std::list<unsigned> root_throwval_options;
  bool loading_work = false;

  // status data to report to Coordinator
  std::vector<std::uint64_t> count;  // count of patterns found at each period
  std::uint64_t nnodes = 0;  // search tree nodes completed
  double secs_working = 0;
  bool running = false;

  // for managing the frequency to check the inbox while running
  static constexpr unsigned STEPS_PER_INBOX_CHECK_INITIAL = 50000u;
  unsigned steps_per_inbox_check = STEPS_PER_INBOX_CHECK_INITIAL;
  static constexpr unsigned CALIBRATIONS_INITIAL = 10;
  unsigned calibrations_remaining = CALIBRATIONS_INITIAL;
  unsigned steps_taken = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> last_ts;

 public:
  void run();

 private:
  void init();
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
  void build_rootpos_throw_options(unsigned from_state, unsigned min_column);
  WorkAssignment split_work_assignment(unsigned split_alg);
  WorkAssignment split_work_assignment_takestartstates();
  WorkAssignment split_work_assignment_takeall();
  WorkAssignment split_work_assignment_takehalf();
  WorkAssignment split_work_assignment_takefraction(double f, bool take_front);
  void gen_patterns();
  void gen_loops();
  void customize_graph();
  void initialize_working_variables();
  void report_pattern() const;

  // main search routines (recursive versions)
  void gen_loops_normal();
  void gen_loops_normal_marking();
  void gen_loops_super();
  void gen_loops_super0();
  unsigned load_one_throw();
  bool mark_off_rootpos_option(unsigned throwval, unsigned to_state);
  bool mark_unreachable_states_throw();
  bool mark_unreachable_states_catch(unsigned to_state);
  void unmark_unreachable_states_throw();
  void unmark_unreachable_states_catch(unsigned to_state);
  void handle_finished_pattern();

  // main search routines (iterative versions; identical in function to above)
  template<bool REPORT> void iterative_gen_loops_normal();
  void iterative_gen_loops_normal_marking();
  template<bool SUPER0> void iterative_gen_loops_super();
  bool iterative_init_workspace(bool marking);
  bool iterative_calc_rootpos_and_options();
  bool iterative_can_split();
  void iterative_update_after_split();
  void iterative_handle_finished_pattern();
};


class JprimeStopException : public std::exception {
};

#endif
