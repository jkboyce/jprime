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
#include "WorkCell.h"
#include "WorkAssignment.h"
#include "WorkSpace.h"

#include <queue>
#include <mutex>
#include <list>
#include <vector>
#include <chrono>
#include <cstdint>


class CoordinatorCPU;

class Worker : public WorkSpace {
 public:
  Worker(const SearchConfig& config, CoordinatorCPU& coord, Graph& g,
      unsigned id, unsigned n_max);
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
  Graph& graph;
  std::vector<std::vector<unsigned>> excludestates_tail;
  std::vector<std::vector<unsigned>> excludestates_head;

  // working variables for search
  std::vector<WorkCell> beat;  // workspace for iterative search
  std::vector<int> pattern;  // throw value at each position
  std::vector<int> used;  // whether a state has been visited
  std::vector<int> cycleused;  // whether cycle has been visited, in SUPER mode
  std::vector<unsigned> deadstates;  // indexed by shift cycle number
  std::vector<unsigned*> deadstates_bystate;  // indexed by state number
  std::vector<int> isexitcycle;
  int pos = 0;  // current index in the pattern
  unsigned from = 1;  // current state number
  unsigned shiftcount = 0;
  unsigned exitcyclesleft = 0;
  int max_possible = 0;  // max period possible from the current position

  // for loading and sharing work assignments
  unsigned start_state = 0;
  unsigned end_state = 0;
  unsigned root_pos = 0;  // lowest `pos` with unexplored tree options
  std::list<unsigned> root_throwval_options;
  bool loading_work = false;  // used for recursive initialization
  unsigned replay_to_pos = 0;  // used for iterative initialization

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
  void do_work_assignment();
  void gen_loops();
  void initialize_working_variables();
  void report_pattern() const;

  // recursive search routines; defined in GenLoopsRecursive.cc
  void gen_loops_normal();
  void gen_loops_normal_marking();
  void gen_loops_super();
  void gen_loops_super0();
  unsigned load_one_throw();
  bool mark_off_rootpos_option(unsigned throwval, unsigned to_state);
  void build_rootpos_throw_options(unsigned from_state, unsigned min_column);
  bool mark_unreachable_states_tail();
  bool mark_unreachable_states_head(unsigned to_state);
  void unmark_unreachable_states_tail();
  void unmark_unreachable_states_head(unsigned to_state);
  void handle_finished_pattern();

  // iterative search routines; defined in GenLoopsIterative.cc
  template<bool MARKING, bool REPORT, bool REPLAY>
    void iterative_gen_loops_normal();
  bool mark(int* const& u, unsigned*& es, unsigned* const& ds);
  void unmark(int* const& u, unsigned*& es, unsigned* const& ds);
  template<bool SUPER0, bool REPLAY> void iterative_gen_loops_super();
  void iterative_init_workspace();
  bool iterative_can_split();
  void iterative_update_after_split();
  void iterative_handle_finished_pattern();

  // WorkSpace methods; defined in GenLoopsIterative.cc
  virtual const Graph& get_graph() const override;
  virtual void set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state) override;
  virtual std::tuple<unsigned, unsigned, unsigned> get_cell(unsigned slot,
    unsigned index) const override;
  virtual void set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos) override;
  virtual std::tuple<unsigned, unsigned, int> get_info(unsigned slot) const
    override;
};


class JprimeStopException : public std::exception {
};

#endif
