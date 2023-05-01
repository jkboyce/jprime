
#ifndef JDEEP_WORKER_H
#define JDEEP_WORKER_H

#include "Coordinator.hpp"
#include "Messages.hpp"
#include "SearchConfig.hpp"

#include <queue>
#include <mutex>
#include <list>
#include <ctime>

class Coordinator;

class Worker {
 public:
  std::queue<MessageC2W> inbox;
  std::mutex inbox_lock;

  Worker(const SearchConfig& config, Coordinator* const coord, int id);
  ~Worker();
  void run();

 private:
  Coordinator* const coordinator;
  const int worker_id;

  // copied from SearchConfig during construction
  int n = 0;
  int h = 0;
  int l = 0;
  int mode = NORMAL_MODE;
  int groundmode = 0;
  bool printflag = true;
  bool invertflag = false;
  bool trimflag = true;
  bool longestflag = true;
  bool exactflag = false;
  bool dualflag = false;
  bool verboseflag = false;
  int skiplimit = 0;
  int shiftlimit = 0;

  // calculated at construction and do not change during search
  int numstates = 0;
  int maxlength = 0;
  int maxoutdegree = 0;
  int maxindegree = 0;
  unsigned long* state;
  int** partners;  // for finding superprime patterns
  int numcycles = 0;  // total number of shift cycles

  // for loading and sharing work assignments
  int start_state = 1;
  int end_state = 1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  bool loading_work = false;
  int loading_pos = 0;

  // variables for search
  int* pattern;
  int pos = 0;
  int from = 1;
  int firstblocklength = -1;
  int skipcount = 0;
  int shiftcount = 0;
  int blocklength = 0;
  int max_possible = 0;

  int** outmatrix;
  int** outthrowval;
  int* outdegree;
  int** inmatrix;
  int* indegree;
  int* used;
  int* cyclenum;  // cycle number for state
  int* cycleperiod;  // indexed by shift cycle number
  int* deadstates;  // indexed by shift cycle number

  // search data to report to Coordinator
  unsigned long ntotal = 0L;
  unsigned long nnodes = 0L;

  static constexpr double secs_per_inbox_check_target = 0.01;
  static constexpr int steps_per_inbox_check_initial = 5000000;

  int steps_per_inbox_check = steps_per_inbox_check_initial;
  int calibrations_remaining = 10;
  int steps_taken = 0;
  timespec last_ts;

  void message_coordinator(const MessageW2C& msg) const;
  void message_coordinator(const MessageW2C& msg1, const MessageW2C& msg2) const;
  void process_inbox();
  void load_work_assignment(const WorkAssignment& wa);
  WorkAssignment split_off_work_assignment();
  WorkAssignment get_work_assignment() const;
  void gen_patterns();
  void gen_loops_normal();
  void gen_loops_block();
  void gen_loops_super();
  void delete_vertices(int statenum);
  void outupdate(int statenum, int slot);
  void inupdate(int statenum, int slot);
  void trim_outgoing(int from_trim, int to_trim, int slot);
  void trim_ingoing(int from_trim, int to_trim, int slot);
  void report_pattern() const;
  void print_throw(std::ostringstream& buffer, int val) const;
  void print_inverse(std::ostringstream& buffer) const;
  void print_inverse_dual(std::ostringstream& buffer) const;
  int reverse_state(int statenum) const;
  static int num_states(int n, int h);
  void prepcorearrays(const std::vector<bool>& xarray);
  static void die();
  static int gen_states(unsigned long* state, int num, int pos, int left, int h, int ns);
  void gen_matrices(const std::vector<bool>& xarray);
};

class JdeepStopException : public std::exception {
};

#endif
