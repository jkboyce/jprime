//
// Worker.cc
//
// Worker thread that executes work assignments given to it by the
// Coordinator thread.
//
// The overall computation is depth first search on multiple worker threads,
// with a work stealing scheme to balance work among the threads. Each worker
// communicates only with the coordinator thread, via a set of message types.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Worker.h"
#include "Coordinator.h"
#include "Messages.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cassert>
#include <ctime>


Worker::Worker(const SearchConfig& config, Coordinator* const coord, int id) :
      coordinator(coord), worker_id(id) {
  n = config.n;
  h = config.h;
  l = config.l;
  printflag = config.printflag;
  invertflag = config.invertflag;
  groundmode = config.groundmode;
  longestflag = config.longestflag;
  exactflag = config.exactflag;
  dualflag = config.dualflag;
  verboseflag = config.verboseflag;
  noplusminusflag = config.noplusminusflag;
  mode = config.mode;
  skiplimit = config.skiplimit;
  shiftlimit = config.shiftlimit;

  numstates = num_states(n, h);
  for (int i = 0; i <= h; ++i) {
    if (!config.xarray[i])
      ++maxoutdegree;
  }
  maxoutdegree = std::min(maxoutdegree, h - n + 1);
  maxindegree = n + 1;
  highestbit = 1L << (h - 1);
  allbits = (1L << h) - 1;

  allocate_arrays();
  int ns = gen_states(state, 0, h - 1, n, h, numstates);
  assert(ns == numstates);
  find_shift_cycles();
  gen_matrices(config.xarray);

  maxlength = (mode == SearchMode::SUPER_MODE) ? (numcycles + shiftlimit)
      : (numstates - numcycles);
  if (l > maxlength) {
    std::cerr << "No patterns longer than " << maxlength << " are possible"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

Worker::~Worker() {
  delete_arrays();
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the main run loop for the worker, which waits until it receives an
// assignment from the coordinator.

void Worker::run() {
  while (true) {
    bool new_assignment = false;
    bool stop_worker = false;

    inbox_lock.lock();
    if (!inbox.empty()) {
      MessageC2W msg = inbox.front();
      inbox.pop();

      if (msg.type == messages_C2W::DO_WORK) {
        load_work_assignment(msg.assignment);
        l = std::max(l, msg.l_current);
        new_assignment = true;
      } else if (msg.type == messages_C2W::UPDATE_METADATA) {
        // ignore in idle state
      } else if (msg.type == messages_C2W::SPLIT_WORK) {
        // leave in the inbox for when we get work
        inbox.push(msg);
      } else if (msg.type == messages_C2W::STOP_WORKER) {
        stop_worker = true;
      } else
        assert(false);
    }
    inbox_lock.unlock();

    if (stop_worker)
      break;
    if (!new_assignment)
      continue;

    // get timestamp so we can report working time to coordinator
    timespec start_ts;
    timespec_get(&start_ts, TIME_UTC);

    // complete the new work assignment
    try {
      gen_patterns();
      record_elapsed_time(start_ts);
    } catch (const JdeepStopException& jdse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the coordinator
      record_elapsed_time(start_ts);
      send_work_to_coordinator(get_work_assignment());
      break;
    }

    // empty the inbox
    inbox_lock.lock();
    inbox = std::queue<MessageC2W>();
    inbox_lock.unlock();

    notify_coordinator_idle();
  }
}

//------------------------------------------------------------------------------
// Handle interactions with the Coordinator thread
//------------------------------------------------------------------------------

void Worker::message_coordinator(const MessageW2C& msg) const {
  coordinator->inbox_lock.lock();
  coordinator->inbox.push(msg);
  coordinator->inbox_lock.unlock();
}

// Handle incoming messages from the coordinator that have queued while the
// worker is running.

void Worker::process_inbox_running() {
  if (calibrations_remaining > 0)
    calibrate_inbox_check();

  bool stopping_work = false;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageC2W msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_C2W::DO_WORK) {
      assert(false);
    } else if (msg.type == messages_C2W::UPDATE_METADATA) {
      l = std::max(l, msg.l_current);
    } else if (msg.type == messages_C2W::SPLIT_WORK) {
      process_split_work_request(msg);
    } else if (msg.type == messages_C2W::STOP_WORKER) {
      stopping_work = true;
    }
  }
  inbox_lock.unlock();

  if (stopping_work) {
    // unwind back to Worker::run()
    throw JdeepStopException();
  }
}

// Get a finishing timestamp and record elapsed-time statistics to report to
// the coordinator later on.

void Worker::record_elapsed_time(const timespec& start_ts) {
  timespec end_ts;
  timespec_get(&end_ts, TIME_UTC);
  double runtime =
      (static_cast<double>(end_ts.tv_sec) + 1.0e-9 * end_ts.tv_nsec) -
      (static_cast<double>(start_ts.tv_sec) + 1.0e-9 * start_ts.tv_nsec);
  secs_working += runtime;
}

void Worker::calibrate_inbox_check() {
  if (calibrations_remaining == calibrations_initial) {
    timespec_get(&last_ts, TIME_UTC);
    --calibrations_remaining;
    return;
  }

  timespec current_ts;
  timespec_get(&current_ts, TIME_UTC);
  double time_spent =
      (static_cast<double>(current_ts.tv_sec) + 1.0e-9 * current_ts.tv_nsec) -
      (static_cast<double>(last_ts.tv_sec) + 1.0e-9 * last_ts.tv_nsec);
  last_ts = current_ts;
  --calibrations_remaining;

  steps_per_inbox_check =
      static_cast<int>(static_cast<double>(steps_per_inbox_check) *
      secs_per_inbox_check_target / time_spent);
}

void Worker::send_work_to_coordinator(const WorkAssignment& wa) {
  MessageW2C msg;
  msg.type = messages_W2C::RETURN_WORK;
  msg.worker_id = worker_id;
  msg.assignment = wa;
  msg.ntotal = ntotal;
  msg.nnodes = nnodes;
  msg.numstates = numstates;
  msg.maxlength = maxlength;
  msg.secs_working = secs_working;
  ntotal = 0;
  nnodes = 0;
  secs_working = 0;
  message_coordinator(msg);
}

void Worker::process_split_work_request(const MessageC2W& msg) {
  send_work_to_coordinator(split_work_assignment(msg.split_alg));

  if (verboseflag) {
    std::ostringstream sstr;
    sstr << "worker " << worker_id
         << " remaining work after split:" << std::endl
         << "  " << get_work_assignment();

    MessageW2C msg2;
    msg2.type = messages_W2C::WORKER_STATUS;
    msg2.worker_id = worker_id;
    msg2.meta = sstr.str();
    message_coordinator(msg2);
  }
}

void Worker::load_work_assignment(const WorkAssignment& wa) {
  loading_work = true;

  start_state = wa.start_state;
  end_state = wa.end_state;
  if (start_state == -1)
    start_state = (groundmode == 2) ? 2 : 1;
  if (end_state == -1)
    end_state = (groundmode == 1) ? 1 : numstates;

  longest_found = 0;
  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  if (wa.start_state == -1 || wa.end_state == -1) {
    // assignment came from the coordinator which doesn't know how to correctly
    // set the throw options, so do that here
    build_rootpos_throw_options(start_state, 0);
  }
  assert(root_throwval_options.size() > 0);
  assert(pos == 0);

  for (int i = 0; i <= numstates; ++i) {
    pattern[i] = (i < wa.partial_pattern.size()) ? wa.partial_pattern[i] : -1;
    assert(mode == SearchMode::SUPER_MODE || used[i] == 0);
  }
}

// Return the work assignment corresponding to the current state of the worker.
// Note this is distinct from split_work_assignment(), which splits off a
// portion of the assignment to give back to the coordinator.

WorkAssignment Worker::get_work_assignment() const {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = end_state;
  wa.root_pos = root_pos;
  wa.root_throwval_options = root_throwval_options;
  for (int i = 0; i <= numstates; ++i) {
    if (pattern[i] == -1)
      break;
    wa.partial_pattern.push_back(pattern[i]);
  }
  return wa;
}

// Notify the coordinator that the worker is idle and ready for another work
// assignment.

void Worker::notify_coordinator_idle() {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_IDLE;
  msg.worker_id = worker_id;
  msg.ntotal = ntotal;
  msg.nnodes = nnodes;
  msg.numstates = numstates;
  msg.maxlength = maxlength;
  msg.secs_working = secs_working;
  ntotal = 0;
  nnodes = 0;
  secs_working = 0;
  message_coordinator(msg);
}

// Notify the coordinator of certain changes in the status of the search. The
// coordinator may use this information to determine which worker to steal work
// from when another worker goes idle.

void Worker::notify_coordinator_rootpos() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.root_pos = root_pos;
  message_coordinator(msg);
}

void Worker::notify_coordinator_longest() const {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.longest_found = longest_found;
  message_coordinator(msg);
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

// Return a work assignment that corresponds to a portion of the worker's
// current work assignment, for handing off to another idle worker. There is no
// single way to do this so we implement a number of strategies and measure
// performance.

WorkAssignment Worker::split_work_assignment(int split_alg) {
  switch (split_alg) {
    case 1:
      return split_work_assignment_takeall();
      break;
    case 2:
      return split_work_assignment_takehalf();
      break;
    default:
      assert(false);
  }
}

WorkAssignment Worker::split_work_assignment_takeall() {
  // strategy: take all of the throw options at root_pos
  return split_work_assignment_takefraction(1, false);
}

WorkAssignment Worker::split_work_assignment_takehalf() {
  // strategy: take half of the throw options at root_pos
  return split_work_assignment_takefraction(0.5, false);
}

WorkAssignment Worker::split_work_assignment_takefraction(double f,
      bool take_front) {
  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  for (int i = 0; i < root_pos; ++i)
    wa.partial_pattern.push_back(pattern[i]);

  // ensure the throw value at `root_pos` isn't on the list of throw options
  std::list<int>::iterator iter = root_throwval_options.begin();
  std::list<int>::iterator end = root_throwval_options.end();
  while (iter != end) {
    if (*iter == pattern[root_pos])
      iter = root_throwval_options.erase(iter);
    else
      ++iter;
  }
  assert(root_throwval_options.size() > 0);

  typedef std::list<int>::size_type li_size_t;
  li_size_t take_count =
      static_cast<li_size_t>(0.51 + f * root_throwval_options.size());
  take_count = std::min(
      std::max(static_cast<li_size_t>(1), take_count),
      root_throwval_options.size());

  li_size_t take_begin_idx = (take_front ? 0
        : root_throwval_options.size() - take_count);
  li_size_t take_end_idx = take_begin_idx + take_count;

  iter = root_throwval_options.begin();
  end = root_throwval_options.end();
  li_size_t index = 0;
  while (iter != end) {
    if (index >= take_begin_idx && index < take_end_idx) {
      wa.root_throwval_options.push_back(*iter);
      iter = root_throwval_options.erase(iter);
    } else
      ++iter;
    ++index;
  }

  if (root_throwval_options.size() == 0) {
    // Gave away all our throw options at this `root_pos`.
    //
    // We need to find the shallowest depth `new_root_pos` where there are
    // unexplored throw options. We have no more options at the current
    // root_pos, so new_root_pos > root_pos.
    //
    // We're also at a point in the search where we know there are unexplored
    // options remaining at the current value of `pos` (by virtue of how we got
    // here), and that pos > root_pos.
    //
    // So we know there must be a value of `new_root_pos` with the properties we
    // need, in the range root_pos < new_root_pos <= pos.

    int from_state = start_state;
    int new_root_pos = -1;
    int col = 0;

    // have to scan from the beginning because we don't record the traversed
    // states as we build the pattern
    for (int pos2 = 0; pos2 <= pos; ++pos2) {
      const int throwval = pattern[pos2];
      for (col = 0; col < outdegree[from_state]; ++col) {
        if (throwval == outthrowval[from_state][col])
          break;
      }
      // diagnostics if there's a problem
      if (col == outdegree[from_state]) {
        std::cerr << "pos2 = " << pos2
                  << ", from_state = " << from_state
                  << ", start_state = " << start_state
                  << ", root_pos = " << root_pos
                  << ", col = " << col
                  << ", throwval = " << throwval
                  << std::endl;
      }
      assert(col != outdegree[from_state]);

      if (pos2 > root_pos && col < outdegree[from_state] - 1) {
        new_root_pos = pos2;
        break;
      }

      from_state = outmatrix[from_state][col];
    }
    assert(new_root_pos != -1);
    root_pos = new_root_pos;
    notify_coordinator_rootpos();
    build_rootpos_throw_options(from_state, col + 1);
    assert(root_throwval_options.size() > 0);
  }

  return wa;
}

//------------------------------------------------------------------------------
// Search the juggling graph for patterns
//------------------------------------------------------------------------------

// Find all patterns within a range of `start_state` values.
//
// We enforce that a prime pattern has no state numbers smaller than the state
// it starts with, which ensures each pattern is generated exactly once.

void Worker::gen_patterns() {
  for (; start_state <= end_state; ++start_state) {
    // check if no way to make a pattern of the target length
    if ((longestflag || exactflag) && (numstates - start_state + 1) < l)
      continue;

    if (verboseflag) {
      std::ostringstream sstr;
      sstr << "worker " << worker_id
           << " starting at start_state: " << start_state;

      MessageW2C msg;
      msg.type = messages_W2C::WORKER_STATUS;
      msg.worker_id = worker_id;
      msg.meta = sstr.str();
      message_coordinator(msg);
    }

    // reset all working variables
    pos = 0;
    from = start_state;
    firstblocklength = -1; // -1 signals unknown
    skipcount = 0;
    shiftcount = 0;
    blocklength = 0;
    max_possible = maxlength;
    for (int i = 0; i <= numstates; ++i) {
      used[i] = 0;
      cycleused[i] = false;
    }
    longest_found = 0;
    notify_coordinator_longest();

    if (!loading_work) {
      // reset `root_pos` and throw options there
      root_pos = 0;
      notify_coordinator_rootpos();
      build_rootpos_throw_options(start_state, 0);
      if (root_throwval_options.size() == 0)
        continue;
    }

    switch (mode) {
      case SearchMode::NORMAL_MODE:
        gen_loops_normal();
        break;
      case SearchMode::BLOCK_MODE:
        gen_loops_block();
        break;
      case SearchMode::SUPER_MODE:
        gen_loops_super();
        break;
    }
  }
}

// Try all allowed throw values at the current pattern position `pos`,
// recursively continuing until (a) a pattern is found, or (b) we determine
// that we can't generate a path of length `l` or longer from our current
// position.
//
// This version is for NORMAL mode.

void Worker::gen_loops_normal() {
  if (exactflag && pos >= l)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < outdegree[from]; ++col) {
    const int to = outmatrix[from][col];
    const int throwval = outthrowval[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    if (to == start_state) {
      pattern[pos] = throwval;
      handle_finished_pattern();
      continue;
    }

    bool valid = true;
    if (throwval != 0 && throwval != h)
      valid = mark_unreachable_states(to);

    if (valid) {
      // we need to go deeper
      pattern[pos] = throwval;
      used[to] = 1;
      ++pos;
      const int old_from = from;
      from = to;
      gen_loops_normal();
      from = old_from;
      --pos;
      used[to] = 0;
    }

    // undo changes made above so we can backtrack
    if (throwval != 0 && throwval != h)
      unmark_unreachable_states(to);

    // see if it's time to check the inbox
    if (++steps_taken >= steps_per_inbox_check && valid && pos > root_pos
          && col < outdegree[from] - 1) {
      // the restrictions on when we enter here are in case we get a message to
      // hand off work to another worker; see split_work_assignment()

      // terminate the pattern at the current position in case we get a
      // STOP_WORKER message and need to unwind back to run()
      pattern[pos + 1] = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    // only a single allowed throw value for `pos` < `root_pos`
    if (pos < root_pos)
      break;
  }
}

// As above, but for BLOCK mode.
//
// Here there is additional structure we impose on the form of the pattern,
// which makes the search generally faster than NORMAL mode.

void Worker::gen_loops_block() {
  if (exactflag && pos >= l)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < outdegree[from]; ++col) {
    const int throwval = outthrowval[from][col];
    const int to = outmatrix[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    const bool linkthrow = (throwval != 0 && throwval != h);
    const int old_blocklength = blocklength;
    const int old_skipcount = skipcount;
    const int old_firstblocklength = firstblocklength;

    // handle checks for link throws and skips
    if (linkthrow) {
      if (firstblocklength >= 0) {
        if (blocklength != (h - 2)) {
          // got a skip
          if (skipcount == skiplimit)
            continue;
          else
            ++skipcount;
        }
      } else {
        // first link throw encountered
        firstblocklength = pos;
      }

      blocklength = 0;
    } else
      ++blocklength;

    bool valid = true;
    if (to == start_state) {
      if (skipcount == skiplimit
            && (blocklength + firstblocklength) != (h - 2))
        valid = false;

      if (valid) {
        pattern[pos] = throwval;
        handle_finished_pattern();
      }
    } else if (valid) {
      if (linkthrow)
        valid = mark_unreachable_states(to);

      if (valid) {
        pattern[pos] = throwval;
        used[to] = 1;
        ++pos;
        const int old_from = from;
        from = to;
        gen_loops_block();
        from = old_from;
        --pos;
        used[to] = 0;
      }

      if (linkthrow)
        unmark_unreachable_states(to);
    }

    // undo changes so we can backtrack
    blocklength = old_blocklength;
    skipcount = old_skipcount;
    firstblocklength = old_firstblocklength;

    if (++steps_taken >= steps_per_inbox_check && valid && pos > root_pos
          && col < outdegree[from] - 1) {
      pattern[pos + 1] = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

// As above, but for SUPER mode.
//
// Since a superprime pattern can only visit a single state in each shift cycle,
// this is the fastest version because so many states are excluded by each
// throw to a new shift cycle.

void Worker::gen_loops_super() {
  if (exactflag && pos >= l)
    return;
  ++nnodes;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < outdegree[from]; ++col) {
    const int throwval = outthrowval[from][col];
    const int to = outmatrix[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;
    if (to < start_state || used[to] != 0)
      continue;

    const bool linkthrow = (throwval != 0 && throwval != h);
    const int to_cycle = cyclenum[to];
    // going to a shift cycle that's already been visited?
    if (linkthrow && cycleused[to_cycle])
      continue;

    const int old_shiftcount = shiftcount;
    // check for shift throw limits
    if (!linkthrow) {
      if (shiftcount == shiftlimit)
        continue;
      else
        ++shiftcount;
    }

    pattern[pos] = throwval;
    if (to == start_state) {
      handle_finished_pattern();
    } else {
      used[to] = 1;
      if (linkthrow)
         cycleused[to_cycle] = true;
      ++pos;
      const int old_from = from;
      from = to;
      gen_loops_super();
      from = old_from;
      --pos;
      if (linkthrow)
        cycleused[to_cycle] = false;
      used[to] = 0;
    }

    // undo changes so we can backtrack
    shiftcount = old_shiftcount;

    if (++steps_taken >= steps_per_inbox_check && pos > root_pos
          && col < outdegree[from] - 1) {
      pattern[pos + 1] = -1;
      process_inbox_running();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

// Return the column number in the `outmatrix[from]` row vector that
// corresponds to the throw value at position `pos` in the pattern. This allows
// us to resume where we left off when loading from a work assignment.

int Worker::load_one_throw() {
  if (pattern[pos] == -1) {
    loading_work = false;
    return 0;
  }

  for (int col = 0; col < outdegree[from]; ++col) {
    if (outthrowval[from][col] == pattern[pos])
      return col;
  }

  // diagnostic information if there's a problem
  std::ostringstream buffer;
  for (int i = 0; i <= pos; ++i)
    print_throw(buffer, pattern[i]);
  std::cerr << "worker: " << worker_id << std::endl
            << "pos: " << pos << std::endl
            << "root_pos: " << root_pos << std::endl
            << "from: " << from << std::endl
            << "state[from]: " << state[from] << std::endl
            << "start_state: " << start_state << std::endl
            << "pattern: " << buffer.str() << std::endl
            << "outthrowval[from][]: ";
  for (int i = 0; i < maxoutdegree; ++i)
    std::cerr << outthrowval[from][i] << ", ";
  std::cerr << std::endl << "outmatrix[from][]: ";
  for (int i = 0; i < maxoutdegree; ++i)
    std::cerr << outmatrix[from][i] << ", ";
  std::cerr << std::endl << "state[outmatrix[from][]]: ";
  for (int i = 0; i < maxoutdegree; ++i)
    std::cerr << state[outmatrix[from][i]] << ", ";
  std::cerr << std::endl;
  assert(false);
}

// Determine the set of throw options available at position `root_pos` in
// the pattern. This list of options is maintained in case we get a request
// to split work.

void Worker::build_rootpos_throw_options(int rootpos_from_state,
      int start_column) {
  root_throwval_options.clear();
  for (int col = start_column; col < outdegree[rootpos_from_state]; ++col)
    root_throwval_options.push_back(outthrowval[rootpos_from_state][col]);
}

// Mark off `throwval` from our set of allowed throw options at position
// `root_pos` in the pattern.
//
// If this exhausts the set of allowed options, then advance `root_pos` by one
// and generate a new set of options. As an invariant we never allow
// `root_throwval_options` to be empty, in case we get a request to split work.
//
// Returns true if the value was found, false otherwise.

bool Worker::mark_off_rootpos_option(int throwval, int to_state) {
  // check to see if this throwval is in our allowed list, and if so remove it
  bool found = false;
  int remaining = 0;
  std::list<int>::iterator iter = root_throwval_options.begin();
  std::list<int>::iterator end = root_throwval_options.end();

  while (iter != end) {
    if (*iter == throwval) {
      found = true;
      iter = root_throwval_options.erase(iter);

      if (verboseflag) {
        std::ostringstream buffer;
        buffer << "worker " << worker_id << " starting root_pos option: ";
        print_throw(buffer, throwval);
        MessageW2C msg;
        msg.type = messages_W2C::WORKER_STATUS;
        msg.worker_id = worker_id;
        msg.meta = buffer.str();
        message_coordinator(msg);
      }
    } else {
      ++iter;
      ++remaining;
    }
  }
  if (!found && !loading_work)
    return false;

  if (remaining == 0) {
    // using our last option at this root level, go one step deeper
    ++root_pos;
    notify_coordinator_rootpos();
    build_rootpos_throw_options(to_state, 0);
  }

  return true;
}

// Mark all of the states as used that are excluded by a throw from state
// `from` to state `to_state`.
//
// Returns false if the number of newly-excluded states implies that we can't
// finish a pattern of at least length `l` from our current position. Returns
// true otherwise.

bool inline Worker::mark_unreachable_states(int to_state) {
  bool valid = true;

  // 1. kill states downstream in `from` shift cycle that end in 'x'
  int j = h - 2;
  unsigned long tempstate = state[from];
  int cnum = cyclenum[from];

  do {
    if (used[cyclepartner[from][j]]++ == 0 && deadstates[cnum]++ >= 1
          && --max_possible < l)
      valid = false;
    --j;
    tempstate >>= 1;
  } while (tempstate & 1L);

  // 2. kill states upstream in 'to' shift cycle that start with '-'
  j = 0;
  tempstate = state[to_state];
  cnum = cyclenum[to_state];

  do {
    if (used[cyclepartner[to_state][j]]++ == 0 && deadstates[cnum]++ >= 1
          && --max_possible < l)
      valid = false;
    ++j;
    tempstate = (tempstate << 1) & allbits;
  } while ((tempstate & highestbit) == 0);

  return valid;
}

// Reverse the marking operation above, so we can backtrack.

void inline Worker::unmark_unreachable_states(int to_state) {
  int j = h - 2;
  unsigned long tempstate = state[from];
  int cnum = cyclenum[from];

  do {
    if (--used[cyclepartner[from][j]] == 0 && --deadstates[cnum] >= 1)
      ++max_possible;
    --j;
    tempstate >>= 1;
  } while (tempstate & 1L);

  j = 0;
  tempstate = state[to_state];
  cnum = cyclenum[to_state];

  do {
    if (--used[cyclepartner[to_state][j]] == 0 && --deadstates[cnum] >= 1)
      ++max_possible;
    ++j;
    tempstate = (tempstate << 1) & allbits;
  } while ((tempstate & highestbit) == 0);
}

void Worker::handle_finished_pattern() {
  ++ntotal;

  if ((pos + 1) >= l) {
    if (longestflag)
      l = pos + 1;
    report_pattern();
  }

  if ((pos + 1) > longest_found) {
    longest_found = pos + 1;
    notify_coordinator_longest();
  }
}

//------------------------------------------------------------------------------
// Output a pattern during run
//------------------------------------------------------------------------------

// Send a message to the coordinator with the completed pattern. Note that all
// console output is done by the coordinator, not the worker threads.

void Worker::report_pattern() const {
  std::ostringstream buffer;

  if (groundmode != 1) {
    if (start_state == 1)
      buffer << "  ";
    else
      buffer << "* ";
  }

  for (int i = 0; i <= pos; ++i) {
    int throwval = (dualflag ? (h - pattern[pos - i]) : pattern[i]);
    print_throw(buffer, throwval);
  }

  if (invertflag) {
    buffer << " : ";
    if (dualflag)
      print_inverse_dual(buffer);
    else
      print_inverse(buffer);
  }

  if (start_state != 1)
    buffer << " *";

  MessageW2C msg;
  msg.type = messages_W2C::SEARCH_RESULT;
  msg.worker_id = worker_id;
  msg.pattern = buffer.str();
  msg.length = pos + 1;
  message_coordinator(msg);
}

void Worker::print_throw(std::ostringstream& buffer, int val) const {
  if (!noplusminusflag && val == 0) {
    buffer << '-';
    return;
  } else if (!noplusminusflag && val == h) {
    buffer << '+';
    return;
  }

  if (val < 10)
    buffer << static_cast<char>(val + '0');
  else
    buffer << static_cast<char>(val - 10 + 'a');
}

// Write the inverse of the current pattern to a buffer. If the inverse is not
// well-defined, indicate this.

void Worker::print_inverse(std::ostringstream& buffer) const {
  // first decide on a starting state
  // state to avoid on first shift cycle:
  int avoid = reverse_state(start_state);

  // how many adjacent states to avoid also?
  int shifts = 0;
  int index = 0;
  while (index <= pos && (pattern[index] == 0 || pattern[index] == h)) {
    ++index;
    ++shifts;
  }

  if (index == pos) {
    buffer << "no inverse";
    return;
  }

  int start = numstates;
  int temp = 0;
  for (int i = cycleperiod[cyclenum[avoid]] - 2; i >= shifts; --i) {
    if (cyclepartner[avoid][i] < start) {
      start = cyclepartner[avoid][i];
      temp = i;
    }
  }
  if (start == numstates) {
    buffer << "no inverse";
    return;
  }

  // number of shift throws printed at beginning
  int shiftthrows = temp - shifts;
  int endshiftthrows = cycleperiod[cyclenum[start]] - shifts - 2 - shiftthrows;
  int current = start;

  do {
    // print shift throws
    while (shiftthrows--) {
      print_throw(buffer, (state[current] & 1) ? h : 0);
      current = cyclepartner[current][cycleperiod[cyclenum[current]] - 2];
    }

    // print the link throw
    const int throwval = h - pattern[index++];
    print_throw(buffer, throwval);

    unsigned long tempstate = (state[current] / 2) | (1L << (throwval - 1));
    bool errorflag = true;
    for (int i = 1; i <= numstates; ++i) {
      if (state[i] == tempstate) {
        current = i;
        errorflag = false;
        break;
      }
    }
    assert(!errorflag);

    // find how many shift throws in the next block
    shifts = 0;
    while (index <= pos && (pattern[index] == 0 || pattern[index] == h)) {
      ++index;
      ++shifts;
    }
    shiftthrows = cycleperiod[cyclenum[current]] - shifts - 2;
  } while (index <= pos);

  // correct for any shift throws at the end, which are part of the first
  // shift cycle
  endshiftthrows -= shifts;

  // finish printing shift throws in first shift cycle
  while (endshiftthrows--) {
    print_throw(buffer, (state[current] & 1) ? h : 0);
    current = cyclepartner[current][cycleperiod[cyclenum[current]] - 2];
  }
}

// As above, but when we're using the dual version of the juggling graph.

void Worker::print_inverse_dual(std::ostringstream& buffer) const {
  // inverse was found in dual space, so we have to transform as we read it
  // first decide on a starting state
  // dual state to avoid on first shift cycle:
  int avoid = reverse_state(start_state);

  // how many adjacent states to avoid also?
  int shifts = 0;
  int index = pos;
  while (index >= 0 && (pattern[index] == 0 || pattern[index] == h)) {
    --index;
    ++shifts;
  }

  if (index < 0) {
    buffer << "no inverse";
    return;
  }

  int start = numstates;
  int temp = 0;
  for (int i = 0; i <= cycleperiod[cyclenum[avoid]] - shifts - 2; ++i) {
    if (cyclepartner[avoid][i] < start) {
      start = cyclepartner[avoid][i];
      temp = i;
    }
  }
  if (start == numstates) {
    buffer << "no inverse";
    return;
  }

  // number of shift throws printed at beginning
  int shiftthrows = cycleperiod[cyclenum[start]] - shifts - 2 - temp;
  int endshiftthrows = temp;
  int current = start;

  do {
    // first print shift throws
    while (shiftthrows--) {
      print_throw(buffer, (state[current] & (1L << (h - 1))) ? 0 : h);
      current = cyclepartner[current][0];
    }

    // print the link throw
    temp = pattern[index--];
    print_throw(buffer, temp);

    unsigned long tempstate = (state[current] * 2 + 1) ^ (1L << (h - temp));
    bool errorflag = true;
    for (int i = 1; i <= numstates; ++i) {
      if (state[i] == tempstate) {
        current = i;
        errorflag = false;
        break;
      }
    }
    assert(!errorflag);

    // how many shift throws in the next block
    shifts = 0;
    while (index >= 0 && (pattern[index] == 0 || pattern[index] == h)) {
      --index;
      ++shifts;
    }
    shiftthrows = cycleperiod[cyclenum[current]] - shifts - 2;
  } while (index >= 0);

  // correct for any shift throws at the end, which are part of the first
  // shift cycle
  endshiftthrows -= shifts;

  // finish printing shift throws in first shift cycle
  while (endshiftthrows--) {
    print_throw(buffer, (state[current] & (1L << (h - 1))) ? 0 : h);
    current = cyclepartner[current][0];
  }
}

// Find the reverse of a given state, where both the input and output are
// referenced to the state number (i.e., index in the state[] array).
//
// For example 'xx-xxx---' becomes '---xxx-xx' under reversal.

int Worker::reverse_state(int statenum) const {
  unsigned long temp = 0;
  unsigned long mask1 = 1L;
  unsigned long mask2 = 1L << (h - 1);

  while (mask2) {
    assert(statenum >= 0 && statenum < numstates);
    if (state[statenum] & mask2)
      temp |= mask1;
    mask1 <<= 1;
    mask2 >>= 1;
  }

  for (int i = 1; i <= numstates; ++i) {
    if (state[i] == temp)
      return i;
  }
  assert(false);
}

// Return a text representation of a given state number

std::string Worker::state_string(int statenum) const {
  std::string result;
  unsigned long value = state[statenum];
  for (int i = 0; i < h; ++i) {
    result += (value & 1 ? 'x' : '-');
    value >>= 1;
  }
  return result;
}

//------------------------------------------------------------------------------
// Prep core data structures during construction
//------------------------------------------------------------------------------

// Allocate all arrays used by the worker and initialize to default values.

void Worker::allocate_arrays() {
  pattern = new int[numstates + 1];
  used = new int[numstates + 1];
  outdegree = new int[numstates + 1];
  indegree = new int[numstates + 1];
  cyclenum = new int[numstates + 1];
  state = new unsigned long[numstates + 1];
  cycleperiod = new int[numstates + 1];
  cycleused = new bool[numstates + 1];
  deadstates = new int[numstates + 1];

  for (int i = 0; i <= numstates; ++i) {
    pattern[i] = -1;
    used[i] = 0;
    outdegree[i] = 0;
    indegree[i] = 0;
    cyclenum[i] = 0;
    state[i] = 0L;
    cycleperiod[i] = 0;
    cycleused[i] = false;
    deadstates[i] = 0;
  }

  outmatrix = new int*[numstates + 1];
  outthrowval = new int*[numstates + 1];
  inmatrix = new int*[numstates + 1];
  cyclepartner = new int*[numstates + 1];

  for (int i = 0; i <= numstates; ++i) {
    outmatrix[i] = new int[maxoutdegree];
    outthrowval[i] = new int[maxoutdegree];
    inmatrix[i] = new int[maxindegree];
    cyclepartner[i] = new int[h];

    for (int j = 0; j < maxoutdegree; ++j) {
      outmatrix[i][j] = 0;
      outthrowval[i][j] = 0;
    }
    for (int j = 0; j < maxindegree; ++j)
      inmatrix[i][j] = 0;
    for (int j = 0; j < h; ++j)
      cyclepartner[i][j] = 0;
  }
}

void Worker::delete_arrays() {
  for (int i = 0; i <= numstates; ++i) {
    if (outmatrix) {
      delete[] outmatrix[i];
      outmatrix[i] = nullptr;
    }
    if (outthrowval) {
      delete[] outthrowval[i];
      outthrowval[i] = nullptr;
    }
    if (inmatrix) {
      delete[] inmatrix[i];
      inmatrix[i] = nullptr;
    }
    if (cyclepartner) {
      delete[] cyclepartner[i];
      cyclepartner[i] = nullptr;
    }
  }

  delete[] outmatrix;
  delete[] outthrowval;
  delete[] inmatrix;
  delete[] cyclepartner;
  delete[] pattern;
  delete[] used;
  delete[] outdegree;
  delete[] indegree;
  delete[] cyclenum;
  delete[] state;
  delete[] cycleperiod;
  delete[] cycleused;
  delete[] deadstates;
  outmatrix = nullptr;
  outthrowval = nullptr;
  inmatrix = nullptr;
  cyclepartner = nullptr;
  pattern = nullptr;
  used = nullptr;
  outdegree = nullptr;
  indegree = nullptr;
  cyclenum = nullptr;
  state = nullptr;
  cycleperiod = nullptr;
  cycleused = nullptr;
  deadstates = nullptr;
}

// Find the number of states (vertices) in the juggling graph, for a given
// number of balls and maximum throw value. This is just (h choose n).

int Worker::num_states(int n, int h) {
  int result = 1;
  for (int denom = 1; denom <= std::min(n, h - n); ++denom)
    result = (result * (h - denom + 1)) / denom;
  return result;
}

// Generate the list of all possible states into the state[] array.
//
// Returns the number of states found.

int Worker::gen_states(unsigned long* state, int num, int pos, int left, int h,
      int ns) {
  if (left > (pos + 1))
    return num;

  if (pos == 0) {
    if (left)
      state[num + 1] |= 1L;
    else
      state[num + 1] &= ~1L;

    if (num < (ns - 1))
      state[num + 2] = state[num + 1];
    return (num + 1);
  }

  state[num + 1] &= ~(1L << pos);
  num = gen_states(state, num, pos - 1, left, h, ns);
  if (left > 0) {
    state[num + 1] |= (1L << pos);
    num = gen_states(state, num, pos - 1, left - 1, h, ns);
  }

  return num;
}

// Generate arrays describing the shift cycles of the juggling graph.
//
// - Which shift cycle number a given state belongs to:
//         cyclenum[statenum] --> cyclenum
// - The period of a given shift cycle number:
//         cycleperiod[cyclenum] --> period
// - The other states on a given state's shift cycle:
//         cyclepartner[statenum][i] --> statenum  (where i < h)

void Worker::find_shift_cycles() {
  const unsigned long lowerbits = highestbit - 1;
  int cycleindex = 0;

  for (int i = 1; i <= numstates; ++i) {
    unsigned long statebits = state[i];
    bool periodfound = false;
    bool newshiftcycle = true;
    int cycleper = h;

    for (int j = 0; j < h; ++j) {
      if (statebits & highestbit)
        statebits = (statebits & lowerbits) << 1 | 1L;
      else
        statebits <<= 1;

      int k = 1;
      for (; k <= numstates; ++k) {
        if (state[k] == statebits)
          break;
      }
      assert(k <= numstates);

      cyclepartner[i][j] = k;
      if (k == i && !periodfound) {
        cycleper = j + 1;
        periodfound = true;
      } else if (k < i)
        newshiftcycle = false;
    }

    if (newshiftcycle) {
      for (int j = 0; j < h; j++)
        cyclenum[cyclepartner[i][j]] = cycleindex;
      cycleperiod[cycleindex++] = cycleper;
    }
  }
  numcycles = cycleindex;
}

// Generate matrices describing the structure of the juggling graph:
//
// - Outward degree from each state (vertex) in the graph:
//         outdegree[statenum] --> degree
// - Outward connections from each state:
//         outmatrix[statenum][col] --> statenum  (where col < outdegree)
// - Throw values corresponding to outward connections from each state:
//         outthrowval[statenum][col] --> throw
// - Inward degree to each state in the graph:
//         indegree[statenum] --> degree
// - Inward connections to each state:
//         inmatrix[statenum][col] --> statenum  (where col < indegree)
//
// outmatrix[][] == 0 indicates no connection.

void Worker::gen_matrices(const std::vector<bool>& xarray) {
  const bool allow_linkthrows_within_cycle = (mode != SearchMode::SUPER_MODE);

  for (int i = 1; i <= numstates; ++i) {
    int outthrownum = 0;
    int inthrownum = 0;

    // first take care of outgoing throws
    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      if (j == 0) {
        if (!(state[i] & 1L)) {
          unsigned long temp = state[i] >> 1;
          bool found = false;

          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              outmatrix[i][outthrownum] = k;
              outthrowval[i][outthrownum++] = j;
              ++outdegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else if (state[i] & 1L) {
        unsigned long temp = (unsigned long)1L << (j - 1);
        unsigned long temp2 = (state[i] >> 1);

        if (!(temp2 & temp)) {
          temp |= temp2;

          bool found = false;
          int k = 1;
          for (; k <= numstates; ++k) {
            if (state[k] == temp) {
              found = true;
              break;
            }
          }
          assert(found);
          if (j == h || allow_linkthrows_within_cycle
              || cyclenum[i] != cyclenum[k]) {
            outmatrix[i][outthrownum] = k;
            outthrowval[i][outthrownum++] = j;
            ++outdegree[i];
          }
        }
      }
    }

    // then take care of ingoing throws
    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      if (j == 0) {
        if (!(state[i] & (1L << (h - 1)))) {
          unsigned long temp = state[i] << 1;

          bool found = false;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              inmatrix[i][inthrownum++] = k;
              ++indegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else if (j == h) {
        if (state[i] & (1L << (h - 1))) {
          unsigned long temp = state[i] ^ (1L << (h - 1));
          temp = (temp << 1) | 1L;

          bool found = false;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              inmatrix[i][inthrownum++] = k;
              ++indegree[i];
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else {
        if ((state[i] & (1L << (j - 1))) && (!(state[i] & (1L << (h - 1))))) {
          unsigned long temp = state[i] ^ (1L << (j - 1));
          temp = (temp << 1) | 1L;

          bool found = false;
          int k = 1;
          for (; k <= numstates; ++k) {
            if (state[k] == temp) {
              found = true;
              break;
            }
          }
          assert(found);
          if (allow_linkthrows_within_cycle
              || cyclenum[i] != cyclenum[k]) {
            inmatrix[i][inthrownum++] = k;
            ++indegree[i];
          }
        }
      }
    }
  }
}
