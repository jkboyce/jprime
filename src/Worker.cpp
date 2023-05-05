
#include "Worker.hpp"
#include "Coordinator.hpp"
#include "Messages.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <cassert>
#include <ctime>
#include <iomanip>

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
  prepcorearrays(config.xarray);
  maxlength = (mode == SUPER_MODE) ? (numcycles + shiftlimit)
      : (numstates - numcycles);
  highmask = 1L << (config.h - 1);
  allmask = (1L << config.h) - 1;

  if (l > maxlength) {
    std::cout << "No patterns longer than " << maxlength << " are possible"
              << std::endl;
    std::exit(0);
  }
}

Worker::~Worker() {
  for (int i = 0; i <= numstates; ++i) {
    delete outmatrix[i];
    delete outthrowval[i];
    delete inmatrix[i];
    delete partners[i];
  }
  delete outmatrix;
  delete outthrowval;
  delete inmatrix;
  delete partners;
  delete used;
  delete outdegree;
  delete indegree;
  delete cyclenum;
  delete state;
  delete pattern;
  delete cycleperiod;
  delete deadstates;
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

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
        // leave in the inbox
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

    timespec start_ts;
    timespec_get(&start_ts, TIME_UTC);
    // for calibrating how often to check the inbox while running
    if (calibrations_remaining > 0)
      last_ts = start_ts;

    // complete the new work assignment
    try {
      gen_patterns();
      record_elapsed_time(start_ts);
    } catch (const JdeepStopException& jdse) {
      // a STOP_WORKER message while running unwinds back here; send any
      // remaining work back to the Coordinator
      record_elapsed_time(start_ts);
      MessageW2C msg;
      msg.type = messages_W2C::RETURN_WORK;
      msg.worker_id = worker_id;
      msg.ntotal = ntotal;
      msg.nnodes = nnodes;
      msg.numstates = numstates;
      msg.maxlength = maxlength;
      msg.secs_elapsed_working = secs_elapsed_working;
      msg.assignment = get_work_assignment();
      message_coordinator(msg);
      break;
    }

    // empty the inbox
    inbox_lock.lock();
    inbox = std::queue<MessageC2W>();
    inbox_lock.unlock();

    // notify coordinator that we're idle
    MessageW2C msg;
    msg.type = messages_W2C::WORKER_IDLE;
    msg.worker_id = worker_id;
    msg.ntotal = ntotal;
    msg.nnodes = nnodes;
    msg.numstates = numstates;
    msg.maxlength = maxlength;
    msg.secs_elapsed_working = secs_elapsed_working;
    message_coordinator(msg);
    ntotal = 0;
    nnodes = 0;
    secs_elapsed_working = 0;
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

void Worker::message_coordinator(const MessageW2C& msg1,
      const MessageW2C& msg2) const {
    coordinator->inbox_lock.lock();
    coordinator->inbox.push(msg1);
    coordinator->inbox.push(msg2);
    coordinator->inbox_lock.unlock();
}

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

  if (stopping_work)
    throw JdeepStopException();
}

void Worker::record_elapsed_time(timespec& start_ts) {
  timespec end_ts;
  timespec_get(&end_ts, TIME_UTC);
  double runtime = ((double)end_ts.tv_sec + 1.0e-9 * end_ts.tv_nsec) -
      ((double)start_ts.tv_sec + 1.0e-9 * start_ts.tv_nsec);
  secs_elapsed_working += runtime;
}

void Worker::calibrate_inbox_check() {
  timespec current_ts;
  timespec_get(&current_ts, TIME_UTC);
  double time_spent =
      ((double)current_ts.tv_sec + 1.0e-9 * current_ts.tv_nsec) -
      ((double)last_ts.tv_sec + 1.0e-9 * last_ts.tv_nsec);
  last_ts = current_ts;
  --calibrations_remaining;

  steps_per_inbox_check =
      static_cast<int>(static_cast<double>(steps_per_inbox_check) *
      secs_per_inbox_check_target / time_spent);
}

void Worker::process_split_work_request(const MessageC2W& msg) {
  MessageW2C msg2;
  msg2.type = messages_W2C::RETURN_WORK;
  msg2.worker_id = worker_id;
  msg2.assignment = split_work_assignment(msg.split_alg);
  msg2.ntotal = ntotal;
  msg2.nnodes = nnodes;
  msg2.numstates = numstates;
  msg2.maxlength = maxlength;
  msg2.secs_elapsed_working = secs_elapsed_working;
  ntotal = 0;
  nnodes = 0;
  secs_elapsed_working = 0;

  if (verboseflag) {
    std::ostringstream sstr;
    sstr << "worker " << worker_id << " remaining work after split:" << std::endl;
    sstr << "  " << get_work_assignment();

    MessageW2C msg3;
    msg3.type = messages_W2C::WORKER_STATUS;
    msg3.worker_id = worker_id;
    msg3.meta = sstr.str();
    message_coordinator(msg2, msg3);
  } else
    message_coordinator(msg2);
}

void Worker::load_work_assignment(const WorkAssignment& wa) {
  loading_work = true;

  start_state = wa.start_state;
  end_state = wa.end_state;
  if (start_state == -1)
    start_state = (groundmode == 2) ? 2 : 1;
  if (end_state == -1)
    end_state = (groundmode == 1) ? 1 : numstates;

  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  longest_found = 0;
  assert(root_throwval_options.size() > 0);
  assert(pos == 0);

  for (int i = 0; i <= numstates; ++i) {
    pattern[i] = (i < wa.partial_pattern.size()) ? wa.partial_pattern[i] : -1;
    assert(mode == SUPER_MODE || used[i] == 0);
  }
}

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


void Worker::notify_coordinator_rootpos() {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.root_pos = root_pos;
  message_coordinator(msg);
}

void Worker::notify_coordinator_longest() {
  MessageW2C msg;
  msg.type = messages_W2C::WORKER_STATUS;
  msg.worker_id = worker_id;
  msg.longest_found = longest_found;
  message_coordinator(msg);
}

//------------------------------------------------------------------------------
// Work-splitting algorithms
//------------------------------------------------------------------------------

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
      static_cast<int>(0.51 + f * root_throwval_options.size());
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
    // Gave away all our throw options at this `root_pos`
    //
    // We need to find the shallowest depth `new_root_pos` where there are
    // unexplored throw options. We have no more options at the current
    // root_pos, so new_root_pos > root_pos
    //
    // We're also at a point in the search where we know there are unexplored
    // options remaining at the current value of `pos` (by virtue of how we got
    // here), and that pos > root_pos.
    //
    // So we know there must be a value of `new_root_pos` with the properties we
    // need, in the range root_pos < new_root_pos <= pos;

    int from_state = start_state;
    int new_root_pos = -1;
    int col = 0;

    // have to start from the beginning because we don't record the traversed
    // states as we build the pattern
    for (int pos2 = 0; pos2 <= pos; ++pos2) {
      const int throwval = pattern[pos2];
      for (col = 0; col < outdegree[from_state]; ++col) {
        if (throwval == outthrowval[from_state][col])
          break;
      }
      if (col == outdegree[from_state]) {
        std::cout << "pos2 = " << pos2
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

    root_throwval_options.clear();
    for (; col < outdegree[from_state]; ++col) {
      const int throwval = outthrowval[from_state][col];
      if (throwval != pattern[root_pos])
        root_throwval_options.push_back(outthrowval[from_state][col]);
    }
    assert(root_throwval_options.size() > 0);
  }

  return wa;
}

//------------------------------------------------------------------------------
// Search the juggling graph for patterns
//------------------------------------------------------------------------------

void Worker::gen_patterns() {
  for (; start_state <= end_state; ++start_state) {
    if (longestflag && start_state > (maxlength - l + 1))
      continue;
    from = start_state;
    pos = 0;
    firstblocklength = -1; // -1 signals unknown
    skipcount = 0;
    longest_found = 0;
    notify_coordinator_longest();

    if (!loading_work) {
      // reset variables for each new starting state
      root_pos = 0;
      notify_coordinator_rootpos();

      root_throwval_options.clear();
      for (int col = 0; col < maxoutdegree; ++col) {
        if (outmatrix[from][col] >= start_state) {
          root_throwval_options.push_back(outthrowval[from][col]);
          if (outthrowval[from][col] == 0)
            break;
        }
      }
      if (root_throwval_options.size() == 0)
        continue;
    }

    for (int i = 0; i <= numstates; ++i)
      used[i] = 0;

    switch (mode) {
      case NORMAL_MODE:
        max_possible = maxlength;
        gen_loops_normal();
        break;
      case BLOCK_MODE:
        if (longestflag) {
          max_possible = numstates - start_state + 1;
          if (l > max_possible)
            continue;
        } else
          max_possible = numstates;
        gen_loops_block();
        break;
      case SUPER_MODE:
        for (int i = 0; i < (n - 1); ++i)
          used[partners[start_state][i]] = 1;
        max_possible = numstates;
        gen_loops_super();
        break;
    }
  }
}

void Worker::gen_loops_normal() {
  if (exactflag && pos >= l)
    return;
  if (pos < root_pos && !loading_work)
    return;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];
    if (used[to] != 0 || to < start_state)
      continue;
    const int throwval = outthrowval[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;

    if (to == start_state) {
      handle_finished_pattern(throwval);
      continue;
    }

    bool valid = true;
    int old_max_possible = max_possible;

    if (throwval > 0 && throwval < h) {
      // throwing to a new shift cycle:
      // 1. kill states downstream in 'from' cycle that end in 'x'
      int j = h - 2;
      unsigned long tempstate = state[from];
      int cnum = cyclenum[from];

      do {
        if (used[partners[from][j]]++ == 0 && deadstates[cnum]++ >= 1
              && --max_possible < l)
          valid = false;

        --j;
        tempstate >>= 1;
      } while (tempstate & 1L);

      // 2. kill states upstream in 'to' cycle that start with '-'
      j = 0;
      tempstate = state[to];
      cnum = cyclenum[to];

      do {
        if (used[partners[to][j]]++ == 0 && deadstates[cnum]++ >= 1
              && --max_possible < l)
          valid = false;

        ++j;
        tempstate = (tempstate << 1) & allmask;
      } while ((tempstate & highmask) == 0);
    }

    if (valid) {
      pattern[pos] = throwval;

      used[to] = 1;
      ++pos;
      int old_from = from;
      from = to;

      if (!loading_work)
        ++nnodes;
      gen_loops_normal();

      from = old_from;
      --pos;
      used[to] = 0;
    }

    // undo changes made above, so we can backtrack
    if (throwval > 0 && throwval < h) {
      int j = h - 2;
      unsigned long tempstate = state[from];
      int cnum = cyclenum[from];

      do {
        if (--used[partners[from][j]] == 0 && --deadstates[cnum] >= 1)
          ++max_possible;

        --j;
        tempstate >>= 1;
      } while (tempstate & 1L);

      j = 0;
      tempstate = state[to];
      cnum = cyclenum[to];

      do {
        if (--used[partners[to][j]] == 0 && --deadstates[cnum] >= 1)
          ++max_possible;

        ++j;
        tempstate = (tempstate << 1) & allmask;
      } while ((tempstate & highmask) == 0);
    }

    assert(old_max_possible == max_possible);

    if (++steps_taken >= steps_per_inbox_check &&
          pos > root_pos && col < outdegree[from] - 1) {
      // the restrictions on when we enter here are in case we get a message to
      // hand off work to another thread; see split_off_work_assignment()

      // terminate the pattern at the current position in case we get a
      // STOP_WORKER message and need to unwind back to run()
      if (valid)
        pattern[pos + 1] = -1;
      else
        pattern[pos] = -1;

      process_inbox_running();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

void Worker::gen_loops_block() {
  if (exactflag && pos >= l)
    return;
  if (pos < root_pos && !loading_work)
    return;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];
    if (used[to] != 0 || to < start_state)
      continue;
    const int throwval = outthrowval[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;

    bool valid = true;

    const int oldblocklength = blocklength;
    const int oldskipcount = skipcount;
    const int oldfirstblocklength = firstblocklength;

    // handle checks for block throws and skips
    if (throwval > 0 && throwval < h) {
      if (firstblocklength >= 0) {
        if (blocklength != (h - 2)) {
          // got a skip
          if (skipcount == skiplimit)
            valid = false;
          else
            ++skipcount;
        }
      } else {
        // first block throw encountered
        firstblocklength = pos;
      }

      blocklength = 0;
    } else
      ++blocklength;

    if (to == start_state) {
      if (skipcount == skiplimit &&
            (blocklength + firstblocklength) != (h - 2))
        valid = false;

      if (valid)
        handle_finished_pattern(throwval);
    } else if (valid) {
      pattern[pos] = throwval;

      used[to] = 1;
      ++pos;
      const int old_from = from;
      from = to;

      if (!loading_work)
        ++nnodes;
      gen_loops_block();

      from = old_from;
      --pos;
      used[to] = 0;
    }

    // undo changes so we can backtrack
    blocklength = oldblocklength;
    skipcount = oldskipcount;
    firstblocklength = oldfirstblocklength;

    if (++steps_taken >= steps_per_inbox_check &&
          pos > root_pos && col < outdegree[from] - 1) {
      if (valid)
        pattern[pos + 1] = -1;
      else
        pattern[pos] = -1;

      process_inbox_running();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

void Worker::gen_loops_super() {
  if (exactflag && pos >= l)
    return;
  if (pos < root_pos && !loading_work)
    return;

  int col = (loading_work ? load_one_throw() : 0);
  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];
    if (used[to] != 0 || to < start_state)
      continue;
    const int throwval = outthrowval[from][col];
    if (pos == root_pos && !mark_off_rootpos_option(throwval, to))
      continue;

    bool valid = true;
    const int oldshiftcount = shiftcount;

    // handle checks for shift throws and limits
    if (throwval == 0 || throwval == h) {
      if (shiftcount == shiftlimit)
        valid = false;
      else
        ++shiftcount;
    }

    if (to == start_state) {
      if (valid)
        handle_finished_pattern(throwval);
    } else if (valid) {
      pattern[pos] = throwval;

      const int oldusedvalue = used[to];
      used[to] = 1;
      for (int j = 0; j < (h - 1); ++j) {
        if (used[partners[to][j]] < 1)
          --used[partners[to][j]];
      }
      ++pos;
      int old_from = from;
      from = to;

      if (!loading_work)
        ++nnodes;
      gen_loops_super();

      from = old_from;
      --pos;
      for (int j = 0; j < (h - 1); ++j) {
        if (used[partners[to][j]] < 0)
          ++used[partners[to][j]];
      }
      used[to] = oldusedvalue;
    }

    // undo changes so we can backtrack
    shiftcount = oldshiftcount;

    if (++steps_taken >= steps_per_inbox_check &&
          pos > root_pos && col < outdegree[from] - 1) {
      if (valid)
        pattern[pos + 1] = -1;
      else
        pattern[pos] = -1;

      process_inbox_running();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

int Worker::load_one_throw() {
  int col = 0;

  if (pos == root_pos) {
    // Coordinator may have given us throw values that don't work at this
    // point in the state graph, so remove any bad elements from the list
    std::list<int>::iterator iter = root_throwval_options.begin();
    std::list<int>::iterator end = root_throwval_options.end();

    while (iter != end) {
      bool allowed = false;
      for (int col2 = 0; col2 < maxoutdegree; ++col2) {
        if (outmatrix[from][col2] < start_state)
          continue;
        if (*iter == outthrowval[from][col2]) {
          allowed = true;
          break;
        }
      }
      if (allowed)
        ++iter;
      else
        iter = root_throwval_options.erase(iter);
    }

    /*
    if (root_throwval_options.size() == 0) {
      std::ostringstream buffer;
      for (int i = 0; i <= pos; ++i) {
        int throwval = (dualflag ? (h - pattern[pos - i]) : pattern[i]);
        print_throw(buffer, throwval);
      }
      std::cout << "worker: " << worker_id << std::endl
                << "pos: " << pos << std::endl
                << "root_pos: " << root_pos << std::endl
                << "from: " << from << std::endl
                << "state[from]: " << state[from] << std::endl
                << "start_state: " << start_state << std::endl
                << "pattern: " << buffer.str() << std::endl
                << "outthrowval[from][]: ";
      for (int i = 0; i < maxoutdegree; ++i)
        std::cout << outthrowval[from][i] << ", ";
      std::cout << std::endl << "outmatrix[from][]: ";
      for (int i = 0; i < maxoutdegree; ++i)
        std::cout << outmatrix[from][i] << ", ";
      std::cout << std::endl;
      std::cout << "throw options: ";
      for (int val : rto_copy)
        std::cout << val << " ";
      std::cout << std::endl;
    }
    */
    assert(root_throwval_options.size() > 0);
  }

  if (pattern[pos] == -1) {
    loading_work = false;
  } else {
    for (; col < maxoutdegree; ++col) {
      if (outmatrix[from][col] < start_state)
        continue;
      if (outthrowval[from][col] == pattern[pos])
        break;
    }

    /*
    if (col == maxoutdegree) {
      std::ostringstream buffer;
      for (int i = 0; i <= pos; ++i) {
        int throwval = (dualflag ? (h - pattern[pos - i]) : pattern[i]);
        print_throw(buffer, throwval);
      }
      std::cout << "worker: " << worker_id << std::endl
                << "pos: " << pos << std::endl
                << "root_pos: " << root_pos << std::endl
                << "from: " << from << std::endl
                << "state[from]: " << state[from] << std::endl
                << "start_state: " << start_state << std::endl
                << "pattern: " << buffer.str() << std::endl
                << "outthrowval[from][]: ";
      for (int i = 0; i < maxoutdegree; ++i)
        std::cout << outthrowval[from][i] << ", ";
      std::cout << std::endl << "outmatrix[from][]: ";
      for (int i = 0; i < maxoutdegree; ++i)
        std::cout << outmatrix[from][i] << ", ";
      std::cout << std::endl << "state[outmatrix[from][]]: ";
      for (int i = 0; i < maxoutdegree; ++i)
        std::cout << state[outmatrix[from][i]] << ", ";
      std::cout << std::endl;
    }
    */
    assert(col != maxoutdegree);
  }

  return col;
}

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

    for (int col = 0; col < maxoutdegree; ++col) {
      if (outmatrix[to_state][col] < start_state)
        continue;
      root_throwval_options.push_back(outthrowval[to_state][col]);
    }
  }

  return true;
}

void Worker::handle_finished_pattern(int throwval) {
  ++ntotal;

  if (pos >= (l - 1) || l == 0) {
    if (longestflag && pos >= l)
      l = pos + 1;
    pattern[pos] = throwval;
    report_pattern();
  }

  if (pos > (longest_found - 1)) {
    longest_found = pos + 1;
    notify_coordinator_longest();
  }
}

//------------------------------------------------------------------------------
// Output a pattern during run
//------------------------------------------------------------------------------

void Worker::report_pattern() const {
  std::ostringstream buffer;

  if (groundmode != 1) {
    if (start_state == 1)
      buffer << "  ";
    else
      buffer << "* ";
  }

  const bool plusminus = ((mode == NORMAL_MODE && longestflag) ||
                          mode == BLOCK_MODE);

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

#ifdef STATELIST
  fprintf(fpout, "  missing states:\n");

  for (int i = start_state + 1; i <= numstates; ++i) {
    if (used[i] == 1)
      continue;

    unsigned long temp2 = state[i];
    for (int j = 0; j < h; ++j) {
      if (temp2 & 1)
        fprintf(fpout, "x");
      else
        fprintf(fpout, "-");
      temp2 >>= 1;
    }
    fprintf(fpout, "\n");
  }
#endif
}

void Worker::print_throw(std::ostringstream& buffer, int val) const {
  const bool plusminus = ((mode == NORMAL_MODE && longestflag) ||
                          mode == BLOCK_MODE);
  if (plusminus && val == 0) {
    buffer << '-';
    return;
  } else if (plusminus && val == h) {
    buffer << '+';
    return;
  }

  if (val < 10)
    buffer << static_cast<char>(val + '0');
  else
    buffer << static_cast<char>(val - 10 + 'a');
}

void Worker::print_inverse(std::ostringstream& buffer) const {
  // first decide on a starting state
  // state to avoid on first shift cycle:
  int avoid = reverse_state(start_state);

  // how many adjacent states to avoid also?
  int shifts = 0;
  int index = 0;
  while (index <= pos &&
         (pattern[index] == 0 || pattern[index] == h)) {
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
    if (partners[avoid][i] < start) {
      start = partners[avoid][i];
      temp = i;
    }
  }
  if (start == numstates) {
    buffer << "no inverse";
    return;
  }

  // number of shift throws printed at beginning
  int numshiftthrows = temp - shifts;
  int endingshiftthrows = cycleperiod[cyclenum[start]] - shifts - 2
      - numshiftthrows;
  int current = start;

  do {
    // print shift throws
    while (numshiftthrows--) {
      if (state[current] & 1)
        buffer << "+";
      else
        buffer << "-";

      current = partners[current][cycleperiod[cyclenum[current]] - 2];
    }

    // print the block throw
    int throwval = h - pattern[index++];

    if (throwval < 10)
      buffer << throwval;
    else
      buffer << static_cast<char>(throwval - 10 + 'a');

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
    while (index <= pos &&
           (pattern[index] == 0 || pattern[index] == h)) {
      ++index;
      ++shifts;
    }
    numshiftthrows = cycleperiod[cyclenum[current]] - shifts - 2;
  } while (index <= pos);

  // finish printing shift throws in first shift cycle
  while (endingshiftthrows--) {
    if (state[current] & 1)
      buffer << "+";
    else
      buffer << "-";

    current = partners[current][cycleperiod[cyclenum[current]] - 2];
  }
}

void Worker::print_inverse_dual(std::ostringstream& buffer) const {
  // inverse was found in dual space, so we have to transform as we read it
  // first decide on a starting state
  // dual state to avoid on first shift cycle:
  int avoid = reverse_state(start_state);

  // how many adjacent states to avoid also?
  int shifts = 0;
  int index = pos;
  while (index >= 0 &&
      (pattern[index] == 0 || pattern[index] == h)) {
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
    if (partners[avoid][i] < start) {
      start = partners[avoid][i];
      temp = i;
    }
  }
  if (start == numstates) {
    buffer << "no inverse";
    return;
  }

  // number of shift throws printed at beginning
  int numshiftthrows = cycleperiod[cyclenum[start]] - shifts - 2 - temp;
  int endingshiftthrows = temp;
  int current = start;

  do {
    // first print shift throws
    while (numshiftthrows--) {
      if (state[current] & (1L << (h - 1)))
        buffer << "-";
      else
        buffer << "+";

      current = partners[current][0];
    }

    // print the block throw
    temp = pattern[index--];

    if (temp < 10)
      buffer << temp;
    else
      buffer << static_cast<char>(temp - 10 + 'a');

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
    while (index >= 0 &&
           (pattern[index] == 0 || pattern[index] == h)) {
      --index;
      ++shifts;
    }
    numshiftthrows = cycleperiod[cyclenum[current]] - shifts - 2;
  } while (index >= 0);

  // finish printing shift throws in first shift cycle
  while (endingshiftthrows--) {
    if (state[current] & (1L << (h - 1)))
      buffer << "-";
    else
      buffer << "+";

    current = partners[current][0];
  }
}

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

//------------------------------------------------------------------------------
// Prep core data structures during startup
//------------------------------------------------------------------------------

// Find the number of states for a given number of balls and maximum throw value
int Worker::num_states(int n, int h) {
  int result = 1;

  for (int denom = 1; denom <= std::min(n, h - n); ++denom)
    result = (result * (h - denom + 1)) / denom;

  return result;
}

void Worker::prepcorearrays(const std::vector<bool>& xarray) {
  int k, cycleindex, periodfound, cycleper, newshiftcycle, *tempperiod;
  unsigned long temp, highmask, lowmask;

  // initialize arrays

  if (!(used = new int[numstates + 1]))
    die();
  for (int i = 0; i <= numstates; ++i)
    used[i] = 0;

  if (!(outmatrix = new int*[numstates + 1]))
    die();
  for (int i = 0; i <= numstates; ++i) {
    if (!(outmatrix[i] = new int[maxoutdegree]))
      die();
  }
  if (!(outthrowval = new int*[numstates + 1]))
    die();
  for (int i = 0; i <= numstates; ++i) {
    if (!(outthrowval[i] = new int[maxoutdegree]))
      die();
  }
  if (!(outdegree = new int[numstates + 1]))
    die();
  if (!(inmatrix = new int*[numstates + 1]))
    die();
  for (int i = 0; i <= numstates; ++i) {
    if (!(inmatrix[i] = new int[maxindegree]))
      die();
  }
  if (!(indegree = new int[numstates + 1]))
    die();

  if (!(partners = new int*[numstates + 1]))
    die();
  for (int i = 0; i <= numstates; ++i) {
    if (!(partners[i] = new int[h - 1]))
      die();
  }
  if (!(cyclenum = new int[numstates + 1]))
    die();

  state = new unsigned long[numstates + 1];
  for (int i = 0; i <= numstates; ++i)
    state[i] = 0L;
  int ns = gen_states(state, 0, h - 1, n, h, numstates);
  assert(ns == numstates);

  for (int i = 0; i <= numstates; ++i) {
    for (int j = 0; j < maxoutdegree; ++j)
      outmatrix[i][j] = outthrowval[i][j] = 0;
    for (int j = 0; j < maxindegree; ++j)
      inmatrix[i][j] = 0;
    indegree[i] = outdegree[i] = 0;
  }
  gen_matrices(xarray);

  // calculate shift cycles

  highmask = 1L << (h - 1);
  lowmask = highmask - 1;
  cycleindex = 0;
  if (!(tempperiod = new int[numstates + 1]))
    die();

  for (int i = 1; i <= numstates; ++i) {
    for (int j = 0; j < (h - 1); ++j)
      partners[i][j] = 0;

    temp = state[i];
    periodfound = 0;
    newshiftcycle = 1;
    cycleper = h; // default period

    for (int j = 0; j < (h - 1); ++j) {
      if (temp & highmask)
        temp = ((temp & lowmask) * 2) + 1;
      else
        temp *= 2;

      int k = 1;
      for (; k <= numstates; k++) {
        if (state[k] == temp)
          break;
      }
      assert(k <= numstates);

      partners[i][j] = k;
      if (k == i && !periodfound) {
        cycleper = j + 1;
        periodfound = 1;
      } else if (k < i)
        newshiftcycle = 0;
    }

    if (newshiftcycle) {
      cyclenum[i] = cycleindex;
      for (int j = 0; j < (h - 1); j++)
        cyclenum[partners[i][j]] = cycleindex;
      tempperiod[cycleindex++] = cycleper;
    }
  }
  numcycles = cycleindex;
  if (!(cycleperiod = new int[cycleindex]))
    die();
  if (!(deadstates = new int[cycleindex]))
    die();
  for (int i = 0; i < cycleindex; ++i) {
    cycleperiod[i] = tempperiod[i];
    deadstates[i] = 0;
  }

  pattern = tempperiod;  // reuse array
  for (int i = 0; i <= numstates; ++i)
    pattern[i] = -1;
}

void Worker::die() {
  std::cout << "Insufficient memory" << std::endl;
  std::exit(0);
}

// Generate the set of all possible states into the state[] array
//
// Returns the number of states found
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

// Generate a matrix containing the outward connections from each state
// (vertex) in the graph, and the throw value for each.
//
// outmatrix[][] == 0 indicates no connection.

void Worker::gen_matrices(const std::vector<bool>& xarray) {
  for (int i = 1; i <= numstates; ++i) {
    int outthrownum = 0;
    int inthrownum = 0;

    for (int j = h; j >= 0; --j) {
      if (xarray[j])
        continue;

      // first take care of outgoing throw
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

          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              if (i != k) {
                outmatrix[i][outthrownum] = k;
                outthrowval[i][outthrownum++] = j;
                ++outdegree[i];
              }
              found = true;
              break;
            }
          }
          assert(found);
        }
      }

      // then take care of ingoing throw
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
          bool found = false;

          temp = (temp << 1) | 1L;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              if (i != k) {
                inmatrix[i][inthrownum++] = k;
                ++indegree[i];
              }
              found = true;
              break;
            }
          }
          assert(found);
        }
      } else {
        if ((state[i] & (1L << (j - 1))) && (!(state[i] & (1L << (h - 1))))) {
          unsigned long temp = state[i] ^ (1L << (j - 1));
          bool found = false;

          temp = (temp << 1) | 1L;
          for (int k = 1; k <= numstates; ++k) {
            if (state[k] == temp) {
              if (i != k) {
                inmatrix[i][inthrownum++] = k;
                ++indegree[i];
              }
              found = true;
              break;
            }
          }
          assert(found);
        }
      }
    }
  }
}
