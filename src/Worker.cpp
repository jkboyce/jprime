
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
  trimflag = config.trimflag;
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
        l = msg.l_current;
        new_assignment = true;
      } else if (msg.type == messages_C2W::UPDATE_METADATA) {
        // ignore here
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

    if (calibrations_remaining > 0)
      timespec_get(&last_ts, TIME_UTC);

    // complete the new work assignment
    try {
      gen_patterns();
    } catch (const JdeepStopException& jdse) {
      // a STOP_WORKER message while running unwinds back here
      MessageW2C msg;
      msg.type = messages_W2C::RETURN_WORK;
      msg.worker_id = worker_id;
      msg.ntotal = ntotal;
      msg.nnodes = nnodes;
      msg.numstates = numstates;
      msg.maxlength = maxlength;
      msg.assignment = get_work_assignment();
      message_coordinator(msg);
      break;
    }

    inbox_lock.lock();
    inbox = std::queue<MessageC2W>();
    inbox_lock.unlock();

    MessageW2C msg;
    msg.type = messages_W2C::WORKER_IDLE;
    msg.worker_id = worker_id;
    msg.ntotal = ntotal;
    msg.nnodes = nnodes;
    msg.numstates = numstates;
    msg.maxlength = maxlength;
    message_coordinator(msg);
    ntotal = 0;
    nnodes = 0;
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

void Worker::process_inbox() {
  if (calibrations_remaining > 0) {
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
      MessageW2C msg;
      msg.type = messages_W2C::RETURN_WORK;
      msg.worker_id = worker_id;
      msg.assignment = split_off_work_assignment();

      if (verboseflag) {
        std::ostringstream sstr;
        sstr << "worker " << worker_id << " remaining work after split:" << std::endl;
        sstr << "  " << get_work_assignment();

        MessageW2C msg2;
        msg2.type = messages_W2C::SEARCH_RESULT;
        msg2.meta = sstr.str();
        message_coordinator(msg, msg2);
      } else
        message_coordinator(msg);
    } else if (msg.type == messages_C2W::STOP_WORKER) {
      stopping_work = true;
    }
  }
  inbox_lock.unlock();

  if (stopping_work)
    throw JdeepStopException();
}

void Worker::load_work_assignment(const WorkAssignment& wa) {
  loading_work = true;
  loading_pos = 0;

  start_state = wa.start_state;
  end_state = wa.end_state;
  if (start_state == -1)
    start_state = (groundmode == 2) ? 2 : 1;
  if (end_state == -1)
    end_state = (groundmode == 1) ? 1 : numstates;

  root_pos = wa.root_pos;
  root_throwval_options = wa.root_throwval_options;
  assert(root_throwval_options.size() > 0);
  assert(pos == 0);

  for (int i = 0; i <= numstates; ++i) {
    pattern[i] = (i < wa.partial_pattern.size()) ? wa.partial_pattern[i] : -1;
    assert(used[i] == 0);
  }
}

WorkAssignment Worker::split_off_work_assignment() {
  // give away all the other throw value options at the current root position,
  // and move this worker to a deeper root position
  assert(root_throwval_options.size() > 0);

  WorkAssignment wa;
  wa.start_state = start_state;
  wa.end_state = start_state;
  wa.root_pos = root_pos;
  wa.root_throwval_options = root_throwval_options;
  for (int i = 0; i < root_pos; ++i)
    wa.partial_pattern.push_back(pattern[i]);

  // remove the throw value at `root_pos` from the list of
  // possibilities we're giving away
  std::list<int>::iterator iter = wa.root_throwval_options.begin();
  std::list<int>::iterator end = wa.root_throwval_options.end();
  while (iter != end) {
    if (*iter == pattern[root_pos])
      iter = wa.root_throwval_options.erase(iter);
    else
      ++iter;
  }

  // Here we need to find the shallowest depth `new_root_pos` where there are
  // unexplored throw options. We're giving away all the options at the current
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
    assert(col != outdegree[from_state]);

    if (pos2 > root_pos && col < outdegree[from_state] - 1) {
      new_root_pos = pos2;
      break;
    }

    from_state = outmatrix[from_state][col];
  }
  assert(new_root_pos != -1);

  root_pos = new_root_pos;
  root_throwval_options.clear();
  for (; col < outdegree[from_state]; ++col) {
    const int throwval = outthrowval[from_state][col];
    if (throwval != pattern[root_pos])
      root_throwval_options.push_back(outthrowval[from_state][col]);
  }
  assert(root_throwval_options.size() > 0);

  return wa;
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

//------------------------------------------------------------------------------
// Search the juggling graph for patterns
//------------------------------------------------------------------------------

void Worker::gen_patterns() {
  const bool pretrim_graph = false;

  for (; start_state <= end_state; ++start_state) {
    from = start_state;
    pos = 0;
    firstblocklength = -1; // -1 signals unknown
    skipcount = 0;

    if (!loading_work) {
      // subsequent values of start_state, reset key variables
      root_pos = 0;

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
        if (pretrim_graph)
          delete_vertices(1);
        else
          gen_loops_normal();
        break;
      case BLOCK_MODE:
        if (longestflag) {
          max_possible = numstates - start_state + 1;
          if (l > max_possible)
            continue;
        } else
          max_possible = numstates;
        if (pretrim_graph)
          delete_vertices(1);
        else
          gen_loops_block();
        break;
      case SUPER_MODE:
        for (int i = 0; i < (n - 1); ++i)
          used[partners[start_state][i]] = 1;
        max_possible = numstates;
        if (pretrim_graph)
          delete_vertices(1);
        else
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

  int col = 0;

  if (loading_work) {
    if (pos == root_pos) {
      // std::list<int> rto_copy = std::list<int>(root_throwval_options);

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

    if (pattern[pos] == -1)
      loading_work = false;
    else {
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
  }

  const unsigned long highmask = 1L << (h - 1);
  const unsigned long allmask = (1L << h) - 1;

  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];
    if (used[to] != 0 || to < start_state)
      continue;
    const int throwval = outthrowval[from][col];

    if (pos == root_pos) {
      // check to see if this throwval is in our allowed list,
      // and if so remove it
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
        continue;

      if (remaining == 0) {
        // using our last option at this root level, go one step deeper
        ++root_pos;
        for (int col2 = 0; col2 < maxoutdegree; ++col2) {
          if (outmatrix[to][col2] < start_state)
            continue;
          root_throwval_options.push_back(outthrowval[to][col2]);
        }
      }
    }

    // finished?
    if (to == start_state) {
      ++ntotal;
      if (pos >= (l - 1) || l == 0) {
        if (longestflag && pos >= l)
          l = pos + 1;
        pattern[pos] = outthrowval[from][col];
        report_pattern();
      }
      continue;
    }

    bool valid = true;
    int old_max_possible = max_possible;

    if (trimflag && throwval > 0 && throwval < h) {
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
      used[to] = 1;
      pattern[pos] = throwval;
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
    if (trimflag && throwval > 0 && throwval < h) {
      // block throw:
      // kill states downstream in 'from' cycle that end in 'x'
      int j = h - 2;
      unsigned long tempstate = state[from];
      int cnum = cyclenum[from];

      do {
        if (--used[partners[from][j]] == 0 && --deadstates[cnum] >= 1)
          ++max_possible;

        --j;
        tempstate >>= 1;
      } while (tempstate & 1L);

      // kill states upstream in 'to' cycle that start with '-'
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

      process_inbox();
      steps_taken = 0;
    }

    if (pos < root_pos)
      break;
  }
}

void Worker::gen_loops_block() {
  if (exactflag && pos >= l)
    return;

  int col = 0;

  if (loading_work) {
    if (pattern[loading_pos] == -1)
      loading_work = false;
    else {
      for (; col < maxoutdegree; ++col) {
        if (outthrowval[from][col] == pattern[loading_pos])
          break;
      }
      assert(col != maxoutdegree);
      ++loading_pos;
    }
  }

  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];

    if (to < start_state || used[to] == 1)
      continue;

    // are we finished?
    if (to == start_state) {
      ++ntotal;

      if (l != 0 && pos < (l - 1))
        continue;

      const int throwval = outthrowval[from][col];
      const int oldblocklength = blocklength;
      const int oldskipcount = skipcount;
      const int oldfirstblocklength = firstblocklength;
      bool valid = true;

      if (throwval > 0 && throwval < h) {
        if (firstblocklength >= 0) {
          if (blocklength != (h - 2)) { // got a skip
            if (skipcount == skiplimit)
              valid = false;
            else
              ++skipcount;
          }
        } else // first block throw encountered
          firstblocklength = pos;

        blocklength = 0;
      } else
        ++blocklength;

      if (skipcount == skiplimit &&
          (blocklength + firstblocklength) != (h - 2))
        valid = false;

      if (valid) {
        if (longestflag && pos >= l)
          l = pos + 1;
        pattern[pos] = throwval;
        report_pattern();
      }

      blocklength = oldblocklength;
      skipcount = oldskipcount;
      firstblocklength = oldfirstblocklength;
      continue;
    }

    const int throwval = outthrowval[from][col];
    const int oldblocklength = blocklength;
    const int oldskipcount = skipcount;
    const int oldfirstblocklength = firstblocklength;
    bool valid = true;

    if (throwval > 0 && throwval < h) { // block throw?
      if (firstblocklength >= 0) {
        if (blocklength != (h - 2))
        { // got a skip
          if (skipcount == skiplimit)
            valid = false;
          else
            ++skipcount;
        }
      } else // first block throw encountered
        firstblocklength = pos;

      blocklength = 0;
    } else
      ++blocklength;

    // continue recursively, if current position is valid
    if (valid) {
      used[to] = 1;
      pattern[pos] = throwval;
      ++pos;
      int old_from = from;
      from = to;
      if (!loading_work)
        ++nnodes;

      if (trimflag)
        trim_outgoing(from, to, 0);
      else
        gen_loops_block();

      used[to] = 0;
      --pos;
      from = old_from;
    }

    // undo changes so we can backtrack
    blocklength = oldblocklength;
    skipcount = oldskipcount;
    firstblocklength = oldfirstblocklength;
  }
}

void Worker::gen_loops_super() {
  if (exactflag && pos >= l)
    return;

  int col = 0;

  if (loading_work) {
    if (pattern[loading_pos] == -1)
      loading_work = false;
    else {
      for (; col < maxoutdegree; ++col) {
        if (outthrowval[from][col] == pattern[loading_pos])
          break;
      }
      assert(col != maxoutdegree);
      ++loading_pos;
    }
  }

  for (; col < maxoutdegree; ++col) {
    const int to = outmatrix[from][col];

    if (to < start_state || used[to] == 1)
      continue;

    if (to == start_state) {
      ++ntotal;
      if (l != 0 && pos < (l - 1))
        continue;

      const int throwval = outthrowval[from][col];

      if ((throwval == 0 || throwval == h) && shiftlimit > 0
            && shiftcount == shiftlimit)
        continue;

      if (longestflag && pos >= l)
        l = pos + 1;
      pattern[pos] = throwval;
      report_pattern();
      continue;
    }

    const int throwval = outthrowval[from][col];
    const int oldshiftcount = shiftcount;
    bool valid = true;

    if (throwval == 0 || throwval == h) {
      if (shiftcount == shiftlimit)
        valid = false;
      else
        ++shiftcount;
    } else {
      if (used[to] < 0)
        valid = false; // block throw into occupied shift cycle
    }

    // if current position is valid then continue recursively
    if (valid) {
      const int oldusedvalue = used[to];
      used[to] = 1;

      for (int j = 0; j < (h - 1); ++j) {
        if (used[partners[to][j]] < 1)
          --used[partners[to][j]];
      }

      pattern[pos] = throwval;
      ++pos;
      int old_from = from;
      from = to;
      if (!loading_work)
        ++nnodes;

      if (trimflag)
        trim_outgoing(from, to, 0);
      else
        gen_loops_super();

      for (int j = 0; j < (h - 1); ++j) {
        if (used[partners[to][j]] < 0)
          ++used[partners[to][j]];
      }

      used[to] = oldusedvalue;
      --pos;
      from = old_from;
    }

    // undo changes so we can backtrack
    shiftcount = oldshiftcount;
  }
}

//------------------------------------------------------------------------------
// Optional graph trimming before finding patterns
//------------------------------------------------------------------------------

// Delete all vertices lower than `start_state` since they can't
// participate in the final pattern.
void Worker::delete_vertices(int statenum) {
  if (statenum >= start_state) {
    outupdate(start_state, 0);
    return;
  }

  int i, j;

  if (indegree[statenum] != 0) {
    for (i = 0; i < maxindegree; ++i) {
      if (inmatrix[statenum][i] > 0)
        break;
    }
    assert(i != maxindegree);
    int temp = inmatrix[statenum][i];

    for (j = 0; j < maxoutdegree; ++j) {
      if (outmatrix[temp][j] == statenum)
        break;
    }
    assert(j != maxoutdegree);

    inmatrix[statenum][i] = -1;
    outmatrix[temp][j] = -1;
    --indegree[statenum];
    --outdegree[temp];
    delete_vertices(statenum);
    inmatrix[statenum][i] = temp;
    outmatrix[temp][j] = statenum;
    ++indegree[statenum];
    ++outdegree[temp];
    return;
  } else if (outdegree[statenum] != 0) {
    for (i = 0; i < maxoutdegree; ++i) {
      if (outmatrix[statenum][i] > 0)
        break;
    }
    assert(i != maxoutdegree);
    int temp = outmatrix[statenum][i];

    for (j = 0; j < maxindegree; ++j) {
      if (inmatrix[temp][j] == statenum)
        break;
    }
    assert(j != maxindegree);

    outmatrix[statenum][i] = -1;
    inmatrix[temp][j] = -1;
    --outdegree[statenum];
    --indegree[temp];
    delete_vertices(statenum);
    outmatrix[statenum][i] = temp;
    inmatrix[temp][j] = statenum;
    ++outdegree[statenum];
    ++indegree[temp];
    return;
  } else {
    delete_vertices(statenum + 1);
    return;
  }
}

void Worker::outupdate(int statenum, int slot) {
  while (statenum <= numstates &&
      (outdegree[statenum] != 0 || indegree[statenum] == 0)) {
    ++statenum;
    slot = 0;
  }

  if (statenum == (numstates + 1)) {
    // finished with current recursion
    inupdate(start_state, 0);
    return;
  }

  // outdegree of current state is zero, indegree is not.  delete a link
  for (; slot < maxindegree; ++slot) {
    int from = inmatrix[statenum][slot];

    if (from == 0)
      return;
    if (from < 0)
      continue;

    int col = 0;
    for (; col < maxoutdegree; ++col) {
      if (outmatrix[from][col] == statenum)
        break;
    }
    if (indegree[statenum] == 1) {
      if (max_possible <= l)
        return;
      --max_possible;
    }
    inmatrix[statenum][slot] = -1;
    --indegree[statenum];
    outmatrix[from][col] = -1;
    --outdegree[from];

    if (outdegree[from] == 0 && from < statenum)
      outupdate(from, 0); // continue earlier
    else
      outupdate(statenum, slot + 1); // continue here

    inmatrix[statenum][slot] = from;
    if (indegree[statenum] == 0)
      ++max_possible;
    ++indegree[statenum];
    outmatrix[from][col] = statenum;
    ++outdegree[from];
    return;
  }
}

void Worker::inupdate(int statenum, int slot) {
  while (statenum <= numstates &&
      (indegree[statenum] != 0 || outdegree[statenum] == 0)) {
    ++statenum;
    slot = 0;
  }

  if (statenum == (numstates + 1)) {
    // finished trimming the juggling graph, now find patterns
    switch (mode) {
      case NORMAL_MODE:
        gen_loops_normal();
        break;
      case BLOCK_MODE:
        gen_loops_block();
        break;
      case SUPER_MODE:
        gen_loops_super();
        break;
    }
    return;
  }

  // indegree of current state is zero, outdegree is not.  delete a link
  for (; slot < maxoutdegree; ++slot) {
    int to = outmatrix[statenum][slot];

    if (to == 0)
      return;
    if (to < 0)
      continue;

    int col = 0;
    for (; col < maxindegree; ++col) {
      if (inmatrix[to][col] == statenum)
        break;
    }

    if (outdegree[statenum] == 1) {
      if (max_possible <= l)
        return;
      --max_possible;
    }
    outmatrix[statenum][slot] = -1;
    --outdegree[statenum];
    inmatrix[to][col] = -1;
    --indegree[to];

    if (indegree[to] == 0 && to < statenum)
      inupdate(to, 0); // continue earlier
    else
      inupdate(statenum, slot + 1); // continue here

    outmatrix[statenum][slot] = to;
    if (outdegree[statenum] == 0)
      ++max_possible;
    ++outdegree[statenum];
    inmatrix[to][col] = statenum;
    ++indegree[to];
    return;
  }
}

//------------------------------------------------------------------------------
// Graph trimming specific to BLOCK and SUPER mode and -trim flag
//------------------------------------------------------------------------------

void Worker::trim_outgoing(int from_trim, int to_trim, int slot) {
  for (; slot < maxoutdegree; ++slot) {
    int to = outmatrix[from_trim][slot];

    if (to == 0)
      break;
    if (to <= 0 || to == to_trim)
      continue;

    outmatrix[from_trim][slot] = -1;

    int col = 0;
    for (; col < maxindegree; ++col) {
      if (inmatrix[to][col] == from_trim) {
        inmatrix[to][col] = -1;
        break;
      }
    }
    --outdegree[from_trim];
    --indegree[to];
    trim_outgoing(from_trim, to_trim, slot + 1);
    outmatrix[from_trim][slot] = to;
    inmatrix[to][col] = from_trim;
    ++outdegree[from_trim];
    ++indegree[to];
    return;
  }

  trim_ingoing(from_trim, to_trim, 0);
}

void Worker::trim_ingoing(int from_trim, int to_trim, int slot) {
  for (; slot < maxindegree; slot++) {
    int from = inmatrix[to_trim][slot];

    if (from == 0)
      break;
    if (from < 0 || from == from_trim)
      continue;

    inmatrix[to_trim][slot] = -1;

    int col = 0;
    for (; col < maxoutdegree; ++col) {
      if (outmatrix[from][col] == to_trim) {
        outmatrix[from][col] = -1;
        break;
      }
    }
    --indegree[to_trim];
    --outdegree[from];
    trim_ingoing(from_trim, to_trim, slot + 1);
    inmatrix[to_trim][slot] = from;
    outmatrix[from][col] = to_trim;
    indegree[to_trim]++;
    outdegree[from]++;
    return;
  }

  outupdate(start_state, 0);
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
    buffer << static_cast<char>(val - 10 + 'A');
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
      buffer << static_cast<char>(throwval - 10 + 'A');

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
      buffer << static_cast<char>(temp - 10 + 'A');

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
