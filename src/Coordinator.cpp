
#include "Coordinator.hpp"
#include "Worker.hpp"
#include "Messages.hpp"

#include <iostream>
#include <thread>
#include <chrono>

Coordinator::Coordinator(const SearchConfig& c) : config(c) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run(int threads) {
  num_threads = threads;
  worker.reserve(num_threads);
  worker_thread.reserve(num_threads);
  for (int id = 0; id < num_threads; ++id) {
    worker[id] = new Worker(config, this, id);
    worker_thread[id] = new std::thread(&Worker::run, worker[id]);
  }

  give_first_assignments();

  while (true) {
    process_inbox();

    if (workers_idle.size() == num_threads) {
      for (int id = 0; id < num_threads; ++id) {
        MessageC2W msg;
        msg.type = messages_C2W::STOP_WORKER;
        message_worker(msg, id);

        if (config.verboseflag)
          std::cout << "worker " << id << " asked to stop" << std::endl;
        worker_thread[id]->join();
      }
      break;
    }
  }

  for (int id = 0; id < num_threads; ++id) {
    delete worker[id];
    delete worker_thread[id];
  }

  print_trailer();
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

void Coordinator::message_worker(const MessageC2W& msg, int worker_id) {
  worker[worker_id]->inbox_lock.lock();
  worker[worker_id]->inbox.push(msg);
  worker[worker_id]->inbox_lock.unlock();
}

void Coordinator::give_first_assignments() {
  // give the entire problem to worker 0
  MessageC2W msg;
  msg.type = messages_C2W::DO_WORK;
  WorkAssignment wa;
  wa.start_state = -1;
  wa.end_state = -1;
  wa.root_pos = 0;
  for (int i = 0; i <= config.h; ++i) {
    if (!config.xarray[i])
      wa.root_throwval_options.push_back(i);
  }
  msg.assignment = wa;
  msg.l_current = l_current;
  message_worker(msg, 0);

  // leave the others idle for now
  for (int id = 1; id < num_threads; ++id)
    workers_idle.push_back(id);
}

void Coordinator::process_inbox() {
  int new_longest_pattern_from_id = -1;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_W2C::SEARCH_RESULT) {
      if (msg.length == 0) {
        if (config.verboseflag)
          std::cout << msg.meta << std::endl;
      } else if (!config.longestflag) {
        print_pattern(msg);
        ++npatterns;
      } else if (msg.length > l_current) {
        print_pattern(msg);
        l_current = msg.length;
        npatterns = 1;
        new_longest_pattern_from_id = msg.worker_id;
      } else if (msg.length == l_current) {
        print_pattern(msg);
        ++npatterns;
      }
      // ignore patterns shorter than current length if longestflag == true
    } else if (msg.type == messages_W2C::WORKER_IDLE) {
      workers_idle.push_back(msg.worker_id);
      ntotal += msg.ntotal;
      numstates = msg.numstates;
      maxlength = msg.maxlength;

      // worker went idle before it could return a work assignment
      if (msg.worker_id == waiting_for_work_from_id)
        waiting_for_work_from_id = -1;

      if (config.verboseflag)
        std::cout << "worker " << msg.worker_id << " went idle" << std::endl;
    } else if (msg.type == messages_W2C::RETURN_WORK_PORTION) {
      assert(workers_idle.size() > 0);
      assert(msg.worker_id == waiting_for_work_from_id);
      waiting_for_work_from_id = -1;

      int id = workers_idle.front();
      workers_idle.pop_front();

      MessageC2W msg2;
      msg2.type = messages_C2W::DO_WORK;
      msg2.assignment = msg.assignment;
      message_worker(msg2, id);

      if (config.verboseflag) {
        std::cout << "worker " << msg.worker_id
                  << " returned work, gave to worker " << id << std::endl;
        std::cout << "  worker " << id << ": " << "{ state:"
                  << msg.assignment.start_state
                  << ", root_pos:" << msg.assignment.root_pos << ", prefix:\"";
        for (int v : msg.assignment.partial_pattern)
          std::cout << print_throw(v);
        std::cout << "\", throws:[";
        for (int v : msg.assignment.root_throwval_options)
          std::cout << print_throw(v);
        std::cout << "] }" << std::endl;
      }
    } else if (msg.type == messages_W2C::NOTIFY_WORKER_STOPPED) {
      // to-do
    } else if (msg.type == messages_W2C::RETURN_WORK_ALL) {
      // to-do
    } else
      assert(false);
  }
  inbox_lock.unlock();

  // notify about new metadata
  if (new_longest_pattern_from_id >= 0) {
    for (int id = 0; id < num_threads; ++id) {
      if (id == new_longest_pattern_from_id)
        continue;
      if (std::find(workers_idle.begin(), workers_idle.end(), id)
            != workers_idle.end())
        continue;

      MessageC2W msg;
      msg.type = messages_C2W::UPDATE_METADATA;
      msg.l_current = l_current;
      message_worker(msg, id);

      if (config.verboseflag)
        std::cout << "worker " << id << " notified of new length " << l_current
                  << std::endl;
    }
  }

  // issue new request for work assignment
  if (waiting_for_work_from_id == -1 && workers_idle.size() > 0) {
    // current strategy: take work from lowest-id worker that's busy

    for (int id = 0; id < num_threads; ++id) {
      if (std::find(workers_idle.begin(), workers_idle.end(), id)
            != workers_idle.end())
        continue;

      waiting_for_work_from_id = id;
      MessageC2W msg;
      msg.type = messages_C2W::TAKE_WORK_PORTION;
      message_worker(msg, id);

      if (config.verboseflag)
        std::cout << "requested work from worker " << id << std::endl;
      break;
    }
  }
}

//------------------------------------------------------------------------------
// Handle output
//------------------------------------------------------------------------------

void Coordinator::print_pattern(const MessageW2C& msg) {
  if (config.printflag) {
    if (config.verboseflag)
      std::cout << msg.worker_id << ": " << msg.pattern << std::endl;
    else
      std::cout << msg.pattern << std::endl;
  }
}

char Coordinator::print_throw(int val) {
  const bool plusminus = ((config.mode == NORMAL_MODE && config.longestflag) ||
                          config.mode == BLOCK_MODE);
  if (plusminus && val == 0)
    return '-';
  if (plusminus && val == config.h)
    return '+';

  if (val < 10)
    return static_cast<char>(val + '0');
  else
    return static_cast<char>(val - 10 + 'A');
}

void Coordinator::print_trailer() {
  std::cout << "balls: " << (config.dualflag ? config.h - config.n : config.n);
  std::cout << ", max throw: " << config.h << std::endl;

  switch (config.mode) {
    case NORMAL_MODE:
      break;
    case BLOCK_MODE:
      std::cout << "block mode, " << config.skiplimit << " skips allowed"
                << std::endl;
      break;
    case SUPER_MODE:
      std::cout << "super mode, " << config.shiftlimit << " shifts allowed";
      if (config.invertflag)
        std::cout << ", inverse output" << std::endl;
      else
        std::cout << std::endl;
      break;
  }

  if (config.longestflag) {
    std::cout << "pattern length: " << l_current << " throws (" << maxlength
              << " maximum, " << numstates << " states)" << std::endl;
  }

  std::cout << npatterns << " patterns found (" << ntotal << " seen)"
            << std::endl;

  if (config.groundmode == 1)
    std::cout << "ground state search" << std::endl;
  if (config.groundmode == 2)
    std::cout << "excited state search" << std::endl;
}
