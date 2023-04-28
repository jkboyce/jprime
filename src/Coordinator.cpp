
#include "Coordinator.hpp"
#include "Worker.hpp"
#include "Messages.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>

bool Coordinator::stopping = false;

Coordinator::Coordinator(const SearchConfig& c) : config(c) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run(int threads, std::list<WorkAssignment>& assignments) {
  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  num_threads = threads;
  worker.reserve(num_threads);
  worker_thread.reserve(num_threads);
  for (int id = 0; id < num_threads; ++id) {
    worker[id] = new Worker(config, this, id);
    worker_thread[id] = new std::thread(&Worker::run, worker[id]);
    workers_idle.push_back(id);
  }

  while (true) {
    give_assignments(assignments);
    steal_assignment();
    process_inbox(assignments);

    if (workers_idle.size() == num_threads || Coordinator::stopping) {
      stop_workers();
      // any worker that was running will have sent back a RETURN_WORK message
      inbox_lock.lock();
      while (!inbox.empty()) {
        MessageW2C msg = inbox.front();
        inbox.pop();
        if (msg.type == messages_W2C::RETURN_WORK) {
          assignments.push_back(msg.assignment);
          ntotal += msg.ntotal;
          numstates = msg.numstates;
          maxlength = msg.maxlength;
        }
      }
      inbox_lock.unlock();
      break;
    }
  }

  for (int id = 0; id < num_threads; ++id) {
    delete worker[id];
    delete worker_thread[id];
  }

  if (assignments.size() > 0)
    std::cout << std::endl << "PARTIAL RESULTS:" << std::endl;
  print_trailer();

  /*
  items we want to retain on disk:
  - SearchConfig
  - num_threads (can be overridden)
  - assignments vector
  - npatterns
  - ntotal
  - time spent
  - text output so far
  */
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

void Coordinator::message_worker(const MessageC2W& msg, int worker_id) {
  worker[worker_id]->inbox_lock.lock();
  worker[worker_id]->inbox.push(msg);
  worker[worker_id]->inbox_lock.unlock();
}

void Coordinator::give_assignments(std::list<WorkAssignment>& assignments) {
  while (workers_idle.size() > 0 && assignments.size() > 0) {
    int id = workers_idle.front();
    WorkAssignment wa = assignments.front();
    workers_idle.pop_front();
    assignments.pop_front();

    MessageC2W msg;
    msg.type = messages_C2W::DO_WORK;
    msg.assignment = wa;
    msg.l_current = l_current;
    message_worker(msg, id);

    if (config.verboseflag) {
      std::cout << "gave work to worker " << id << ":" << std::endl;
      std::cout << "  " << msg.assignment << std::endl;
    }
  }
}

void Coordinator::steal_assignment() {
  if (waiting_for_work_from_id != -1 || workers_idle.size() == 0)
    return;

  // current strategy: take work from lowest-id worker that's busy

  for (int id = 0; id < num_threads; ++id) {
    if (std::find(workers_idle.begin(), workers_idle.end(), id)
          != workers_idle.end())
      continue;

    waiting_for_work_from_id = id;
    MessageC2W msg;
    msg.type = messages_C2W::SPLIT_WORK;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << "requested work from worker " << id << std::endl;
    break;
  }
}

void Coordinator::process_inbox(std::list<WorkAssignment>& assignments) {
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
    } else if (msg.type == messages_W2C::RETURN_WORK) {
      assert(workers_idle.size() > 0);
      assert(msg.worker_id == waiting_for_work_from_id);
      waiting_for_work_from_id = -1;
      assignments.push_back(msg.assignment);

      if (config.verboseflag) {
        std::cout << "worker " << msg.worker_id << " returned work:" << std::endl;
        std::cout << "  " << msg.assignment << std::endl;
      }
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
}

void Coordinator::stop_workers() {
  for (int id = 0; id < num_threads; ++id) {
    MessageC2W msg;
    msg.type = messages_C2W::STOP_WORKER;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << "worker " << id << " asked to stop" << std::endl;
    worker_thread[id]->join();
  }
}

void Coordinator::signal_handler(int signum) {
  stopping = true;
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
