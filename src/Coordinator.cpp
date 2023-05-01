
#include "SearchConfig.hpp"
#include "SearchContext.hpp"
#include "Coordinator.hpp"
#include "Worker.hpp"
#include "Messages.hpp"

#include <iostream>
#include <thread>
#include <csignal>

bool Coordinator::stopping = false;

Coordinator::Coordinator(const SearchConfig& a, SearchContext& b)
    : config(a), context(b) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run() {
  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  // start worker threads
  worker.reserve(context.num_threads);
  worker_thread.reserve(context.num_threads);
  for (int id = 0; id < context.num_threads; ++id) {
    worker[id] = new Worker(config, this, id);
    worker_thread[id] = new std::thread(&Worker::run, worker[id]);
    workers_idle.push_back(id);
  }

  while (true) {
    give_assignments();
    steal_work();
    process_inbox();

    if ((workers_idle.size() == context.num_threads
          && context.assignments.size() == 0) || Coordinator::stopping) {
      stop_workers();
      // any worker that was running will have sent back a RETURN_WORK message
      inbox_lock.lock();
      while (!inbox.empty()) {
        MessageW2C msg = inbox.front();
        inbox.pop();
        if (msg.type == messages_W2C::SEARCH_RESULT) {
          process_search_result(msg);
        } else if (msg.type == messages_W2C::RETURN_WORK) {
          context.assignments.push_back(msg.assignment);
          context.ntotal += msg.ntotal;
          context.nnodes += msg.nnodes;
          context.numstates = msg.numstates;
          context.maxlength = msg.maxlength;
        }
        // ignore other message types
      }
      inbox_lock.unlock();
      break;
    }
  }

  for (int id = 0; id < context.num_threads; ++id) {
    delete worker[id];
    delete worker_thread[id];
  }

  if (context.assignments.size() > 0)
    std::cout << std::endl << "PARTIAL RESULTS:" << std::endl;
  print_trailer();
}

//------------------------------------------------------------------------------
// Handle interactions with the Worker threads
//------------------------------------------------------------------------------

void Coordinator::message_worker(const MessageC2W& msg, int worker_id) const {
  worker[worker_id]->inbox_lock.lock();
  worker[worker_id]->inbox.push(msg);
  worker[worker_id]->inbox_lock.unlock();
}

void Coordinator::give_assignments() {
  while (workers_idle.size() > 0 && context.assignments.size() > 0) {
    int id = workers_idle.front();
    WorkAssignment wa = context.assignments.front();
    workers_idle.pop_front();
    context.assignments.pop_front();

    MessageC2W msg;
    msg.type = messages_C2W::DO_WORK;
    msg.assignment = wa;
    msg.l_current = context.l_current;
    message_worker(msg, id);

    if (config.verboseflag) {
      std::cout << "gave work to worker " << id << ":" << std::endl;
      std::cout << "  " << msg.assignment << std::endl;
    }
  }
}

void Coordinator::steal_work() {
  if (waiting_for_work_from_id != -1 || workers_idle.size() == 0)
    return;

  // current strategy: take work from lowest-id worker that's busy

  for (int id = 0; id < context.num_threads; ++id) {
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

void Coordinator::process_inbox() {
  int new_longest_pattern_from_id = -1;

  inbox_lock.lock();
  while (!inbox.empty()) {
    MessageW2C msg = inbox.front();
    inbox.pop();

    if (msg.type == messages_W2C::SEARCH_RESULT) {
      new_longest_pattern_from_id = process_search_result(msg);
    } else if (msg.type == messages_W2C::WORKER_IDLE) {
      workers_idle.push_back(msg.worker_id);
      context.ntotal += msg.ntotal;
      context.nnodes += msg.nnodes;
      context.numstates = msg.numstates;
      context.maxlength = msg.maxlength;

      // worker went idle before it could return a work assignment
      if (msg.worker_id == waiting_for_work_from_id)
        waiting_for_work_from_id = -1;

      if (config.verboseflag)
        std::cout << "worker " << msg.worker_id << " went idle" << std::endl;
    } else if (msg.type == messages_W2C::RETURN_WORK) {
      assert(workers_idle.size() > 0);
      assert(msg.worker_id == waiting_for_work_from_id);
      waiting_for_work_from_id = -1;
      context.assignments.push_back(msg.assignment);

      if (config.verboseflag) {
        std::cout << "worker " << msg.worker_id << " returned work:"
                  << std::endl
                  << "  " << msg.assignment << std::endl;
      }
    } else
      assert(false);
  }
  inbox_lock.unlock();

  if (new_longest_pattern_from_id >= 0)
    notify_metadata(new_longest_pattern_from_id);
}

int Coordinator::process_search_result(const MessageW2C& msg) {
  int new_longest_pattern_from_id = -1;

  if (msg.length == 0) {
    if (config.verboseflag)
      std::cout << msg.meta << std::endl;
  } else if (!config.longestflag) {
    print_pattern(msg);
    ++context.npatterns;
  } else if (msg.length > context.l_current) {
    context.patterns.clear();
    print_pattern(msg);
    context.l_current = msg.length;
    context.npatterns = 1;
    new_longest_pattern_from_id = msg.worker_id;
  } else if (msg.length == context.l_current) {
    print_pattern(msg);
    ++context.npatterns;
  }
  // ignore patterns shorter than current length if longestflag == true

  return new_longest_pattern_from_id;
}

void Coordinator::notify_metadata(int skip_id) const {
  for (int id = 0; id < context.num_threads; ++id) {
    if (id == skip_id)
      continue;
    if (std::find(workers_idle.begin(), workers_idle.end(), id)
          != workers_idle.end())
      continue;

    MessageC2W msg;
    msg.type = messages_C2W::UPDATE_METADATA;
    msg.l_current = context.l_current;
    message_worker(msg, id);

    if (config.verboseflag)
      std::cout << "worker " << id << " notified of new length "
                << context.l_current << std::endl;
  }
}

void Coordinator::stop_workers() const {
  for (int id = 0; id < context.num_threads; ++id) {
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

  context.patterns.push_back(msg.pattern);
}

void Coordinator::print_trailer() const {
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
    std::cout << "pattern length: " << context.l_current
              << " throws (" << context.maxlength << " maximum, "
              << context.numstates << " states)" << std::endl;
  }

  std::cout << context.npatterns << " patterns found (" << context.ntotal
            << " seen, " << context.nnodes << " nodes)" << std::endl;

  if (config.groundmode == 1)
    std::cout << "ground state search" << std::endl;
  if (config.groundmode == 2)
    std::cout << "excited state search" << std::endl;
}
