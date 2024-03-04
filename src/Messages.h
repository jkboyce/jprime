//
// Messages.h
//
// Defines messages that may be sent from the worker to the coordinator and
// vice versa.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_MESSAGES_H_
#define JPRIME_MESSAGES_H_

#include "WorkAssignment.h"

#include <vector>
#include <string>
#include <cstdint>


// Message types from the coordinator to the worker

enum class messages_C2W {
  NONE,
  DO_WORK,
  SPLIT_WORK,
  SEND_STATS,
  STOP_WORKER,
};

struct MessageC2W {
  // for all message types
  messages_C2W type = messages_C2W::NONE;

  // for type DO_WORK
  WorkAssignment assignment;

  // for type SPLIT_WORK
  int split_alg = 1;
};


// Message types from the worker to the coordinator

enum class messages_W2C {
  NONE,
  SEARCH_RESULT,
  WORKER_IDLE,
  RETURN_WORK,
  RETURN_STATS,
  WORKER_STATUS,
};

struct MessageW2C {
  // for all message types
  messages_W2C type = messages_W2C::NONE;
  int worker_id = 0;

  // for type SEARCH_RESULT
  std::string pattern;
  int length = 0;

  // for types WORKER_IDLE and RETURN_WORK and RETURN_STATS
  std::uint64_t ntotal = 0;
  std::vector<std::uint64_t> count;
  std::uint64_t nnodes = 0;
  int numstates = 0;
  int numcycles = 0;
  int numshortcycles = 0;
  int l_bound = 0;
  double secs_working = 0;
  int longest_found = 0;

  // for type RETURN_WORK
  WorkAssignment assignment;

  // for type RETURN_STATS
  bool running = false;
  int start_state = 0;
  std::vector<int> worker_throw;
  std::vector<int> worker_optionsleft;

  // for type WORKER_STATUS
  std::string meta;
  int root_pos = -1;
};

#endif
