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
  unsigned int split_alg = 1;
};


// Message types from the worker to the coordinator

enum class messages_W2C {
  NONE,
  SEARCH_RESULT,
  WORKER_IDLE,
  RETURN_WORK,
  RETURN_STATS,
  WORKER_UPDATE,
};

struct MessageW2C {
  // for all message types
  messages_W2C type = messages_W2C::NONE;
  unsigned int worker_id = 0;

  // for type SEARCH_RESULT
  std::string pattern;
  unsigned int length = 0;

  // for types WORKER_IDLE and RETURN_WORK and RETURN_STATS
  std::vector<std::uint64_t> count;
  std::uint64_t nnodes = 0;
  double secs_working = 0;

  // for type RETURN_WORK
  WorkAssignment assignment;

  // for type RETURN_STATS
  bool running = false;
  std::vector<unsigned int> worker_throw;
  std::vector<unsigned int> worker_options_left;
  std::vector<unsigned int> worker_deadstates_extra;

  // for type WORKER_UPDATE
  std::string meta;
  unsigned int start_state = 0;
  unsigned int end_state = 0;
  unsigned int root_pos = 0;
};

#endif
