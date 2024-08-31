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


// Message from the coordinator to the worker

struct MessageC2W {
  enum class Type {
    NONE,
    DO_WORK,
    SPLIT_WORK,
    SEND_STATS,
    STOP_WORKER,
  };

  // for all message types
  Type type = Type::NONE;

  // for type DO_WORK
  WorkAssignment assignment;
};


// Message from the worker to the coordinator

struct MessageW2C {
  enum class Type {
    NONE,
    SEARCH_RESULT,
    WORKER_IDLE,
    RETURN_WORK,
    RETURN_STATS,
    WORKER_UPDATE,
  };

  // for all message types
  Type type = Type::NONE;
  unsigned worker_id = -1;

  // for type SEARCH_RESULT
  std::string pattern;
  unsigned length = 0;

  // for types WORKER_IDLE and RETURN_WORK and RETURN_STATS
  std::vector<std::uint64_t> count;
  std::uint64_t nnodes = 0;
  double secs_working = 0;

  // for type RETURN_WORK
  WorkAssignment assignment;

  // for type RETURN_STATS
  bool running = false;
  std::vector<unsigned> worker_throw;
  std::vector<unsigned> worker_options_left;
  std::vector<unsigned> worker_deadstates_extra;

  // for type WORKER_UPDATE
  std::string meta;
  unsigned start_state = 0;
  unsigned end_state = 0;
  unsigned root_pos = 0;
};

#endif
