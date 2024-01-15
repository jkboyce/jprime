//
// Messages.h
//
// Defines messages that may be sent from the worker to the coordinator and
// vice versa.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_MESSAGES_H_
#define JPRIME_MESSAGES_H_

#include "WorkAssignment.h"

#include <vector>
#include <string>


// Message types from the coordinator to the worker

enum class messages_C2W {
  DO_WORK,
  UPDATE_METADATA,
  SPLIT_WORK,
  SEND_STATS,
  STOP_WORKER,
};

struct MessageC2W {
  // for all message types
  messages_C2W type;

  // for type DO_WORK
  WorkAssignment assignment;

  // for types DO_WORK and UPDATE_METADATA
  int l_current = 0;

  // for type SPLIT_WORK
  int split_alg = 1;
};


// Message types from the worker to the coordinator

enum class messages_W2C {
  SEARCH_RESULT,
  WORKER_IDLE,
  RETURN_WORK,
  RETURN_STATS,
  WORKER_STATUS,
};

struct MessageW2C {
  // for all message types
  messages_W2C type;
  int worker_id;

  // for type SEARCH_RESULT
  std::string pattern;
  int length = 0;

  // for types WORKER_IDLE and RETURN_WORK and RETURN_STATS
  unsigned long ntotal = 0L;
  std::vector<unsigned long> count;
  unsigned long nnodes = 0L;
  int numstates = 0;
  int numcycles = 0;
  int numshortcycles = 0;
  int maxlength = 0;
  double secs_working = 0;

  // for type RETURN_WORK
  WorkAssignment assignment;

  // for type WORKER_STATUS
  std::string meta;
  int root_pos = -1;
  int longest_found = -1;
};

#endif
