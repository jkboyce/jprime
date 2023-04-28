
#ifndef JDEEP_MESSAGES_H
#define JDEEP_MESSAGES_H

#include "WorkAssignment.hpp"

#include <string>
#include <list>

enum messages_C2W {
  DO_WORK,
  UPDATE_METADATA,
  SPLIT_WORK,
  STOP_WORKER,
};

struct MessageC2W {
  messages_C2W type;

  // for type DO_WORK
  WorkAssignment assignment;

  // for types DO_WORK and UPDATE_METADATA
  int l_current = 0;
};

enum messages_W2C {
  SEARCH_RESULT,
  WORKER_IDLE,
  RETURN_WORK,
};

struct MessageW2C {
  messages_W2C type;
  int worker_id;

  // for type SEARCH_RESULT
  std::string pattern;
  std::string meta;
  int length = 0;

  // for types WORKER_IDLE and RETURN_WORK
  unsigned long ntotal = 0L;
  int numstates = 0;
  int maxlength = 0;

  // for type RETURN_WORK
  WorkAssignment assignment;
};

#endif
