
#ifndef JDEEP_MESSAGES_H
#define JDEEP_MESSAGES_H

#include <string>
#include <list>

// Defines a work assignment that can be handed off
struct WorkAssignment {
  int start_state = 1;
  int end_state = 1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  std::vector<int> partial_pattern;
};


enum messages_C2W {
  DO_WORK,
  UPDATE_METADATA,
  TAKE_WORK_PORTION,
  STOP_WORKER,
  TAKE_WORK_ALL,
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
  RETURN_WORK_PORTION,
  NOTIFY_WORKER_STOPPED,
  RETURN_WORK_ALL,
};

struct MessageW2C {
  messages_W2C type;
  int worker_id;

  // for type SEARCH_RESULT
  std::string pattern;
  std::string meta;
  int length = 0;

  // for type WORKER_IDLE
  unsigned long ntotal = 0L;
  int numstates = 0;
  int maxlength = 0;

  // for types RETURN_WORK_PORTION and RETURN_WORK_ALL
  WorkAssignment assignment;
};

#endif
