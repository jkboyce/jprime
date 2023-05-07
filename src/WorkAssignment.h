
#ifndef JDEEP_WORKASSIGNMENT_H
#define JDEEP_WORKASSIGNMENT_H

#include <iostream>
#include <list>
#include <vector>
#include <string>

// Defines a work assignment to hand off between workers. Together with
// SearchConfig these completely define the computation.

struct WorkAssignment {
  int start_state = -1;
  int end_state = -1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  std::vector<int> partial_pattern;

  bool from_string(std::string str);
};

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
