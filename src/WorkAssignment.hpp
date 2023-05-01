#ifndef JDEEP_WORKASSIGNMENT_H
#define JDEEP_WORKASSIGNMENT_H

#include <iostream>
#include <list>
#include <vector>
#include <string>

// Defines a work assignment that can be handed off
struct WorkAssignment {
  int start_state = 1;
  int end_state = 1;
  int root_pos = 0;
  std::list<int> root_throwval_options;
  std::vector<int> partial_pattern;

  // not saved and loaded
  //int l_current = 0;

  bool from_string(std::string str);
};

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa);

#endif
