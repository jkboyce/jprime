
#include "WorkAssignment.hpp"

#include <iostream>

char throw_char(int val) {
  if (val < 10)
    return static_cast<char>(val + '0');
  else
    return static_cast<char>(val - 10 + 'a');
}

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa) {
  ost << "{ start_state:" << wa.start_state
      << ", end_state:" << wa.end_state
      << ", root_pos:" << wa.root_pos
      << ", root_options:[";
  for (int v : wa.root_throwval_options)
    ost << throw_char(v);
  ost << "], current:\"";
  for (int i = 0; i < wa.partial_pattern.size(); ++i)
    ost << throw_char(wa.partial_pattern[i]);
  ost << "\" }";
  return ost;
}
