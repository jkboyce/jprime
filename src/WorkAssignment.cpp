
#include "WorkAssignment.hpp"

#include <iostream>
#include <string>
#include <regex>

static char throw_char(int val) {
  if (val < 10)
    return static_cast<char>(val + '0');
  else
    return static_cast<char>(val - 10 + 'a');
}

static int throw_value(char ch) {
  if (ch >= '0' && ch <= '9')
    return static_cast<int>(ch - '0');
  else if (ch >= 'a' && ch <= 'z')
    return static_cast<int>(ch - 'a') + 10;
  else if (ch >= 'A' && ch <= 'Z')
    return static_cast<int>(ch - 'A') + 10;
  else
    return -1;
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

bool WorkAssignment::from_string(std::string str) {
  std::regex rgx(
      "\\{ start_state:([0-9]+), end_state:([0-9]+), root_pos:([0-9]+), "
      "root_options:\\[([0-9a-z]+)\\], current:\\\"([0-9a-z]*)\\\" \\}"
  );
  std::smatch matches;

  if (!std::regex_search(str, matches, rgx) || matches.size() != 6)
    return false;

  start_state = std::stoi(matches[1].str());
  end_state = std::stoi(matches[2].str());
  root_pos = std::stoi(matches[3].str());

  root_throwval_options.clear();
  for (char c : matches[4].str())
    root_throwval_options.push_back(throw_value(c));

  partial_pattern.clear();
  for (char c : matches[5].str())
    partial_pattern.push_back(throw_value(c));

  // std::cout << "result:" << std::endl << *this << std::endl;
  return true;
}
