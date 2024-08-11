//
// WorkAssignment.cc
//
// Functions for reading a WorkAssignment from a string representation, and
// printing to an output stream.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "WorkAssignment.h"

#include <iostream>
#include <string>
#include <regex>


// Initialize from a string; return true on success, false otherwise.

bool WorkAssignment::from_string(const std::string& str) {
  std::regex rgx(
      "\\{ start_state:([0-9]+), end_state:([0-9]+), root_pos:([0-9]+), "
      "root_options:\\[([0-9,]+)\\], current:\\\"([0-9,]*)\\\" \\}"
  );
  std::smatch matches;

  if (!std::regex_search(str, matches, rgx) || matches.size() != 6)
    return false;

  start_state = std::stoi(matches[1].str());
  end_state = std::stoi(matches[2].str());
  root_pos = std::stoi(matches[3].str());

  root_throwval_options.clear();
  std::string tvo{matches[4].str()};
  auto x = tvo.cbegin();
  while (true) {
    auto y = std::find(x, tvo.cend(), ',');
    std::string s{x, y};
    root_throwval_options.push_back(std::stoi(s));
    if (y == tvo.cend())
      break;
    x = y + 1;
  }

  partial_pattern.clear();
  std::string pp{matches[5].str()};
  x = pp.cbegin();
  while (true) {
    auto y = std::find(x, pp.cend(), ',');
    std::string s{x, y};
    partial_pattern.push_back(std::stoi(s));
    if (y == pp.cend())
      break;
    x = y + 1;
  }

  // std::cout << "result:" << std::endl << *this << std::endl;
  return true;
}

// Output a text representation of a WorkAssignment.

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa) {
  ost << "{ start_state:" << wa.start_state
      << ", end_state:" << wa.end_state
      << ", root_pos:" << wa.root_pos
      << ", root_options:[";
  for (unsigned v : wa.root_throwval_options) {
    if (v != wa.root_throwval_options.front())
      ost << ',';
    ost << v;
  }
  ost << "], current:\"";
  for (size_t i = 0; i < wa.partial_pattern.size(); ++i) {
    if (i > 0)
      ost << ',';
    ost << wa.partial_pattern.at(i);
  }
  ost << "\" }";
  return ost;
}
