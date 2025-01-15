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
#include <sstream>
#include <regex>


// Initialize from a string.
//
// Return true on success, false otherwise.

bool WorkAssignment::from_string(const std::string& str) {
  std::regex rgx(
      "\\{ start_state:([0-9]+), end_state:([0-9]+), root_pos:([0-9]+), "
      "root_options:\\[([0-9,]+)\\], current:\\\"([0-9,]*)\\\" \\}"
  );
  std::smatch matches;

  if (!std::regex_search(str, matches, rgx) || matches.size() != 6) {
    return false;
  }

  start_state = std::stoi(matches[1].str());
  end_state = std::stoi(matches[2].str());
  root_pos = std::stoi(matches[3].str());

  root_throwval_options.clear();
  const std::string tvo{matches[4].str()};
  auto x = tvo.cbegin();
  while (true) {
    const auto y = std::find(x, tvo.cend(), ',');
    const std::string s{x, y};
    root_throwval_options.push_back(std::stoi(s));
    if (y == tvo.cend())
      break;
    x = y + 1;
  }

  partial_pattern.clear();
  const std::string pp{matches[5].str()};
  x = pp.cbegin();
  while (true) {
    const auto y = std::find(x, pp.cend(), ',');
    const std::string s{x, y};
    partial_pattern.push_back(std::stoi(s));
    if (y == pp.cend())
      break;
    x = y + 1;
  }

  // std::cout << "result:" << std::endl << *this << std::endl;
  return true;
}

// Return a text representation.

std::string WorkAssignment::to_string() const {
  std::ostringstream buffer;

  buffer << "{ start_state:" << start_state
         << ", end_state:" << end_state
         << ", root_pos:" << root_pos
         << ", root_options:[";
  for (const unsigned v : root_throwval_options) {
    if (v != root_throwval_options.front()) {
      buffer << ',';
    }
    buffer << v;
  }
  buffer << "], current:\"";
  for (size_t i = 0; i < partial_pattern.size(); ++i) {
    if (i > 0) {
      buffer << ',';
    }
    buffer << partial_pattern.at(i);
  }
  buffer << "\" }";
  return buffer.str();
}

// Print to an output stream using the default text format.

std::ostream& operator<<(std::ostream& ost, const WorkAssignment& wa) {
  ost << wa.to_string();
  return ost;
}
