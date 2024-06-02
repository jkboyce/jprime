//
// Pattern.h
//
// Represents a single pattern in async siteswap notation.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_PATTERN_H_
#define JPRIME_PATTERN_H_

#include "Graph.h"

#include <vector>
#include <string>
#include <memory>
#include <sstream>


class Pattern {
 public:
  Pattern(const std::vector<int>& p);
  Pattern(const std::string& p);
  Pattern() = default;

 public:
  std::vector<int> throwval;

 public:
  int objects() const;
  int length() const;
  bool is_valid() const;

  // pattern transformations
  Pattern dual(int h = 0) const;
  Pattern inverse(const Graph& graph) const;

  // string output
  std::string to_string(unsigned int throwdigits = 1, unsigned int h = 0) const;
  static void print_throw(std::ostringstream& buffer, unsigned int val,
      unsigned int throwdigits = 1, unsigned int h = 0);
  static char throw_char(unsigned int val);
};

#endif
