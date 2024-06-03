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
#include "State.h"

#include <vector>
#include <string>
#include <memory>
#include <sstream>


class Pattern {
 public:
  Pattern(const std::vector<int>& p, int hmax = 0);
  Pattern(const std::string& p, int hmax = 0);
  Pattern() = default;

 private:
  std::vector<int> throwval;
  std::vector<State> states;
  int h = 0;  // number of beats in a state (max. throw value)

 public:
  int objects() const;
  size_t length() const;
  int throwvalue(size_t index) const;
  bool is_valid() const;
  State state_before(size_t index);

  // pattern transformations
  Pattern dual() const;
  Pattern inverse(const Graph& graph) const;

  // string output
  std::string to_string(unsigned int throwdigits = 1,
      unsigned int hmax = 0) const;
  static void print_throw(std::ostringstream& buffer, unsigned int val,
      unsigned int throwdigits = 1, unsigned int hmax = 0);
  static char throw_char(unsigned int val);

  // analysis
  std::string make_analysis();

 private:
  // helper methods
  void check_have_states();
};

#endif
