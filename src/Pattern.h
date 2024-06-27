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

#include "State.h"

#include <vector>
#include <string>
#include <sstream>


class Pattern {
 public:
  Pattern(const std::vector<int>& p, int hmax = 0);
  Pattern(const std::string& p);
  Pattern() = default;

 private:
  std::vector<int> throwval;
  int h = 0;  // number of beats in a state (>= max. throw value)
  std::vector<State> states;
  std::vector<State> cyclestates;

 public:
  int objects() const;
  size_t length() const;
  State state_before(size_t index);
  bool is_valid() const;
  bool is_prime();
  bool is_superprime();

  // pattern transformations
  Pattern dual() const;
  Pattern inverse();

  // operator overrides
  int operator[](size_t index) const;
  bool operator==(const Pattern& s2) const;
  bool operator!=(const Pattern& s2) const;

  // string output
  std::string to_string(int throwdigits = 0, bool plusminus = false) const;
  static void print_throw(std::ostringstream& buffer, int val,
      int throwdigits = 1, int plusval = 0);
  static char throw_char(int val);

  // analysis
  std::string make_analysis();

 private:
  // helper methods
  void check_have_states();
};

std::ostream& operator<<(std::ostream& ost, const Pattern& p);

#endif
