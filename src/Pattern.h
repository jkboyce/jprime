//
// Pattern.h
//
// Class representing a single siteswap pattern.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_PATTERN_H_
#define JPRIME_PATTERN_H_

#include <string>


class Pattern {
 public:
  const std::string pattern;
  const int n;
  const int h;
  const int l;

 public:
  Pattern(const std::string& pat);
  void print_analysis();

 private:
  static int find_pattern_n(const std::string& pat);
  static int find_pattern_h(const std::string& pat);
  static bool find_pattern_nh(const std::string& pat, int& n, int& h);
};

#endif
