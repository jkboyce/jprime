//
// State.h
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_STATE_H_
#define JPRIME_STATE_H_

#include <iostream>
#include <string>
#include <vector>


class State {
 public:
  State(int balls, int height);
  State(std::string s);

 public:
  int n;  // number of objects
  int h;  // max throw height
  std::vector<int> slot;  // 0 or 1

  State advance_with_throw(int throwval) const;
  State downstream() const;
  State upstream() const;
  State reverse() const;
  int& operator[](size_t i);
  bool operator==(const State& s2) const;
  std::string to_string() const;
};

std::ostream& operator<<(std::ostream& ost, const State& s);

#endif
