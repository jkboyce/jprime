//
// State.h
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
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
  explicit State(unsigned h);
  explicit State(const std::string& s);
  State() = delete;

 private:
  std::vector<unsigned> _slot;  // 0 or 1

 public:
  size_t size() const;
  unsigned& slot(size_t i);
  const unsigned& slot(size_t i) const;
  State advance_with_throw(unsigned throwval) const;
  State downstream() const;
  State upstream() const;
  State reverse() const;
  bool operator==(const State& s2) const;
  bool operator!=(const State& s2) const;
  bool operator<(const State& s2) const;
  std::string to_string() const;
};

std::ostream& operator<<(std::ostream& ost, const State& s);
bool state_compare(const State& s1, const State& s2);

#endif
