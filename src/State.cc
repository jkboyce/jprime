//
// State.cc
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "State.h"

#include <algorithm>


State::State(int balls, int height)
    : n(balls), h(height), slot(height, 0) {}

State::State(std::string s) {
  h = s.size();
  n = 0;
  for (size_t i = 0; i < h; ++i) {
    int val = (s[i] == 'x' || s[i] == '1') ? 1 : 0;
    slot.push_back(val);
    n += val;
  }
}

State State::advance_with_throw(int throwval) const {
  State s = *this;
  int head = *s.slot.begin();
  s.slot.erase(s.slot.begin());
  s.slot.push_back(0);

  if ((head == 0 && throwval != 0) || (head != 0 && throwval == 0) ||
      (throwval > 0 && s.slot[throwval - 1] != 0)) {
    std::cerr << "cannot throw " << throwval
              << " from state " << (*this) << "\n";
    std::exit(EXIT_FAILURE);
  }

  s.slot[throwval - 1] = 1;
  return s;
}

// Return the next state downstream in the state's shift cycle

State State::downstream() const {
  State s = *this;
  int head = *s.slot.begin();
  s.slot.erase(s.slot.begin());
  s.slot.push_back(head);
  return s;
}

// Return the next state upstream in the state's shift cycle

State State::upstream() const {
  State s = *this;
  int tail = s.slot.back();
  s.slot.pop_back();
  s.slot.insert(s.slot.begin(), tail);
  return s;
}

// Return the reverse of this state

State State::reverse() const {
  State s = *this;
  std::reverse(s.slot.begin(), s.slot.end());
  return s;
}

int& State::operator[](size_t i) {
  return slot.at(i);
}

bool State::operator==(const State& s2) const {
  return (n == s2.n && h == s2.h && slot == s2.slot);
}

std::string State::to_string() const {
  std::string result;
  for (int i = 0; i < h; ++i) {
    result += (slot[i] ? 'x' : '-');
  }
  return result;
}

std::ostream& operator<<(std::ostream& ost, const State& s) {
  ost << s.to_string();
  return ost;
}