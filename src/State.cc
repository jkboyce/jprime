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
#include <cassert>
#include <stdexcept>


// Initialize an empty state with the given number of objects and slots.

State::State(unsigned int b, unsigned int h)
    : b(b), h(h), _slot(h, 0) {}

// Initialize from a string representation.

State::State(std::string s) {
  h = static_cast<unsigned int>(s.size());
  b = 0;
  for (char ch : s) {
    int val = (ch == 'x' || ch == '1') ? 1 : 0;
    _slot.push_back(val);
    b += val;
  }
}

// Return a reference to the i'th slot in the state.

unsigned int& State::slot(size_t i) {
  return _slot.at(i);
}

const unsigned int& State::slot(size_t i) const {
  return _slot.at(i);
}

// Return a new State object that corresponds to the current State advanced by
// throw `throwval`.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

State State::advance_with_throw(unsigned int throwval) const {
  State s = *this;
  unsigned int head = *s._slot.begin();

  s._slot.erase(s._slot.begin());
  s._slot.push_back(0);
  assert(s._slot.size() == h);

  if (throwval > h) {
    std::string err = "Throw value " + std::to_string(throwval) +
        " exceeds the number of slots in state (" +
        std::to_string(h) + ")";
    throw std::invalid_argument(err);
  }

  if ((head == 0 && throwval != 0) || (head != 0 && throwval == 0) ||
      (throwval > 0 && s._slot.at(throwval - 1) != 0)) {
    std::string err = "Throw value " + std::to_string(throwval) +
        " is not valid from state " + to_string();
    throw std::invalid_argument(err);
  }

  if (throwval > 0) {
    s._slot.at(throwval - 1) = 1;
  }

  return s;
}

// Return the next state downstream in the state's shift cycle.

State State::downstream() const {
  State s = *this;
  unsigned int head = *s._slot.begin();
  s._slot.erase(s._slot.begin());
  s._slot.push_back(head);
  return s;
}

// Return the next state upstream in the state's shift cycle.

State State::upstream() const {
  State s = *this;
  unsigned int tail = s._slot.back();
  s._slot.pop_back();
  s._slot.insert(s._slot.begin(), tail);
  return s;
}

// Return the reverse of this state.

State State::reverse() const {
  State s = *this;
  std::reverse(s._slot.begin(), s._slot.end());
  return s;
}

// Perform comparisons on States.

bool State::operator==(const State& s2) const {
  return (b == s2.b && h == s2.h && _slot == s2._slot);
}

bool State::operator!=(const State& s2) const {
  return (b != s2.b || h != s2.h || _slot != s2._slot);
}

bool State::operator<(const State& s2) const {
  return state_compare(*this, s2);
}

// Return a string representation.

std::string State::to_string() const {
  std::string result;
  for (size_t i = 0; i < h; ++i) {
    result += (_slot.at(i) ? 'x' : '-');
  }
  return result;
}

std::ostream& operator<<(std::ostream& ost, const State& s) {
  ost << s.to_string();
  return ost;
}

// Standard library compliant Compare relation for States
//
// Returns true if the first argument appears before the second in a strict
// weak ordering, and false otherwise.

bool state_compare(const State& s1, const State& s2) {
  if (s1.b < s2.b)
    return true;
  if (s1.b > s2.b)
    return false;
  if (s1.h < s2.h)
    return true;
  if (s1.h > s2.h)
    return false;

  for (int i = s1.h - 1; i >= 0; --i) {
    if (s1.slot(i) < s2.slot(i))
      return true;
    if (s1.slot(i) > s2.slot(i))
      return false;
  }
  return false;
}
