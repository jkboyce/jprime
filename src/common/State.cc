//
// State.cc
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "State.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <stdexcept>


// Initialize an empty state with `h` slots.

State::State(unsigned h)
    : _slot(h, 0)
{}

// Initialize from a string representation.

State::State(const std::string& s)
{
  for (const char ch : s) {
    _slot.push_back((ch == 'x' || ch == '1') ? 1 : 0);
  }
}

// Return the number of slots in the state.

size_t State::size() const
{
  return _slot.size();
}

// Return a reference to the i'th slot in the state, indexing from 0.

unsigned& State::slot(size_t i)
{
  return _slot.at(i);
}

const unsigned& State::slot(size_t i) const
{
  return _slot.at(i);
}

// Return raw pointer to the slot array.

const unsigned* State::data() const
{
  return _slot.data();
}

unsigned* State::data()
{
  return _slot.data();
}

// Return a new State object that corresponds to the current State advanced by
// throw `throwval`.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

State State::advance_with_throw(unsigned throwval) const
{
  State s = *this;
  const unsigned head = *s._slot.begin();

  s._slot.erase(s._slot.begin());
  s._slot.push_back(0);

  if (throwval > size()) {
    throw std::invalid_argument(
        std::format("Throw value {} exceeds the number of slots in state ({})",
        throwval, size()));
  }

  if ((head == 0 && throwval != 0) || (head != 0 && throwval == 0) ||
      (throwval > 0 && s.slot(throwval - 1) != 0)) {
    throw std::invalid_argument(std::format(
        "Throw value {} is not valid from state {}", throwval, to_string()));
  }

  if (throwval > 0) {
    s.slot(throwval - 1) = 1;
  }

  return s;
}

// Shift the state downstream in place (rotate slots left by 1).

void State::shift_downstream()
{
  if (_slot.empty()) {
    return;
  }
  const unsigned head = _slot.front();
  std::memmove(_slot.data(), _slot.data() + 1, (_slot.size() - 1) * sizeof(unsigned));
  _slot.back() = head;
}

// Return the next state downstream in the state's shift cycle.

State State::downstream() const
{
  State s = *this;
  s.shift_downstream();
  return s;
}

// Shift the state upstream in place (rotate slots right by 1).

void State::shift_upstream()
{
  if (_slot.empty()) {
    return;
  }
  const unsigned tail = _slot.back();
  std::memmove(_slot.data() + 1, _slot.data(), (_slot.size() - 1) * sizeof(unsigned));
  _slot[0] = tail;
}

// Return the next state upstream in the state's shift cycle.

State State::upstream() const
{
  State s = *this;
  s.shift_upstream();
  return s;
}

// Return the reverse of this state.

State State::reverse() const
{
  State s = *this;
  std::ranges::reverse(s._slot);
  return s;
}

// Perform comparisons on States.

// 3-way comparison of two states of the same size and ball count.
// Returns -1 if *this < s2, 0 if *this == s2, and 1 if *this > s2.

int State::compare_same_size(const State& s2) const
{
  for (int i = static_cast<int>(_slot.size()) - 1; i >= 0; --i) {
    if (_slot[i] < s2._slot[i]) {
      return -1;
    }
    if (_slot[i] > s2._slot[i]) {
      return 1;
    }
  }
  return 0;
}

bool State::operator==(const State& s2) const
{
  return (size() == s2.size() && _slot == s2._slot);
}

bool State::operator!=(const State& s2) const
{
  return (size() != s2.size() || _slot != s2._slot);
}

bool State::operator<(const State& s2) const
{
  return state_compare(*this, s2);
}

// Return a string representation.

std::string State::to_string() const
{
  std::string result;
  for (size_t i = 0; i < size(); ++i) {
    result += (slot(i) != 0 ? 'x' : '-');
  }
  return result;
}

std::ostream& operator<<(std::ostream& ost, const State& s)
{
  ost << s.to_string();
  return ost;
}

// Standard library compliant Compare relation for States
//
// Returns true if the first argument appears before the second in a strict
// weak ordering, and false otherwise.

bool state_compare(const State& s1, const State& s2)
{
  unsigned b1 = 0;
  for (size_t i = 0; i < s1.size(); ++i) {
    b1 += s1.slot(i);
  }
  unsigned b2 = 0;
  for (size_t i = 0; i < s2.size(); ++i) {
    b2 += s2.slot(i);
  }

  if (b1 < b2) {
    return true;
  }
  if (b1 > b2) {
    return false;
  }
  if (s1.size() < s2.size()) {
    return true;
  }
  if (s1.size() > s2.size()) {
    return false;
  }

  return s1.compare_same_size(s2) < 0;
}
