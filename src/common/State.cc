//
// State.cc
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2026 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "State.h"

#include <bit>
#include <cassert>
#include <format>
#include <iostream>
#include <stdexcept>


// Initialize an empty state with `h` slots.

State::State(unsigned h)
    : _h(h)
{
  if (h > MAX_BITS) {
    throw std::invalid_argument(std::format(
        "State size h = {} exceeds State::MAX_BITS ({})", h, MAX_BITS));
  }
}

// Initialize from a string representation.

State::State(const std::string& s)
    : _h(static_cast<unsigned>(s.size()))
{
  if (s.size() > MAX_BITS) {
    throw std::invalid_argument(std::format(
        "State string size {} exceeds State::MAX_BITS ({})", s.size(), MAX_BITS));
  }
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == 'x' || s[i] == '1') {
      _words[i / 64] |= (1ULL << (i % 64));
    }
  }
}

// Return a new State object that corresponds to the current State advanced by
// throw `throwval`.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

State State::advance_with_throw(unsigned throwval) const
{
  State s = *this;
  const unsigned head = s.slot(0);

  s.shift_downstream_zero();

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
  if (_h == 0) {
    return;
  }
  const uint64_t head = _words[0] & 1ULL;
  shift_downstream_zero();
  if (head != 0U) {
    const size_t last_slot = _h - 1;
    _words[last_slot / 64] |= (1ULL << (last_slot % 64));
  }
}

// Shift the state downstream by 1 without wrap-around (slot h - 1 is set to 0).

void State::shift_downstream_zero()
{
  if (_h == 0) {
    return;
  }
  const size_t words_used = (_h + 63) / 64;
  for (size_t w = 0; w < words_used; ++w) {
    _words[w] >>= 1;
    if (w + 1 < words_used) {
      _words[w] |= (_words[w + 1] << 63);
    }
  }
}

// Return a state shifted downstream by 1 without wrap-around.

State State::downstream_zero() const
{
  State s = *this;
  s.shift_downstream_zero();
  return s;
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
  if (_h == 0) {
    return;
  }
  const size_t tail_slot = _h - 1;
  const uint64_t tail = (_words[tail_slot / 64] >> (tail_slot % 64)) & 1ULL;

  const size_t words_used = (_h + 63) / 64;
  for (int w = static_cast<int>(words_used) - 1; w >= 0; --w) {
    _words[w] <<= 1;
    if (w > 0) {
      _words[w] |= (_words[w - 1] >> 63);
    }
  }

  // Clear any bits >= _h that might have been shifted into the top active word
  if (_h % 64 != 0) {
    const uint64_t mask = (1ULL << (_h % 64)) - 1;
    _words[words_used - 1] &= mask;
  }

  // Put tail into slot 0
  if (tail != 0U) {
    _words[0] |= 1ULL;
  }
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
  State s(_h);
  for (size_t i = 0; i < _h; ++i) {
    if (((_words[i / 64] >> (i % 64)) & 1ULL) != 0U) {
      const size_t rev_i = _h - 1 - i;
      s._words[rev_i / 64] |= (1ULL << (rev_i % 64));
    }
  }
  return s;
}

// Perform comparisons on States.

// 3-way comparison of two states of the same size and ball count.
// Returns -1 if *this < s2, 0 if *this == s2, and 1 if *this > s2.

int State::compare_same_size(const State& s2) const
{
  const size_t words_used = (_h + 63) / 64;
  for (int w = static_cast<int>(words_used) - 1; w >= 0; --w) {
    if (_words[w] < s2._words[w]) {
      return -1;
    }
    if (_words[w] > s2._words[w]) {
      return 1;
    }
  }
  return 0;
}

bool State::operator==(const State& s2) const
{
  if (_h != s2._h) {
    return false;
  }
  for (size_t w = 0; w < NUM_WORDS; ++w) {
    if (_words[w] != s2._words[w]) {
      return false;
    }
  }
  return true;
}

bool State::operator!=(const State& s2) const
{
  return !(*this == s2);
}

bool State::operator<(const State& s2) const
{
  return state_compare(*this, s2);
}

// Return a string representation.

std::string State::to_string() const
{
  std::string result;
  result.reserve(_h);
  for (size_t i = 0; i < _h; ++i) {
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
  const size_t w1 = (s1._h + 63) / 64;
  for (size_t w = 0; w < w1; ++w) {
    b1 += std::popcount(s1._words[w]);
  }
  unsigned b2 = 0;
  const size_t w2 = (s2._h + 63) / 64;
  for (size_t w = 0; w < w2; ++w) {
    b2 += std::popcount(s2._words[w]);
  }

  if (b1 < b2) {
    return true;
  }
  if (b1 > b2) {
    return false;
  }
  if (s1._h < s2._h) {
    return true;
  }
  if (s1._h > s2._h) {
    return false;
  }

  return s1.compare_same_size(s2) < 0;
}
