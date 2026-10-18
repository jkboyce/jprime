//
// State.h
//
// Representation of an individual state in the juggling graph.
//
// Copyright (C) 1998-2026 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_STATE_H_
#define JPRIME_STATE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>


class State {
 public:
  static constexpr size_t NUM_WORDS = 4;
  static constexpr size_t MAX_BITS = NUM_WORDS * 64;

  struct BitRef {
    uint64_t& word;
    uint64_t mask;

    operator unsigned() const
    {
      return ((word & mask) != 0U) ? 1U : 0U;
    }

    BitRef& operator=(unsigned val)
    {
      if (val != 0U) {
        word |= mask;
      } else {
        word &= ~mask;
      }
      return *this;
    }

    BitRef& operator=(const BitRef& other)
    {
      return *this = static_cast<unsigned>(other);
    }
  };

  explicit State(unsigned h);
  explicit State(const std::string& s);
  State() = delete;

 private:
  unsigned _h = 0;
  uint64_t _words[NUM_WORDS] = {0, 0, 0, 0};

 public:
  size_t size() const
  {
    return _h;
  }

  unsigned slot(size_t i) const
  {
    if (i >= _h) [[unlikely]] {
      throw std::out_of_range("State::slot index out of range");
    }
    return static_cast<unsigned>((_words[i / 64] >> (i % 64)) & 1ULL);
  }

  BitRef slot(size_t i)
  {
    if (i >= _h) [[unlikely]] {
      throw std::out_of_range("State::slot index out of range");
    }
    return BitRef{.word=_words[i / 64], .mask=1ULL << (i % 64)};
  }

  uint64_t word(size_t w) const
  {
    return _words[w];
  }

  State advance_with_throw(unsigned throwval) const;
  State downstream() const;
  State upstream() const;
  void shift_downstream();
  void shift_downstream_zero();
  State downstream_zero() const;
  void shift_upstream();
  State reverse() const;
  int compare_same_size(const State& s2) const;
  bool operator==(const State& s2) const;
  bool operator!=(const State& s2) const;
  bool operator<(const State& s2) const;
  std::string to_string() const;

  friend bool state_compare(const State& s1, const State& s2);
};

std::ostream& operator<<(std::ostream& ost, const State& s);
bool state_compare(const State& s1, const State& s2);

#endif
