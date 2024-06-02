//
// Pattern.cc
//
// Represents a single pattern in async siteswap notation.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Pattern.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
#include <ios>
#include <cassert>
#include <stdexcept>


// Initialize from a vector of throw values.

Pattern::Pattern(const std::vector<int>& p) {
  for (int val : p) {
    if (val < 0)
      break;
    throwval.push_back(val);
  }
}

// Initialize from a string representation.

Pattern::Pattern(const std::string& p) {
  const bool has_comma = (std::find(p.begin(), p.end(), ',') != p.end());

  if (has_comma) {
    auto x = p.begin();
    while (true) {
      auto y = std::find(x, p.end(), ',');
      std::string s{x, y};
      int val;
      try {
        val = std::stoi(s);
      } catch (const std::invalid_argument& ie) {
        throw std::invalid_argument("Could not parse as number: " + s);
      }
      if (val < 0) {
        throw std::invalid_argument("Disallowed throw value: " + s);
      }
      throwval.push_back(val);
      if (y == p.end())
        break;
      x = y + 1;
    }
    return;
  }

  int plusses = 0;
  int sum = 0;
  int maxval = 0;

  for (char ch : p) {
    if (ch == ' ') {
    } else if (ch == '-') {
      throwval.push_back(0);
    } else if (ch == '+') {
      throwval.push_back(-1);
      ++plusses;
    } else {
      int val = -1;
      if (ch >= '0' && ch <= '9') {
        val = static_cast<int>(ch - '0');
      } else if (ch >= 'a' && ch <= 'z') {
        val = 10 + static_cast<int>(ch - 'a');
      } else if (ch >= 'A' && ch <= 'Z') {
        val = 10 + static_cast<int>(ch - 'A');
      }
      if (val == -1) {
        throw std::invalid_argument(
            std::string("Unrecognized character: ") + ch);
      }
      throwval.push_back(val);
      sum += val;
      maxval = std::max(val, maxval);
    }
  }

  if (plusses > 0) {
    // solve the equation: plusses * h + sum = n * length
    // for minimal values of `n` and `h`
    //
    // `h` must be at least as large as ceiling(sum / (length - plusses))

    int h = 1;
    if (length() > plusses) {
      h = 1 + std::max(maxval, (sum - 1) / (length() - plusses));
      while (true) {
        if ((plusses * h + sum) % length() == 0) {
          assert(h > maxval);
          assert((plusses * h + sum) / length() <= h);
          break;
        }
        ++h;
      }
    }

    for (size_t i = 0; i < throwval.size(); ++i) {
      if (throwval.at(i) < 0) {
        throwval.at(i) = h;
      }
    }
  }
}

//------------------------------------------------------------------------------
// Pattern properties
//------------------------------------------------------------------------------

// Return the number of objects in the pattern.

int Pattern::objects() const {
  int sum = 0;
  for (int val : throwval)
    sum += val;
  return (sum / length());
}

// Return the pattern length (period).

int Pattern::length() const {
  return throwval.size();
}

// Return true if the pattern is a valid siteswap, false otherwise.

bool Pattern::is_valid() const {
  if (throwval.size() == 0)
    return false;

  // collisions in the pattern?
  std::vector<bool> taken(length(), false);
  for (int i = 0; i < length(); ++i) {
    int index = (i + throwval.at(i)) % length();
    if (taken.at(index))
      return false;
    taken.at(index) = true;
  }

  // sanity check (should always pass): integer average?
  int sum = 0;
  for (int val : throwval)
    sum += val;
  assert(sum % length() == 0);

  return true;
}

//------------------------------------------------------------------------------
// Pattern transformations
//------------------------------------------------------------------------------

// Return the dual of the pattern.
//
// The duality transform is with respect to a maximum throw value, which is
// taken to be the greater of the supplied parameter `h`, and the throw values
// in the pattern.

Pattern Pattern::dual(int h) const {
  const size_t l = throwval.size();
  if (l == 0)
    return {};

  for (int val : throwval) {
    h = std::max(h, val);
  }

  std::vector<int> dual_throws(l);
  for (size_t i = 0; i < l; ++i) {
    dual_throws[i] = h - throwval[l - 1 - i];
  }
  return {dual_throws};
}

// Return the inverse of the given pattern on the juggling graph. If the inverse
// does not exist, return an empty pattern.

Pattern Pattern::inverse(const Graph& graph) const {
  std::vector<int> patternstate(graph.numstates + 1);
  std::vector<bool> state_used(graph.numstates + 1, false);
  std::vector<bool> cycle_used(graph.numstates + 1, false);

  // `start_state` is assumed to be the lowest active state in the graph
  unsigned int start_state = 1;
  for (; start_state <= graph.numstates; ++start_state) {
    if (graph.state_active.at(start_state))
      break;
  }
  if (start_state > graph.numstates)
    return {};

  // Step 1. build a vector of state numbers traversed by the pattern, and
  // determine if an inverse exists.
  //
  // a pattern has an inverse if and only if:
  // - it visits more than one shift cycle on the state graph, and
  // - it never revisits a shift cycle, and
  // - it never does a link throw (0 < t < h) within a single cycle

  unsigned int state_current = start_state;
  unsigned int cycle_current = graph.cyclenum.at(start_state);
  bool cycle_multiple = false;

  for (size_t i = 0; i < throwval.size(); ++i) {
    patternstate.at(i) = state_current;
    state_used.at(state_current) = true;

    const int state_next = graph.advance_state(state_current, throwval.at(i));
    assert(state_next > 0);
    const unsigned int cycle_next = graph.cyclenum.at(state_next);

    if (cycle_next != cycle_current) {
      // mark a shift cycle as used only when we transition off it
      if (cycle_used.at(cycle_current)) {
        // revisited cycle number `cycle_current` --> no inverse
        return {};
      }
      cycle_used.at(cycle_current) = true;
      cycle_multiple = true;
    } else if (throwval.at(i) != 0 &&
        static_cast<unsigned int>(throwval.at(i)) != graph.h) {
      // link throw within a single cycle --> no inverse
      return {};
    }

    state_current = state_next;
    cycle_current = cycle_next;
  }
  patternstate.at(throwval.size()) = start_state;

  if (!cycle_multiple) {
    // never left starting shift cycle --> no inverse
    return {};
  }

  // Step 2. Find the states and throws of the inverse.
  //
  // Iterate through the link throws in the pattern to build up a list of
  // states and throws for the inverse.
  //
  // Note the inverse may go through states that aren't in memory so we can't
  // refer to them by state number.

  std::vector<unsigned int> inversepattern;
  std::vector<State> inversestate;

  for (size_t i = 0; i < throwval.size(); ++i) {
    // continue until `throwval[i]` is a link throw
    if (graph.cyclenum.at(patternstate.at(i)) ==
        graph.cyclenum.at(patternstate.at(i + 1))) {
      continue;
    }

    if (inversestate.size() == 0) {
      // the inverse pattern starts at the (reversed version of) the next state
      // 'downstream' from `patternstate[i]`
      inversestate.push_back(
          graph.state.at(patternstate.at(i)).downstream().reverse());
    }

    const unsigned int inversethrow = graph.h -
        static_cast<unsigned int>(throwval.at(i));
    inversepattern.push_back(inversethrow);
    inversestate.push_back(
        inversestate.back().advance_with_throw(inversethrow));

    // advance the inverse pattern along the shift cycle until it gets to a
    // state whose reverse is used by the original pattern

    while (true) {
      State trial_state = inversestate.back().downstream();
      unsigned int trial_statenum = graph.get_statenum(trial_state.reverse());
      if (trial_statenum > 0 && state_used.at(trial_statenum))
        break;

      inversepattern.push_back(trial_state.slot.at(graph.h - 1) ? graph.h : 0);
      inversestate.push_back(trial_state);
    }

    if (inversestate.back() == inversestate.front())
      break;
  }
  assert(inversestate.size() > 0);
  assert(inversestate.back() == inversestate.front());

  // Step 3. Create the final inverse pattern.
  //
  // By convention we start all patterns with their smallest state.

  std::vector<int> inverse_final;

  size_t min_index = 0;
  for (size_t i = 1; i < inversestate.size(); ++i) {
    if (inversestate.at(i) < inversestate.at(min_index)) {
      min_index = i;
    }
  }

  const size_t inverselength = inversepattern.size();
  for (size_t i = 0; i < inverselength; ++i) {
    size_t j = (i + min_index) % inverselength;
    inverse_final.push_back(inversepattern.at(j));
  }

  return {inverse_final};
}

//------------------------------------------------------------------------------
// Pattern output
//------------------------------------------------------------------------------

// Return the string representation of the pattern.
//
// Parameter `throwdigits` determines the field width to use, and whether to
// print as single-character alphanumeric or as a number.
//
// Parameter `h`, if nonzero, causes throws of value 0 and `h` to be output as
// `-` and `+` respectively.

std::string Pattern::to_string(unsigned int throwdigits, unsigned int h) const {
  std::ostringstream buffer;

  for (size_t i = 0; i < throwval.size(); ++i) {
    if (throwdigits > 1 && i != 0)
      buffer << ',';
    const unsigned int val = static_cast<unsigned int>(throwval.at(i));
    print_throw(buffer, val, throwdigits, h);
  }

  return buffer.str();
}

// Output a single throw to a string buffer.
//
// For a parameter description see Pattern::to_string().

void Pattern::print_throw(std::ostringstream& buffer, unsigned int val,
    unsigned int throwdigits, unsigned int h) {
  if (h > 0 && val == 0) {
    buffer << '-';
    return;
  } else if (h > 0 && val == h) {
    buffer << '+';
    return;
  }

  if (throwdigits == 1) {
    buffer << throw_char(val);
  } else {
    buffer << std::setw(throwdigits) << val;
  }
}

// Return a character for a given integer throw value (0 = '0', 1 = '1',
// 10 = 'a', 11 = 'b', ...

char Pattern::throw_char(unsigned int val) {
  if (val < 10) {
    return static_cast<char>(val + '0');
  } else if (val < 36) {
    return static_cast<char>(val - 10 + 'a');
  } else {
    return '?';
  }
}
