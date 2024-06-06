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
#include <set>
#include <cassert>
#include <stdexcept>


// Initialize from a vector of throw values.
//
// Parameter `hmax` determines the number of beats to use in each state; it
// must be at least as large as the largest throw value. If omitted then the
// number of beats per state is set equal to the largest throw value.

Pattern::Pattern(const std::vector<int>& p, int hmax) {
  int maxval = 0;
  for (int val : p) {
    if (val < 0)
      break;
    throwval.push_back(val);
    maxval = std::max(val, maxval);
  }

  if (hmax > 0) {
    assert(hmax >= maxval);
    h = hmax;
  } else {
    h = maxval;
  }
}

// Initialize from a string representation.
//
// Parameter `hmax` determines the number of beats to use in each state; it
// must be at least as large as the largest throw value. If omitted then the
// number of beats per state is set equal to the largest throw value.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

Pattern::Pattern(const std::string& p, int hmax) {
  const bool has_comma = (std::find(p.begin(), p.end(), ',') != p.end());
  int maxval = 0;

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
      maxval = std::max(val, maxval);
      if (y == p.end())
        break;
      x = y + 1;
    }

    if (hmax > 0) {
      assert(hmax >= maxval);
      h = hmax;
    } else {
      h = maxval;
    }
    return;
  }

  int plusses = 0;
  int sum = 0;

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

  if (plusses > 0) {  // pattern was in block form
    // solve the equation: plusses * h + sum = b * length
    // for minimal values of `b` and `h`
    //
    // `h` must be at least as large as ceiling(sum / (length - plusses))

    h = 1;
    int len = static_cast<int>(length());
    if (len > plusses) {
      h = 1 + std::max(maxval, (sum - 1) / (len - plusses));
      while (true) {
        if ((plusses * h + sum) % len == 0) {
          assert(h > maxval);
          assert((plusses * h + sum) / len <= h);
          break;
        }
        ++h;
      }
    }
    assert(hmax == 0 || hmax == h);

    // fill in throw values of `h`
    for (size_t i = 0; i < throwval.size(); ++i) {
      if (throwval.at(i) < 0) {
        throwval.at(i) = h;
      }
    }
    return;
  }

  if (hmax > 0) {
    assert(hmax >= maxval);
    h = hmax;
  } else {
    h = maxval;
  }
}

//------------------------------------------------------------------------------
// Pattern properties
//------------------------------------------------------------------------------

// Return the number of objects in the pattern.
//
// If the pattern is not valid, return 0.

int Pattern::objects() const {
  if (!is_valid())
    return 0;

  int sum = 0;
  for (int val : throwval)
    sum += val;
  return (sum / length());
}

// Return the pattern length (period).

size_t Pattern::length() const {
  return throwval.size();
}

// Return the throw value on beat `index`.

int Pattern::throwvalue(size_t index) const {
  assert(index >= 0 && index < length());
  return throwval.at(index);
}

// Return the State value immediately before the throw on beat `index`.

State Pattern::state_before(size_t index) {
  check_have_states();
  return states.at(index);
}

// Return true if the pattern is a valid siteswap, false otherwise.

bool Pattern::is_valid() const {
  if (length() == 0)
    return false;

  // look for collisions
  std::vector<bool> taken(length(), false);
  for (size_t i = 0; i < length(); ++i) {
    size_t index = (i + static_cast<size_t>(throwval.at(i))) % length();
    if (taken.at(index))
      return false;
    taken.at(index) = true;
  }

  // confirm integer average (should always pass)
  int sum = 0;
  for (int val : throwval)
    sum += val;
  assert(sum % length() == 0);
  return true;
}

// Return true if the pattern is prime, false otherwise.

bool Pattern::is_prime() {
  check_have_states();
  std::set<State> s(states.begin(), states.end());
  return (s.size() == states.size());
}

// Return true if the pattern is superprime, false otherwise.

bool Pattern::is_superprime() {
  if (length() == 0)
    return true;
  check_have_states();

  std::vector<State> deduped;
  deduped.push_back(cyclestates.at(0));
  for (size_t i = 1; i < cyclestates.size(); ++i) {
    // check for link throw within a cycle
    if (cyclestates.at(i) == cyclestates.at((i + 1) % length()) &&
        throwval.at(i) != 0 && throwval.at(i) != h)
      return false;
    // remove runs of the same cyclestate
    if (cyclestates.at(i) != deduped.back()) {
      deduped.push_back(cyclestates.at(i));
    }
  }

  // remove the ones at the end that are duplicates of the start
  while (deduped.size() != 0 && deduped.back() == deduped.front()) {
    deduped.pop_back();
  }

  std::set<State> s(deduped.begin(), deduped.end());
  return (s.size() == deduped.size());
}

//------------------------------------------------------------------------------
// Pattern transformations
//------------------------------------------------------------------------------

// Return the dual of the pattern.
//
// The duality transform is with respect to a maximum throw value, which is
// taken to be the greater of the supplied parameter `h`, and the throw values
// in the pattern.

Pattern Pattern::dual() const {
  if (length() == 0)
    return {};

  std::vector<int> dual_throws(length());
  for (size_t i = 0; i < length(); ++i) {
    dual_throws.at(i) = h - throwval.at(length() - 1 - i);
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
// Helper methods
//------------------------------------------------------------------------------

// Ensure the `states` vector is populated with the states in the pattern. If it
// isn't then generate them, along with the `cyclestates` vector describing the
// shift cycles traversed.
//
// This should only be called for a valid pattern!

void Pattern::check_have_states() {
  if (states.size() == length())
    return;
  assert(states.size() == 0);
  assert(cyclestates.size() == 0);

  // find the starting state
  State start_state{static_cast<unsigned int>(objects()),
      static_cast<unsigned int>(h)};
  for (size_t i = 0; i < length(); ++i) {
    int fillslot = throwval.at(i) - static_cast<int>(length()) +
        static_cast<int>(i);
    while (fillslot >= 0) {
      if (fillslot < static_cast<int>(h)) {
        assert(start_state.slot.at(fillslot) == 0);
        start_state.slot.at(fillslot) = 1;
      }
      fillslot -= length();
    }
  }

  states.push_back(start_state);
  State state = start_state;
  for (size_t i = 0; i < length(); ++i) {
    state = state.advance_with_throw(throwval.at(i));
    if (i != length() - 1) {
      states.push_back(state);
    }
  }
  assert(state == start_state);

  for (State s : states) {
    State cyclestate = s;
    State s2 = s.downstream();
    while (s2 != s) {
      if (s2 < cyclestate)
        cyclestate = s2;
      s2 = s2.downstream();
    }
    cyclestates.push_back(cyclestate);
  }

  assert(states.size() == length());
  assert(cyclestates.size() == length());
}

//------------------------------------------------------------------------------
// Pattern output
//------------------------------------------------------------------------------

// Return the string representation of the pattern.
//
// Parameter `throwdigits` determines the field width to use, and whether to
// print as single-character alphanumeric or as a number.
//
// Parameter `hmax`, if nonzero, causes throws of value 0 and `h` to be output
// as `-` and `+` respectively.

std::string Pattern::to_string(unsigned int throwdigits,
      unsigned int hmax) const {
  std::ostringstream buffer;

  for (size_t i = 0; i < throwval.size(); ++i) {
    if (throwdigits > 1 && i != 0)
      buffer << ',';
    const unsigned int val = static_cast<unsigned int>(throwval.at(i));
    print_throw(buffer, val, throwdigits, hmax);
  }

  return buffer.str();
}

// Output a single throw to a string buffer.
//
// For a parameter description see Pattern::to_string().

void Pattern::print_throw(std::ostringstream& buffer, unsigned int val,
    unsigned int throwdigits, unsigned int hmax) {
  if (hmax > 0 && val == 0) {
    buffer << '-';
    return;
  } else if (hmax > 0 && val == hmax) {
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

//------------------------------------------------------------------------------
// Pattern analysis
//------------------------------------------------------------------------------

// Return an analysis of the pattern in string format.

std::string Pattern::make_analysis() {
  std::ostringstream buffer;

  int maxval = 0;
  for (size_t i = 0; i < length(); ++i) {
    maxval = std::max(throwvalue(i), maxval);
  }
  int throwdigits = 1;
  for (int temp = 10; temp <= maxval; temp *= 10) {
    ++throwdigits;
  }
  std::string patstring = to_string(throwdigits + 1);

  if (!is_valid()) {
    buffer << "Input is not a valid siteswap due to collision:\n  ";

    int collision_start = -1;
    int collision_end = -1;
    std::vector<int> landing_index(length(), -1);
    for (size_t i = 0; i < length(); ++i) {
      size_t index = (i + static_cast<size_t>(throwvalue(i))) % length();
      if (landing_index[index] != -1) {
        collision_start = landing_index[index];
        collision_end = i;
        break;
      }
      landing_index[index] = i;
    }
    assert(collision_start != -1 && collision_end != -1);

    int index = 0;
    auto x = patstring.begin();
    while (true) {
      auto y = std::find(x, patstring.end(), ',');
      std::string s{x, y};
      if (index == collision_start || index == collision_end) {
        buffer << '[' << s << ']';
      } else {
        buffer << s;
      }
      if (y == patstring.end())
        break;
      buffer << ',';
      x = y + 1;
      ++index;
    }
    buffer << '\n';
    buffer << "------------------------------------------------------------";
    return buffer.str();
  }

  buffer << "Pattern representations:\n";
  if (maxval < 36) {
    buffer << "  short form     " << to_string(1) << '\n'
           << "  block form     " << to_string(1, maxval) << '\n';
  }
  buffer << "  standard form " << patstring << "\n\n";

  buffer << "Properties:\n"
         << "  objects        " << objects() << '\n'
         << "  length         " << length() << '\n'
         << "  max. throw     " << maxval << '\n'
         << "  is_prime       " << std::boolalpha << is_prime() << '\n'
         << "  is_superprime  " << std::boolalpha << is_superprime() << "\n\n";

  buffer << "States and shift cycles:\n";
  check_have_states();

  std::vector<State> shiftcycles_visited;
  for (size_t i = 0; i < length(); ++i) {
    if (throwvalue(i) != 0 && throwvalue(i) != h) {
      shiftcycles_visited.push_back(cyclestates.at((i + 1) % length()));
    }
  }
  for (size_t i = 0; i < length(); ++i) {
    if (std::count(states.begin(), states.end(), states.at(i)) == 1) {
      buffer << "  ";
    } else {
      buffer << "R ";
    }
    buffer << states.at(i) << "  "
           << std::setw(throwdigits) << throwvalue(i) << "   ";
    int prev_throwvalue = throwvalue(i == 0 ? length() - 1 : i - 1);
    bool print_cyclestate = (prev_throwvalue != 0 && prev_throwvalue != h);
    if (!print_cyclestate) {
      buffer << "   .\n";
      continue;
    }
    buffer << "  (" << cyclestates.at(i) << ')';
    if (std::count(shiftcycles_visited.begin(), shiftcycles_visited.end(),
        cyclestates.at(i)) == 1) {
      buffer << '\n';
    } else {
      buffer << " R\n";
    }
  }
  buffer << '\n';

  Graph graph(objects(), maxval);

  buffer << "Graph (" << objects() << ',' << maxval << "):\n"
         << "  states         " << graph.numstates << '\n'
         << "  shift cycles   " << graph.numcycles << '\n'
         << "  short cycles   " << graph.numshortcycles << '\n';

  // is_prime
  // is_superprime
  // list of states traversed, with duplicates shown
  // list of states missed, if short
  // table of shift cycles: number, representative state, period, pattern states on it
  // inverse, if it exists

  buffer << "------------------------------------------------------------";
  return buffer.str();
}
