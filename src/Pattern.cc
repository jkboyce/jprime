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
#include "Graph.h"

#include <iostream>
#include <algorithm>
#include <numeric>
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
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

Pattern::Pattern(const std::vector<int>& p, int hmax) {
  int maxval = 0;
  for (int val : p) {
    if (val < 0)
      break;
    throwval.push_back(val);
    maxval = std::max(val, maxval);
  }

  if (hmax > 0) {
    if (hmax < maxval) {
      std::string err = "Supplied `hmax` value (" + std::to_string(hmax)
          + ") is too small";
      throw std::invalid_argument(err);
    }
    h = hmax;
  } else {
    h = std::max(maxval, 1);
  }
}

// Initialize from a string representation.
//
// The input string can be a comma-separated list of integer throw values or a
// pattern in short form with one alphanumeric character per throw. It accepts
// the +/- substitutions for h and 0. It also looks for an optional suffix in
// the form "/<h>" where <h> is an integer; this allows one to set the
// graph (n,h) that is used for e.g. determining superprimality and finding an
// inverse.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

Pattern::Pattern(const std::string& p) {
  int maxval = 0;
  int hmax = 0;
  const bool has_slash = (std::find(p.begin(), p.end(), '/') != p.end());
  const bool has_comma = (std::find(p.begin(), p.end(), ',') != p.end());
  std::string pat;

  if (has_slash) {
    auto x = std::find(p.begin(), p.end(), '/');
    std::string hstr{x + 1, p.end()};
    int val;
    try {
      val = std::stoi(hstr);
    } catch (const std::invalid_argument& ie) {
      throw std::invalid_argument("Not a number: " + hstr);
    }
    if (val < 1) {
      throw std::invalid_argument("Disallowed `h` value: " + hstr);
    }
    hmax = val;
    pat = {p.begin(), x};
  } else {
    pat = {p};
  }

  if (has_comma) {
    auto x = pat.begin();
    while (true) {
      auto y = std::find(x, pat.end(), ',');
      std::string s{x, y};
      int val;
      try {
        val = std::stoi(s);
      } catch (const std::invalid_argument& ie) {
        throw std::invalid_argument("Not a number: " + s);
      }
      if (val < 0) {
        throw std::invalid_argument("Disallowed throw value: " + s);
      }
      throwval.push_back(val);
      maxval = std::max(val, maxval);
      if (y == pat.end())
        break;
      x = y + 1;
    }

    if (hmax > 0) {
      if (hmax < maxval) {
        std::string err = "Supplied `h` value (" + std::to_string(hmax)
            + ") is too small";
        throw std::invalid_argument(err);
      }
      h = hmax;
    } else {
      h = std::max(maxval, 1);
    }
    return;
  }

  int plusses = 0;
  int sum = 0;

  for (char ch : pat) {
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

  if (plusses == 0) {
    if (hmax > 0) {
      if (hmax < maxval) {
        std::string err = "Supplied `h` value (" + std::to_string(hmax)
            + ") is too small";
        throw std::invalid_argument(err);
      }
      h = hmax;
    } else {
      h = std::max(maxval, 1);
    }
    return;
  }

  // pattern is in block form.
  //
  // we need to solve the equation: plusses * h + sum = b * length
  // for minimal (positive integer) values of `h` and `b`, where h >= b.
  //
  // from Bizout's identity this has a solution if and only if `sum` is
  // divisible by gcd(plusses, length).
  //
  // also since h >= b, then `h` must be at least as large as
  // ceiling(sum / (length - plusses))

  int len = static_cast<int>(length());
  if (sum % std::gcd(plusses, len) != 0) {
    throw std::invalid_argument("No solution for `+` value in pattern");
  }

  h = (len > plusses) ? 1 + std::max(maxval, (sum - 1) / (len - plusses)) : 1;

  for (; h <= len + maxval; ++h) {
    if ((plusses * h + sum) % len != 0)
      continue;
    assert(h > maxval);
    assert(h >= (plusses * h + sum) / len);  // h >= b

    // fill in throw values for + placeholders
    for (size_t i = 0; i < throwval.size(); ++i) {
      if (throwval.at(i) < 0) {
        throwval.at(i) = h;
      }
    }

    // validity check here catches cases like ++4+--++4-+-++2-++-+1+-++-3-
    // where the minimal solution h=5,b=3 is not valid due to collisions
    // between link throws and + throws --> correct solution is h=7,b=4
    if (is_valid()) {
      if (hmax > 0 && hmax != h) {
        std::string err = "Solution for `+` value (" + std::to_string(h)
            + ") does not match the supplied `h` value ("
            + std::to_string(hmax) + ")";
        throw std::invalid_argument(err);
      }
      return;
    }

    // didn't work; revert back
    for (size_t i = 0; i < throwval.size(); ++i) {
      if (throwval.at(i) == h) {
        throwval.at(i) = -1;
      }
    }
  }

  throw std::invalid_argument("Collision between link throws in pattern");
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
  for (int val : throwval) {
    sum += val;
  }
  assert(sum % length() == 0);  // always true if no collisions
  return (sum / length());
}

// Return the pattern length (period).

size_t Pattern::length() const {
  return throwval.size();
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

  return true;
}

// Return true if the pattern is prime, false otherwise.

bool Pattern::is_prime() {
  if (!is_valid())
    return false;
  check_have_states();

  std::set<State> s(states.begin(), states.end());
  return (s.size() == states.size());
}

// Return true if the pattern is composite, false otherwise.

bool Pattern::is_composite() {
  if (!is_valid())
    return false;
  return !is_prime();
}

// Return true if the pattern is superprime, false otherwise.

bool Pattern::is_superprime() {
  if (!is_valid())
    return false;
  check_have_states();

  std::vector<State> deduped;
  deduped.push_back(cyclestates.at(0));
  for (size_t i = 1; i < cyclestates.size(); ++i) {
    // check for link throw within a cycle
    if (cyclestates.at(i) == cyclestates.at((i + 1) % length()) &&
        throwval.at(i) != 0 && throwval.at(i) != h) {
      return false;
    }
    // remove runs of the same cyclestate
    if (cyclestates.at(i) != deduped.back()) {
      deduped.push_back(cyclestates.at(i));
    }
  }

  // remove the states at the end that are duplicates of the start
  while (deduped.size() != 0 && deduped.back() == deduped.front()) {
    deduped.pop_back();
  }

  std::set<State> s(deduped.begin(), deduped.end());
  return (deduped.size() > 1 && s.size() == deduped.size());
}

//------------------------------------------------------------------------------
// Pattern transformations
//------------------------------------------------------------------------------

// Return the dual of the pattern.

Pattern Pattern::dual() const {
  if (length() == 0) {
    std::vector<int> empty_throwval;
    return {empty_throwval, h};
  }

  std::vector<int> dual_throwval(length());
  for (size_t i = 0; i < length(); ++i) {
    dual_throwval.at(i) = h - throwval.at(length() - 1 - i);
  }
  return {dual_throwval, h};
}

// Return the inverse of the pattern.
//
// If the inverse does not exist, return an empty pattern.

Pattern Pattern::inverse() {
  check_have_states();

  // Step 1. Determine if an inverse exists.
  //
  // A pattern has an inverse if and only if:
  // - it visits more than one shift cycle on the state graph, and
  // - it never revisits a shift cycle, and
  // - it never does a link throw (0 < t < h) within a single cycle

  std::set<State> states_used;
  std::set<State> cycles_used;
  std::vector<int> empty_throwval;
  State cycle_current = cyclestates.at(0);
  bool cycles_multiple = false;

  for (size_t i = 0; i < throwval.size(); ++i) {
    states_used.insert(states.at(i));
    const State cycle_next = cyclestates.at((i + 1) % length());

    if (cycle_next != cycle_current) {
      // mark a shift cycle as used only when we transition off it
      if (cycles_used.count(cycle_current) != 0) {
        // revisited cycle number `cycle_current` --> no inverse
        return {empty_throwval, h};
      }
      cycles_used.insert(cycle_current);
      cycles_multiple = true;
    } else if (throwval.at(i) != 0 && throwval.at(i) != h) {
      // link throw within a single cycle --> no inverse
      return {empty_throwval, h};
    }

    cycle_current = cycle_next;
  }

  if (!cycles_multiple) {
    // never left starting shift cycle --> no inverse
    return {empty_throwval, h};
  }

  // Step 2. Find the states and throws of the inverse.
  //
  // Iterate through the link throws in the pattern to build up a list of
  // states and throws for the inverse.

  std::vector<int> inverse_throwval;
  std::vector<State> inverse_states;

  for (size_t i = 0; i < throwval.size(); ++i) {
    // continue until `throwval[i]` is a link throw
    if (cyclestates.at(i) == cyclestates.at((i + 1) % length()))
      continue;

    if (inverse_states.size() == 0) {
      // the inverse pattern starts at the (reversed version of) the next state
      // 'downstream' from `patternstate[i]`
      inverse_states.push_back(states.at(i).downstream().reverse());
    }

    const int inverse_throw = h - throwval.at(i);
    inverse_throwval.push_back(inverse_throw);
    inverse_states.push_back(
        inverse_states.back().advance_with_throw(inverse_throw));

    // advance the inverse pattern along the shift cycle until it gets to a
    // state whose reverse is used by the original pattern

    while (true) {
      State trial_state = inverse_states.back().downstream();
      if (states_used.count(trial_state.reverse()) != 0)
        break;

      inverse_states.push_back(trial_state);
      inverse_throwval.push_back(trial_state.slot(h - 1) ? h : 0);
    }
  }
  assert(inverse_states.size() > 0);
  assert(inverse_states.back() == inverse_states.front());

  // Step 3. Create the final inverse pattern.
  //
  // By convention we start all patterns with their smallest state.

  std::vector<int> inverse_final;

  size_t min_index = 0;
  for (size_t i = 1; i < inverse_states.size(); ++i) {
    if (inverse_states.at(i) < inverse_states.at(min_index)) {
      min_index = i;
    }
  }

  const size_t inverse_length = inverse_throwval.size();
  for (size_t i = 0; i < inverse_length; ++i) {
    size_t j = (i + min_index) % inverse_length;
    inverse_final.push_back(inverse_throwval.at(j));
  }

  return {inverse_final, h};
}

//------------------------------------------------------------------------------
// Operator overrides
//------------------------------------------------------------------------------

// Return the throw value on beat `index`.

int Pattern::operator[](size_t index) const {
  assert(index < length());
  return throwval.at(index);
}

bool Pattern::operator==(const Pattern& p2) const {
  return (h == p2.h && throwval == p2.throwval);
}

bool Pattern::operator!=(const Pattern& p2) const {
  return (h != p2.h || throwval != p2.throwval);
}

//------------------------------------------------------------------------------
// Pattern output
//------------------------------------------------------------------------------

// Return the string representation of the pattern.
//
// Parameter `throwdigits` determines the field width to use when printing.
// When equal to 0, each throw is printed as a single character without
// separating commas. When > 0, each throw is printed as an integer using the
// given field width (with padding spaces), including separating commas. If the
// supplied value of `throwdigits` cannot print each throw without truncation,
// it defaults to a value that avoids truncation.
//
// Parameter `plusminus`, if true, causes throws of value 0 and `h` to be output
// as `-` and `+` respectively. This is only active when `throwdigits` == 0.

std::string Pattern::to_string(int throwdigits, bool plusminus) const {
  int maxval = 0;
  for (int val : throwval) {
    maxval = std::max(maxval, val);
  }
  int min_throwdigits = 1;
  for (int temp = 10; temp <= maxval; temp *= 10) {
    ++min_throwdigits;
  }

  if (throwdigits == 0) {
    if (maxval > 35) {  // 'z' = 35
      throwdigits = min_throwdigits;
    }
  } else {
    throwdigits = std::max(throwdigits, min_throwdigits);
  }

  std::ostringstream buffer;
  for (size_t i = 0; i < throwval.size(); ++i) {
    if (throwdigits > 0 && i != 0) {
      buffer << ',';
    }
    print_throw(buffer, throwval.at(i), throwdigits,
        (throwdigits == 0 && plusminus) ? h : 0);
  }
  return buffer.str();
}

// Output a single throw to a string buffer.
//
// For a parameter description see Pattern::to_string().

void Pattern::print_throw(std::ostringstream& buffer, int val, int throwdigits,
    int plusval) {
  if (throwdigits == 0) {
    if (plusval > 0 && val == 0) {
      buffer << '-';
    } else if (plusval > 0 && val == plusval) {
      buffer << '+';
    } else {
      buffer << throw_char(val);
    }
  } else {
    buffer << std::setw(throwdigits) << val;
  }
}

// Return a character for a given integer throw value (0 = '0', 1 = '1',
// 10 = 'a', 11 = 'b', ...

char Pattern::throw_char(int val) {
  if (val < 0 || val > 35) {
    return '?';
  } else if (val < 10) {
    return static_cast<char>(val + '0');
  } else {
    return static_cast<char>(val - 10 + 'a');
  }
}

//------------------------------------------------------------------------------
// Pattern analysis
//------------------------------------------------------------------------------

// Return an analysis of the pattern in string format.

std::string Pattern::make_analysis() {
  std::ostringstream buffer;

  if (!is_valid()) {
    buffer << "Input is not a valid siteswap due to colliding throws:\n  ";

    int collision_start = -1;
    int collision_end = -1;
    std::vector<int> landing_index(length(), -1);
    for (size_t i = 0; i < length(); ++i) {
      size_t index = (i + static_cast<size_t>(throwval.at(i))) % length();
      if (landing_index[index] != -1) {
        collision_start = landing_index[index];
        collision_end = i;
        break;
      }
      landing_index[index] = i;
    }
    assert(collision_start != -1 && collision_end != -1);

    int index = 0;
    std::string patstring = to_string(1);
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

  // basic information

  int maxval = 0;
  for (int val : throwval) {
    maxval = std::max(maxval, val);
  }

  buffer << "Pattern:\n";
  if (h < 36) {
    buffer << "  short form          " << to_string(0) << '\n';
    if (maxval == h) {
      buffer << "  block form          " << to_string(0, true) << '\n';
    }
  }
  buffer << "  standard form       " << to_string(1) << "\n\n";

  check_have_states();
  buffer << "Properties:\n"
         << "  objects             " << objects() << '\n'
         << "  length              " << length() << '\n'
         << "  maximum throw       " << maxval << '\n'
         << "  beats in state      " << h << '\n'
         << "  is_prime            " << std::boolalpha << is_prime() << '\n'
         << "  is_superprime       " << std::boolalpha << is_superprime()
         << "\n\n";

  // graph information

  std::uint64_t full_numstates = Graph::combinations(h, objects());
  std::uint64_t full_numcycles = 0;
  std::uint64_t full_numshortcycles = 0;
  int max_cycle_period = 0;

  for (int p = 1; p <= h; ++p) {
    const std::uint64_t cycles = Graph::shift_cycle_count(objects(), h, p);
    full_numcycles += cycles;
    if (p < h) {
      full_numshortcycles += cycles;
    }
    if (cycles > 0) {
      max_cycle_period = p;
    }
  }

  buffer << "Graph (" << objects() << ',' << h << "):\n"
         << "  states              " << full_numstates << '\n'
         << "  shift cycles        " << full_numcycles << '\n'
         << "  short cycles        " << full_numshortcycles << '\n'
         << "  prime length bound  "
         << std::max(static_cast<std::uint64_t>(max_cycle_period),
              full_numstates - full_numcycles)
         << "\n\n";

  // table of states, shift cycles, and excluded states

  bool show_sc = is_prime();
  int separation = std::max(15, 4 + h + 4);

  int throwdigits = 1;
  for (int temp = 10; temp <= h; temp *= 10) {
    ++throwdigits;
  }

  if (show_sc) {
    buffer << "States:";
    for (int j = 0; j < (h + throwdigits + 1); ++j) {
      buffer << ' ';
    }
    buffer << "Shift cycles:";
    for (int j = 0; j < (separation - 13); ++j) {
      buffer << ' ';
    }
    buffer << "Excluded states:\n";
  } else {
    buffer << "States:\n";
  }

  std::vector<State> shiftcycles_visited;
  bool any_linkthrow = false;
  for (size_t i = 0; i < length(); ++i) {
    if (throwval.at(i) != 0 && throwval.at(i) != h) {
      shiftcycles_visited.push_back(cyclestates.at((i + 1) % length()));
    }
    if (throwval.at(i) != 0 && throwval.at(i) != h) {
      any_linkthrow = true;
    }
  }
  std::set<State> printed;

  for (size_t i = 0; i < length(); ++i) {
    // state and throw value out of it
    if (std::count(states.begin(), states.end(), states.at(i)) == 1) {
      buffer << "  ";
    } else {
      buffer << "R ";
    }
    buffer << states.at(i) << "  "
           << std::setw(throwdigits) << throwval.at(i);

    if (!show_sc) {
      buffer << '\n';
      continue;
    }

    // shift cycle visited
    int prev_throwvalue = throwval.at(i == 0 ? length() - 1 : i - 1);
    int curr_throwvalue = throwval.at(i);
    bool prev_linkthrow = (prev_throwvalue != 0 && prev_throwvalue != h);
    bool curr_linkthrow = (curr_throwvalue != 0 && curr_throwvalue != h);

    if (prev_linkthrow) {
      if (std::count(shiftcycles_visited.begin(), shiftcycles_visited.end(),
          cyclestates.at(i)) == 1) {
        buffer << "      ";
      } else {
        buffer << "    R ";
      }
      buffer << '(' << cyclestates.at(i) << ") ";
    } else if (!any_linkthrow && i == 0) {
      buffer << "      (" << cyclestates.at(i) << ") ";
    } else {
      buffer << "         .";
      for (int j = 0; j < (h - 1); ++j) {
        buffer << ' ';
      }
    }
    for (int j = 0; j < (separation - (h + 3)); ++j) {
      buffer << ' ';
    }

    // excluded states, if any
    bool es_printed = false;
    if (prev_linkthrow) {
      State excluded = states.at(i).upstream();
      if (printed.count(excluded) == 0) {
        buffer << excluded;
        printed.insert(excluded);
        es_printed = true;
      }
    }
    if (curr_linkthrow) {
      State excluded = states.at(i).downstream();
      if (printed.count(excluded) == 0) {
        if (es_printed) {
          buffer << ", ";
        }
        buffer << excluded;
        printed.insert(excluded);
      }
    }

    buffer << '\n';
  }

  // inverse pattern, if one exists

  Pattern inverse_pattern = inverse();
  assert((inverse_pattern.length() != 0) == is_superprime());
  assert(is_superprime() == inverse_pattern.is_superprime());
  if (inverse_pattern.length() != 0) {
    buffer << "\nInverse pattern:\n  " << inverse_pattern.to_string(0, true)
           << " /" << h << '\n';
  }

  buffer << "------------------------------------------------------------";
  return buffer.str();
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
  State start_state{static_cast<unsigned>(h)};
  for (size_t i = 0; i < length(); ++i) {
    int fillslot = throwval.at(i) - static_cast<int>(length()) +
        static_cast<int>(i);
    while (fillslot >= 0) {
      if (fillslot < static_cast<int>(h)) {
        assert(start_state.slot(fillslot) == 0);
        start_state.slot(fillslot) = 1;
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
      if (s2 < cyclestate) {
        cyclestate = s2;
      }
      s2 = s2.downstream();
    }
    cyclestates.push_back(cyclestate);
  }

  assert(states.size() == length());
  assert(cyclestates.size() == length());
}

// Print the pattern to an output stream using the default output format.

std::ostream& operator<<(std::ostream& ost, const Pattern& p) {
  ost << p.to_string();
  return ost;
}
