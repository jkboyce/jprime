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
        throw std::invalid_argument("Error parsing pattern: " + s);
      }
      if (val < 0) {
        throw std::invalid_argument("Error disallowed throw value: " + s);
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
            std::string("Error parsing pattern: ") + ch);
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

  int sum = 0;
  for (int val : throwval)
    sum += val;
  if (sum % length() != 0)
    return false;

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

//------------------------------------------------------------------------------
// Pattern analysis
//------------------------------------------------------------------------------

std::string Pattern::do_analysis() const {
  std::string result;

  result += "pattern: " + to_string();

  return result;
}
