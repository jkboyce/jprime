//
// Pattern.cc
//
// Class representing a single siteswap pattern.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Pattern.h"
#include "Graph.h"

#include <iostream>
#include <string>


Pattern::Pattern(const std::string& pat)
    : pattern(pat),
      n(find_pattern_n(pat)),
      h(find_pattern_h(pat)),
      l(pat.length()) {
}

int Pattern::find_pattern_n(const std::string& pat) {
  int n_, h_;
  if (!find_pattern_nh(pat, n_, h_))
    std::exit(EXIT_FAILURE);
  return n_;
}

int Pattern::find_pattern_h(const std::string& pat) {
  int n_, h_;
  if (!find_pattern_nh(pat, n_, h_))
    std::exit(EXIT_FAILURE);
  return h_;
}

bool Pattern::find_pattern_nh(const std::string& pat, int& n, int& h) {
  const int l = pat.length();
  int sum = 0;
  int pluscount = 0;
  int maxvalue = 0;

  for (int i = 0; i < l; ++i) {
    const char ch = pat[i];
    int value = 0;

    if (ch >= '0' && ch <= '9') {
      value = static_cast<int>(ch - '0');
    } else if (ch >= 'a' && ch <= 'z') {
      value = static_cast<int>(ch - 'a') + 10;
    } else if (ch >= 'A' && ch <= 'Z') {
      value = static_cast<int>(ch - 'A') + 10;
    } else if (ch == '-') {
      value = 0;
    } else if (ch == '+') {
      value = 0;
      ++pluscount;
    } else {
      std::cerr << "Error: Unrecognized character '" << ch << "' in pattern"
                << std::endl;
      return false;
    }

    sum += value;
    maxvalue = std::max(maxvalue, value);
  }

  if (pluscount == 0) {
    if (sum % l != 0) {
      std::cerr << "Error: Pattern is not valid (sum=" << sum
                << " is not divisible by length=" << l << ")" << std::endl;
      return false;
    }
    n = sum / l;
    h = maxvalue;
    return true;
  }

  // Find the smallest `h` > `maxvalue` that satisfies the equation
  //   (pluscount * h + sum) % l == 0
  //
  // e.g.,  +++++--  : pluscount = 5, sum = 0, l = 7 --> h = 7

  h = maxvalue + 1;
  while (true) {
    if ((pluscount * h + sum) % l == 0) {
      n = (pluscount * h + sum) / l;
      return true;
    }
    ++h;
    assert(h < 36);  // 'z' = 35
  }
}

void Pattern::print_analysis() {
  // Graph graph(n, h, )
  std::cout << "n = " << n << ", h = " << h << std::endl;
  std::cout << "analyze pattern here" << std::endl;
}
