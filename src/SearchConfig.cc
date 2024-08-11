//
// SearchConfig.cc
//
// Methods for intializing a SearchConfig structure from command line arguments.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "SearchConfig.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cstring>


// Initialize SearchConfig from command line arguments.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

void SearchConfig::from_args(size_t argc, char** argv) {
  int val;
  try {
    val = std::stoi(argv[1]);
  } catch (const std::invalid_argument& ie) {
    std::string msg("Error parsing number of objects: ");
    msg.append(argv[1]);
    throw std::invalid_argument(msg);
  }
  if (val < 1) {
    throw std::invalid_argument("Must have at least 1 object");
  }
  b = static_cast<unsigned>(val);

  try {
    val = std::stoi(argv[2]);
  } catch (const std::invalid_argument& ie) {
    std::string msg("Error parsing max. throw value: ");
    msg.append(argv[2]);
    throw std::invalid_argument(msg);
  }
  if (val < 1) {
    throw std::invalid_argument("Max. throw value must be at least 1");
  }
  h = static_cast<unsigned>(val);
  if (h < b) {
    throw std::invalid_argument(
        "Max. throw value must equal or exceed the number of objects");
  }

  // defaults
  l_min = 1;
  l_max = 0;
  xarray.assign(h + 1, false);

  // default for using dual graph
  if (h > (2 * b)) {
    dualflag = true;
    b = h - b;
  }

  for (size_t i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-noprint")) {
      printflag = false;
    } else if (!strcmp(argv[i], "-inverse")) {
      invertflag = true;
    } else if (!strcmp(argv[i], "-g")) {
      groundmode = GroundMode::GROUND_SEARCH;
    } else if (!strcmp(argv[i], "-ng")) {
      groundmode = GroundMode::EXCITED_SEARCH;
    } else if (!strcmp(argv[i], "-noplus")) {
      noplusminusflag = true;
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        ++i;
        mode = RunMode::SUPER_SEARCH;
        try {
          val = std::stoi(argv[i]);
        } catch (const std::invalid_argument& ie) {
          std::string msg("Error parsing shift limit: ");
          msg.append(argv[i]);
          throw std::invalid_argument(msg);
        }
        if (val < 0) {
          throw std::invalid_argument("Shift limit must be non-negative");
        }
        shiftlimit = static_cast<unsigned>(val);
        if (shiftlimit == 0) {
          xarray.at(0) = xarray.at(h) = true;
        }
      } else {
        throw std::invalid_argument("No shift limit provided after -super");
      }
    } else if (!strcmp(argv[i], "-file")) {
      if ((i + 1) < argc) {
        ++i;
        fileoutputflag = true;
        outfile = std::string(argv[i]);
      } else {
        throw std::invalid_argument("No filename provided after -file");
      }
    } else if (!strcmp(argv[i], "-steal_alg")) {
      if ((i + 1) < argc) {
        ++i;
        val = std::stoi(argv[i]);
        if (val < 1 || val > 1) {
          throw std::invalid_argument("Steal_alg out of range");
        }
        steal_alg = static_cast<unsigned>(val);
      } else {
        throw std::invalid_argument("No number provided after -steal_alg");
      }
    } else if (!strcmp(argv[i], "-split_alg")) {
      if ((i + 1) < argc) {
        ++i;
        val = std::stoi(argv[i]);
        if (val < 1 || val > 2) {
          throw std::invalid_argument("Split_alg out of range");
        }
        split_alg = static_cast<unsigned>(val);
      } else {
        throw std::invalid_argument("No number provided after -split_alg");
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        try {
          val = std::stoi(argv[i]);
        } catch (const std::invalid_argument& ie) {
          std::string msg("Error parsing excluded throw value: ");
          msg.append(argv[i]);
          throw std::invalid_argument(msg);
        }
        if (val < 0) {
          throw std::invalid_argument("Excluded throws must be non-negative");
        }
        unsigned j = static_cast<unsigned>(val);
        if (j == h) {
          throw std::invalid_argument(
              "Cannot exclude max. throw value using -x");
        }
        if (j < h) {
          xarray.at(dualflag ? (h - j) : j) = true;
        }
        ++i;
      }
      --i;
    } else if (!strcmp(argv[i], "-verbose")) {
      verboseflag = true;
    } else if (!strcmp(argv[i], "-count")) {
      countflag = true;
    } else if (!strcmp(argv[i], "-info")) {
      infoflag = true;
    } else if (!strcmp(argv[i], "-status")) {
      statusflag = true;
    } else if (!strcmp(argv[i], "-threads")) {
      if ((i + 1) < argc) {
        ++i;
        try {
          val = std::stoi(argv[i]);
        } catch (const std::invalid_argument& ie) {
          std::string msg("Error parsing number of threads: ");
          msg.append(argv[i]);
          throw std::invalid_argument(msg);
        }
        if (val < 1) {
          throw std::invalid_argument("Must have at least one worker thread");
        }
        num_threads = static_cast<unsigned>(val);
      } else {
        throw std::invalid_argument("No number provided after -threads");
      }
    } else if (i == 3) {
      // try to parse argument as a length or length range
      bool success = false;
      std::string s{argv[i]};
      int hyphens = static_cast<int>(std::count(s.cbegin(), s.cend(), '-'));
      if (hyphens == 0) {
        try {
          val = std::stoi(argv[i]);
        } catch (const std::invalid_argument& ie) {
          std::string msg("Error parsing pattern length: ");
          msg.append(argv[i]);
          throw std::invalid_argument(msg);
        }
        if (val > 0) {
          l_min = l_max = static_cast<unsigned>(val);
          success = true;
        }
      } else if (hyphens == 1) {
        success = true;
        auto hpos = s.find('-');
        std::string s1 = s.substr(0, hpos);
        std::string s2 = s.substr(hpos + 1);
        if (s1.size() > 0) {
          try {
            val = std::stoi(s1);
          } catch (const std::invalid_argument& ie) {
            std::string msg("Error parsing pattern length: ");
            msg.append(s1);
            throw std::invalid_argument(msg);
          }
          if (val > 0)
            l_min = static_cast<unsigned>(val);
          else
            success = false;
        }
        if (s2.size() > 0) {
          try {
            val = std::stoi(s2);
          } catch (const std::invalid_argument& ie) {
            std::string msg("Error parsing pattern length: ");
            msg.append(s2);
            throw std::invalid_argument(msg);
          }
          if (val > 0)
            l_max = static_cast<unsigned>(val);
          else
            success = false;
        }
      }

      if (!success) {
        std::string msg("Error parsing pattern length: ");
        msg.append(argv[i]);
        throw std::invalid_argument(msg);
      }
    } else if (i > 3) {
      std::string msg("Unrecognized input: ");
      msg.append(argv[i]);
      throw std::invalid_argument(msg);
    }
  }

  // graph type
  if (l_min == l_max && l_min < h) {
    graphmode = GraphMode::SINGLE_PERIOD_GRAPH;
  } else {
    graphmode = GraphMode::FULL_GRAPH;
  }

  // output throws as letters (a, b, ...) or numbers (10, 11, ...)?
  unsigned max_throw_value = h;
  if (l_max > 0) {
    if (dualflag) {
      max_throw_value = std::min(max_throw_value, (h - b) * l_max);
    } else {
      max_throw_value = std::min(max_throw_value, b * l_max);
    }
  }
  if (max_throw_value < 36) {
    throwdigits = 0;  // 'z' = 35
  } else {
    noplusminusflag = true;
    throwdigits = 1;
    for (unsigned temp = 10; temp <= max_throw_value; temp *= 10) {
      ++throwdigits;
    }
  }

  // for efficiency, default to -count mode when -noprint is active and no
  // output file
  if (!printflag && !fileoutputflag) {
    countflag = true;
  }
}

// Initialize SearchConfig from concatenated command line arguments.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

void SearchConfig::from_args(const std::string& str) {
  // tokenize the argslist string
  std::stringstream ss(str);
  std::string s;
  std::vector<std::string> args;
  while (std::getline(ss, s, ' ')) {
    args.push_back(s);
  }

  const size_t argc = args.size();
  std::vector<char*> argv;
  for (size_t i = 0; i < argc; ++i) {
    argv.push_back(&(args[i][0]));
  }

  from_args(argc, argv.data());
}
