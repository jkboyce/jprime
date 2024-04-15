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
#include <stdexcept>


// Initialize SearchConfig from command line arguments.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

void SearchConfig::from_args(size_t argc, char** argv) {
  n = atoi(argv[1]);
  if (n < 1) {
    throw std::invalid_argument("Must have at least 1 object");
  }
  h = atoi(argv[2]);
  if (h < n) {
    throw std::invalid_argument(
        "Max. throw value must equal or exceed the number of objects");
  }

  // defaults
  l_min = 1;
  l_max = 0;
  xarray.assign(h + 1, false);

  // default for using dual graph
  if (h > (2 * n)) {
    dualflag = true;
    n = h - n;
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
        shiftlimit = atoi(argv[i]);
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
        steal_alg = atoi(argv[i]);
      } else {
        throw std::invalid_argument("No number provided after -steal_alg");
      }
    } else if (!strcmp(argv[i], "-split_alg")) {
      if ((i + 1) < argc) {
        ++i;
        split_alg = atoi(argv[i]);
      } else {
        throw std::invalid_argument("No number provided after -split_alg");
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        unsigned int j = static_cast<unsigned int>(atoi(argv[i]));
        if (j == h) {
          throw std::invalid_argument(
              "Cannot exclude max. throw value using -x");
        }
        if (j >= 0 && j < h) {
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
        num_threads = static_cast<unsigned int>(atoi(argv[i]));
      } else {
        throw std::invalid_argument("No number provided after -threads");
      }
    } else if (i == 3) {
      // try to parse argument as a length or length range
      bool success = false;
      std::string s{argv[i]};
      int hyphens = std::count(s.begin(), s.end(), '-');
      if (hyphens == 0) {
        int num = atoi(argv[i]);
        if (num > 0) {
          l_min = l_max = static_cast<unsigned int>(num);
          success = true;
        }
      } else if (hyphens == 1) {
        success = true;
        auto hpos = s.find('-');
        std::string s1 = s.substr(0, hpos);
        std::string s2 = s.substr(hpos + 1);
        if (s1.size() > 0) {
          int num = atoi(s1.c_str());
          if (num > 0)
            l_min = static_cast<unsigned int>(num);
          else
            success = false;
        }
        if (s2.size() > 0) {
          int num = atoi(s2.c_str());
          if (num > 0)
            l_max = static_cast<unsigned int>(num);
          else
            success = false;
        }
      }

      if (!success) {
        std::string msg("Error parsing length: ");
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
  unsigned int max_throw_value = h;
  if (l_max > 0) {
    if (dualflag) {
      max_throw_value = std::min(max_throw_value, (h - n) * l_max);
    } else {
      max_throw_value = std::min(max_throw_value, n * l_max);
    }
  }
  if (max_throw_value < 36) {
    throwdigits = 1;  // 'z' = 35
  } else {
    noplusminusflag = true;
    throwdigits = 2;
    for (unsigned int temp = 100; temp <= max_throw_value; temp *= 10) {
      ++throwdigits;
    }
  }

  // for efficiency, default to -count mode when -noprint is active and no
  // output file
  if (!printflag && !fileoutputflag) {
    countflag = true;
  }

  // consistency checks
  if (num_threads < 1) {
    throw std::invalid_argument("Must have at least one worker thread");
  }
}

// Initialize SearchConfig from concatenated command line arguments.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

void SearchConfig::from_args(std::string str) {
  // tokenize the argslist string
  std::stringstream ss(str);
  std::string s;
  std::vector<std::string> args;
  while (std::getline(ss, s, ' '))
    args.push_back(s);

  const size_t argc = args.size();
  char** argv = new char*[argc];
  for (size_t i = 0; i < argc; ++i)
    argv[i] = &(args[i][0]);

  from_args(argc, argv);

  delete[] argv;
}
