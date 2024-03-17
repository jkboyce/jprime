//
// SearchConfig.h
//
// Defines items that are constant during the duration of the search
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHCONFIG_H_
#define JPRIME_SEARCHCONFIG_H_

#include <string>
#include <vector>


enum class RunMode {
  NORMAL_SEARCH,
  SUPER_SEARCH,
};

enum class GroundMode {
  GROUND_SEARCH,
  EXCITED_SEARCH,
  ALL_SEARCH,
};

enum class GraphMode {
  FULL_GRAPH,
  SINGLE_PERIOD_GRAPH,
};

struct SearchConfig {
  // number of objects
  int n = 0;

  // maximum throw value
  int h = 0;

  // minimum pattern length to find
  int l_min = 1;

  // maximum pattern length to find
  int l_max = -1;

  // worker mode
  RunMode mode = RunMode::NORMAL_SEARCH;

  // ground state, excited state, or all
  GroundMode groundmode = GroundMode::ALL_SEARCH;

  // type of graph to build
  GraphMode graphmode = GraphMode::FULL_GRAPH;

  // print patterns to console
  bool printflag = true;

  // print inverses in super mode
  bool invertflag = false;

  // find patterns in dual graph
  bool dualflag = false;

  // print worker diagnostic information
  bool verboseflag = false;

  // print live search status
  bool statusflag = false;

  // print info about search, but do not execute
  bool infoflag = false;

  // if 1 then print as letter (a=10, b=11, ...), if >1 then print as an integer
  // with the given field width
  unsigned int throwdigits = 1;

  // print without using +, - for h and 0 (when throwdigits = 1)
  bool noplusminusflag = false;

  // keep a record of patterns seen at each length
  bool countflag = false;

  // for super mode
  unsigned int shiftlimit = 0;

  // throw values to exclude from search
  std::vector<bool> xarray;
};

#endif
