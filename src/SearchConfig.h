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
  BLOCK_SEARCH,
  SUPER_SEARCH,
};


struct SearchConfig {
  // number of objects
  int n = 0;

  // maximum throw value
  int h = 0;

  // (min) pattern length to find
  int l = 0;

  // worker mode
  RunMode mode = RunMode::NORMAL_SEARCH;

  // ground state, excited state, or both
  int groundmode = 0;

  // print patterns to console
  bool printflag = true;

  // print inverses in super mode
  bool invertflag = false;

  // search for the longest pattern(s)
  bool longestflag = true;

  // search for an exact pattern length
  bool exactflag = false;

  // find patterns in dual graph
  bool dualflag = false;

  // print worker diagnostic information
  bool verboseflag = false;

  // print live search status
  bool statusflag = false;

  // print without using +, - for h and 0
  bool noplusminusflag = false;

  // keep a record of patterns seen at each length
  bool countsflag = false;

  // for block mode
  int skiplimit = 0;

  // for super mode
  int shiftlimit = 0;

  // throw values to exclude from search
  std::vector<bool> xarray;
};

#endif
