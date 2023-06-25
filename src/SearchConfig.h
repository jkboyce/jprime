//
// SearchConfig.h
//
// Defines items that are constant during the duration of the search
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHCONFIG_H_
#define JPRIME_SEARCHCONFIG_H_

#include <vector>

enum class SearchMode {
  NORMAL_MODE,
  BLOCK_MODE,
  SUPER_MODE
};


struct SearchConfig {
  // number of objects
  int n = 0;

  // maximum throw value
  int h = 0;

  // (min) pattern length to find
  int l = 0;

  // search mode
  SearchMode mode = SearchMode::NORMAL_MODE;

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

  // print search metadata
  bool verboseflag = false;

  // print without using +, - for h and 0
  bool noplusminusflag = false;

  // for block mode
  int skiplimit = 0;

  // for super mode
  int shiftlimit = 0;

  // throw values to exclude from search
  std::vector<bool> xarray;
};

#endif
