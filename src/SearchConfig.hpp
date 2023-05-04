
#ifndef JDEEP_SEARCHCONFIG_H
#define JDEEP_SEARCHCONFIG_H

#include <vector>

// Defines items that are constant during the duration of the search

#define NORMAL_MODE     1
#define BLOCK_MODE      2
#define SUPER_MODE      3

struct SearchConfig {
  // number of objects
  int n = 0;
  // maximum throw value
  int h = 0;
  // (min) pattern length to find
  int l = 0;

  // search mode
  int mode = NORMAL_MODE;
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
  // for block mode
  int skiplimit = 0;
  // for super mode
  int shiftlimit = 0;
  // throw values to exclude from search
  std::vector<bool> xarray;
};

#endif
