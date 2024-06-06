//
// SearchConfig.h
//
// This structure defines the calculation requested by the user, as specified
// by command line arguments.
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
  unsigned int b = 0;

  // maximum throw value
  unsigned int h = 0;

  // minimum pattern length to find
  unsigned int l_min = 1;

  // maximum pattern length to find; 0 means open-ended range like "5-"
  unsigned int l_max = 0;

  // search type
  RunMode mode = RunMode::NORMAL_SEARCH;

  // ground state, excited state, or all
  GroundMode groundmode = GroundMode::ALL_SEARCH;

  // type of graph to build
  GraphMode graphmode = GraphMode::FULL_GRAPH;

  // print patterns to console?
  bool printflag = true;

  // also find pattern inverses, to print/save with patterns?
  bool invertflag = false;

  // find patterns in dual graph?
  bool dualflag = false;

  // print worker diagnostic information?
  bool verboseflag = false;

  // print live search status?
  bool statusflag = false;

  // print info about search, but do not execute?
  bool infoflag = false;

  // print without using +, - for h and 0 (when throwdigits = 1)?
  bool noplusminusflag = false;

  // keep a record of patterns seen at each length?
  bool countflag = false;

  // use a file to save, resume after interruption, and record the final
  // results?
  bool fileoutputflag = false;

  // filename to use when `fileoutputflag`==true
  std::string outfile;

  // number of worker threads to use
  unsigned int num_threads = 1;

  // for super mode, number of shift throws to allow
  unsigned int shiftlimit = 0;

  // throw values to exclude from search
  std::vector<bool> xarray;

  // if 1 then print as letter (a=10, b=11, ...), if >1 then print as an integer
  // with the given field width
  unsigned int throwdigits = 1;

  // work stealing algorithm to use
  unsigned int steal_alg = 1;

  // work splitting algorithm to use
  unsigned int split_alg = 1;

  // methods to initialize from command line arguments
  void from_args(size_t argc, char** argv);
  void from_args(const std::string& str);
};

#endif
