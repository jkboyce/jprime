//
// SearchContext.h
//
// Defines items that change during the search, some of which are saved to disk
// if the calculation is interrupted so that it may be resumed.
//
// Only the Coordinator thread has access to this data structure; the Workers
// report all changes to the coordinator via messages.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHCONTEXT_H_
#define JPRIME_SEARCHCONTEXT_H_

#include "WorkAssignment.h"

#include <string>
#include <vector>
#include <list>
#include <cstdint>


struct SearchContext {
  // original invocation command line arguments, concatenated
  std::string arglist;

  // precalculated quantities for the full (n,h) graph
  std::uint64_t full_numstates = 0;
  std::uint64_t full_numcycles = 0;
  std::uint64_t full_numshortcycles = 0;

  // precalculated maximum length possible for a pattern of the type we're
  // searching for, in the full graph (does not change)
  unsigned int l_bound = 0;

  // number of states in constructed juggling graph (does not change)
  unsigned int numstates = 0;

  // number of shift cycles in constructed juggling graph
  unsigned int numcycles = 0;

  // number of short (period < h) shift cycles in constructed juggling graph
  unsigned int numshortcycles = 0;

  // number of patterns found in the range [l_min, l_max]
  std::uint64_t npatterns = 0;

  // total number of patterns seen, of any length
  std::uint64_t ntotal = 0;

  // number of patterns seen at each length
  std::vector<std::uint64_t> count;

  // total number of nodes visited in the search tree
  std::uint64_t nnodes = 0;

  // wall clock time elapsed
  double secs_elapsed = 0;

  // sum of working (not idle) time for all workers
  double secs_working = 0;

  // sum of available working time for all workers (busy or idle)
  double secs_available = 0;

  // record of patterns found, or if `longestflag`==true then the longest ones
  std::vector<std::string> patterns;

  // work assignments remaining not assigned to a worker
  std::list<WorkAssignment> assignments;

  // number of worker threads to use
  unsigned int num_threads = 1;

  // whether to use a file to save, resume after interruption, and record the
  // final results
  bool fileoutputflag = false;

  // filename to use when `fileoutputflag`==true
  std::string outfile;

  // work stealing algorithm to use
  unsigned int steal_alg = 1;

  // work splitting algorithm to use
  unsigned int split_alg = 1;
};

#endif
