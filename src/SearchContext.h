//
// SearchContext.h
//
// This structure captures the progression and results of the search. These
// items are saved to disk in file output mode, to record results and/or allow
// calculations to be interrupted and resumed.
//
// Only the Coordinator thread has access to this data structure; the Workers
// report all results and changes to the coordinator via messages.
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
  // whether to use a file to save, resume after interruption, and record the
  // final results
  bool fileoutputflag = false;

  // filename to use when `fileoutputflag`==true
  std::string outfile;

  // original invocation command line arguments, concatenated
  std::string arglist;

  // precalculated quantities for the full (n,h) graph
  std::uint64_t full_numstates = 0;
  std::uint64_t full_numcycles = 0;
  std::uint64_t full_numshortcycles = 0;

  // precalculated maximum length possible for a pattern of the type we're
  // searching for, in the full graph (does not change)
  std::uint64_t l_bound = 0;

  // precalculated quantity for the actual graph in memory
  std::uint64_t memory_numstates = 0;

  // number of patterns found in the range [l_min, l_max]
  std::uint64_t npatterns = 0;

  // total number of patterns seen, of any length
  std::uint64_t ntotal = 0;

  // total number of nodes visited in the search tree
  std::uint64_t nnodes = 0;

  // number of worker threads to use
  unsigned int num_threads = 1;

  // wall clock time elapsed
  double secs_elapsed = 0;

  // sum of working (not idle) time for all workers
  double secs_working = 0;

  // sum of available working time for all workers (busy or idle)
  double secs_available = 0;

  // record of patterns found with length in target range
  std::vector<std::string> patterns;

  // number of patterns found at all lengths
  std::vector<std::uint64_t> count;

  // work assignments remaining not assigned to a worker
  std::list<WorkAssignment> assignments;
};

#endif
