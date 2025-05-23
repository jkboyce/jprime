//
// SearchContext.h
//
// This structure captures the progression and results of the search. In file
// output mode these items are saved to disk, recording results and/or allowing
// calculations to be interrupted and resumed.
//
// Only the coordinator has access to this data structure; the workers report
// all results and changes to the coordinator via messages.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
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

  // precalculated quantities for the full (b,h) graph
  std::uint64_t full_numstates = 0;
  std::uint64_t full_numcycles = 0;
  std::uint64_t full_numshortcycles = 0;

  // precalculated maximum period possible for a pattern of the type we're
  // searching for, in the full graph (does not change)
  std::uint64_t n_bound = 0;

  // precalculated quantity for the actual graph in memory
  std::uint64_t memory_numstates = 0;

  // number of patterns found in the range [n_min, n_max]
  std::uint64_t npatterns = 0;

  // total number of patterns seen, of any period
  std::uint64_t ntotal = 0;

  // total number of nodes completed in the search tree
  std::uint64_t nnodes = 0;

  // wall clock time elapsed
  double secs_elapsed = 0;

  // sum of working (not idle) time for all workers (core-seconds)
  double secs_working = 0;

  // sum of available working time for all workers (busy or idle)
  double secs_available = 0;

  // patterns found with period in the range [n_min, n_max]
  std::vector<std::string> patterns;

  // count of patterns found at each period, for all periods
  std::vector<std::uint64_t> count;

  // work assignments remaining not assigned to a worker
  std::list<WorkAssignment> assignments;

  // number of times work assignments have been split
  unsigned splits_total = 0;

  // methods to save and load on disk
  void to_file(const std::string& file);
  void from_file(const std::string& file);
};

#endif
