//
// SearchContext.h
//
// Defines items that change during the search, some of which are saved to disk
// if the calculation is interrupted so that it may be resumed.
//
// Only the Coordinator thread has access to this data structure; the Workers
// report all changes to the coordinator via messages.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHCONTEXT_H_
#define JPRIME_SEARCHCONTEXT_H_

#include "WorkAssignment.h"

#include <string>
#include <vector>
#include <list>


struct SearchContext {
  // original invocation command line arguments, concatenated
  std::string arglist;

  // longest pattern reported by any worker so far
  int l_current = 0;

  // number of states in the juggling graph (does not change)
  int numstates = 0;

  // number of shift cycles in the juggling graph
  int numcycles = 0;

  // number of short (length < h) shift cycles
  int numshortcycles = 0;

  // maximum length possible for a prime pattern of the type we're searching
  // (does not change)
  int maxlength = 0;

  // number of patterns found, either in total or (if `longestflag`==true) at
  // the current value of `l_current`
  unsigned long npatterns = 0L;

  // total number of valid patterns seen
  unsigned long ntotal = 0L;

  // total number of nodes visited in the search tree
  unsigned long nnodes = 0L;

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
  int num_threads = 1;

  // whether to use a file to save, resume after interruption, and record the
  // final results
  bool fileoutputflag = false;

  // filename to use when `fileoutputflag`==true
  std::string outfile;

  // work stealing algorithm to use
  int steal_alg = 1;

  // work splitting algorithm to use
  int split_alg = 1;
};

#endif
