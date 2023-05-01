
#ifndef JDEEP_SEARCHCONTEXT_H
#define JDEEP_SEARCHCONTEXT_H

#include "WorkAssignment.hpp"

#include <list>
#include <vector>
#include <string>

struct SearchContext {
  std::string arglist;
  int l_current = 0;
  int maxlength = 0;
  int numstates = 0;
  unsigned long npatterns = 0L;
  unsigned long ntotal = 0L;
  unsigned long nnodes = 0L;
  double secs_elapsed = 0;
  std::vector<std::string> patterns;
  std::list<WorkAssignment> assignments;

  int num_threads = 1;
  bool fileoutputflag = false;
  std::string outfile;
};

#endif
