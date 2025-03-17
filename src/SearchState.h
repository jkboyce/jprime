//
// SearchState.h
//
// State variables specific to a particular time step (position) in the pattern,
// used for the iterative search algorithms.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHSTATE_H_
#define JPRIME_SEARCHSTATE_H_


struct SearchState {
  unsigned col = 0;
  unsigned col_limit = 0;
  unsigned from_state = 0;
  unsigned to_state = 0;
  //unsigned* outmatrix = nullptr;
  unsigned* excludes_throw = nullptr;
  unsigned* deadstates_throw = nullptr;
  unsigned* excludes_catch = nullptr;
  unsigned* deadstates_catch = nullptr;
  int to_cycle = -1;
  unsigned shiftcount = 0;
  unsigned exitcycles_remaining = 0;
};

#endif
