//
// SearchState.h
//
// State variables specific to a particular time step (position) in the pattern,
// used for the iterative search process.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_SEARCHSTATE_H_
#define JPRIME_SEARCHSTATE_H_


struct SearchState {
  unsigned int col = 0;
  unsigned int col_limit = 0;
  unsigned int from_state = 0;
  unsigned int to_state = 0;
  unsigned int* outmatrix = nullptr;
  unsigned int* excludes_throw = nullptr;
  unsigned int* deadstates_throw = nullptr;
  unsigned int* excludes_catch = nullptr;
  unsigned int* deadstates_catch = nullptr;
  int to_cycle = -1;
  unsigned int shifts_remaining = 0;
  unsigned int exitcycles_remaining = 0;
};

#endif
