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
  int col = 0;
  int col_limit = 0;
  int from_state = 0;
  int to_state = 0;
  int* outmatrix = nullptr;
  int* excludes_throw = nullptr;
  int* deadstates_throw = nullptr;
  int* excludes_catch = nullptr;
  int* deadstates_catch = nullptr;
  // int to_cycle = 0;
  // int shifts_remaining = 0;
};

#endif
