//
// WorkCell.h
//
// State variables specific to a particular time step (position) in the pattern,
// used for the iterative search algorithms.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_WORKCELL_H_
#define JPRIME_WORKCELL_H_


struct WorkCell {
  // initialized by iterative_init_workspace()
  unsigned col = 0;
  unsigned col_limit = 0;
  unsigned from_state = 0;

  // initialized by replay versions of algorithms
  unsigned* excludes_throw = nullptr;
  unsigned* excludes_catch = nullptr;
  unsigned shiftcount = 0;
  unsigned exitcycles_remaining = 0;
};

#endif
