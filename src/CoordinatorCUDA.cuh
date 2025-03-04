//
// CoordinatorCUDA.cuh
//
// Defines data types and helper functions for executing the search on a CUDA-
// enabled GPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATORCUDA_CUH_
#define JPRIME_COORDINATORCUDA_CUH_

#include "Graph.h"
#include "SearchConfig.h"
#include "SearchContext.h"
#include "WorkAssignment.h"

#include <cuda_runtime.h>


//------------------------------------------------------------------------------
// Data types
//------------------------------------------------------------------------------

using statenum_t = uint16_t;


struct WorkerInfo {  // 16 bytes
  statenum_t start_state = 0;  // current value of `start_state` (input/output)
  statenum_t end_state = 0;  // highest value of `start_state` (input)
  uint16_t pos = 0;  // position in WorkAssignmentCell array (input/output)
  uint64_t nnodes = 0;  // number of nodes completed (output)
  uint16_t done = 1;  // 1 if worker is done, 0 otherwise (output)
};


struct WorkAssignmentCell {  // 8 bytes
  uint8_t col = 0;
  uint8_t col_limit = 0;
  statenum_t from_state = 0;
  uint32_t count = 0;  // output
};


struct WorkAssignmentLine {
  unsigned id;
  WorkAssignment wa;
};


struct ThreadStorageUsed {  // 128 bytes
  uint32_t used;
  uint32_t unused[31];
};


struct ThreadStorageWorkCell {  // 256 bytes
  uint8_t col;
  uint8_t col_limit;
  statenum_t from_state;
  uint32_t unused1[31];
  uint32_t count;
  uint32_t unused2[31];
};


enum class CudaAlgorithm {
  NONE,
  NORMAL,
  NORMAL2,
  NORMAL_GLOBAL,
  NORMAL_MARKING,
  SUPER,
  SUPER0,
};

constexpr std::array cuda_algs = {
  "no_algorithm",
  "cuda_gen_loops_normal()",
  "cuda_gen_loops_normal2()",
  "cuda_gen_loops_normal_global()",
  "cuda_gen_loops_normal_marking()",
  "cuda_gen_loops_super()",
  "cuda_gen_loops_super0()",
};

#endif

