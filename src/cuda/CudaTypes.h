//
// CudaTypes.h
//
// Defines data types for searching on a CUDA GPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_CUDATYPES_H_
#define JPRIME_CUDATYPES_H_

#include "WorkAssignment.h"

#include <array>
#include <vector>
#include <cstdint>


// state numbers
using statenum_t = uint16_t;


// information about the state of a single worker
struct WorkerInfo {  // 20 bytes
  statenum_t start_state = 0;  // current value of `start_state` (input/output)
  statenum_t end_state = 0;  // highest value of `start_state` (input)
  uint16_t pos = 0;  // position in WorkAssignmentCell array (input/output)
  uint64_t nnodes = 0;  // number of nodes completed (output)
  uint16_t status = 1;  // bit 0 = is worker done, other bits unused
  uint32_t cycles_startup = 0;  // measured GPU clock cycles to initialize
};


// storage for used[] bitarray, for 32 threads = 32 bits per thread; each state
// in the graph maps onto a single bit.
//
// Data layout gives each thread its own bank in shared memory
struct ThreadStorageUsed {  // 128 bytes
  uint32_t data;
  uint32_t unused[31];
};


// storage for a single work cell (single value of `pos`), for 32 threads =
// 8 bytes per thread
//
// Data layout gives each thread its own bank in shared memory
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
  NORMAL_MARKING,
  SUPER,
  SUPER0,
};

constexpr std::array cuda_algs = {
  "no_algorithm",
  "cuda_gen_loops_normal()",
  "cuda_gen_loops_normal_marking()",
  "cuda_gen_loops_super()",
  "cuda_gen_loops_super0()",
};


struct CudaRuntimeParams {
  // defines how the algorithms are mapped onto the GPU hardware
  unsigned num_blocks = 1;
  unsigned num_threadsperblock = 32;
  size_t pattern_buffer_size = 1;
  size_t shared_memory_size = 0;
  bool used_in_shared = true;
  unsigned window_lower = 0;
  unsigned window_upper = 0;

  // configures the search algorithm
  unsigned n_min = 0;
  unsigned n_max = 0;
  bool report = false;
  unsigned shiftlimit = -1u;
};


struct CudaMemoryPointers {
  // statically allocated items in GPU memory
  statenum_t* graphmatrix_c = nullptr;
  uint8_t* maxoutdegree_d = nullptr;
  uint16_t* numstates_d = nullptr;
  uint16_t* numcycles_d = nullptr;
  uint32_t* pattern_buffer_size_d = nullptr;
  uint32_t* pattern_index_d[2] = { nullptr, nullptr };

  // dynamically allocated memory blocks in GPU global memory
  WorkerInfo* wi_d[2] = { nullptr, nullptr };
  ThreadStorageWorkCell* wc_d[2] = { nullptr, nullptr };
  statenum_t* pb_d[2] = { nullptr, nullptr };  // if needed
  statenum_t* graphmatrix_d = nullptr;  // if needed
  uint32_t* used_d = nullptr;  // if needed
};


struct CudaWorkerSummary {
  unsigned root_pos_min;  // minimum `root_pos` across all active workers
  statenum_t max_start_state;  // maximum `start_state` across all workers
  uint32_t cycles_startup = 0;  // average GPU clock cycles to initialize

  // vectors containing ids of workers in various states; note that all ids in
  // multiple_start_states are in other vectors as well!
  std::vector<unsigned> workers_idle;  // idle workers
  std::vector<unsigned> workers_multiple_start_states;
  std::vector<unsigned> workers_rpm_plus0;  // root_pos == root_pos_min
  std::vector<unsigned> workers_rpm_plus1;  // root_pos == root_pos_min + 1
  std::vector<unsigned> workers_rpm_plus2;
  std::vector<unsigned> workers_rpm_plus3;
  std::vector<unsigned> workers_rpm_plus4p;

  // vectors containing counts of active workers, indexed by start_state
  std::vector<unsigned> count_rpm_plus0;
  std::vector<unsigned> count_rpm_plus1;
  std::vector<unsigned> count_rpm_plus2;
  std::vector<unsigned> count_rpm_plus3;
  std::vector<unsigned> count_rpm_plus4p;

  // values from `context` captured at a point in time
  uint64_t npatterns = 0;
  uint64_t ntotal = 0;
  uint64_t nnodes = 0;
};


#endif
