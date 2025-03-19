//
// CoordinatorCUDA.cu
//
// Coordinator that executes the search on a CUDA GPU. This file should be
// compiled with `nvcc`, part of the CUDA Toolkit.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "CoordinatorCUDA.cuh"

#include <iostream>
#include <vector>
#include <format>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>


//------------------------------------------------------------------------------
// GPU memory layout
//------------------------------------------------------------------------------

// GPU constant memory
//
// Every NVIDIA GPU from capability 5.0 through 12.0 has 64 KB of constant
// memory. This is where we place the juggling graph data.

__device__ __constant__ statenum_t graphmatrix_c[65536 / sizeof(statenum_t)];

// GPU global memory

__device__ uint8_t maxoutdegree_d;
__device__ uint8_t unused_d;
__device__ uint16_t numstates_d;
__device__ uint16_t numcycles_d;
__device__ uint32_t pattern_buffer_size_d;
__device__ uint32_t pattern_index_d = 0;


//------------------------------------------------------------------------------
// GPU kernels
//------------------------------------------------------------------------------

// Normal mode

__global__ void cuda_gen_loops_normal(
        // execution setup
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wc_d,
        statenum_t* const patterns_d,
        const statenum_t* const graphmatrix_d, ThreadStorageUsed* const used_d,
        unsigned pos_lower_s, unsigned pos_upper_s, uint64_t cycles,
        // algorithm config
        bool report, unsigned n_min, unsigned n_max) {
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }

  const auto end_clock = clock64() + cycles;

  // set up register variables
  auto st_state = wi_d[id].start_state;
  auto pos = wi_d[id].pos;
  auto nnodes = wi_d[id].nnodes;
  const auto outdegree = maxoutdegree_d;
  const statenum_t* const graphmatrix = (graphmatrix_d == nullptr ?
      graphmatrix_c : graphmatrix_d);

  // find base address of workcell[] in device memory, for this thread
  ThreadStorageWorkCell* workcell_d = nullptr;
  {
    ThreadStorageWorkCell* const warp_start = &wc_d[(id / 32) * n_max];
    uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
    workcell_d = reinterpret_cast<ThreadStorageWorkCell*>(
        &warp_start_u32[id & 31]);
  }

  // if used[] is in device memory, set up base address for this thread
  ThreadStorageUsed* used = nullptr;
  if (used_d != nullptr) {
    ThreadStorageUsed* const warp_start =
          &used_d[(id / 32) * (((numstates_d + 1) + 31) / 32)];
    uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
    used = reinterpret_cast<ThreadStorageUsed*>(&warp_start_u32[id & 31]);
  }

  // set up shared memory
  //
  // if used_d is nullptr then we put used[] into shared memory. It is stored
  // as bitfields for 32 threads, in (numstates_d + 1)/32 instances of
  // ThreadStorageUsed, each of which is 32 uint32s
  //
  // workcell[] arrays for 32 threads are stored in (pos_upper_s - pos_lower_s)
  // instances of ThreadStorageWorkCell, each of which is 64 uint32s
  extern __shared__ uint32_t shared[];

  size_t shared_base_u32 = 0;
  if (used_d == nullptr) {
    used = (ThreadStorageUsed*)
        &shared[(threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
              (((numstates_d + 1) + 31) / 32) + (threadIdx.x & 31)];
    shared_base_u32 = ((blockDim.x + 31) / 32) *
        (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
  }

  const unsigned upper = (n_max < pos_upper_s ? n_max : pos_upper_s);
  ThreadStorageWorkCell* const workcell_s =
      (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
      (ThreadStorageWorkCell*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageWorkCell) / 4) *
                (upper - pos_lower_s) + (threadIdx.x & 31)
      ] : nullptr;

  // initialize used[]
  for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].data = 0;
  }
  for (unsigned i = 1; i <= pos; ++i) {
    const statenum_t from_state = workcell_d[i].from_state;
    used[from_state / 32].data |= (1u << (from_state & 31));
  }

  // initialize workcell_s[]
  for (unsigned i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_s[i - pos_lower_s].col = workcell_d[i].col;
      workcell_s[i - pos_lower_s].col_limit = workcell_d[i].col_limit;
      workcell_s[i - pos_lower_s].from_state = workcell_d[i].from_state;
      workcell_s[i - pos_lower_s].count = workcell_d[i].count;
    }
  }

  // set up four pointers to indicate when we're moving between the portions of
  // workcell[] in device memory and shared memory
  ThreadStorageWorkCell* const workcell_pos_lower_minus1 =
      (pos_lower_s > 0 && pos_lower_s <= n_max &&
            pos_lower_s < pos_upper_s) ?
      &workcell_d[pos_lower_s - 1] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_lower =
      (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
      &workcell_s[0] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_upper =
      (pos_lower_s < pos_upper_s && pos_upper_s < n_max &&
          pos_lower_s < pos_upper_s) ?
      &workcell_s[pos_upper_s - pos_lower_s - 1] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_upper_plus1 =
      (pos_upper_s < n_max && pos_lower_s < pos_upper_s) ?
      &workcell_d[pos_upper_s] : nullptr;

  // initialize current workcell pointer
  ThreadStorageWorkCell* ss =
      (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];

  unsigned from_state = ss->from_state;

  while (true) {
    statenum_t to_state = 0;

    if (ss->col == ss->col_limit || (to_state =
          graphmatrix[(from_state - 1) * outdegree + ss->col]) == 0) {
      // beat is finished, go back to previous one
      used[from_state / 32].data &= ~(1u << (from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = from_state = st_state;
        continue;
      }

      --pos;
      if (ss == workcell_pos_lower) {
        ss = workcell_pos_lower_minus1;
      } else if (ss == workcell_pos_upper_plus1) {
        ss = workcell_pos_upper;
      } else {
        --ss;
      }
      from_state = ss->from_state;
      ++ss->col;
      continue;
    }

    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(&pattern_index_d, 1);
        if (idx < pattern_buffer_size_d) {
          // write to the pattern buffer
          for (unsigned j = 0; j <= pos; ++j) {
            patterns_d[idx * n_max + j] =
              (j >= pos_lower_s && j < pos_upper_s) ?
              workcell_s[j - pos_lower_s].from_state :
              workcell_d[j].from_state;
          }
          if (pos + 1 < n_max) {
            patterns_d[idx * n_max + pos + 1] = 0;
          }
        }
      }
      ++ss->count;
      ++ss->col;
      continue;
    }

    if (to_state < st_state) {
      ++ss->col;
      continue;
    }

    if (used[to_state / 32].data & (1u << (to_state & 31))) {
      ++ss->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++ss->col;
      continue;
    }

    // current throw is valid, so advance to next beat

    // invariant: only exit when we're about to move to the next beat
    if (clock64() > end_clock)
      break;

    used[to_state / 32].data |= (1u << (to_state & 31));

    ++pos;
    if (ss == workcell_pos_lower_minus1) {
      ss = workcell_pos_lower;
    } else if (ss == workcell_pos_upper) {
      ss = workcell_pos_upper_plus1;
    } else {
      ++ss;
    }
    ss->col = 0;
    ss->col_limit = outdegree;
    ss->from_state = from_state = to_state;
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;

  // save workcell_s[] to device memory
  for (unsigned i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_d[i].col = workcell_s[i - pos_lower_s].col;
      workcell_d[i].col_limit = workcell_s[i - pos_lower_s].col_limit;
      workcell_d[i].from_state = workcell_s[i - pos_lower_s].from_state;
      workcell_d[i].count = workcell_s[i - pos_lower_s].count;
    }
  }
}

// Super mode

__global__ void cuda_gen_loops_super(
        // execution setup
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wc_d,
        statenum_t* const patterns_d,
        const statenum_t* const graphmatrix_d, ThreadStorageUsed* const used_d,
        unsigned pos_lower_s, unsigned pos_upper_s, uint64_t cycles,
        // algorithm config
        bool report, unsigned n_min, unsigned n_max, unsigned shiftlimit) {
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }

  const auto end_clock = clock64() + cycles;

  // set up register variables
  auto st_state = wi_d[id].start_state;
  auto pos = wi_d[id].pos;
  auto nnodes = wi_d[id].nnodes;
  const auto outdegree = maxoutdegree_d;
  const statenum_t* const graphmatrix = (graphmatrix_d == nullptr ?
      graphmatrix_c : graphmatrix_d);

  // register-based state variables during search
  unsigned from_state = 0;
  unsigned from_cycle = 0;
  unsigned shiftcount = 0;
  unsigned exitcycles_left = 0;

  // find base address of workcell[] in device memory, for this thread
  ThreadStorageWorkCell* workcell_d = nullptr;
  {
    ThreadStorageWorkCell* const warp_start = &wc_d[(id / 32) * n_max];
    uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
    workcell_d =
        reinterpret_cast<ThreadStorageWorkCell*>(&warp_start_u32[id & 31]);
  }

  // arrays that are in device memory or shared memory
  ThreadStorageUsed* used = nullptr;
  ThreadStorageUsed* cycleused = nullptr;
  ThreadStorageUsed* isexitcycle = nullptr;

  // if used[] arrays in device memory, set up base addresses for this thread
  if (used_d != nullptr) {
    size_t device_base_u32 = 0;

    if (shiftlimit != 0) {
      // used[]
      ThreadStorageUsed* const warp_start = &used_d[
            (id / 32) * (((numstates_d + 1) + 31) / 32)];
      uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
      used = reinterpret_cast<ThreadStorageUsed*>(&warp_start_u32[id & 31]);
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
    }
    {
      // cycleused[]
      ThreadStorageUsed* const warp_start = &used_d[device_base_u32 +
            (id / 32) * (((numstates_d + 1) + 31) / 32)];
      uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
      cycleused = reinterpret_cast<ThreadStorageUsed*>(
          &warp_start_u32[id & 31]);
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }
    {
      // isexitcycle[]
      ThreadStorageUsed* const warp_start = &used_d[device_base_u32 +
            (id / 32) * (((numstates_d + 1) + 31) / 32)];
      uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
      isexitcycle = reinterpret_cast<ThreadStorageUsed*>(
          &warp_start_u32[id & 31]);
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }
  }

  // set up shared memory
  //
  // if used_d is nullptr then we put used[] into shared memory. It is stored
  // as bitfields for 32 threads, in (numstates_d + 1)/32 instances of
  // ThreadStorageUsed, each of which is 32 uint32s
  //
  // workcell[] arrays for 32 threads are stored in (pos_upper_s - pos_lower_s)
  // instances of ThreadStorageWorkCell, each of which is 64 uint32s
  extern __shared__ uint32_t shared[];
  ThreadStorageWorkCell* workcell_s = nullptr;

  {
    size_t shared_base_u32 = 0;

    if (used_d == nullptr) {
      // put used[], cycleused[], isexitcycle[] in shared memory
      if (shiftlimit != 0) {
        used = (ThreadStorageUsed*)&shared[
            (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
                  (((numstates_d + 1) + 31) / 32) + (threadIdx.x & 31)];
        shared_base_u32 += ((blockDim.x + 31) / 32) *
            (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
      }
      cycleused = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
                ((numcycles_d + 31) / 32) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
      isexitcycle = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
                ((numcycles_d + 31) / 32) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }

    const unsigned upper = (n_max < pos_upper_s ? n_max : pos_upper_s);
    workcell_s = (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
        (ThreadStorageWorkCell*)&shared[shared_base_u32 +
            (threadIdx.x / 32) * (sizeof(ThreadStorageWorkCell) / 4) *
                  (upper - pos_lower_s) + (threadIdx.x & 31)
        ] : nullptr;
  }

  // initialize used[]
  if (used != nullptr) {
    for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
      used[i].data = 0;
    }
    for (unsigned i = 1; i <= pos; ++i) {
      const statenum_t from_state = workcell_d[i].from_state;
      used[from_state / 32].data |= (1u << (from_state & 31));
    }
  }

  // initialize cycleused[], isexitcycle[], exitcycles_left, and shiftcount
  {
    for (unsigned i = 0; i < ((numcycles_d + 31) / 32); ++i) {
      cycleused[i].data = 0;
      isexitcycle[i].data = 0;
    }
    for (unsigned i = st_state + 1; i <= numstates_d; ++i) {
      for (unsigned j = 0; j < outdegree; ++j) {
        if (graphmatrix[(i - 1) * (outdegree + 1) + j] == st_state) {
          const auto cyc = graphmatrix[(i - 1) * (outdegree + 1) + outdegree];
          isexitcycle[cyc / 32].data |= (1u << (cyc & 31));
          break;
        }
      }
    }
    const auto st_cyc = graphmatrix[(st_state - 1) * (outdegree + 1) +
        outdegree];
    isexitcycle[st_cyc / 32].data &= ~(1u << (st_cyc & 31));
    for (unsigned i = 0; i < numcycles_d; ++i) {
      if (isexitcycle[i / 32].data & (1u << (i & 31))) {
        ++exitcycles_left;
      }
    }

    for (unsigned i = 0; i < pos; ++i) {
      const statenum_t from_state = workcell_d[i].from_state;
      const statenum_t from_cycle =
          graphmatrix[(from_state - 1) * (outdegree + 1) + outdegree];
      const statenum_t to_state = workcell_d[i + 1].from_state;
      const statenum_t to_cycle =
          graphmatrix[(to_state - 1) * (outdegree + 1) + outdegree];
      if (from_cycle == to_cycle) {
        ++shiftcount;
      } else {
        cycleused[to_cycle / 32].data |= (1u << (to_cycle & 31));
        if (isexitcycle[to_cycle / 32].data & (1u << (to_cycle & 31))) {
          --exitcycles_left;
        }
      }
    }
  }

  // initialize workcell_s[]
  for (unsigned i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_s[i - pos_lower_s].col = workcell_d[i].col;
      workcell_s[i - pos_lower_s].col_limit = workcell_d[i].col_limit;
      workcell_s[i - pos_lower_s].from_state = workcell_d[i].from_state;
      workcell_s[i - pos_lower_s].count = workcell_d[i].count;
    }
  }

  // set up four pointers to indicate when we're moving between the portions of
  // workcell[] in device memory and shared memory
  ThreadStorageWorkCell* const workcell_pos_lower_minus1 =
      (pos_lower_s > 0 && pos_lower_s <= n_max &&
            pos_lower_s < pos_upper_s) ?
      &workcell_d[pos_lower_s - 1] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_lower =
      (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
      &workcell_s[0] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_upper =
      (pos_lower_s < pos_upper_s && pos_upper_s < n_max &&
          pos_lower_s < pos_upper_s) ?
      &workcell_s[pos_upper_s - pos_lower_s - 1] : nullptr;
  ThreadStorageWorkCell* const workcell_pos_upper_plus1 =
      (pos_upper_s < n_max && pos_lower_s < pos_upper_s) ?
      &workcell_d[pos_upper_s] : nullptr;

  // initialize current workcell pointer
  ThreadStorageWorkCell* ss =
      (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];

  from_state = ss->from_state;
  from_cycle = graphmatrix[(from_state - 1) * (outdegree + 1) + outdegree];

  while (true) {
    statenum_t to_state = 0;

    if (ss->col == ss->col_limit || (to_state =
          graphmatrix[(from_state - 1) * (outdegree + 1) + ss->col]) == 0) {
      // beat is finished, go back to previous one
      if (shiftlimit != 0) {
        used[from_state / 32].data &= ~(1u << (from_state & 31));
      }
      ++nnodes;

      if (pos == 0) {
        // done with search starting at `st_state`
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;

        // rebuild isexitcycle[] for new start state
        exitcycles_left = 0;
        shiftcount = 0;
        for (unsigned i = 0; i < ((numcycles_d + 31) / 32); ++i) {
          cycleused[i].data = 0;
          isexitcycle[i].data = 0;
        }
        for (unsigned i = st_state + 1; i <= numstates_d; ++i) {
          for (unsigned j = 0; j < outdegree; ++j) {
            if (graphmatrix[(i - 1) * (outdegree + 1) + j] == st_state) {
              const auto cyc = graphmatrix[(i - 1) * (outdegree + 1) +
                    outdegree];
              isexitcycle[cyc / 32].data |= (1u << (cyc & 31));
              break;
            }
          }
        }
        const auto st_cyc = graphmatrix[(st_state - 1) * (outdegree + 1) +
            outdegree];
        isexitcycle[st_cyc / 32].data &= ~(1u << (st_cyc & 31));
        for (unsigned i = 0; i < numcycles_d; ++i) {
          if (isexitcycle[i / 32].data & (1u << (i & 31))) {
            ++exitcycles_left;
          }
        }

        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = from_state = st_state;
        from_cycle = graphmatrix[(from_state - 1) * (outdegree + 1) +
            outdegree];
        continue;
      }

      --pos;
      if (ss == workcell_pos_lower) {
        ss = workcell_pos_lower_minus1;
      } else if (ss == workcell_pos_upper_plus1) {
        ss = workcell_pos_upper;
      } else {
        --ss;
      }

      const unsigned to_cycle = from_cycle;
      from_state = ss->from_state;
      from_cycle = graphmatrix[(from_state - 1) * (outdegree + 1) + outdegree];
      if (from_cycle == to_cycle) {  // unwinding a shift throw
        --shiftcount;
      } else {  // link throw
        cycleused[to_cycle / 32].data &= ~(1u << (to_cycle & 31));
        if (isexitcycle[to_cycle / 32].data & (1u << (to_cycle & 31))) {
          ++exitcycles_left;
        }
      }
      ++ss->col;
      continue;
    }

    if (to_state < st_state) {
      ++ss->col;
      continue;
    }

    if (shiftlimit != 0 &&
          (used[to_state / 32].data & (1u << (to_state & 31)))) {
      ++ss->col;
      continue;
    }

    const unsigned to_cycle = graphmatrix[(to_state - 1) * (outdegree + 1) +
        outdegree];

    if (/* shiftlimit == 0 ||*/ to_cycle != from_cycle) {  // link throw
      if (to_state == st_state) {
        // found a valid pattern
        if (report && pos + 1 >= n_min) {
          const uint32_t idx = atomicAdd(&pattern_index_d, 1);
          if (idx < pattern_buffer_size_d) {
            // write to the pattern buffer
            for (unsigned j = 0; j <= pos; ++j) {
              patterns_d[idx * n_max + j] =
                (j >= pos_lower_s && j < pos_upper_s) ?
                workcell_s[j - pos_lower_s].from_state :
                workcell_d[j].from_state;
            }
            if (pos + 1 < n_max) {
              patterns_d[idx * n_max + pos + 1] = 0;
            }
          }
        }
        ++ss->count;
        ++ss->col;
        continue;
      }

      if (cycleused[to_cycle / 32].data & (1u << (to_cycle & 31))) {
        ++ss->col;
        continue;
      }

      if ((/* shiftlimit == 0 ||*/ shiftcount == shiftlimit) &&
            exitcycles_left == 0) {
        ++ss->col;
        continue;
      }

      if (pos + 1 == n_max) {
        ++ss->col;
        continue;
      }

      if (clock64() > end_clock)
        break;

      // go to next beat
      if (shiftlimit != 0) {
        used[to_state / 32].data |= (1u << (to_state & 31));
      }

      cycleused[to_cycle / 32].data |= (1u << (to_cycle & 31));
      if (isexitcycle[to_cycle / 32].data & (1u << (to_cycle & 31))) {
        --exitcycles_left;
      }
    } else {  // shift throw
      if (shiftcount == shiftlimit) {
        ++ss->col;
        continue;
      }

      if (to_state == st_state) {
        if (shiftcount < pos) {
          // don't allow all shift throws in superprime pattern
          if (report && pos + 1 >= n_min) {
            const uint32_t idx = atomicAdd(&pattern_index_d, 1);
            if (idx < pattern_buffer_size_d) {
              // write to the pattern buffer
              for (unsigned j = 0; j <= pos; ++j) {
                patterns_d[idx * n_max + j] =
                  (j >= pos_lower_s && j < pos_upper_s) ?
                  workcell_s[j - pos_lower_s].from_state :
                  workcell_d[j].from_state;
              }
              if (pos + 1 < n_max) {
                patterns_d[idx * n_max + pos + 1] = 0;
              }
            }
          }
          ++ss->count;
        }
        ++ss->col;
        continue;
      }

      if (pos + 1 == n_max) {
        ++ss->col;
        continue;
      }

      // go to next beat
      used[to_state / 32].data |= (1u << (to_state & 31));
      ++shiftcount;
    }

    ++pos;
    if (ss == workcell_pos_lower_minus1) {
      ss = workcell_pos_lower;
    } else if (ss == workcell_pos_upper) {
      ss = workcell_pos_upper_plus1;
    } else {
      ++ss;
    }
    ss->col = 0;
    ss->col_limit = outdegree;
    ss->from_state = from_state = to_state;
    from_cycle = to_cycle;
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;

  // save workcell_s[] to device memory
  for (unsigned i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_d[i].col = workcell_s[i - pos_lower_s].col;
      workcell_d[i].col_limit = workcell_s[i - pos_lower_s].col_limit;
      workcell_d[i].from_state = workcell_s[i - pos_lower_s].from_state;
      workcell_d[i].count = workcell_s[i - pos_lower_s].count;
    }
  }
}


//------------------------------------------------------------------------------
// Host code
//------------------------------------------------------------------------------

CoordinatorCUDA::CoordinatorCUDA(SearchConfig& a, SearchContext& b,
    std::ostream& c) : Coordinator(a, b, c) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void CoordinatorCUDA::run_search() {
  const auto prop = initialize_cuda_device();
  const auto graph = build_and_reduce_graph();
  const auto alg = select_cuda_search_algorithm(graph);
  const auto graph_buffer = make_graph_buffer(graph, alg);

  const auto params = find_runtime_params(prop, alg, graph);
  configure_cuda_shared_memory(params);
  allocate_memory(alg, params, graph_buffer, graph);
  copy_graph_to_gpu(graph_buffer);
  copy_static_vars_to_gpu(params, graph);
  load_initial_work_assignments(graph);

  // timing setup
  std::chrono::time_point<std::chrono::system_clock> before_kernel;
  std::chrono::time_point<std::chrono::system_clock> after_kernel;
  after_kernel = std::chrono::system_clock::now();
  uint32_t cycles = 1000000;

  // idle workers at kernel start
  auto summary = summarize_worker_status(graph);
  unsigned idle_start = summary.workers_idle.size();
  int kernel_runs = 0;
  bool startup = true;

  while (true) {
    const auto prev_after_kernel = after_kernel;
    copy_worker_data_to_gpu(startup, max_active_idx);

    before_kernel = std::chrono::high_resolution_clock::now();
    launch_cuda_kernel(params, alg, cycles);
    after_kernel = std::chrono::high_resolution_clock::now();
    ++kernel_runs;

    // process worker results
    copy_worker_data_from_gpu(max_active_idx);
    process_worker_counters();
    const auto pattern_count = process_pattern_buffer(pb_d, graph,
      params.pattern_buffer_size);

    // timekeeping
    const auto host_time = calc_duration_secs(prev_after_kernel, before_kernel);
    const auto kernel_time = calc_duration_secs(before_kernel, after_kernel);
    record_working_time(host_time, kernel_time, idle_start,
        summary.workers_idle.size());

    const auto last_summary = std::move(summary);
    summary = summarize_worker_status(graph);
    do_status_display(summary, last_summary, host_time, kernel_time);

    if (Coordinator::stopping || (summary.workers_idle.size() ==
        config.num_threads && context.assignments.size() == 0)) {
      break;
    }

    // prepare for next run
    cycles = calc_next_kernel_cycles(cycles, host_time, kernel_time,
        idle_start, summary.workers_idle.size(), pattern_count, params);
    idle_start = assign_new_jobs(summary, graph);
    startup = false;
  }

  gather_unfinished_work_assignments(graph);
  cleanup_memory();

  total_host_time += calc_duration_secs(after_kernel,
      std::chrono::high_resolution_clock::now());
  if (config.verboseflag) {
    erase_status_output();
    jpout << "total kernel time = " << total_kernel_time
          << "\ntotal host time = " << total_host_time << '\n';
    jpout << "host time per kernel run = " << (total_host_time / kernel_runs)
          << '\n';
  }
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

// Initialize CUDA device and check properties.

cudaDeviceProp CoordinatorCUDA::initialize_cuda_device() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  if (config.verboseflag) {
    erase_status_output();
    jpout << "Device Number: " << 0
          << "\n  device name: " << prop.name
          << "\n  multiprocessor count: " << prop.multiProcessorCount
          << "\n  total global memory (bytes): " << prop.totalGlobalMem
          << "\n  total constant memory (bytes): " << prop.totalConstMem
          << "\n  shared memory per block (bytes): " << prop.sharedMemPerBlock
          << "\n  shared memory per block, maximum opt-in (bytes): "
          << prop.sharedMemPerBlockOptin << std::endl;
    print_status_output();
  }

  return prop;
}

// Build and reduce the juggling graph.

Graph CoordinatorCUDA::build_and_reduce_graph() {
  Graph graph = {
      config.b,
      config.h,
      config.xarray,
      config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH
                     ? config.n_min : 0
  };
  graph.build_graph();
  customize_graph(graph);
  graph.reduce_graph();
  return graph;
}

// Choose a search algorithm to use.

CudaAlgorithm CoordinatorCUDA::select_cuda_search_algorithm(
      const Graph& graph) {
  unsigned max_possible = (config.mode == SearchConfig::RunMode::SUPER_SEARCH)
      ? graph.superprime_period_bound(config.shiftlimit)
      : graph.prime_period_bound();

  CudaAlgorithm alg = CudaAlgorithm::NONE;

  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    if (config.graphmode == SearchConfig::GraphMode::FULL_GRAPH &&
        static_cast<double>(config.n_min) >
        0.66 * static_cast<double>(max_possible)) {
      // the overhead of marking is only worth it for long-period patterns
      alg = CudaAlgorithm::NORMAL_MARKING;
    } else if (config.countflag) {
      alg = CudaAlgorithm::NORMAL;
    } else {
      alg = CudaAlgorithm::NORMAL;
    }
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    if (config.shiftlimit == 0) {
      alg = CudaAlgorithm::SUPER0;
    } else {
      alg = CudaAlgorithm::SUPER;
    }
  }

  return alg;
}

// Return a version of the graph for the GPU.

std::vector<statenum_t> CoordinatorCUDA::make_graph_buffer(const Graph& graph,
      CudaAlgorithm alg) {
  std::vector<statenum_t> graph_buffer;

  for (unsigned i = 1; i <= graph.numstates; ++i) {
    for (unsigned j = 0; j < graph.maxoutdegree; ++j) {
      if (j < graph.outdegree.at(i)) {
        graph_buffer.push_back(graph.outmatrix.at(i).at(j));
      } else {
        graph_buffer.push_back(0);
      }
    }

    // add an extra column with needed information
    if (alg == CudaAlgorithm::NORMAL_MARKING) {
      graph_buffer.push_back(graph.upstream_state(i));
    }
    if (alg == CudaAlgorithm::SUPER || alg == CudaAlgorithm::SUPER0) {
      graph_buffer.push_back(graph.cyclenum.at(i));
    }
  }

  return graph_buffer;
}

// Speedup as a function of number of warps per block, when the workcell[] array
// is placed in shared memory or global memory
//
// columns are {warps, shared memory speedup, global memory speedup}

const double throughput[33][3] = {
  {  0,  0.000, 0.000 },  // unused
  {  1,  1.000, 0.984 },  // measured
  {  2,  1.987, 1.944 },
  {  3,  2.936, 2.875 },
  {  4,  3.873, 3.784 },
  {  5,  4.715, 4.607 },
  {  6,  5.600, 5.438 },
  {  7,  6.421, 6.225 },
  {  8,  7.223, 6.982 },
  {  9,  7.961, 7.675 },
  { 10,  8.691, 8.291 },
  { 11,  9.354, 8.634 },
  { 12, 10.060, 8.893 },
  { 13, 10.618, 8.850 },
  { 14, 11.186, 8.988 },
  { 15, 11.758, 9.068 },
  { 16, 12.260, 9.131 },
  { 17, 12.683, 9.077 },
  { 18, 13.155, 9.260 },
  { 19, 13.522, 9.204 },
  { 20, 13.869, 9.186 },
  { 21, 14.234, 9.186 },
  { 22, 14.390, 9.077 },
  { 23, 14.618, 9.122 },
  { 24, 14.927, 9.077 },
  { 25, 15.049, 8.919 },
  { 26, 15.148, 8.774 },

  { 27, 15.307, 8.774 },  // extrapolated
  { 28, 15.466, 8.774 },
  { 29, 15.624, 8.774 },
  { 30, 15.783, 8.774 },
  { 31, 15.941, 8.774 },
  { 32, 16.100, 8.774 },
};

// Determine an optimal runtime configuration for the GPU hardware available.

CudaRuntimeParams CoordinatorCUDA::find_runtime_params(
      const cudaDeviceProp& prop, CudaAlgorithm alg, const Graph& graph) {
  CudaRuntimeParams params;
  params.num_blocks = prop.multiProcessorCount;
  params.pattern_buffer_size = (config.countflag ? 0 :
    (prop.totalGlobalMem / 16) / sizeof(statenum_t) / n_max);

  // heuristic: see if used[] arrays for 10 warps will fit into shared memory;
  // if not then put into device memory
  params.num_threadsperblock = 32 * 10;
  params.used_in_shared = true;
  params.window_lower = params.window_upper = 0;
  size_t shared_mem = calc_shared_memory_size(alg, graph, n_max, params);
  if (shared_mem > prop.sharedMemPerBlockOptin) {
    params.used_in_shared = false;
  }

  // create a model for what fraction of memory accesses will occur at each
  // position in the workcell array; accesses follow a normal distribution
  const double pos_mean = 0.48 * static_cast<double>(graph.numstates) - 1;
  const double pos_fwhm = sqrt(pos_mean + 1) * (config.b == 2 ? 3.25 : 2.26);
  const double pos_sigma = pos_fwhm / (2 * sqrt(2 * log(2)));

  std::vector<double> access_fraction(n_max, 0);
  double maxval = 0;
  for (int i = 0; i < n_max; ++i) {
    const auto x = static_cast<double>(i);
    // be careful to avoid underflowing exp(-x^2)
    const double val = -(x - pos_mean) * (x - pos_mean) /
        (2 * pos_sigma * pos_sigma);
    access_fraction.at(i) = val;
    if (i == 0 || val > maxval) {
      maxval = val;
    }
  }
  double sum = 0;
  for (int i = 0; i < n_max; ++i) {
    access_fraction.at(i) = exp(access_fraction.at(i) - maxval);
    sum += access_fraction.at(i);
  }
  for (int i = 0; i < n_max; ++i) {
    access_fraction.at(i) /= sum;
  }

  // consider each warp value in turn, and estimate throughput for each
  unsigned best_warps = 1;
  unsigned best_lower = 0;
  unsigned best_upper = 0;
  double best_throughput = -1;
  const int max_warps = prop.maxThreadsPerBlock / 32;

  for (int warps = 1; warps <= max_warps; ++warps) {
    // jpout << "calculating for " << warps << " warps:\n";

    // find the maximum window size that fits into shared memory
    unsigned lower = 0;
    unsigned upper = n_max;
    for (; upper != 0; --upper) {
      params.num_threadsperblock = 32 * warps;
      params.window_lower = lower;
      params.window_upper = upper;
      shared_mem = calc_shared_memory_size(alg, graph, n_max, params);
      // jpout << "  upper = " << upper << ", mem = " << shared_mem << '\n';

      if (shared_mem <= prop.sharedMemPerBlockOptin)
        break;
    }
    if (upper == 0) {
      // used[] array is in shared, and too big for this many warps
      break;
    }

    // slide the window to maximize the area contained by it
    while (upper < n_max &&
          access_fraction.at(lower) < access_fraction.at(upper)) {
      ++lower;
      ++upper;
    }
    double S = 0;
    for (int i = lower; i < upper; ++i) {
      S += access_fraction.at(i);
    }

    // estimate throughput
    const double throughput_est = (warps <= 32 ?
        throughput[warps][1] * S + throughput[warps][2] * (1 - S) :
        0.5 * warps * S + 0.25 * warps * (1 - S));
    /* jpout << "  window [" << lower << ',' << upper
          << "), S = " << S << ", T = " << throughput_avg << '\n'; */

    if (throughput_est > best_throughput + 0.25) {
      // jpout << "  new best warps: " << warps << '\n';
      best_throughput = throughput_est;
      best_warps = warps;
      best_lower = lower;
      best_upper = upper;
    }

    /*
    jpout << std::format("warps {}: window [{},{}) S = {}, throughput = {}\n",
          warps, lower, upper, S, throughput_avg);*/
  }

  params.num_threadsperblock = 32 * best_warps;
  params.window_lower = best_lower;
  params.window_upper = best_upper;
  params.shared_memory_size = calc_shared_memory_size(alg, graph, n_max,
      params);
  config.num_threads = params.num_blocks * params.num_threadsperblock;

  if (config.verboseflag) {
    erase_status_output();
    jpout << "Execution parameters:\n"
          << "  algorithm: " << cuda_algs[static_cast<int>(alg)]
          << "\n  blocks: " << params.num_blocks
          << "\n  warps per block: " << best_warps
          << "\n  threads per block: " << params.num_threadsperblock
          << "\n  worker count: " << config.num_threads
          << "\n  pattern buffer size: " << params.pattern_buffer_size
          << " patterns"
          << "\n  shared memory size: " << params.shared_memory_size
          << " bytes"
          << std::format("\n  placing used[] into {} memory",
                params.used_in_shared ? "shared" : "device")
          << "\n  workcell[] window in shared memory = ["
          << params.window_lower << ',' << params.window_upper << ')'
          << std::endl;
    print_status_output();
  }

  return params;
}

// Return the amount of shared memory needed per block, in bytes, to support a
// set of runtime parameters.

size_t CoordinatorCUDA::calc_shared_memory_size(CudaAlgorithm alg,
        const Graph& graph, unsigned n_max, const CudaRuntimeParams& p) {
  size_t shared_bytes = 0;

  switch (alg) {
    case CudaAlgorithm::NORMAL:
      if (p.used_in_shared) {
        // used[] as bitfields in shared memory
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
      }
      if (p.window_lower < p.window_upper && p.window_lower < n_max) {
        // workcell[] partially in shared memory
        const unsigned upper = std::min(n_max, p.window_upper);
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageWorkCell) * (upper - p.window_lower);
      }
      break;
    case CudaAlgorithm::SUPER:
      if (p.used_in_shared) {
        // used[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
        // cycleused[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
        // isexitcycle[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
      }
      if (p.window_lower < p.window_upper && p.window_lower < n_max) {
        // workcell[] partially in shared memory
        const unsigned upper = std::min(n_max, p.window_upper);
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageWorkCell) * (upper - p.window_lower);
      }
      break;
    case CudaAlgorithm::SUPER0:
      if (p.used_in_shared) {
        // cycleused[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
        // isexitcycle[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
      }
      if (p.window_lower < p.window_upper && p.window_lower < n_max) {
        // workcell[] partially in shared memory
        const unsigned upper = std::min(n_max, p.window_upper);
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageWorkCell) * (upper - p.window_lower);
      }
      break;
    default:
      break;
  }

  return shared_bytes;
}

// Set up CUDA shared memory configuration.

void CoordinatorCUDA::configure_cuda_shared_memory(const CudaRuntimeParams& p) {
  cudaFuncSetAttribute(cuda_gen_loops_normal,
    cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared_memory_size);
  cudaFuncSetAttribute(cuda_gen_loops_super,
    cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared_memory_size);
}

// Allocate memory in the GPU and the host.

void CoordinatorCUDA::allocate_memory(CudaAlgorithm alg,
      const CudaRuntimeParams& params,
      const std::vector<statenum_t>& graph_buffer, const Graph& graph) {
  // GPU memory

  if (!config.countflag) {
    throw_on_cuda_error(
        cudaMalloc(&pb_d, sizeof(statenum_t) * n_max *
            params.pattern_buffer_size),
        __FILE__, __LINE__);
  }
  throw_on_cuda_error(
      cudaMalloc(&wi_d, sizeof(WorkerInfo) * config.num_threads),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMalloc(&wc_d, sizeof(ThreadStorageWorkCell) * n_max *
          ((config.num_threads + 31) / 32)),
      __FILE__, __LINE__);
  if (graph_buffer.size() * sizeof(statenum_t) > sizeof(graphmatrix_c)) {
    // graph doesn't fit in constant memory
    throw_on_cuda_error(
        cudaMalloc(&graphmatrix_d, graph_buffer.size() * sizeof(statenum_t)),
        __FILE__, __LINE__);
  }
  if (!params.used_in_shared) {
    // put used[], cycleused[], and isexitcycle[] arrays in device memory
    size_t used_size = 0;
    switch (alg) {
      case CudaAlgorithm::NORMAL:
        // used[] only
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            (((graph.numstates + 1) + 31) / 32) * sizeof(ThreadStorageUsed);
        break;
      case CudaAlgorithm::SUPER:
        // used[] not needed for SUPER0
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            (((graph.numstates + 1) + 31) / 32) * sizeof(ThreadStorageUsed);
      case CudaAlgorithm::SUPER0:
        // cycleused[] and isexitcycle[]
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            ((graph.numcycles + 31) / 32) * sizeof(ThreadStorageUsed);
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            ((graph.numcycles + 31) / 32) * sizeof(ThreadStorageUsed);
        break;
      default:
        break;
    }
    if (used_size != 0) {
      throw_on_cuda_error(
        cudaMalloc(&used_d, used_size), __FILE__, __LINE__);
    }
  }

  // Host memory

  throw_on_cuda_error(
    cudaHostAlloc(&wi_h, sizeof(WorkerInfo) * config.num_threads,
        cudaHostAllocDefault),
    __FILE__, __LINE__);
  throw_on_cuda_error(
    cudaHostAlloc(&wc_h, sizeof(ThreadStorageWorkCell) * n_max *
        ((config.num_threads + 31) / 32), cudaHostAllocDefault),
    __FILE__, __LINE__);
}

// Copy graph data to GPU.

void CoordinatorCUDA::copy_graph_to_gpu(
      const std::vector<statenum_t>& graph_buffer) {
  if (graphmatrix_d != nullptr) {
    if (config.verboseflag) {
      erase_status_output();
      jpout << "  placing graph into device memory ("
            << sizeof(statenum_t) * graph_buffer.size() << " bytes)\n";
      print_status_output();
    }
    throw_on_cuda_error(
        cudaMemcpy(graphmatrix_d, graph_buffer.data(),
            sizeof(statenum_t) * graph_buffer.size(), cudaMemcpyHostToDevice),
        __FILE__, __LINE__);

  } else {
    if (config.verboseflag) {
      erase_status_output();
      jpout << "  placing graph into constant memory ("
            << sizeof(statenum_t) * graph_buffer.size() << " bytes)\n";
      print_status_output();
    }
    throw_on_cuda_error(
        cudaMemcpyToSymbol(graphmatrix_c, graph_buffer.data(),
                          sizeof(statenum_t) * graph_buffer.size()),
        __FILE__, __LINE__);
  }
}

// Copy static global variables to GPU global memory.

void CoordinatorCUDA::copy_static_vars_to_gpu(const CudaRuntimeParams& params,
      const Graph& graph) {
  uint8_t maxoutdegree_h = static_cast<uint8_t>(graph.maxoutdegree);
  uint16_t numstates_h = static_cast<uint16_t>(graph.numstates);
  uint16_t numcycles_h = static_cast<uint16_t>(graph.numcycles);
  uint32_t pattern_buffer_size_h = params.pattern_buffer_size;
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
      cudaMemcpyToSymbol(maxoutdegree_d, &maxoutdegree_h, sizeof(uint8_t)),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyToSymbol(numstates_d, &numstates_h, sizeof(uint16_t)),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyToSymbol(numcycles_d, &numcycles_h, sizeof(uint16_t)),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyToSymbol(pattern_buffer_size_d, &pattern_buffer_size_h,
                         sizeof(uint32_t)),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyToSymbol(pattern_index_d, &pattern_index_h, sizeof(uint32_t)),
      __FILE__, __LINE__);
}

//------------------------------------------------------------------------------
// Main loop
//------------------------------------------------------------------------------

// Copy worker data to the GPU. For the workcells, copy only the worker data for
// threads [0, max_idx].

void CoordinatorCUDA::copy_worker_data_to_gpu(bool startup, unsigned max_idx) {
  throw_on_cuda_error(
      cudaMemcpy(wi_d, wi_h, sizeof(WorkerInfo) *
          (startup ? config.num_threads : max_idx + 1),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(wc_d, wc_h, sizeof(ThreadStorageWorkCell) *
          (max_idx / 32 + 1) * n_max, cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

// Launch the appropriate CUDA kernel.

void CoordinatorCUDA::launch_cuda_kernel(const CudaRuntimeParams& p,
    CudaAlgorithm alg, unsigned cycles) {
  switch (alg) {
    case CudaAlgorithm::NORMAL:
      cuda_gen_loops_normal
        <<<p.num_blocks, p.num_threadsperblock, p.shared_memory_size>>>(
          wi_d, wc_d, pb_d, graphmatrix_d, used_d, p.window_lower,
          p.window_upper, cycles,
          !config.countflag, config.n_min, n_max
        );
      break;
    case CudaAlgorithm::SUPER:
    case CudaAlgorithm::SUPER0:
      cuda_gen_loops_super
        <<<p.num_blocks, p.num_threadsperblock, p.shared_memory_size>>>(
          wi_d, wc_d, pb_d, graphmatrix_d, used_d, p.window_lower,
          p.window_upper, cycles,
          !config.countflag, config.n_min, n_max, config.shiftlimit
        );
      break;
    default:
      throw std::runtime_error("CUDA error: algorithm not implemented");
  }

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::format("CUDA Error in kernel: {}",
        cudaGetErrorString(err)));
  }
}

// Copy worker data from the GPU. For the workcells, copy only the worker data
// for threads [0, max_idx].

void CoordinatorCUDA::copy_worker_data_from_gpu(unsigned max_idx) {
  throw_on_cuda_error(
      cudaMemcpy(wi_h, wi_d, sizeof(WorkerInfo) * (max_idx + 1),
          cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(wc_h, wc_d, sizeof(ThreadStorageWorkCell) *
          (max_idx / 32 + 1) * n_max, cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
}

// Process the worker counters after a kernel run.

void CoordinatorCUDA::process_worker_counters() {
  if (longest_by_startstate_ever.size() > 0) {
    longest_by_startstate_current.assign(longest_by_startstate_ever.size(), 0);
  }

  for (int id = 0; id < config.num_threads; ++id) {
    context.nnodes += wi_h[id].nnodes;
    wi_h[id].nnodes = 0;

    const statenum_t st_state = wi_h[id].start_state;
    if (st_state >= longest_by_startstate_ever.size()) {
      longest_by_startstate_ever.resize(st_state + 1, 0);
      longest_by_startstate_current.resize(st_state + 1, 0);
    }

    for (size_t i = 0; i < n_max; ++i) {
      auto& cell = workcell(id, i);
      if (cell.count == 0)
        continue;

      context.count.at(i + 1) += cell.count;
      context.ntotal += cell.count;
      if (i + 1 >= config.n_min && i + 1 <= n_max) {
        context.npatterns += cell.count;
      }
      if (i + 1 > longest_by_startstate_current.at(st_state)) {
        longest_by_startstate_current.at(st_state) = i + 1;
        if (i + 1 > longest_by_startstate_ever.at(st_state)) {
          longest_by_startstate_ever.at(st_state) = i + 1;
        }
      }
      cell.count = 0;
    }
  }
}

// Process the pattern buffer. Copy any patterns in the buffer to `context`, and
// print them to the console if needed. Then clear the buffer.
//
// Returns the count of patterns retrieved from the buffer.
//
// In the event of a pattern buffer overflow, throw a `std::runtime_error`
// exception with a relevant error message.

uint32_t CoordinatorCUDA::process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size) {
  if (config.countflag) {
    return 0;
  }

  // get the number of patterns in the buffer
  uint32_t pattern_count;
  throw_on_cuda_error(
    cudaMemcpyFromSymbol(&pattern_count, pattern_index_d, sizeof(uint32_t)),
    __FILE__, __LINE__
  );
  if (pattern_count == 0) {
    return 0;
  } else if (pattern_count > pattern_buffer_size) {
    throw std::runtime_error("CUDA error: pattern buffer overflow");
  }

  // copy pattern data to host
  std::vector<statenum_t> patterns_h(n_max * pattern_count);
  throw_on_cuda_error(
    cudaMemcpy(patterns_h.data(), pb_d, sizeof(statenum_t) * n_max *
        pattern_count, cudaMemcpyDeviceToHost),
    __FILE__, __LINE__
  );

  // work out each pattern's throw values from the list of state numbers
  // traversed, and process them
  std::vector<int> pattern_throws(n_max + 1);

  for (int i = 0; i < pattern_count; ++i) {
    const statenum_t start_state = patterns_h.at(i * n_max);
    statenum_t from_state = start_state;

    for (int j = 0; j < n_max; ++j) {
      statenum_t to_state = (j == n_max - 1) ? start_state :
                              patterns_h.at(i * n_max + j + 1);
      if (to_state == 0) {
        to_state = start_state;
      }

      int throwval = -1;
      for (unsigned k = 0; k < graph.outdegree.at(from_state); ++k) {
        if (graph.outmatrix.at(from_state).at(k) == to_state) {
          throwval = graph.outthrowval.at(from_state).at(k);
          break;
        }
      }
      if (throwval == -1) {
        // diagnostic information in case of a problem
        std::cerr << "pattern count = " << pattern_count << '\n';
        std::cerr << "i = " << i << '\n';
        std::cerr << "j = " << j << '\n';
        for (unsigned k = 0; k < n_max; ++k) {
          statenum_t st = patterns_h.at(i * n_max + k);
          if (st == 0)
            break;
          std::cerr << "state(" << k << ") = " << graph.state.at(st) << '\n';
        }
        std::cerr << "from_state = " << from_state << " ("
                  << graph.state.at(from_state) << ")\n"
                  << "to_state = " << to_state << '\n'
                  << "outdegree(from_state) = "
                  << graph.outdegree.at(from_state) << '\n';
        for (unsigned k = 0; k < graph.outdegree.at(from_state); ++k) {
          std::cerr << "outmatrix(from_state)[" << k << "] = "
                    << graph.outmatrix.at(from_state).at(k)
                    << " ("
                    << graph.state.at(graph.outmatrix.at(from_state).at(k))
                    << ")\n";
        }
        throw std::runtime_error("CUDA error: invalid pattern");
      }
      pattern_throws.at(j) = throwval;

      if (to_state == start_state) {
        pattern_throws.at(j + 1) = -1;  // signals end of the pattern
        break;
      }
      from_state = to_state;
    }

    const std::string pattern =
        pattern_output_format(pattern_throws, start_state);
    process_search_result(pattern);
  }

  // reset the pattern buffer index
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
    cudaMemcpyToSymbol(pattern_index_d, &pattern_index_h, sizeof(uint32_t)),
    __FILE__, __LINE__
  );

  return pattern_count;
}

// Update the global time counters.

void CoordinatorCUDA::record_working_time(double host_time, double kernel_time,
    unsigned idle_before, unsigned idle_after) {
  total_kernel_time += kernel_time;
  total_host_time += host_time;

  // assume that the workers that went idle during the last kernel run did so
  // with uniform probability over the kernel runtime
  assert(idle_after >= idle_before);
  context.secs_working += kernel_time *
      (config.num_threads - idle_before / 2 - idle_after / 2);
}

// Calculate the next number of kernel cycles to run, based on timing and
// progress.

uint64_t CoordinatorCUDA::calc_next_kernel_cycles(uint64_t last_cycles,
      double host_time, double kernel_time, unsigned idle_start,
      unsigned idle_end, uint32_t pattern_count, CudaRuntimeParams p) {
  // minimum cycles we give to the kernel
  const uint64_t min_cycles = 1000000ul;

  // calculate the normalized completion rate of jobs (probability/sec)
  /*
  double beta = log(static_cast<double>(config.num_threads - idle_start) /
          static_cast<double>(config.num_threads - idle_end)) / kernel_time;
  double c = beta * host_time + 1;
  double x = -Wm1(-exp(-c)) - c
  double target_kernel_time = (beta == 0 ? 2 * kernel_time : x / beta);

  jpout << "beta = " << beta << ", target kernel time = "
        << target_kernel_time << '\n';
  */
  double target_kernel_time = (idle_end == 0) ? 2 * kernel_time :
      sqrt(host_time * host_time +
      2 * static_cast<double>(config.num_threads) * host_time *
      kernel_time / static_cast<double>(idle_end)) - host_time;
  target_kernel_time = std::min(1.0, target_kernel_time);  // 1 sec max

  auto target_cycles = static_cast<uint64_t>(static_cast<double>(last_cycles) *
      target_kernel_time / kernel_time);
  target_cycles = std::max(min_cycles, target_cycles);

  // try to keep the pattern buffer from overflowing
  if (pattern_count > p.pattern_buffer_size / 3) {
    const auto frac = static_cast<double>(p.pattern_buffer_size / 3) /
        static_cast<double>(pattern_count);
    const auto max_cycles = static_cast<uint64_t>(
        static_cast<double>(last_cycles) * frac);
    target_cycles = std::min(target_cycles, max_cycles);
  }

  if (config.verboseflag) {
    erase_status_output();
    jpout << std::format(
        "kernel = {:.5}, host = {:.5}, idle = {}\n", kernel_time,
        host_time, idle_end);
    print_status_output();
  }

  return target_cycles;
  //return (idle_end > config.num_threads / 2 ? min_cycles : target_cycles);
}

//------------------------------------------------------------------------------
// Cleanup
//------------------------------------------------------------------------------

// Clean up GPU and host memory.

void CoordinatorCUDA::cleanup_memory() {
  if (!config.countflag) {
    cudaFree(pb_d);
  }
  cudaFree(wi_d);
  cudaFree(wc_d);
  if (graphmatrix_d != nullptr) {
    cudaFree(graphmatrix_d);
    graphmatrix_d = nullptr;
  }
  if (used_d != nullptr) {
    cudaFree(used_d);
    used_d = nullptr;
  }
  cudaFreeHost(wi_h);
  wi_h = nullptr;
  cudaFreeHost(wc_h);
  wc_h = nullptr;
}

// Gather unfinished work assignments.

void CoordinatorCUDA::gather_unfinished_work_assignments(const Graph& graph) {
  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (!wi_h[id].status & 1) {
      WorkAssignment wa = read_work_assignment(id, graph);
      context.assignments.push_back(wa);
    }
  }
}

//------------------------------------------------------------------------------
// Manage work assignments
//------------------------------------------------------------------------------

// Load initial work assignments.

void CoordinatorCUDA::load_initial_work_assignments(const Graph& graph) {
  for (int id = 0; id < config.num_threads; ++id) {
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, graph);
      max_active_idx = id;
    } else {
      wi_h[id].status |= 1;
    }
  }
}

// Load a work assignment into a worker's slot in the `WorkerInfo` and
// `ThreadStorageWorkCell` arrays.

void CoordinatorCUDA::load_work_assignment(const unsigned id,
    const WorkAssignment& wa, const Graph& graph) {
  unsigned start_state = wa.start_state;
  unsigned end_state = wa.end_state;
  if (start_state == 0) {
    start_state = (config.groundmode ==
        SearchConfig::GroundMode::EXCITED_SEARCH ? 2 : 1);
  }
  if (end_state == 0) {
    end_state = (config.groundmode ==
        SearchConfig::GroundMode::GROUND_SEARCH ? 1 : graph.numstates);
  }

  wi_h[id].start_state = start_state;
  wi_h[id].end_state = end_state;
  wi_h[id].pos = wa.partial_pattern.size();
  wi_h[id].nnodes = 0;
  wi_h[id].status &= ~1u;

  // set up workcells

  for (unsigned i = 0; i < n_max; ++i) {
    workcell(id, i).count = 0;
  }

  // default if `wa.partial_pattern` is empty
  workcell(id, 0).col = 0;
  workcell(id, 0).col_limit =
      static_cast<uint8_t>(graph.maxoutdegree);
  workcell(id, 0).from_state = start_state;

  unsigned from_state = start_state;

  for (unsigned i = 0; i < wa.partial_pattern.size(); ++i) {
    const unsigned tv = wa.partial_pattern.at(i);
    unsigned to_state = 0;

    for (unsigned j = 0; j < graph.outdegree.at(from_state); ++j) {
      if (graph.outthrowval.at(from_state).at(j) != tv)
        continue;

      to_state = graph.outmatrix.at(from_state).at(j);

      workcell(id, i).col = static_cast<uint8_t>(j);
      workcell(id, i).col_limit = (i < wa.root_pos ?
          static_cast<uint8_t>(j + 1) :
          static_cast<uint8_t>(graph.maxoutdegree));

      workcell(id, i + 1).col = 0;
      workcell(id, i + 1).col_limit =
          static_cast<uint8_t>(graph.maxoutdegree);
      workcell(id, i + 1).from_state = to_state;
      break;
    }
    if (to_state == 0) {
      std::cerr << "problem loading work assignment:\n   "
                << wa
                << "\nat position " << i
                << std::endl;

      throw std::runtime_error("CUDA error: problem loading work assignment");
    }

    from_state = to_state;
  }

  // fix `col` and `col_limit` at position `root_pos`
  if (wa.root_throwval_options.size() > 0) {
    statenum_t from_state = workcell(id, wa.root_pos).from_state;
    unsigned col = graph.maxoutdegree;
    unsigned col_limit = 0;

    for (unsigned i = 0; i < graph.outdegree.at(from_state); ++i) {
      const unsigned tv = graph.outthrowval.at(from_state).at(i);
      auto it = std::find(wa.root_throwval_options.begin(),
          wa.root_throwval_options.end(), tv);
      if (it != wa.root_throwval_options.end()) {
        col = std::min(i, col);
        col_limit = i + 1;
      }
      if (wa.root_pos < wa.partial_pattern.size() &&
          tv == wa.partial_pattern.at(wa.root_pos)) {
        col = std::min(i, col);
        col_limit = i + 1;
      }
    }

    workcell(id, wa.root_pos).col = col;
    workcell(id, wa.root_pos).col_limit = col_limit;
  }

  /*
  if (config.statusflag) {
    worker_options_left_start.at(id).resize(0);
    worker_options_left_last.at(id).resize(0);
    worker_longest_start.at(id) = 0;
    worker_longest_last.at(id) = 0;
  }
  */
}

// Read out the current work assignment for worker `id`.

WorkAssignment CoordinatorCUDA::read_work_assignment(unsigned id,
    const Graph& graph) {
  WorkAssignment wa;

  wa.start_state = wi_h[id].start_state;
  wa.end_state = wi_h[id].end_state;

  bool root_pos_found = false;

  for (unsigned i = 0; i <= wi_h[id].pos; ++i) {
    const unsigned from_state = workcell(id, i).from_state;
    unsigned col = workcell(id, i).col;
    const unsigned col_limit = std::min(graph.outdegree.at(from_state),
        static_cast<unsigned>(workcell(id, i).col_limit));

    wa.partial_pattern.push_back(graph.outthrowval.at(from_state).at(col));

    if (col < col_limit - 1 && !root_pos_found) {
      wa.root_pos = i;
      root_pos_found = true;

      ++col;
      while (col < col_limit) {
        wa.root_throwval_options.push_back(
            graph.outthrowval.at(from_state).at(col));
        ++col;
      }
    }
  }

  return wa;
}

// Assign new jobs to idle workers.
//
// Returns the number of idle workers with no jobs assigned.

unsigned CoordinatorCUDA::assign_new_jobs(const CudaWorkerSummary& summary,
    const Graph& graph) {
  if (summary.workers_idle.size() == 0)
    return 0;

  unsigned idle_remaining = summary.workers_idle.size();
  std::vector<int> has_split(config.num_threads, 0);
  auto it = summary.workers_multiple_start_states.begin();

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if ((wi_h[id].status & 1) == 0)
      continue;

    // first try assigning from our list of unassigned jobs
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, graph);
      --idle_remaining;
      continue;
    }

    // otherwise, split one of the running jobs
    bool success = false;
    while (!success) {
      if (it == summary.workers_multiple_start_states.end()) {
        it = summary.workers_rpm_plus0.begin();
      }
      if (it == summary.workers_rpm_plus0.end()) {
        it = summary.workers_rpm_plus1.begin();
      }
      if (it == summary.workers_rpm_plus1.end()) {
        it = summary.workers_rpm_plus2.begin();
      }
      if (it == summary.workers_rpm_plus2.end()) {
        it = summary.workers_rpm_plus3.begin();
      }
      if (it == summary.workers_rpm_plus3.end()) {
        it = summary.workers_rpm_plus4p.begin();
      }
      if (it == summary.workers_rpm_plus4p.end()) {
        return idle_remaining;
      }

      if (has_split.at(*it)) {
        ++it;
        continue;
      }
      has_split.at(*it) = 1;

      WorkAssignment wa = read_work_assignment(*it, graph);

      try {
        // split() throws an exception if the WorkAssignment isn't splittable
        WorkAssignment wa2 = wa.split(graph, config.split_alg);
        load_work_assignment(*it, wa, graph);
        load_work_assignment(id, wa2, graph);

        // Avoid double counting nodes: Each of the "prefix" nodes up to and
        // including `wa2.root_pos` will be reported twice: by the worker that
        // was running, and by the worker `id` that just got job `wa2`.
        if (wa.start_state == wa2.start_state) {
          wi_h[id].nnodes -= (wa2.root_pos + 1);
        }

        ++context.splits_total;
        --idle_remaining;
        max_active_idx = std::max(max_active_idx, id);
        success = true;
      } catch (const std::invalid_argument& ia) {
      }
      ++it;
    }
  }
  return idle_remaining;
}

//------------------------------------------------------------------------------
// Summarization and status display
//------------------------------------------------------------------------------

// Produce a summary of the current worker status.

CudaWorkerSummary CoordinatorCUDA::summarize_worker_status(const Graph& graph) {
  unsigned root_pos_min = -1;
  statenum_t max_start_state = 0;
  max_active_idx = 0;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (wi_h[id].status & 1) {
      continue;
    }

    max_active_idx = id;
    max_start_state = std::max(max_start_state, wi_h[id].start_state);

    for (unsigned i = 0; i <= wi_h[id].pos; ++i) {
      const auto& cell = workcell(id, i);
      unsigned col = cell.col;
      const unsigned from_state = cell.from_state;
      const unsigned col_limit = std::min(graph.outdegree.at(from_state),
          static_cast<unsigned>(cell.col_limit));

      if (col < col_limit - 1) {
        // `root_pos` == i for this worker
        if (i < root_pos_min || root_pos_min == -1) {
          root_pos_min = i;
        }
        break;
      }
    }
  }

  CudaWorkerSummary summary;
  summary.root_pos_min = root_pos_min;
  summary.max_start_state = max_start_state;
  summary.count_rpm_plus0.assign(max_start_state + 1, 0);
  summary.count_rpm_plus1.assign(max_start_state + 1, 0);
  summary.count_rpm_plus2.assign(max_start_state + 1, 0);
  summary.count_rpm_plus3.assign(max_start_state + 1, 0);
  summary.count_rpm_plus4p.assign(max_start_state + 1, 0);
  summary.npatterns = context.npatterns;
  summary.nnodes = context.nnodes;
  summary.ntotal = context.ntotal;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (wi_h[id].status & 1) {
      summary.workers_idle.push_back(id);
      continue;
    }

    if (wi_h[id].start_state != wi_h[id].end_state) {
      summary.workers_multiple_start_states.push_back(id);
    }

    for (unsigned i = 0; i <= wi_h[id].pos; ++i) {
      const auto& cell = workcell(id, i);
      unsigned col = cell.col;
      const unsigned from_state = cell.from_state;
      const unsigned col_limit = std::min(graph.outdegree.at(from_state),
          static_cast<unsigned>(cell.col_limit));

      if (col < col_limit - 1) {
        switch (i - root_pos_min) {
          case 0:
            summary.workers_rpm_plus0.push_back(id);
            summary.count_rpm_plus0.at(wi_h[id].start_state) += 1;
            break;
          case 1:
            summary.workers_rpm_plus1.push_back(id);
            summary.count_rpm_plus1.at(wi_h[id].start_state) += 1;
            break;
          case 2:
            summary.workers_rpm_plus2.push_back(id);
            summary.count_rpm_plus2.at(wi_h[id].start_state) += 1;
            break;
          case 3:
            summary.workers_rpm_plus3.push_back(id);
            summary.count_rpm_plus3.at(wi_h[id].start_state) += 1;
            break;
          default:
            summary.workers_rpm_plus4p.push_back(id);
            summary.count_rpm_plus4p.at(wi_h[id].start_state) += 1;
            break;
        }
        break;
      }
    }
  }
  return summary;
}

// Create and display the live status indicator, if needed.

void CoordinatorCUDA::do_status_display(const CudaWorkerSummary& summary,
    const CudaWorkerSummary& last_summary, double host_time,
    double kernel_time) {
  if (!config.statusflag)
    return;

  erase_status_output();
  status_lines.clear();
  status_lines.push_back("Status on: " + current_time_string());

  if (summary.root_pos_min != -1) {
    std::string total_str = std::to_string(config.num_threads);
    std::string period_str = "                       period";
    if (total_str.length() < 7) {
      period_str.insert(0, 7 - total_str.length(), ' ');
    }
    status_lines.push_back(std::format(
      " state  root_pos and worker count ({} total) {}", total_str, period_str));

    auto format1 = [](unsigned a, unsigned b, bool plus) {
      if (b == 0) {
        if (plus) {
          return std::string("             ");
        } else {
          return std::string("            ");
        }
      }
      if (plus) {
        return std::format("{:4}+: {: <6}", a, b);
      }
      return std::format("{:4}: {: <6}", a, b);
    };

    const statenum_t MAX_START = 9;
    for (unsigned start_state = 1; start_state <=
          std::min(MAX_START, summary.max_start_state); ++start_state) {
      unsigned num_rpm_plus0 = summary.count_rpm_plus0.at(start_state);
      unsigned num_rpm_plus1 = summary.count_rpm_plus1.at(start_state);
      unsigned num_rpm_plus2 = summary.count_rpm_plus2.at(start_state);
      unsigned num_rpm_plus3 = summary.count_rpm_plus3.at(start_state);
      unsigned num_rpm_plus4p = summary.count_rpm_plus4p.at(start_state);

      unsigned longest_now = longest_by_startstate_current.at(start_state);
      unsigned longest_ever = longest_by_startstate_ever.at(start_state);

      char ch = ' ';
      if (start_state == MAX_START && summary.max_start_state > MAX_START) {
        ch = '+';
        for (int st = start_state + 1; st <= summary.max_start_state; ++st) {
          longest_now = std::max(longest_now,
              longest_by_startstate_current.at(st));
          longest_ever = std::max(longest_ever,
              longest_by_startstate_ever.at(st));
        }
      }

      status_lines.push_back(std::format(
        "{:4}{}  {}{}{}{}{}  {:4} {:4}",
        start_state,
        ch,
        format1(summary.root_pos_min, num_rpm_plus0, false),
        format1(summary.root_pos_min + 1, num_rpm_plus1, false),
        format1(summary.root_pos_min + 2, num_rpm_plus2, false),
        format1(summary.root_pos_min + 3, num_rpm_plus3, false),
        format1(summary.root_pos_min + 4, num_rpm_plus4p, true),
        longest_now,
        longest_ever
      ));
    }
  }

  auto format2 = [](double a) {
    if (a < 1 || a > 9999000000000) {
      return std::string("-----");
    }
    if (a < 99999) {
      auto result = std::format("{:5g}", a);
      return result.substr(0, 5);
    } else if (a < 1000000) {
      auto result = std::format("{:4g}", a / 1000);
      return result.substr(0, 4) + "K";
    } else if (a < 1000000000) {
      auto result = std::format("{:4g}", a / 1000000);
      return result.substr(0, 4) + "M";
    } else {
      auto result = std::format("{:4g}", a / 1000000000);
      return result.substr(0, 4) + "B";
    }
  };

  const double nodespersec =
      static_cast<double>(summary.nnodes - last_summary.nnodes) /
          (host_time + kernel_time);
  const double patspersec =
      static_cast<double>(summary.ntotal - last_summary.ntotal) /
          (host_time + kernel_time);

  status_lines.push_back(std::format(
    "idled:{:7}, nodes/s: {}, pats/s: {}, pats in range:{:19}",
    summary.workers_idle.size(),
    format2(nodespersec),
    format2(patspersec),
    context.npatterns
  ));

  print_status_output();
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Return a reference to the ThreadStorageWorkCell for thread `id`, position
// `pos`, with `n_max` workcells per thread.

ThreadStorageWorkCell& CoordinatorCUDA::workcell(unsigned id, unsigned pos) {
  ThreadStorageWorkCell* start_warp = &wc_h[(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}

// Handle CUDA errors by throwing a `std::runtime_error` exception with a
// relevant error message.

void CoordinatorCUDA::throw_on_cuda_error(cudaError_t code, const char *file,
      int line) {
  if (code != cudaSuccess) {
    std::ostringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << " in file "
       << file << " at line " << line;
    throw std::runtime_error(ss.str());
  }
}
