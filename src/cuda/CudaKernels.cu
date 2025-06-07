//
// CudaKernels.cu
//
// GPU kernels for executing the core search functions of `jprime`. This file
// should be compiled with `nvcc`, part of the CUDA Toolkit.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Coordinator.h"
#include "CudaTypes.h"

#include <stdexcept>

#include <cuda_runtime.h>


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
__device__ uint32_t pattern_index_bank0_d = 0;
__device__ uint32_t pattern_index_bank1_d = 0;


//------------------------------------------------------------------------------
// Normal mode
//------------------------------------------------------------------------------

__global__ void cuda_gen_loops_normal(
        // execution setup
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wc_d,
        statenum_t* const patterns_d, uint32_t* const pattern_index_d,
        const statenum_t* const graphmatrix_d, uint32_t* const used_d,
        unsigned pos_lower_s, unsigned pos_upper_s, uint64_t cycles,
        // algorithm config
        bool report, unsigned n_min, unsigned n_max)
{
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }
  const auto start_clock = clock64();

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
    used = (ThreadStorageUsed*)&used_d[
          (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
          (((numstates_d + 1) + 31) / 32) + (id & 31)];
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

  // starting a new value of `start_state`
  if (pos == -1) {
    pos = 0;
  }

  // initialize used[]
  for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].data = 0;
  }
  for (int i = 0; i < st_state; ++i) {
    used[i / 32].data |= (1u << (i & 31));
  }
  for (int i = 1; i <= pos; ++i) {
    const statenum_t from_state = workcell_d[i].from_state;
    used[from_state / 32].data |= (1u << (from_state & 31));
  }

  // initialize current workcell pointer
  ThreadStorageWorkCell* wc =
      (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];

  unsigned from_state = wc->from_state;
  const auto init_clock = clock64();
  const auto end_clock = init_clock + cycles;
  wi_d[id].cycles_startup = init_clock - start_clock;

  // main loop -----------------------------------------------------------------

  while (true) {
    statenum_t to_state = 0;

    if (wc->col == wc->col_limit || (to_state =
          graphmatrix[(from_state - 1) * outdegree + wc->col]) == 0) {
      // beat is finished, go back to previous one
      used[from_state / 32].data &= ~(1u << (from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
        } else {
          ++st_state;
        }
        pos = -1;
        break;
      }

      --pos;
      if (wc == workcell_pos_lower) {
        wc = workcell_pos_lower_minus1;
      } else if (wc == workcell_pos_upper_plus1) {
        wc = workcell_pos_upper;
      } else {
        --wc;
      }
      from_state = wc->from_state;
      ++wc->col;
      continue;
    }

    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(pattern_index_d, 1);
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
      ++wc->count;
      ++wc->col;
      continue;
    }

    if (used[to_state / 32].data & (1u << (to_state & 31))) {
      ++wc->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++wc->col;
      continue;
    }

    // current throw is valid, so advance to next beat

    // invariant: only exit when we're about to move to the next beat
    if (clock64() > end_clock)
      break;

    used[to_state / 32].data |= (1u << (to_state & 31));

    ++pos;
    if (wc == workcell_pos_lower_minus1) {
      wc = workcell_pos_lower;
    } else if (wc == workcell_pos_upper) {
      wc = workcell_pos_upper_plus1;
    } else {
      ++wc;
    }
    wc->col = 0;
    wc->col_limit = outdegree;
    wc->from_state = from_state = to_state;
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
// Super mode
//------------------------------------------------------------------------------

__global__ void cuda_gen_loops_super(
        // execution setup
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wc_d,
        statenum_t* const patterns_d, uint32_t* const pattern_index_d,
        const statenum_t* const graphmatrix_d, uint32_t* const used_d,
        unsigned pos_lower_s, unsigned pos_upper_s, uint64_t cycles,
        // algorithm config
        bool report, unsigned n_min, unsigned n_max, unsigned shiftlimit)
{
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }
  const auto start_clock = clock64();

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
      used = (ThreadStorageUsed*)&used_d[
            (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
            (((numstates_d + 1) + 31) / 32) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
    }
    {
      // cycleused[]
      cycleused = (ThreadStorageUsed*)&used_d[device_base_u32 +
            (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
            ((numcycles_d + 31) / 32) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }
    {
      // isexitcycle[]
      isexitcycle = (ThreadStorageUsed*)&used_d[device_base_u32 +
            (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
            ((numcycles_d + 31) / 32) + (id & 31)];
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

  // starting a new value of `start_state`
  if (pos == -1) {
    pos = 0;
  }

  // initialize used[]
  if (used != nullptr) {
    for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
      used[i].data = 0;
    }
    for (int i = 0; i < st_state; ++i) {
      used[i / 32].data |= (1u << (i & 31));
    }
    for (int i = 1; i <= pos; ++i) {
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

    for (int i = 0; i < pos; ++i) {
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

  // initialize current workcell pointer
  ThreadStorageWorkCell* wc =
      (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];

  from_state = wc->from_state;
  from_cycle = graphmatrix[(from_state - 1) * (outdegree + 1) + outdegree];
  const auto init_clock = clock64();
  const auto end_clock = init_clock + cycles;
  wi_d[id].cycles_startup = init_clock - start_clock;

  // main loop -----------------------------------------------------------------

  while (true) {
    statenum_t to_state = 0;

    if (wc->col == wc->col_limit || (to_state =
          graphmatrix[(from_state - 1) * (outdegree + 1) + wc->col]) == 0) {
      // beat is finished, go back to previous one
      if (shiftlimit != 0) {
        used[from_state / 32].data &= ~(1u << (from_state & 31));
      }
      ++nnodes;

      if (pos == 0) {
        // done with search starting at `st_state`
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
        } else {
          ++st_state;
        }
        pos = -1;
        break;
      }

      --pos;
      if (wc == workcell_pos_lower) {
        wc = workcell_pos_lower_minus1;
      } else if (wc == workcell_pos_upper_plus1) {
        wc = workcell_pos_upper;
      } else {
        --wc;
      }

      const unsigned to_cycle = from_cycle;
      from_state = wc->from_state;
      from_cycle = graphmatrix[(from_state - 1) * (outdegree + 1) + outdegree];
      if (from_cycle == to_cycle) {  // unwinding a shift throw
        --shiftcount;
      } else {  // link throw
        cycleused[to_cycle / 32].data &= ~(1u << (to_cycle & 31));
        if (isexitcycle[to_cycle / 32].data & (1u << (to_cycle & 31))) {
          ++exitcycles_left;
        }
      }
      ++wc->col;
      continue;
    }

    if (to_state < st_state) {
      ++wc->col;
      continue;
    }

    if (shiftlimit != 0 &&
          (used[to_state / 32].data & (1u << (to_state & 31)))) {
      ++wc->col;
      continue;
    }

    const unsigned to_cycle = graphmatrix[(to_state - 1) * (outdegree + 1) +
        outdegree];

    if (/* shiftlimit == 0 ||*/ to_cycle != from_cycle) {  // link throw
      if (to_state == st_state) {
        // found a valid pattern
        if (report && pos + 1 >= n_min) {
          const uint32_t idx = atomicAdd(pattern_index_d, 1);
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
        ++wc->count;
        ++wc->col;
        continue;
      }

      if (cycleused[to_cycle / 32].data & (1u << (to_cycle & 31))) {
        ++wc->col;
        continue;
      }

      if ((/* shiftlimit == 0 ||*/ shiftcount == shiftlimit) &&
            exitcycles_left == 0) {
        ++wc->col;
        continue;
      }

      if (pos + 1 == n_max) {
        ++wc->col;
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
        ++wc->col;
        continue;
      }

      if (to_state == st_state) {
        if (shiftcount < pos) {
          // don't allow all shift throws in superprime pattern
          if (report && pos + 1 >= n_min) {
            const uint32_t idx = atomicAdd(pattern_index_d, 1);
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
          ++wc->count;
        }
        ++wc->col;
        continue;
      }

      if (pos + 1 == n_max) {
        ++wc->col;
        continue;
      }

      // go to next beat
      used[to_state / 32].data |= (1u << (to_state & 31));
      ++shiftcount;
    }

    ++pos;
    if (wc == workcell_pos_lower_minus1) {
      wc = workcell_pos_lower;
    } else if (wc == workcell_pos_upper) {
      wc = workcell_pos_upper_plus1;
    } else {
      ++wc;
    }
    wc->col = 0;
    wc->col_limit = outdegree;
    wc->from_state = from_state = to_state;
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
// Host functions for interfacing
//------------------------------------------------------------------------------

// Set up CUDA shared memory configuration for gpu kernels.

void configure_cuda_shared_memory(const CudaRuntimeParams& p)
{
  cudaFuncSetAttribute(cuda_gen_loops_normal,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(p.shared_memory_used));
  cudaFuncSetAttribute(cuda_gen_loops_super,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(p.shared_memory_used));
}

// Return pointers to statically allocated items in GPU memory that are
// declared in this file.

CudaMemoryPointers get_gpu_static_pointers()
{
  CudaMemoryPointers ptrs;
  cudaGetSymbolAddress((void **)&(ptrs.graphmatrix_c), graphmatrix_c);
  cudaGetSymbolAddress((void **)&(ptrs.maxoutdegree_d), maxoutdegree_d);
  cudaGetSymbolAddress((void **)&(ptrs.numstates_d), numstates_d);
  cudaGetSymbolAddress((void **)&(ptrs.numcycles_d), numcycles_d);
  cudaGetSymbolAddress((void **)&(ptrs.pattern_buffer_size_d),
      pattern_buffer_size_d);
  cudaGetSymbolAddress((void **)&(ptrs.pattern_index_d[0]),
      pattern_index_bank0_d);
  cudaGetSymbolAddress((void **)&(ptrs.pattern_index_d[1]),
      pattern_index_bank1_d);
  return ptrs;
}

// Launch the appropriate CUDA kernel.
//
// In the event of an error, throw a `std::runtime_error` exception with an
// appropriate error message.

void launch_kernel(const CudaRuntimeParams& p, const CudaMemoryPointers& ptrs,
    Coordinator::SearchAlgorithm alg, unsigned bank, uint64_t cycles,
    cudaStream_t& stream)
{
  switch (alg) {
    case Coordinator::SearchAlgorithm::NORMAL:
      cuda_gen_loops_normal
        <<<p.num_blocks, p.num_threadsperblock, p.shared_memory_used, stream>>>(
          ptrs.wi_d[bank], ptrs.wc_d[bank], ptrs.pb_d[bank],
          ptrs.pattern_index_d[bank], ptrs.graphmatrix_d, ptrs.used_d,
          p.window_lower, p.window_upper, cycles,
          p.report, p.n_min, p.n_max
        );
      break;
    case Coordinator::SearchAlgorithm::SUPER:
    case Coordinator::SearchAlgorithm::SUPER0:
      cuda_gen_loops_super
        <<<p.num_blocks, p.num_threadsperblock, p.shared_memory_used, stream>>>(
          ptrs.wi_d[bank], ptrs.wc_d[bank], ptrs.pb_d[bank],
          ptrs.pattern_index_d[bank], ptrs.graphmatrix_d, ptrs.used_d,
          p.window_lower, p.window_upper, cycles,
          p.report, p.n_min, p.n_max, p.shiftlimit
        );
      break;
    default:
      throw std::runtime_error("CUDA error: algorithm not implemented");
  }
}
