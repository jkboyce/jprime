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
// Helper functions
//------------------------------------------------------------------------------

__device__ __forceinline__ void set_bit(ThreadStorageUsed* arr, unsigned st)
{
  arr[st / 32].data |= (static_cast<uint32_t>(1) << (st & 31));
}

__device__ __forceinline__ void clear_bit(ThreadStorageUsed* arr, unsigned st)
{
  arr[st / 32].data &= ~(static_cast<uint32_t>(1) << (st & 31));
}

__device__ __forceinline__ bool is_bit_set(ThreadStorageUsed* arr, unsigned st)
{
  return (arr[st / 32].data & (static_cast<uint32_t>(1) << (st & 31))) != 0;
}

// Return pointers to working buffers for this CUDA thread

__device__ CudaThreadPointers get_thread_pointers(
    const CudaGlobalPointers& gptrs, unsigned bank,
    Coordinator::SearchAlgorithm alg, const CudaRuntimeParams& params)
{
  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  CudaThreadPointers ptrs;

  // shared memory for this SM
  extern __shared__ uint32_t shared[];
  size_t shared_base_u32 = 0;

  // set up pointer to graph matrix
  statenum_t* graphmatrix_d = (gptrs.graphmatrix_d == nullptr ?
      graphmatrix_c : gptrs.graphmatrix_d);  // source of data

  if (params.graph_size_s != 0) {
    // put into shared memory
    ptrs.graphmatrix = reinterpret_cast<statenum_t*>(shared);
    shared_base_u32 += params.graph_size_s / 4;

    uint32_t* graphmatrix_u32 = reinterpret_cast<uint32_t*>(graphmatrix_d);
    for (int i = threadIdx.x; i < params.graph_size_s / 4; i += blockDim.x) {
      shared[i] = graphmatrix_u32[i];
    }
    __syncthreads();
  } else {
    ptrs.graphmatrix = graphmatrix_d;
  }

  // find base address of workcell[] in device memory, for this thread
  {
    ThreadStorageWorkCell* const warp_start =
        &gptrs.wc_d[bank][(id / 32) * params.n_max];
    uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
    ptrs.workcell_d = reinterpret_cast<ThreadStorageWorkCell*>(
        &warp_start_u32[id & 31]);
  }

  if (gptrs.used_d != nullptr) {
    // used[] and other arrays are in device memory; set up base addresses for
    // this thread

    size_t device_base_u32 = 0;

    if (alg != Coordinator::SearchAlgorithm::SUPER0) {
      // used[]
      ptrs.used = (ThreadStorageUsed*)&gptrs.used_d[device_base_u32 +
          (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
          (((numstates_d + 1) + 31) / 32) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
    }

    if (alg == Coordinator::SearchAlgorithm::NORMAL_MARKING) {
      // deadstates[]
      ptrs.deadstates = (ThreadStorageUsed*)&gptrs.used_d[device_base_u32 +
          (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 3) / 4) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 3) / 4);
    }

    if (alg == Coordinator::SearchAlgorithm::SUPER || alg ==
        Coordinator::SearchAlgorithm::SUPER0) {
      // cycleused[]
      ptrs.cycleused = (ThreadStorageUsed*)&gptrs.used_d[device_base_u32 +
          (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 31) / 32) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);

      // isexitcycle[]
      ptrs.isexitcycle = (ThreadStorageUsed*)&gptrs.used_d[device_base_u32 +
          (id / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 31) / 32) + (id & 31)];
      device_base_u32 += gridDim.x * ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }
  } else {
    // used[] and other arrays are in shared memory. used[] is stored as a
    // bitfield for 32 threads, in (numstates_d + 1)/32 instances of
    // ThreadStorageUsed, each of which is 32 uint32s

    if (alg != Coordinator::SearchAlgorithm::SUPER0) {
      ptrs.used = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
          (((numstates_d + 1) + 31) / 32) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * (((numstates_d + 1) + 31) / 32);
    }

    if (alg == Coordinator::SearchAlgorithm::NORMAL_MARKING) {
      ptrs.deadstates = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 3) / 4) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 3) / 4);
    }

    if (alg == Coordinator::SearchAlgorithm::SUPER || alg ==
        Coordinator::SearchAlgorithm::SUPER0) {
      ptrs.cycleused = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 31) / 32) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
      ptrs.isexitcycle = (ThreadStorageUsed*)&shared[shared_base_u32 +
          (threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
          ((numcycles_d + 31) / 32) + (threadIdx.x & 31)];
      shared_base_u32 += ((blockDim.x + 31) / 32) *
          (sizeof(ThreadStorageUsed) / 4) * ((numcycles_d + 31) / 32);
    }
  }

  // workcells in shared memory, if any
  //
  // workcell[] arrays for 32 threads are stored in (pos_upper_s - pos_lower_s)
  // instances of ThreadStorageWorkCell, each of which is 64 uint32s
  const unsigned pos_lower_s = params.window_lower;
  const unsigned pos_upper_s = params.window_upper;
  const unsigned n_max = params.n_max;
  const unsigned upper = (n_max < pos_upper_s ? n_max : pos_upper_s);

  if (pos_lower_s < upper) {
    ptrs.workcell_s =
        (ThreadStorageWorkCell*)&shared[shared_base_u32 +
        (threadIdx.x / 32) * (sizeof(ThreadStorageWorkCell) / 4) *
        (upper - pos_lower_s) + (threadIdx.x & 31)];
    shared_base_u32 += ((blockDim.x + 31) / 32) *
        (sizeof(ThreadStorageWorkCell) / 4) * (upper - pos_lower_s);

    // initialize
    for (unsigned i = pos_lower_s; i < upper; ++i) {
      ptrs.workcell_s[i - pos_lower_s].col = ptrs.workcell_d[i].col;
      ptrs.workcell_s[i - pos_lower_s].col_limit = ptrs.workcell_d[i].col_limit;
      ptrs.workcell_s[i - pos_lower_s].from_state =
          ptrs.workcell_d[i].from_state;
      ptrs.workcell_s[i - pos_lower_s].count = ptrs.workcell_d[i].count;
    }
  } else {
    ptrs.workcell_s = nullptr;
  }

  // set up four pointers to indicate when we're moving between the portions of
  // workcell[] in device memory and shared memory
  ptrs.workcell_pos_lower_minus1 =
      (pos_lower_s > 0 && pos_lower_s <= n_max &&
            pos_lower_s < pos_upper_s) ?
      &ptrs.workcell_d[pos_lower_s - 1] : nullptr;
  ptrs.workcell_pos_lower =
      (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
      &ptrs.workcell_s[0] : nullptr;
  ptrs.workcell_pos_upper =
      (pos_lower_s < pos_upper_s && pos_upper_s < n_max &&
          pos_lower_s < pos_upper_s) ?
      &ptrs.workcell_s[pos_upper_s - pos_lower_s - 1] : nullptr;
  ptrs.workcell_pos_upper_plus1 =
      (pos_upper_s < n_max && pos_lower_s < pos_upper_s) ?
      &ptrs.workcell_d[pos_upper_s] : nullptr;

  return ptrs;
}

// Helper for debugging

__device__ void dump_info(int16_t pos, ThreadStorageWorkCell* workcell_d,
      ThreadStorageUsed* used, ThreadStorageUsed* deadstates,
      int max_possible)
{
  const auto nstates = numstates_d;
  const auto ncycles = numcycles_d;

  printf("  pos = %d, max_possible = %d\n", pos, max_possible);
  printf("  workcells:  ");
  for (unsigned i = 0; i <= pos; ++i) {
    printf("(%d,%d,%d,%d), ", i, workcell_d[i].col, workcell_d[i].col_limit,
        workcell_d[i].from_state);
  }
  printf("\n");
  printf("  used states:  ");
  for (unsigned i = 1; i <= nstates; ++i) {
    if (is_bit_set(used, i)) {
      printf("%d, ", i);
    }
  }
  printf("\n");
  if (deadstates != nullptr) {
    printf("  deadstates:  ");
    for (unsigned i = 0; i < ncycles; ++i) {
      const uint32_t ds = (deadstates[i / 4].data >> ((i & 3) * 8)) & 255;
      printf("(%d,%d), ", i, ds);
    }
    printf("\n");
  }
}

//------------------------------------------------------------------------------
// Helper functions for NORMAL_MARKING mode
//------------------------------------------------------------------------------

__device__ __forceinline__ bool mark_tail(statenum_t from_st,
    statenum_t from_cy, const statenum_t* const gr, uint8_t od,
    ThreadStorageUsed* u, ThreadStorageUsed* ds, int& maxp, unsigned nmin)
{
  // unusable states in excludestates_tail[]
  const uint16_t idx_low = gr[from_st * (od + 5) + (od + 1)];
  const uint16_t idx_high = gr[from_st * (od + 5) + (od + 2)];
  uint32_t idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
  if (idx == 0)
    return true;

  bool valid = true;
  statenum_t st = 0;
  const uint32_t mask = static_cast<uint32_t>(255) << ((from_cy & 3) * 8);
  uint32_t& ds_ref = ds[from_cy / 4].data;

  while ((st = gr[idx]) != 0) {
    const uint32_t old_used = u[st / 32].data;
    const uint32_t new_used = old_used ^
        (static_cast<uint32_t>(1) << (st & 31));
    u[st / 32].data = new_used;

    if (new_used > old_used) {
      // state flipped from used==0 to used==1
      const uint32_t old_ds = (ds_ref & mask);
      if (old_ds != 0) {
        --maxp;
        if (maxp < static_cast<int>(nmin)) {
          valid = false;
        }
      }
      ds_ref += (static_cast<uint32_t>(1) << ((from_cy & 3) * 8));
    }

    ++idx;
  }

  return valid;
}

__device__ __forceinline__ bool mark_head(statenum_t to_st,
    statenum_t to_cy, const statenum_t* const gr, uint8_t od,
    ThreadStorageUsed* u, ThreadStorageUsed* ds, int& maxp, unsigned nmin)
{
  // unusable states in excludestates_head[]
  const uint16_t idx_low = gr[to_st * (od + 5) + (od + 3)];
  const uint16_t idx_high = gr[to_st * (od + 5) + (od + 4)];
  uint32_t idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
  if (idx == 0)
    return true;

  bool valid = true;
  statenum_t st = 0;
  const uint32_t mask = static_cast<uint32_t>(255) << ((to_cy & 3) * 8);
  uint32_t& ds_ref = ds[to_cy / 4].data;

  while ((st = gr[idx]) != 0) {
    const uint32_t old_used = u[st / 32].data;
    const uint32_t new_used = old_used ^ (1u << (st & 31));
    u[st / 32].data = new_used;

    if (new_used > old_used) {
      const uint32_t old_ds = (ds_ref & mask);
      if (old_ds != 0) {
        --maxp;
        if (maxp < static_cast<int>(nmin)) {
          valid = false;
        }
      }
      ds_ref += (static_cast<uint32_t>(1) << ((to_cy & 3) * 8));
    }

    ++idx;
  }

  return valid;
}

__device__ __forceinline__ void unmark_tail(statenum_t from_st,
    statenum_t from_cy, const statenum_t* const gr, uint8_t od,
    ThreadStorageUsed* u, ThreadStorageUsed* ds, int& maxp, unsigned nmin)
{
  // unusable states in excludestates_tail[]
  const uint16_t idx_low = gr[from_st * (od + 5) + (od + 1)];
  const uint16_t idx_high = gr[from_st * (od + 5) + (od + 2)];
  uint32_t idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
  if (idx == 0)
    return;

  statenum_t st = 0;
  const uint32_t mask = static_cast<uint32_t>(255) << ((from_cy & 3) * 8);
  uint32_t& ds_ref = ds[from_cy / 4].data;

  while ((st = gr[idx]) != 0) {
    const uint32_t old_used = u[st / 32].data;
    const uint32_t new_used = old_used ^
        (static_cast<uint32_t>(1) << (st & 31));
    u[st / 32].data = new_used;

    if (new_used < old_used) {
      const uint32_t new_ds = ds_ref -
          (static_cast<uint32_t>(1) << ((from_cy & 3) * 8));
      if ((new_ds & mask) != 0) {
        ++maxp;
      }
      ds_ref = new_ds;
    }

    ++idx;
  }
}

__device__ __forceinline__ void unmark_head(statenum_t to_st,
    statenum_t to_cy, const statenum_t* const gr, uint8_t od,
    ThreadStorageUsed* u, ThreadStorageUsed* ds, int& maxp, unsigned nmin)
{
  // unusable states in excludestates_head[]
  const uint16_t idx_low = gr[to_st * (od + 5) + (od + 3)];
  const uint16_t idx_high = gr[to_st * (od + 5) + (od + 4)];
  uint32_t idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
  if (idx == 0)
    return;

  statenum_t st = 0;
  const uint32_t mask = static_cast<uint32_t>(255) << ((to_cy & 3) * 8);
  uint32_t& ds_ref = ds[to_cy / 4].data;

  while ((st = gr[idx]) != 0) {
    const uint32_t old_used = u[st / 32].data;
    const uint32_t new_used = old_used ^
        (static_cast<uint32_t>(1) << (st & 31));
    u[st / 32].data = new_used;

    if (new_used < old_used) {
      const uint32_t new_ds = ds_ref -
          (static_cast<uint32_t>(1) << ((to_cy & 3) * 8));
      if ((new_ds & mask) != 0) {
        ++maxp;
      }
      ds_ref = new_ds;
    }

    ++idx;
  }
}

//------------------------------------------------------------------------------
// NORMAL mode kernel
//------------------------------------------------------------------------------

__global__ void cuda_gen_loops_normal(
    CudaGlobalPointers gptrs, unsigned bank, Coordinator::SearchAlgorithm alg,
    CudaRuntimeParams params, uint64_t cycles)
{
  const auto start_clock = clock64();
  const auto ptrs = get_thread_pointers(gptrs, bank, alg, params);

  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  WorkerInfo& wi = gptrs.wi_d[bank][id];
  if ((wi.status & 1) != 0) {
    return;
  }

  // local variables for fast access
  const auto outdegree = maxoutdegree_d;
  const auto graphmatrix = ptrs.graphmatrix;
  const auto pos_lower_s = params.window_lower;
  const auto pos_upper_s = params.window_upper;
  const auto report = params.report;
  const auto n_min = params.n_min;
  const auto n_max = params.n_max;

  auto st_state = wi.start_state;
  auto pos = wi.pos;
  auto nnodes = wi.nnodes;
  auto used = ptrs.used;
  auto workcell_d = ptrs.workcell_d;
  auto workcell_s = ptrs.workcell_s;
  auto workcell_pos_lower_minus1 = ptrs.workcell_pos_lower_minus1;
  auto workcell_pos_lower = ptrs.workcell_pos_lower;
  auto workcell_pos_upper = ptrs.workcell_pos_upper;
  auto workcell_pos_upper_plus1 = ptrs.workcell_pos_upper_plus1;

  // initialize workspace ------------------------------------------------------
  // (identical to CPU version in Worker::initialize_working_variables)

  for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].data = 0;
  }
  for (int i = 1; i < st_state; ++i) {
    set_bit(used, i);
  }

  // replay partial pattern ----------------------------------------------------

  if (pos == -1) {
    pos = 0;  // starting a new value of `start_state`
  }

  for (int i = 0; i < pos; ++i) {
    const statenum_t to_st = workcell_d[i + 1].from_state;
    if (is_bit_set(used, to_st)) {
      wi.status |= 2;  // initialization error
      return;
    }
    set_bit(used, to_st);
  }

  // current workcell pointer
  auto wc = (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];
  auto from_state = wc->from_state;

  // record cycles used during initialization
  const auto init_clock = clock64();
  const auto end_clock = init_clock + cycles;
  wi.cycles_startup = init_clock - start_clock;

  // main loop -----------------------------------------------------------------

  while (true) {
    statenum_t to_state = 0;

    if (wc->col == wc->col_limit || (to_state =
          graphmatrix[from_state * outdegree + wc->col]) == 0) {
      // beat is finished, go back to previous one
      clear_bit(used, from_state);
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi.end_state) {
          wi.status |= 1;
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
        const uint32_t idx = atomicAdd(gptrs.pattern_index_d[bank], 1);
        statenum_t*& patterns_d = gptrs.pb_d[bank];
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

    if (is_bit_set(used, to_state)) {
      ++wc->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++wc->col;
      continue;
    }

    // invariant: only exit when we're about to move to the next beat
    if (clock64() > end_clock)
      break;

    // advance to next beat
    set_bit(used, to_state);

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

  wi.start_state = st_state;
  wi.pos = pos;
  wi.nnodes = nnodes;

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
// NORMAL_MARKING mode kernel
//------------------------------------------------------------------------------

// graphmatrix elements:
//   [from_state * (outdegree + 5) + outdegree] = cyclenum
//   [from_state * (outdegree + 5) + outdegree + 1] = estail_index_lower
//   [from_state * (outdegree + 5) + outdegree + 2] = estail_index_upper
//   [from_state * (outdegree + 5) + outdegree + 3] = eshead_index_lower
//   [from_state * (outdegree + 5) + outdegree + 4] = eshead_index_upper

__global__ void cuda_gen_loops_normal_marking(
    CudaGlobalPointers gptrs, unsigned bank, Coordinator::SearchAlgorithm alg,
    CudaRuntimeParams params, uint64_t cycles)
{
  const auto start_clock = clock64();
  const auto ptrs = get_thread_pointers(gptrs, bank, alg, params);
  constexpr bool debugprint = false;

  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  WorkerInfo& wi = gptrs.wi_d[bank][id];
  if ((wi.status & 1) != 0) {
    return;
  }

  // local variables for fast access
  const auto outdegree = maxoutdegree_d;
  const auto graphmatrix = ptrs.graphmatrix;
  const auto pos_lower_s = params.window_lower;
  const auto pos_upper_s = params.window_upper;
  const auto report = params.report;
  const auto n_min = params.n_min;
  const auto n_max = params.n_max;

  auto st_state = wi.start_state;
  auto pos = wi.pos;
  auto nnodes = wi.nnodes;
  auto used = ptrs.used;
  auto workcell_d = ptrs.workcell_d;
  auto workcell_s = ptrs.workcell_s;
  auto workcell_pos_lower_minus1 = ptrs.workcell_pos_lower_minus1;
  auto workcell_pos_lower = ptrs.workcell_pos_lower;
  auto workcell_pos_upper = ptrs.workcell_pos_upper;
  auto workcell_pos_upper_plus1 = ptrs.workcell_pos_upper_plus1;

  auto deadstates = ptrs.deadstates;
  int max_possible = numstates_d - numcycles_d;

  // initialize workspace ------------------------------------------------------
  // (identical to CPU version in Worker::initialize_working_variables)

  for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].data = 0;
  }
  for (int i = 1; i < st_state; ++i) {
    set_bit(used, i);

    // unusable states in excludestates_tail[]
    uint16_t idx_low = graphmatrix[i * (outdegree + 5) + (outdegree + 1)];
    uint16_t idx_high = graphmatrix[i * (outdegree + 5) + (outdegree + 2)];
    uint32_t idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
    if (idx != 0) {
      while (true) {
        const statenum_t st = graphmatrix[idx];
        if (st == 0)
          break;
        set_bit(used, st);
        ++idx;
      }
    }

    // unusable states in excludestates_head[]
    idx_low = graphmatrix[i * (outdegree + 5) + (outdegree + 3)];
    idx_high = graphmatrix[i * (outdegree + 5) + (outdegree + 4)];
    idx = (static_cast<uint32_t>(idx_high) << 16) | idx_low;
    if (idx != 0) {
      while (true) {
        const statenum_t st = graphmatrix[idx];
        if (st == 0)
          break;
        set_bit(used, st);
        ++idx;
      }
    }
  }

  // initialize deadstates[] and maxpossible
  for (unsigned i = 0; i < ((numcycles_d + 3) / 4); ++i) {
    deadstates[i].data = 0;
  }
  for (unsigned i = 1; i <= numstates_d; ++i) {
    if (is_bit_set(used, i)) {
      const auto cyc = graphmatrix[i * (outdegree + 5) + (outdegree)];
      const uint32_t mask = static_cast<uint32_t>(1) << ((cyc & 3) * 8);
      deadstates[cyc / 4].data += mask;
    }
  }

  for (unsigned i = 0; i < numcycles_d; ++i) {
    const uint32_t ds = (deadstates[i / 4].data >> ((i & 3) * 8)) & 255;
    if (ds > 1) {
      max_possible -= (static_cast<int>(ds) - 1);
    }
  }

  // replay partial pattern ----------------------------------------------------

  if (pos == -1) {
    pos = 0;  // starting a new value of `start_state`
  }

  for (int i = 0; i < pos; ++i) {
    const statenum_t from_st = workcell_d[i].from_state;
    const statenum_t to_st =
        graphmatrix[from_st * (outdegree + 5) + workcell_d[i].col];
    const statenum_t from_cy =
        graphmatrix[from_st * (outdegree + 5) + outdegree];
    const statenum_t to_cy =
        graphmatrix[to_st * (outdegree + 5) + outdegree];

    if (is_bit_set(used, to_st)) {
      wi.status |= 2;  // initialization error
      return;
    }
    set_bit(used, to_st);

    if (workcell_d[i].col != 0) {
      // link throw from `from_st` to `to_st`
      if (!mark_tail(from_st, from_cy, graphmatrix, outdegree, used,
          deadstates, max_possible, n_min)) {
        wi.status |= 2;  // initialization error
        return;
      }
      if (!mark_head(to_st, to_cy, graphmatrix, outdegree, used, deadstates,
          max_possible, n_min)) {
        wi.status |= 2;  // initialization error
        return;
      }
    }
  }

  // current workcell pointer
  auto wc = (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];
  auto from_state = wc->from_state;
  bool firstworkcell = true;

  if constexpr(debugprint) {
    if (id == 0) {
      printf("State after initialization:\n");
      dump_info(pos, workcell_d, used, deadstates, max_possible);
    }
  }

  const auto init_clock = clock64();
  const auto end_clock = init_clock + cycles;
  wi.cycles_startup = init_clock - start_clock;

  // main loop -----------------------------------------------------------------

  while (true) {
    statenum_t to_state = 0;

    if (wc->col == wc->col_limit || (to_state =
          graphmatrix[from_state * (outdegree + 5) + wc->col]) == 0) {
      // beat is finished, backtrack after cleaning up marking operations
      const statenum_t from_cycle =
          graphmatrix[from_state * (outdegree + 5) + (outdegree)];
      if (wc->col > 1) {
        unmark_tail(from_state, from_cycle, graphmatrix, outdegree, used,
            deadstates, max_possible, n_min);
      }
      clear_bit(used, from_state);
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi.end_state) {
          wi.status |= 1;
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
      // unmark head of link throw from `wc->from_state` into `from_state`
      if (wc->col != 0) {
        unmark_head(from_state, from_cycle, graphmatrix, outdegree, used,
            deadstates, max_possible, n_min);
      }
      firstworkcell = false;
      from_state = wc->from_state;
      ++wc->col;
      continue;
    }

    if (wc->col == 1 || (firstworkcell && wc->col != 0)) {
      // First link throw at this position; mark states on the `from_state`
      // shift cycle that are excluded by a link throw. Only need to do this
      // once since the excluded states are independent of link throw value.
      firstworkcell = false;

      const statenum_t from_cycle =
          graphmatrix[from_state * (outdegree + 5) + (outdegree)];
      if (!mark_tail(from_state, from_cycle, graphmatrix, outdegree, used,
          deadstates, max_possible, n_min)) {
        // not valid, bail to previous beat
        wc->col = wc->col_limit;
        continue;
      }
    }

    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(gptrs.pattern_index_d[bank], 1);
        statenum_t*& patterns_d = gptrs.pb_d[bank];
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

    if (is_bit_set(used, to_state)) {
      ++wc->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++wc->col;
      continue;
    }

    if (wc->col != 0) {  // link throw
      // mark states excluded by head of link throw
      const statenum_t to_cycle =
          graphmatrix[to_state * (outdegree + 5) + (outdegree)];
      if (!mark_head(to_state, to_cycle, graphmatrix, outdegree, used,
          deadstates, max_possible, n_min)) {
        // couldn't advance to next beat
        unmark_head(to_state, to_cycle, graphmatrix, outdegree, used,
            deadstates, max_possible, n_min);
        ++wc->col;
        continue;
      }
    }

    if (clock64() > end_clock)
      break;

    // advance to next beat
    set_bit(used, to_state);

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
    firstworkcell = false;
  }

  wi.start_state = st_state;
  wi.pos = pos;
  wi.nnodes = nnodes;

  // save workcell_s[] to device memory
  for (unsigned i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_d[i].col = workcell_s[i - pos_lower_s].col;
      workcell_d[i].col_limit = workcell_s[i - pos_lower_s].col_limit;
      workcell_d[i].from_state = workcell_s[i - pos_lower_s].from_state;
      workcell_d[i].count = workcell_s[i - pos_lower_s].count;
    }
  }

  if constexpr (debugprint) {
    if (id == 0 && (wi.status & 1) == 0) {
      printf("State after execution:\n");
      dump_info(pos, workcell_d, used, deadstates, max_possible);
    }
  }
}

//------------------------------------------------------------------------------
// SUPER and SUPER0 mode kernel
//------------------------------------------------------------------------------

__global__ void cuda_gen_loops_super(
    CudaGlobalPointers gptrs, unsigned bank, Coordinator::SearchAlgorithm alg,
    CudaRuntimeParams params, uint64_t cycles)
{
  const auto start_clock = clock64();
  const auto ptrs = get_thread_pointers(gptrs, bank, alg, params);

  const unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
  WorkerInfo& wi = gptrs.wi_d[bank][id];
  if ((wi.status & 1) != 0) {
    return;
  }

  // local variables for fast access
  const auto outdegree = maxoutdegree_d;
  const auto graphmatrix = ptrs.graphmatrix;
  const auto pos_lower_s = params.window_lower;
  const auto pos_upper_s = params.window_upper;
  const auto report = params.report;
  const auto n_min = params.n_min;
  const auto n_max = params.n_max;
  const auto shiftlimit = params.shiftlimit;

  auto st_state = wi.start_state;
  auto pos = wi.pos;
  auto nnodes = wi.nnodes;
  auto used = ptrs.used;
  auto workcell_d = ptrs.workcell_d;
  auto workcell_s = ptrs.workcell_s;
  auto workcell_pos_lower_minus1 = ptrs.workcell_pos_lower_minus1;
  auto workcell_pos_lower = ptrs.workcell_pos_lower;
  auto workcell_pos_upper = ptrs.workcell_pos_upper;
  auto workcell_pos_upper_plus1 = ptrs.workcell_pos_upper_plus1;

  auto cycleused = ptrs.cycleused;
  auto isexitcycle = ptrs.isexitcycle;
  unsigned shiftcount = 0;
  unsigned exitcycles_left = 0;

  // set up working variables --------------------------------------------------
  // identical to CPU version in Worker::initialize_working_variables()

  if (used != nullptr) {
    for (unsigned i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
      used[i].data = 0;
    }
    for (int i = 1; i < st_state; ++i) {
      set_bit(used, i);
    }
  }

  // initialize cycleused[], isexitcycle[], and exitcycles_left
  {
    for (unsigned i = 0; i < ((numcycles_d + 31) / 32); ++i) {
      cycleused[i].data = 0;
      isexitcycle[i].data = 0;
    }
    for (unsigned i = st_state + 1; i <= numstates_d; ++i) {
      for (unsigned j = 0; j < outdegree; ++j) {
        if (graphmatrix[i * (outdegree + 1) + j] == st_state) {
          const auto cyc = graphmatrix[i * (outdegree + 1) + outdegree];
          set_bit(isexitcycle, cyc);
          break;
        }
      }
    }
    const auto st_cyc = graphmatrix[st_state * (outdegree + 1) + outdegree];
    clear_bit(isexitcycle, st_cyc);
    for (unsigned i = 0; i < numcycles_d; ++i) {
      if (is_bit_set(isexitcycle, i)) {
        ++exitcycles_left;
      }
    }
  }

  // replay partial pattern ----------------------------------------------------

  if (pos == -1) {
    pos = 0;  // starting a new value of `start_state`
  }

  for (int i = 0; i < pos; ++i) {
    const statenum_t from_st = workcell_d[i].from_state;
    const statenum_t from_cy =
        graphmatrix[from_st * (outdegree + 1) + outdegree];
    const statenum_t to_st = workcell_d[i + 1].from_state;
    const statenum_t to_cy = graphmatrix[to_st * (outdegree + 1) + outdegree];

    if (used != nullptr) {
      if (is_bit_set(used, to_st)) {
        wi.status |= 2;  // initialization error
        return;
      }
      set_bit(used, to_st);
    }

    if (from_cy == to_cy) {
      ++shiftcount;
    } else {
      set_bit(cycleused, to_cy);
      if (is_bit_set(isexitcycle, to_cy)) {
        --exitcycles_left;
      }
    }
  }

  // current workcell pointer
  auto wc = (pos >= pos_lower_s && pos < pos_upper_s) ?
      &workcell_s[pos - pos_lower_s] : &workcell_d[pos];
  auto from_state = wc->from_state;
  auto from_cycle = graphmatrix[from_state * (outdegree + 1) + outdegree];

  const auto init_clock = clock64();
  const auto end_clock = init_clock + cycles;
  wi.cycles_startup = init_clock - start_clock;

  // main loop -----------------------------------------------------------------

  while (true) {
    statenum_t to_state = 0;

    if (wc->col == wc->col_limit || (to_state =
          graphmatrix[from_state * (outdegree + 1) + wc->col]) == 0) {
      // beat is finished, go back to previous one
      if (shiftlimit != 0) {
        clear_bit(used, from_state);
      }
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi.end_state) {
          wi.status |= 1;
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
      from_cycle = graphmatrix[from_state * (outdegree + 1) + outdegree];
      if (from_cycle == to_cycle) {  // unwinding a shift throw
        --shiftcount;
      } else {  // link throw
        clear_bit(cycleused, to_cycle);
        if (is_bit_set(isexitcycle, to_cycle)) {
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

    if (shiftlimit != 0 && is_bit_set(used, to_state)) {
      ++wc->col;
      continue;
    }

    const unsigned to_cycle = graphmatrix[to_state * (outdegree + 1) +
        outdegree];

    if (/* shiftlimit == 0 ||*/ to_cycle != from_cycle) {  // link throw
      if (to_state == st_state) {
        // found a valid pattern
        if (report && pos + 1 >= n_min) {
          const uint32_t idx = atomicAdd(gptrs.pattern_index_d[bank], 1);
          statenum_t*& patterns_d = gptrs.pb_d[bank];
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

      if (is_bit_set(cycleused, to_cycle)) {
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
        set_bit(used, to_state);
      }

      set_bit(cycleused, to_cycle);
      if (is_bit_set(isexitcycle, to_cycle)) {
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
            const uint32_t idx = atomicAdd(gptrs.pattern_index_d[bank], 1);
            statenum_t*& patterns_d = gptrs.pb_d[bank];
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
      set_bit(used, to_state);
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

  wi.start_state = st_state;
  wi.pos = pos;
  wi.nnodes = nnodes;

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
    static_cast<int>(p.total_allocated_s));
  cudaFuncSetAttribute(cuda_gen_loops_normal_marking,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(p.total_allocated_s));
  cudaFuncSetAttribute(cuda_gen_loops_super,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    static_cast<int>(p.total_allocated_s));
}

// Return pointers to statically allocated items in GPU memory that are
// declared in this file.

CudaGlobalPointers get_gpu_static_pointers()
{
  CudaGlobalPointers ptrs;
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

void launch_kernel(const CudaGlobalPointers& gptrs, unsigned bank,
    Coordinator::SearchAlgorithm alg, const CudaRuntimeParams& p,
    uint64_t cycles, cudaStream_t& stream)
{
  const auto blocks = p.num_blocks;
  const auto threads = p.num_threadsperblock;
  const auto shared_mem = p.total_allocated_s;

  switch (alg) {
    case Coordinator::SearchAlgorithm::NORMAL:
      cuda_gen_loops_normal<<<blocks, threads, shared_mem, stream>>>(
          gptrs, bank, alg, p, cycles
      );
      break;
    case Coordinator::SearchAlgorithm::NORMAL_MARKING:
      cuda_gen_loops_normal_marking<<<blocks, threads, shared_mem, stream>>>(
          gptrs, bank, alg, p, cycles
      );
      break;
    case Coordinator::SearchAlgorithm::SUPER:
    case Coordinator::SearchAlgorithm::SUPER0:
      cuda_gen_loops_super<<<blocks, threads, shared_mem, stream>>>(
          gptrs, bank, alg, p, cycles
      );
      break;
    default:
      throw std::runtime_error("CUDA error: algorithm not implemented");
  }
}
