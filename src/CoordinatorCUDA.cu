//
// CoordinatorCUDA.cu
//
// Routines for executing the search on a CUDA-enabled GPU. This file should
// be compiled with `nvcc`, part of the CUDA Toolkit.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Coordinator.h"
#include "Graph.h"

#include "CoordinatorCUDA.cuh"

#include <iostream>
#include <vector>
#include <format>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cassert>

#include <cuda_runtime.h>



//------------------------------------------------------------------------------
// GPU memory layout
//------------------------------------------------------------------------------

// GPU constant memory
//
// Every NVIDIA GPU from capability 5.0 through 12.0 has 64 KB of constant
// memory. This is where we place the juggling graph data.

__device__ __constant__ statenum_t graphmatrix_d[65536 / sizeof(statenum_t)];


// GPU global memory

__device__ uint8_t maxoutdegree_d;
__device__ uint8_t unused_d;
__device__ uint16_t numstates_d;
__device__ uint32_t pattern_buffer_size_d;
__device__ uint32_t pattern_index_d = 0;


//------------------------------------------------------------------------------
// GPU kernels
//------------------------------------------------------------------------------


__global__ void cuda_gen_loops_normal_shared(statenum_t* const patterns_d,
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wa_d,
        const unsigned n_min, const unsigned n_max, const bool report,
        uint64_t cycles) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }

  const auto end_clock = clock64() + cycles;

  // set up register variables
  statenum_t st_state = wi_d[id].start_state;
  int pos = wi_d[id].pos;
  uint64_t nnodes = wi_d[id].nnodes;
  const uint8_t outdegree = maxoutdegree_d;

  // find base address of ThreadStorageWorkCell[] for this thread
  ThreadStorageWorkCell* start_warp = &wa_d[(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* const tswc =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);

  // set up shared memory
  //
  // used[] bitfields for 32 threads are stored in (numstates_d + 1)/32
  // instances of ThreadStorageUsed, each of which is 32 uint32s
  //
  // WorkAssignmentCell[] arrays for 32 threads are stored in (n_max)
  // instances of ThreadStorageWorkCell, each of which is 64 uint32s

  extern __shared__ uint32_t shared[];
  ThreadStorageUsed* used = (ThreadStorageUsed*)
      &shared[(threadIdx.x / 32) * 32 * (((numstates_d + 1) + 31) / 32) +
            (threadIdx.x & 31)];
  ThreadStorageWorkCell* workcell = (ThreadStorageWorkCell*)
      &shared[
          ((blockDim.x + 31) / 32) * 32 * (((numstates_d + 1) + 31) / 32) +
          (threadIdx.x / 32) * 64 * n_max + (threadIdx.x & 31)
      ];

  // initialize workcell[] array
  for (int i = 0; i < n_max; ++i) {
    workcell[i].col = tswc[i].col;
    workcell[i].col_limit = tswc[i].col_limit;
    workcell[i].from_state = tswc[i].from_state;
    workcell[i].count = tswc[i].count;
  }

  // initialize used[] array
  for (int i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].used = 0;
  }
  for (int i = 1; i <= pos; ++i) {
    const statenum_t from = workcell[i].from_state;
    used[from / 32].used |= (1u << (from & 31));
  }

  ThreadStorageWorkCell* ss = &workcell[pos];

  while (true) {
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      used[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        --ss;
        ++ss->col;
        continue;
      }
    }

    const statenum_t to_state = graphmatrix_d[(ss->from_state - 1) *
          outdegree + ss->col];

    if (to_state == 0) {
      // beat is finished, go back to previous one
      used[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        --ss;
        ++ss->col;
        continue;
      }
    }
    
    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(&pattern_index_d, 1);
        if (idx < pattern_buffer_size_d) {
          for (int j = 0; j <= pos; ++j) {
            patterns_d[idx * n_max + j] = workcell[j].from_state;
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

    if (used[to_state / 32].used & (1u << (to_state & 31))) {
      ++ss->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++ss->col;
      continue;
    }

    // current throw is valid, so advance to next beat

    if (clock64() > end_clock)
      break;

    ++pos;
    ++ss;
    ss->col = 0;
    ss->col_limit = outdegree;
    ss->from_state = to_state;
    used[to_state / 32].used |= (1u << (to_state & 31));
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;

  // save workcell[] array
  for (int i = 0; i < n_max; ++i) {
    tswc[i].col = workcell[i].col;
    tswc[i].col_limit = workcell[i].col_limit;
    tswc[i].from_state = workcell[i].from_state;
    tswc[i].count = workcell[i].count;
  }
}


__global__ void cuda_gen_loops_normal_hybrid(statenum_t* const patterns_d,
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wa_d,
        const unsigned n_min, const unsigned n_max,
        const unsigned pos_lower_s, const unsigned pos_upper_s,
        const bool report, uint64_t cycles) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }

  const auto end_clock = clock64() + cycles;

  // set up register variables
  statenum_t st_state = wi_d[id].start_state;
  int pos = wi_d[id].pos;
  uint64_t nnodes = wi_d[id].nnodes;
  const uint8_t outdegree = maxoutdegree_d;

  // find base address of workcell[] in device memory, for this thread

  ThreadStorageWorkCell* const warp_start = &wa_d[(id / 32) * n_max];
  uint32_t* const warp_start_u32 = reinterpret_cast<uint32_t*>(warp_start);
  ThreadStorageWorkCell* const workcell_d =
      reinterpret_cast<ThreadStorageWorkCell*>(&warp_start_u32[id & 31]);

  // set up shared memory
  //
  // used[] bitfields for 32 threads are stored in (numstates_d + 1)/32
  // instances of ThreadStorageUsed, each of which is 32 uint32s
  //
  // workcell[] arrays for 32 threads are stored in
  // (n_upper_shared - n_lower_shared + 1) instances of ThreadStorageWorkCell,
  // each of which is 64 uint32s
  //
  // find base addresses of used[] and workcell[] in shared memory, for this
  // thread

  extern __shared__ uint32_t shared[];
  ThreadStorageUsed* const used_s = (ThreadStorageUsed*)
      &shared[(threadIdx.x / 32) * (sizeof(ThreadStorageUsed) / 4) *
            (((numstates_d + 1) + 31) / 32) + (threadIdx.x & 31)];
  const unsigned upper = (n_max < pos_upper_s ? n_max : pos_upper_s);
  ThreadStorageWorkCell* const workcell_s =
      (pos_lower_s < n_max && pos_lower_s < pos_upper_s) ?
      (ThreadStorageWorkCell*)&shared[
          ((blockDim.x + 31) / 32) * (sizeof(ThreadStorageUsed) / 4) *
                (((numstates_d + 1) + 31) / 32) +
          (threadIdx.x / 32) * (sizeof(ThreadStorageWorkCell) / 4) *
                (upper - pos_lower_s) + (threadIdx.x & 31)
      ] : nullptr;

  // initialize used_s[]
  for (int i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used_s[i].used = 0;
  }
  for (int i = 1; i <= pos; ++i) {
    const statenum_t from = workcell_d[i].from_state;
    used_s[from / 32].used |= (1u << (from & 31));
  }

  // initialize workcell_s[]
  for (int i = pos_lower_s; i < pos_upper_s; ++i) {
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

  while (true) {
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      used_s[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;  // mark done
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        if (ss == workcell_pos_lower) {
          ss = workcell_pos_lower_minus1;
        } else if (ss == workcell_pos_upper_plus1) {
          ss = workcell_pos_upper;
        } else {
          --ss;
        }
        ++ss->col;
        continue;
      }
    }

    const statenum_t to_state = graphmatrix_d[(ss->from_state - 1) *
          outdegree + ss->col];

    if (to_state == 0) {
      // beat is finished, go back to previous one
      used_s[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        if (ss == workcell_pos_lower) {
          ss = workcell_pos_lower_minus1;
        } else if (ss == workcell_pos_upper_plus1) {
          ss = workcell_pos_upper;
        } else {
          --ss;
        }
        ++ss->col;
        continue;
      }
    }
    
    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(&pattern_index_d, 1);
        if (idx < pattern_buffer_size_d) {
          // write to the pattern buffer
          for (int j = 0; j <= pos; ++j) {
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

    if (used_s[to_state / 32].used & (1u << (to_state & 31))) {
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
    ss->from_state = to_state;
    used_s[to_state / 32].used |= (1u << (to_state & 31));
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;

  // save workcell_s[] to device memory
  for (int i = pos_lower_s; i < pos_upper_s; ++i) {
    if (workcell_s != nullptr && i < n_max) {
      workcell_d[i].col = workcell_s[i - pos_lower_s].col;
      workcell_d[i].col_limit = workcell_s[i - pos_lower_s].col_limit;
      workcell_d[i].from_state = workcell_s[i - pos_lower_s].from_state;
      workcell_d[i].count = workcell_s[i - pos_lower_s].count;
    }
  }
}


__global__ void cuda_gen_loops_normal_global(statenum_t* const patterns_d,
        WorkerInfo* const wi_d, ThreadStorageWorkCell* const wa_d,
        const unsigned n_min, const unsigned n_max, const bool report,
        const uint64_t cycles) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].status & 1) {
    return;
  }

  const auto end_clock = clock64() + cycles;

  // set up register variables
  statenum_t st_state = wi_d[id].start_state;
  int pos = wi_d[id].pos;
  uint64_t nnodes = wi_d[id].nnodes;
  const uint8_t outdegree = maxoutdegree_d;

  // find base address of ThreadStorageWorkCell[] for this thread
  ThreadStorageWorkCell* start_warp = &wa_d[(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* const tswc =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);

  // set up shared memory
  //
  // used[] bitfields for 32 threads are stored in (numstates_d + 1)/32
  // instances of ThreadStorageUsed, each of which is 32 uint32s

  extern __shared__ uint32_t shared[];
  ThreadStorageUsed* used = (ThreadStorageUsed*)
      &shared[(threadIdx.x / 32) * 32 * (((numstates_d + 1) + 31) / 32) +
            (threadIdx.x & 31)];

  // initialize used[] array
  for (int i = 0; i < (((numstates_d + 1) + 31) / 32); ++i) {
    used[i].used = 0;
  }
  for (int i = 1; i <= pos; ++i) {
    const statenum_t from = tswc[i].from_state;
    used[from / 32].used |= (1u << (from & 31));
  }

  ThreadStorageWorkCell* ss = &tswc[pos];

  while (true) {
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      used[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        --ss;
        ++ss->col;
        continue;
      }
    }

    const statenum_t to_state = graphmatrix_d[(ss->from_state - 1) *
          outdegree + ss->col];

    if (to_state == 0) {
      // beat is finished, go back to previous one
      used[ss->from_state / 32].used &= ~(1u << (ss->from_state & 31));
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].status |= 1;
          break;
        }
        ++st_state;
        ss->col = 0;
        ss->col_limit = outdegree;
        ss->from_state = st_state;
        continue;
      } else {
        --pos;
        --ss;
        ++ss->col;
        continue;
      }
    }
    
    if (to_state == st_state) {
      // found a valid pattern
      if (report && pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(&pattern_index_d, 1);
        if (idx < pattern_buffer_size_d) {
          for (int j = 0; j <= pos; ++j) {
            patterns_d[idx * n_max + j] = tswc[j].from_state;
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

    if (used[to_state / 32].used & (1u << (to_state & 31))) {
      ++ss->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++ss->col;
      continue;
    }

    // current throw is valid, so advance to next beat

    if (clock64() > end_clock)
      break;

    ++pos;
    ++ss;
    ss->col = 0;
    ss->col_limit = outdegree;
    ss->from_state = to_state;
    used[to_state / 32].used |= (1u << (to_state & 31));
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;
}


//------------------------------------------------------------------------------
// Benchmarks
//------------------------------------------------------------------------------

/*
jprime 3 8 -cuda -count
11906414 patterns in range (11906414 seen, 49962563 nodes)
1.53 sec (300000 steps, normal, 56 x 96) 38400 bytes


jprime 3 9 -cuda -count
30513071763 patterns in range (30513071763 seen, 141933075458 nodes)
--> 61.0728 sec on 10 CPU cores
22.2 sec (shared, 56 x 160) 96640 bytes
32.9 sec (global, 56 x 160) 1920 bytes
27.7 sec (global, 56 x 480) 5760 bytes
56.7 sec (hybrid, 56 x 64) 11520 bytes [29,49]
16.9 sec (hybrid, 56 x 320) 57600 bytes [29,49]
14.8 sec (hybrid, 56 x 480) 86400 bytes [29,49]
15.2 sec (hybrid, 56 x 384) 99840 bytes [24,54]
27.4 sec (hybrid, 56 x 480) 5760 bytes [74,74]  --> same result as _global
26.6 sec (hybrid, 56 x 160) 96640 bytes [0,73]  --> slower than _shared
14.7 sec (hybrid, 56 x 480) 97920 bytes [27,50]


jprime 3 11 1-25 -count
19638164481 patterns in range (19638164481 seen, 94368010897 nodes)
--> 62.8 sec on 10 CPU cores
18.10 sec (shared, 56 x 160) 34560 bytes
10.57 sec (shared, 56 x 320) 71680 bytes
8.69 sec (shared, 56 x 448) 100352 bytes
10.76 sec (hybrid, 56 x 384) 49152 bytes [12,24]
8.67 sec (hybrid, 56 x 640) 81920 bytes [12,24]
11.67 sec (hybrid, 56 x 448) 100352 bytes [0,24]

*/

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run_cuda() {
  // 1. Initialization

  const auto prop = initialize_cuda_device();
  const auto graph = build_and_reduce_graph();
  const auto alg = select_cuda_search_algorithm(graph);
  const auto graph_buffer = make_graph_buffer(graph, alg);

  set_runtime_params(prop, alg, graph.numstates);
  configure_cuda_shared_memory(shared_memory_size);
  allocate_gpu_device_memory();
  copy_graph_to_gpu(graph_buffer);
  copy_static_vars_to_gpu(graph);

  std::vector<WorkerInfo> wi_h(num_workers);
  std::vector<ThreadStorageWorkCell> wa_h(((num_workers + 31) / 32) * n_max);
  load_initial_work_assignments(graph, wi_h, wa_h);

  // 2. Main loop

  std::chrono::time_point<std::chrono::system_clock> before_kernel;
  std::chrono::time_point<std::chrono::system_clock> after_kernel;
  after_kernel = std::chrono::system_clock::now();
  uint32_t cycles = 1000000;

  while (true) {
    const auto prev_after_kernel = after_kernel;
    copy_worker_data_to_gpu(wi_h, wa_h);

    before_kernel = std::chrono::high_resolution_clock::now();
    launch_cuda_kernel(num_blocks, num_threadsperblock, shared_memory_size, alg,
        cycles);
    after_kernel = std::chrono::high_resolution_clock::now();

    copy_worker_data_from_gpu(wi_h, wa_h);
    process_worker_results(graph, wi_h, wa_h);
    process_pattern_buffer(pb_d, graph, pattern_buffer_size);

    unsigned num_done = 0;
    bool all_done = true;
    for (const auto& wi : wi_h) {
      if (wi.status & 1) {
        ++num_done;
      } else {
        all_done = false;
      }
    }
  
    if (Coordinator::stopping || all_done)
      break;

    if (num_done != 0) {
      assign_new_jobs(graph, wi_h, wa_h);
    }

    // update kernel cycles for next run, based on timing and progress
    cycles = calc_next_kernel_cycles(cycles, prev_after_kernel, before_kernel,
        after_kernel, num_done);
  }

  jpout << "total kernel time = " << total_kernel_time
        << "\ntotal host time = " << total_host_time << '\n';

  // 3. Cleanup

  cleanup_gpu_memory();
  gather_unfinished_work_assignments(graph, wi_h, wa_h);
}

//------------------------------------------------------------------------------
// Setup
//------------------------------------------------------------------------------

// Initialize CUDA device and check properties.

cudaDeviceProp Coordinator::initialize_cuda_device() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  jpout << "Device Number: " << 0
        << "\n  device name: " << prop.name
        << "\n  multiprocessor count: " << prop.multiProcessorCount
        << "\n  total global memory (bytes): " << prop.totalGlobalMem
        << "\n  total constant memory (bytes): " << prop.totalConstMem
        << "\n  shared memory per block (bytes): " << prop.sharedMemPerBlock
        << "\n  shared memory per block, maximum opt-in (bytes): "
        << prop.sharedMemPerBlockOptin << std::endl;

  return prop;
}

// Build and reduce the juggling graph.

Graph Coordinator::build_and_reduce_graph() {
  Graph graph = {
      config.b,
      config.h,
      config.xarray,
      config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH
                     ? config.n_min : 0
  };
  graph.build_graph();
  // TODO: call customize_graph() here
  graph.reduce_graph();
  return graph;
}

// choose a search algorithm to use

CudaAlgorithm Coordinator::select_cuda_search_algorithm(const Graph& graph) {
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
      alg = CudaAlgorithm::NORMAL2;
    } else {
      alg = CudaAlgorithm::NORMAL2;
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
//
// If the resulting graph is too large to fit into the GPU's constant memory,
// throw a `std::runtime_error` exception with a relevant error message.

std::vector<statenum_t> Coordinator::make_graph_buffer(const Graph& graph,
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
    if (alg == CudaAlgorithm::SUPER0 || alg == CudaAlgorithm::SUPER) {
      graph_buffer.push_back(graph.cyclenum.at(i));
    }
  }

  if (graph_buffer.size() * sizeof(statenum_t) > sizeof(graphmatrix_d)) {
    throw std::runtime_error("CUDA error: Juggling graph too large");
  }

  return graph_buffer;
}

// Speedup as a function of number of warps per block, for different different
// locations for the workcell[] array
//
// columns are {warps, shared memory, global memory}

const double throughput[33][3] = {
  {  0,  0.000, 0.000 },
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

// Determine an optimal runtime configuration for the GPU hardware available
//
// If the calculation cannot run in any case, throw a `std::runtime_error`
// exception with a relevant error message.

void Coordinator::set_runtime_params(const cudaDeviceProp& prop,
      CudaAlgorithm alg, unsigned num_states) {
  // error if we can't run with one warp and zero window
  size_t shared_mem = calc_shared_memory_size(alg, num_states,
      32, n_max, n_max, n_max);
  if (shared_mem > prop.sharedMemPerBlockOptin) {
    throw std::runtime_error("CUDA error: Not enough shared memory");
  }  
  
  // create a model for what fraction of memory accesses will occur at each
  // position in the workcell array; accesses follow a normal distribution
  const double pos_mean = 0.48 * static_cast<double>(num_states) - 1;
  const double pos_fwhm = sqrt(pos_mean + 1) * (config.b == 2 ? 3.25 : 2.26);
  const double pos_sigma = pos_fwhm / (2 * sqrt(2 * log(2)));

  std::vector<double> access_fraction(n_max, 0);
  double sum = 0;
  for (int i = 0; i < n_max; ++i) {
    const auto x = static_cast<double>(i);
    const double val = exp(-(x - pos_mean) * (x - pos_mean) /
        (2 * pos_sigma * pos_sigma));
    access_fraction.at(i) = val;
    sum += val;
  }
  for (int i = 0; i < n_max; ++i) {
    access_fraction.at(i) /= sum;
  }

  // consider each warp value in turn, and estimate throughput for each
  unsigned best_warps = 1;
  unsigned best_lower = 0;
  unsigned best_upper = 0;
  double best_throughput = 0;
  const int max_warps = prop.maxThreadsPerBlock / 32;

  for (int warps = 1; warps <= max_warps; ++warps) {
    // find the maximum window size that fits into shared memory
    unsigned lower = 0;
    unsigned upper = n_max;
    for (; upper != 0; --upper) {
      shared_mem = calc_shared_memory_size(alg, num_states, 32 * warps, n_max,
            lower, upper);
      if (shared_mem <= prop.sharedMemPerBlockOptin)
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
    double throughput_avg = throughput[warps][1] * S +
        throughput[warps][2] * (1 - S);
    if (throughput_avg > best_throughput + 0.25) {
      best_throughput = throughput_avg;
      best_warps = warps;
      best_lower = lower;
      best_upper = upper;
    }

    /*
    jpout << std::format("warps {}: window [{},{}) S = {}, throughput = {}\n",
          warps, lower, upper, S, throughput_avg);*/
  }

  num_blocks = prop.multiProcessorCount;
  num_threadsperblock = 32 * best_warps;
  window_lower = best_lower;
  window_upper = best_upper;
  num_workers = num_blocks * num_threadsperblock;

  shared_memory_size = calc_shared_memory_size(alg, num_states,
      num_threadsperblock, n_max, window_lower, window_upper);
  pattern_buffer_size = (prop.totalGlobalMem / 16) / sizeof(statenum_t) / n_max;

  jpout << "Execution parameters:\n"
        << "  algorithm = " << cuda_algs[static_cast<int>(alg)]
        << "\n  num_blocks = " << num_blocks
        << "\n  warpsperblock = " << best_warps
        << "\n  num_threadsperblock = " << num_threadsperblock
        << "\n  num_workers = " << num_workers
        << "\n  pattern buffer size = " << pattern_buffer_size << " patterns"
        << "\n  shared memory size = " << shared_memory_size << " bytes"
        << "\n  pos window in shared = [" << window_lower << ','
        << window_upper << ')'
        << std::endl;
}

// Return the amount of shared memory needed per block, in bytes.

size_t Coordinator::calc_shared_memory_size(CudaAlgorithm alg,
        unsigned num_states, unsigned num_threadsperblock, unsigned n_max,
        unsigned window_lower, unsigned window_upper) {
  size_t shared_bytes = 0;
  if (alg == CudaAlgorithm::NORMAL) {
    // used[] as individual bits in uint32s
    // workcell[] in shared memory
    shared_bytes = ((num_threadsperblock + 31) / 32) * (
        sizeof(ThreadStorageUsed) * (((num_states + 1) + 31) / 32) +  // used[]
        sizeof(ThreadStorageWorkCell) * n_max  // WorkAssignentCell[]
    );
  } else if (alg == CudaAlgorithm::NORMAL2) {
    // used[] as individual bits in uint32s
    shared_bytes = ((num_threadsperblock + 31) / 32) * (
        sizeof(ThreadStorageUsed) * (((num_states + 1) + 31) / 32)  // used[]
    );
    if (window_lower < window_upper && window_lower < n_max) {
      // workcell[] partially in shared memory
      const unsigned upper = std::min(n_max, window_upper);
      shared_bytes += ((num_threadsperblock + 31) / 32) * (
        sizeof(ThreadStorageWorkCell) * (upper - window_lower)
      );
    }
  } else if (alg == CudaAlgorithm::NORMAL_GLOBAL) {
    // used[] as uint32s
    // workcell[] in global memory
    shared_bytes = ((num_threadsperblock + 31) / 32) * (
      sizeof(ThreadStorageUsed) * (((num_states + 1) + 31) / 32)    // used[]
    );
  }

  return shared_bytes;
}

// Set up CUDA shared memory configuration.

void Coordinator::configure_cuda_shared_memory(size_t shared_memory_size) {
  cudaFuncSetAttribute(cuda_gen_loops_normal_shared,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
  cudaFuncSetAttribute(cuda_gen_loops_normal_hybrid,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
  cudaFuncSetAttribute(cuda_gen_loops_normal_global,
    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
}

// Allocate GPU memory for patterns, WorkerInfo, and WorkAssignmentCells.

void Coordinator::allocate_gpu_device_memory() {
  if (!config.countflag) {
    throw_on_cuda_error(
        cudaMalloc(&pb_d, sizeof(statenum_t) * n_max * pattern_buffer_size),
        __FILE__, __LINE__);
  }
  throw_on_cuda_error(
      cudaMalloc(&wi_d, sizeof(WorkerInfo) * num_workers),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMalloc(&wa_d, sizeof(ThreadStorageWorkCell) * n_max *
          ((num_workers + 31) / 32)),
      __FILE__, __LINE__);
}

// Copy graph data to GPU constant memory.

void Coordinator::copy_graph_to_gpu(
      const std::vector<statenum_t>& graph_buffer) {
  throw_on_cuda_error(
      cudaMemcpyToSymbol(graphmatrix_d, graph_buffer.data(),
                         sizeof(statenum_t) * graph_buffer.size()),
      __FILE__, __LINE__);
}

// Copy static global variables to GPU global memory.

void Coordinator::copy_static_vars_to_gpu(const Graph& graph) {
  uint8_t maxoutdegree_h = static_cast<uint8_t>(graph.maxoutdegree);
  uint16_t numstates_h = static_cast<uint16_t>(graph.numstates);
  uint32_t pattern_buffer_size_h = pattern_buffer_size;
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
      cudaMemcpyToSymbol(maxoutdegree_d, &maxoutdegree_h, sizeof(uint8_t)),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyToSymbol(numstates_d, &numstates_h, sizeof(uint16_t)),
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

// Return a reference to the ThreadStorageWorkCell for thread `id`, position
// `pos`, with `n_max` workcells per thread.

ThreadStorageWorkCell& workcell(std::vector<ThreadStorageWorkCell>& wa_h,
      unsigned n_max, unsigned id, unsigned pos) {
  ThreadStorageWorkCell* start_warp = &wa_h.at((id / 32) * n_max);
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);

  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}


// Copy worker data to the GPU.

void Coordinator::copy_worker_data_to_gpu(std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h) {
  throw_on_cuda_error(
      cudaMemcpy(wi_d, wi_h.data(), sizeof(WorkerInfo) * wi_h.size(),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(wa_d, wa_h.data(), sizeof(ThreadStorageWorkCell) * wa_h.size(),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

// Launch the appropriate CUDA kernel.

void Coordinator::launch_cuda_kernel(unsigned num_blocks,
    unsigned num_threadsperblock, size_t shared_memory_size, CudaAlgorithm alg,
    unsigned cycles) {
  switch (alg) {
    case CudaAlgorithm::NORMAL:
      cuda_gen_loops_normal_shared
        <<<num_blocks, num_threadsperblock, shared_memory_size>>>
        (pb_d, wi_d, wa_d, config.n_min, n_max, !config.countflag, cycles);
      break;
    case CudaAlgorithm::NORMAL2:
      cuda_gen_loops_normal_hybrid
        <<<num_blocks, num_threadsperblock, shared_memory_size>>>
        (pb_d, wi_d, wa_d, config.n_min, n_max, window_lower, window_upper,
        !config.countflag, cycles);
      break;
    case CudaAlgorithm::NORMAL_GLOBAL:
      cuda_gen_loops_normal_global
        <<<num_blocks, num_threadsperblock, shared_memory_size>>>
        (pb_d, wi_d, wa_d, config.n_min, n_max, !config.countflag, cycles);
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

// Copy worker data from the GPU.

void Coordinator::copy_worker_data_from_gpu(std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h) {
  throw_on_cuda_error(
      cudaMemcpy(wi_h.data(), wi_d, sizeof(WorkerInfo) * wi_h.size(),
          cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(wa_h.data(), wa_d, sizeof(ThreadStorageWorkCell) * wa_h.size(),
          cudaMemcpyDeviceToHost),
      __FILE__, __LINE__);
}

// Process worker results and handle pattern buffer.

void Coordinator::process_worker_results(const Graph& graph,
      std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h) {
  int num_working = 0;
  int num_idle = 0;

  for (int id = 0; id < num_workers; ++id) {
    if (wi_h.at(id).status & 1) {
      ++num_idle;
    } else {
      ++num_working;
    }

    MessageW2C msg;
    msg.worker_id = id;
    msg.count.assign(n_max + 1, 0);
    for (unsigned j = 0; j < n_max; ++j) {
      msg.count.at(j + 1) = workcell(wa_h, n_max, id, j).count;
      workcell(wa_h, n_max, id, j).count = 0;
    }
    msg.nnodes = wi_h.at(id).nnodes;
    wi_h.at(id).nnodes = 0;
    record_data_from_message(msg);
  }
}

// Process the pattern buffer, copying any patterns to `context` and printing
// them to the console if needed. Then clear the buffer.
//
// In the event of a buffer overflow, throw a `std::runtime_error` exception
// with a relevant error message.

void Coordinator::process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size) {
  if (config.countflag)
      return;

  // get the number of patterns in the buffer
  uint32_t pattern_count;
  throw_on_cuda_error(
    cudaMemcpyFromSymbol(&pattern_count, pattern_index_d, sizeof(uint32_t)),
    __FILE__, __LINE__
  );

  if (pattern_count == 0) {
    return;
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
    unsigned period = 0;

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
        std::cerr << "from_state = " << from_state << " (" << graph.state.at(from_state)
                  << ")\n";
        std::cerr << "to_state = " << to_state << '\n';
        std::cerr << "outdegree(from_state) = " << graph.outdegree.at(from_state) << '\n';
        for (unsigned k = 0; k < graph.outdegree.at(from_state); ++k) {
          std::cerr << "outmatrix(from_state)[" << k << "] = "
                    << graph.outmatrix.at(from_state).at(k)
                    << " (" << graph.state.at(graph.outmatrix.at(from_state).at(k)) << ")\n";
        }
        throw std::runtime_error("CUDA error: invalid pattern");
      }
      pattern_throws.at(j) = throwval;

      ++period;
      if (to_state == start_state) {
        pattern_throws.at(j + 1) = -1;  // signals end of the pattern
        break;
      }
      from_state = to_state;
    }

    MessageW2C msg;
    msg.worker_id = 0;
    msg.pattern = pattern_output_format(config, pattern_throws, start_state);
    msg.period = period;
    process_search_result(msg);
  }

  // reset the pattern buffer index

  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
    cudaMemcpyToSymbol(pattern_index_d, &pattern_index_h, sizeof(uint32_t)),
    __FILE__, __LINE__
  );
}

// Calculate the next number of kernel cycles to run, based on timing and
// progress.

uint64_t Coordinator::calc_next_kernel_cycles(uint64_t last_cycles,
      std::chrono::time_point<std::chrono::system_clock> prev_after_kernel,
      std::chrono::time_point<std::chrono::system_clock> before_kernel,
      std::chrono::time_point<std::chrono::system_clock> after_kernel,
      unsigned num_done) {
  const std::chrono::duration<double> kernel_diff =
      after_kernel - before_kernel;
  const std::chrono::duration<double> host_diff =
      before_kernel - prev_after_kernel;
  const double kernel_runtime = kernel_diff.count();
  const double host_runtime = host_diff.count();

  total_kernel_time += kernel_runtime;
  total_host_time += host_runtime;

  // calculate kernel runtime that will maximize work done per unit time
  double target_kernel_runtime = (num_done == 0) ? 2 * kernel_runtime :
      sqrt(host_runtime * host_runtime +
      2 * static_cast<double>(num_workers) * host_runtime * kernel_runtime /
      static_cast<double>(num_done)) - host_runtime;
  target_kernel_runtime = std::min(1.0, target_kernel_runtime);  // 1 sec max

  jpout << std::format(
      "kernel = {:.5}, host = {:.5}, done = {}\n", kernel_runtime,
      host_runtime, num_done);

  auto new_cycles = static_cast<uint64_t>(static_cast<double>(last_cycles) *
      target_kernel_runtime / kernel_runtime);
  new_cycles = std::max(100000ul, new_cycles);

  return new_cycles;
}

//------------------------------------------------------------------------------
// Cleanup
//------------------------------------------------------------------------------

// Clean up GPU memory.

void Coordinator::cleanup_gpu_memory() {
  if (!config.countflag) {
    cudaFree(pb_d);
  }
  cudaFree(wi_d);
  cudaFree(wa_d);
}

// Gather unfinished work assignments.

void Coordinator::gather_unfinished_work_assignments(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h) {
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).status & 1) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph);
      context.assignments.push_back(wa);
    }
  }
}

//------------------------------------------------------------------------------
// Manage work assignments
//------------------------------------------------------------------------------

// Load initial work assignments.

void Coordinator::load_initial_work_assignments(const Graph& graph,
      std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h)  {
  for (int id = 0; id < num_workers; ++id) {
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, wi_h, wa_h, graph);

      if (config.verboseflag) {
        erase_status_output();
        jpout << std::format("worker {} given work:\n  ", id)
              << wa << std::endl;
        print_status_output();
      }
    } else {
      wi_h.at(id).status |= 1;
    }
  }
}

// Load a work assignment into a worker's slot in the `WorkerInfo` and
// `ThreadStorageWorkCell` arrays.

void Coordinator::load_work_assignment(const unsigned id,
    const WorkAssignment& wa, std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h, const Graph& graph) {
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

  wi_h.at(id).start_state = start_state;
  wi_h.at(id).end_state = end_state;
  wi_h.at(id).pos = wa.partial_pattern.size();
  wi_h.at(id).nnodes = 0;
  wi_h.at(id).status &= ~1u;

  // set up workcells

  for (unsigned i = 0; i < n_max; ++i) {
    workcell(wa_h, n_max, id, i).count = 0;
  }

  // default if `wa.partial_pattern` is empty
  workcell(wa_h, n_max, id, 0).col = 0;
  workcell(wa_h, n_max, id, 0).col_limit = static_cast<uint8_t>(graph.maxoutdegree);
  workcell(wa_h, n_max, id, 0).from_state = start_state;

  unsigned from_state = start_state;

  for (unsigned i = 0; i < wa.partial_pattern.size(); ++i) {
    const unsigned tv = wa.partial_pattern.at(i);
    unsigned to_state = 0;

    for (unsigned j = 0; j < graph.outdegree.at(from_state); ++j) {
      if (graph.outthrowval.at(from_state).at(j) != tv)
        continue;

      to_state = graph.outmatrix.at(from_state).at(j);

      workcell(wa_h, n_max, id, i).col = static_cast<uint8_t>(j);
      workcell(wa_h, n_max, id, i).col_limit = (i < wa.root_pos ?
          static_cast<uint8_t>(j + 1) :
          static_cast<uint8_t>(graph.maxoutdegree));

      workcell(wa_h, n_max, id, i + 1).col = 0;
      workcell(wa_h, n_max, id, i + 1).col_limit =
          static_cast<uint8_t>(graph.maxoutdegree);
      workcell(wa_h, n_max, id, i + 1).from_state = to_state;
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
    statenum_t from_state = workcell(wa_h, n_max, id, wa.root_pos).from_state;
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

    workcell(wa_h, n_max, id, wa.root_pos).col = col;
    workcell(wa_h, n_max, id, wa.root_pos).col_limit = col_limit;
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

WorkAssignment Coordinator::read_work_assignment(unsigned id,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h,
    const Graph& graph) {
  WorkAssignment wa;

  wa.start_state = wi_h.at(id).start_state;
  wa.end_state = wi_h.at(id).end_state;

  bool root_pos_found = false;

  for (unsigned i = 0; i <= wi_h.at(id).pos; ++i) {
    const unsigned from_state = workcell(wa_h, n_max, id, i).from_state;
    unsigned col = workcell(wa_h, n_max, id, i).col;
    const unsigned col_limit = std::min(graph.outdegree.at(from_state),
        static_cast<unsigned>(workcell(wa_h, n_max, id, i).col_limit));

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

// Assign new jobs to idle workers

void Coordinator::assign_new_jobs(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h) {

  // sort the running work assignments to find the best ones to split
  std::vector<WorkAssignmentLine> sorted_assignments;
  for (unsigned id = 0; id < num_workers; ++id) {
    if ((wi_h.at(id).status & 3) == 0) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph);
      sorted_assignments.push_back({id, wa});
    }
  }

  // compare function returns true if the first argument appears before the
  // second in a strict weak ordering, and false otherwise
  std::sort(sorted_assignments.begin(), sorted_assignments.end(),
      [](WorkAssignmentLine wal1, WorkAssignmentLine wal2) {
        return work_assignment_compare(wal1.wa, wal2.wa);
      }
  );

  unsigned index = 0;
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).status & 1)
      continue;

    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, wi_h, wa_h, graph);

      if (config.verboseflag) {
        jpout << std::format("worker {} given work:\n   ", id)
              << wa << '\n';
      }
      continue;
    }
  
    // split one of the running jobs and give it to worker `id`

    bool success = false;
    while (!success) {
      if (index == sorted_assignments.size())
        break;

      WorkAssignmentLine& wal = sorted_assignments.at(index);
      WorkAssignment wa = wal.wa;

      if (config.verboseflag) {
        jpout << std::format("worker {} went idle\n", id)
              << std::format("stealing from worker {}\n", wal.id)
              << "work before:\n" << wa << '\n';
      }

      try {
        WorkAssignment wa2 = wa.split(graph, config.split_alg);
        load_work_assignment(wal.id, wa, wi_h, wa_h, graph);
        load_work_assignment(id, wa2, wi_h, wa_h, graph);
        
        if (config.verboseflag) {
          jpout << "work after:\n" << wa << '\n'
                << std::format("new work for worker {}:\n", id)
                << wa2 << '\n';
        }

        // Avoid double counting nodes: Each of the "prefix" nodes up to and
        // including `wa2.root_pos` will be reported twice: by the worker that
        // was running, and by the worker `id` who just got job `wa2`.
        if (wa.start_state == wa2.start_state) {
          wi_h.at(id).nnodes -= (wa2.root_pos + 1);
        }
        ++context.splits_total;
        success = true;
      } catch (const std::invalid_argument& ia) {
      }
      ++index;
    }

    if (index == sorted_assignments.size())
      break;
  }
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Handle CUDA errors by throwing a `std::runtime_error` exception with a
// relevant error message.

void Coordinator::throw_on_cuda_error(cudaError_t code, const char *file,
      int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << " in file "
      << file << " at line " << line;
    throw std::runtime_error(ss.str());
  }
}
