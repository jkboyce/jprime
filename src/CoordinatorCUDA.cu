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

__global__ void cuda_gen_loops_normal(statenum_t* const patterns_d,
        WorkerInfo* const wi_d, WorkAssignmentCell* const wa_d,
        const unsigned n_min, const unsigned n_max, const unsigned steps,
        const bool report) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (wi_d[id].done) {
    return;
  }

  // set up register variables
  statenum_t st_state = wi_d[id].start_state;
  int pos = wi_d[id].pos;
  uint64_t nnodes = wi_d[id].nnodes;
  const uint8_t outdegree = maxoutdegree_d;

  // set up shared memory
  //
  // unused[] arrays for 32 threads are stored in (numstates_d + 1) instances
  // of ThreadStorageUsed, each of which is 32 uint32s
  //
  // WorkAssignmentCell[] arrays for 32 threads are stored in (n_max)
  // instances of ThreadStorageWorkCell, each of which is 64 uint32s

  extern __shared__ uint32_t shared[];
  ThreadStorageUsed* used = (ThreadStorageUsed*)
      &shared[(threadIdx.x / 32) * 32 * (numstates_d + 1) + (threadIdx.x % 32)];
  ThreadStorageWorkCell* workcell = (ThreadStorageWorkCell*)
      &shared[
          ((blockDim.x + 31) / 32) * 32 * (numstates_d + 1) +
          (threadIdx.x / 32) * 64 * n_max + (threadIdx.x % 32)
      ];

  /*
  int shared_memory_size_bytes =
      ((blockDim.x + 31) / 32) * 128 * (numstates_d + 1) +
      ((blockDim.x + 31) / 32) * 256 * n_max;
  printf("shared memory size (device) = %d bytes\n", shared_memory_size_bytes);
  */

  // initialize workcell[] array
  for (int i = 0; i < n_max; ++i) {
    workcell[i].col = wa_d[id * n_max + i].col;
    workcell[i].col_limit = wa_d[id * n_max + i].col_limit;
    workcell[i].from_state = wa_d[id * n_max + i].from_state;
    workcell[i].count = wa_d[id * n_max + i].count;
  }

  // initialize used[] array
  for (int i = 0; i <= numstates_d; ++i) {
    used[i].used = 0;
  }
  for (int i = 1; i <= pos; ++i) {
    used[workcell[i].from_state].used = 1;
  }

  ThreadStorageWorkCell* ss = &workcell[pos];

  for (unsigned step = 0; ; ++step) {
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      used[ss->from_state].used = 0;
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].done = 1;
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
      used[ss->from_state].used = 0;
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[id].end_state) {
          wi_d[id].done = 1;
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

    if (used[to_state].used) {
      ++ss->col;
      continue;
    }

    if (pos + 1 == n_max) {
      ++ss->col;
      continue;
    }

    // current throw is valid, so advance to next beat

    if (step > steps)
      break;

    ++pos;
    ++ss;
    ss->col = 0;
    ss->col_limit = outdegree;
    ss->from_state = to_state;
    used[to_state].used = 1;
  }

  wi_d[id].start_state = st_state;
  wi_d[id].pos = pos;
  wi_d[id].nnodes = nnodes;

  // save workcell[] array
  for (int i = 0; i < n_max; ++i) {
    wa_d[id * n_max + i].col = workcell[i].col;
    wa_d[id * n_max + i].col_limit = workcell[i].col_limit;
    wa_d[id * n_max + i].from_state = workcell[i].from_state;
    wa_d[id * n_max + i].count = workcell[i].count;
  }
}


//------------------------------------------------------------------------------
// Benchmarks
//------------------------------------------------------------------------------

/*
20 blocks, 32 threads/block:
jprime 3 9 -cuda -count
30513071763 patterns in range (30513071763 seen, 141933075458 nodes)
runtime = 238.0548 sec (596.2M nodes/sec, 0.0 % util, 14803 splits)

1 block, 32 threads/block:
jprime 3 8 -cuda -count
11906414 patterns in range (11906414 seen, 49962563 nodes)
runtime = 1.7509 sec (28.5M nodes/sec, 0.0 % util, 306 splits)

1 block, 64 threads/block:
jprime 3 8 -cuda -count
11906414 patterns in range (11906414 seen, 49962563 nodes)
runtime = 1.0728 sec (46.6M nodes/sec, 0.0 % util, 533 splits)

1 block, 96 threads/block:
jprime 3 8 -cuda -count
11906414 patterns in range (11906414 seen, 49962563 nodes)
runtime = 0.8166 sec (61.2M nodes/sec, 0.0 % util, 765 splits)

2 blocks, 32 threads/block:
jprime 3 8 -cuda -count
11906414 patterns in range (11906414 seen, 49962563 nodes)
runtime = 1.0862 sec (46.0M nodes/sec, 0.0 % util, 533 splits)

50 blocks, 96 threads/block:
jprime 3 9 -cuda -count
shared memory size = 89472 bytes
steps per kernel call = 200000
30513071763 patterns in range (30513071763 seen, 141933075458 nodes)
runtime = 36.7760 sec (3859.4M nodes/sec, 0.0 % util, 81821 splits)
--> 61.0728 sec on 10 CPU cores
*/

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Run the search on a CUDA-enabled GPU.
//
// In the event of an error, throw a `std::runtime_error` exception with a
// relevant error message.
/*
void Coordinator::run_cuda() {
  const unsigned num_blocks = 1;
  const unsigned num_threadsperblock = 64;
  const unsigned num_workers = num_blocks * num_threadsperblock;
  const unsigned num_steps = 20000;
  const unsigned pattern_buffer_size = 100000;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device Number: %d\n", 0);
  printf("  Device name: %s\n", prop.name);
  printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
  printf("  Total global memory (bytes): %ld\n", prop.totalGlobalMem);
  printf("  Total constant memory (bytes): %ld\n", prop.totalConstMem);
  printf("  Shared memory per block (bytes): %ld\n", prop.sharedMemPerBlock);
  printf("  Shared memory per block, maximum opt-in (bytes): %ld\n",
      prop.sharedMemPerBlockOptin);
  
  // build juggling graph

  Graph graph = {config.b, config.h, config.xarray,
      config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH ?
      config.n_min : 0};
  graph.build_graph();
  // TODO: call customize_graph() here
  graph.reduce_graph();

  // choose which search algorithm to use
  const unsigned alg = select_CUDA_search_algorithm(graph);

  // will graph fit into GPU constant memory? Calculate constant memory
  // needed, in bytes
  const unsigned graphcols = (alg == 0) ? graph.maxoutdegree :
                    graph.maxoutdegree + 1;
  const size_t graph_buffer_size =
      graph.numstates * graphcols * sizeof(statenum_t);
  if (graph_buffer_size > sizeof(graphmatrix_d)) {
    throw std::runtime_error("CUDA error: Juggling graph too large");
  }

  // will used[] and WorkAssignmentCell[] arrays fit into shared (block-local)
  // memory? Calculate shared memory needed, in bytes

  const size_t shared_memory_size = ((num_threadsperblock + 31) / 32) * (
      128 * (graph.numstates + 1) +  // used[]
      256 * n_max                    // WorkAssignentCell[]
  );
  std::cout << "num_workers = " << num_workers << '\n'
            << "shared memory size = " << shared_memory_size << " bytes\n"
            << "steps per kernel call = " << num_steps << std::endl;
  if (shared_memory_size > 99 * 1024) {
    throw std::runtime_error("CUDA error: Not enough shared memory");
  }
  cudaFuncSetAttribute(cuda_gen_loops_normal,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

  // Graph matrix data in GPU constant memory

  std::vector<statenum_t> graph_buffer(graph_buffer_size, 0);
  for (unsigned i = 1; i <= graph.numstates; ++i) {
    for (unsigned j = 0; j < graph.outdegree.at(i); ++j) {
      graph_buffer.at((i - 1) * graphcols + j) = graph.outmatrix.at(i).at(j);
    }
    if (alg == 1) {
      graph_buffer.at((i - 1) * graphcols + graph.maxoutdegree) =
          graph.upstream_state(i);
    }
    if (alg == 2 || alg == 3) {
      graph_buffer.at((i - 1) * graphcols + graph.maxoutdegree) =
          graph.cyclenum.at(i);
    }
  }
  throw_on_cuda_error(
    cudaMemcpyToSymbol(graphmatrix_d, graph_buffer.data(),
        sizeof(statenum_t) * graph_buffer.size()),
    __FILE__, __LINE__
  );

  // Buffer to hold finished patterns (global memory)

  statenum_t* pb_d;
  throw_on_cuda_error(
    cudaMalloc(&pb_d, sizeof(statenum_t) * n_max * pattern_buffer_size),
    __FILE__, __LINE__
  );

  // Static global variables (global memory)

  uint8_t maxoutdegree_h = static_cast<uint8_t>(graph.maxoutdegree);
  uint16_t numstates_h = static_cast<uint16_t>(graph.numstates);
  uint32_t pattern_buffer_size_h = pattern_buffer_size;
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
    cudaMemcpyToSymbol(maxoutdegree_d, &maxoutdegree_h, sizeof(uint8_t)),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMemcpyToSymbol(numstates_d, &numstates_h, sizeof(uint16_t)),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMemcpyToSymbol(pattern_buffer_size_d, &pattern_buffer_size_h,
        sizeof(uint32_t)), __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMemcpyToSymbol(pattern_index_d, &pattern_index_h, sizeof(uint32_t)),
    __FILE__, __LINE__
  );

  // Arrays for WorkerInfo and WorkAssignmentCells (global memory)

  WorkerInfo* wi_d;
  WorkAssignmentCell* wa_d;
  throw_on_cuda_error(
    cudaMalloc(&wi_d, sizeof(WorkerInfo) * num_workers),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMalloc(&wa_d, sizeof(WorkAssignmentCell) * num_workers * n_max),
    __FILE__, __LINE__
  );

  std::vector<WorkerInfo> wi_h(num_workers);
  std::vector<WorkAssignmentCell> wa_h(num_workers * n_max);
  
  for (int id = 0; id < num_workers; ++id) {
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, wi_h, wa_h, graph, config, n_max);
    
      if (config.verboseflag) {
        erase_status_output();
        jpout << std::format("worker {} given work:\n  ", id)
              << wa << std::endl;
        print_status_output();
      }
    } else {
      wi_h.at(id).done = 1;
    }
  }

  while (true) {
    throw_on_cuda_error(
      cudaMemcpy(wi_d, wi_h.data(), sizeof(WorkerInfo) * num_workers,
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__
    );
    throw_on_cuda_error(
      cudaMemcpy(wa_d, wa_h.data(), sizeof(WorkAssignmentCell) * num_workers *
          n_max, cudaMemcpyHostToDevice),
      __FILE__, __LINE__
    );
    
    // ----------------------- launch kernel -----------------------

    switch (alg) {
      case 0:
        cuda_gen_loops_normal
            <<<num_blocks, num_threadsperblock, shared_memory_size>>>
            (pb_d, wi_d, wa_d, config.n_min, n_max, num_steps, !config.countflag);
        break;
      default:
        throw std::runtime_error("CUDA error: algorithm not implemented");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      throw std::runtime_error(std::format("CUDA Error: {}",
          cudaGetErrorString(err)));
    }

    cudaDeviceSynchronize();

    // ----------------- copy results back to host -----------------

    throw_on_cuda_error(
      cudaMemcpy(wi_h.data(), wi_d, sizeof(WorkerInfo) * num_workers,
          cudaMemcpyDeviceToHost),
      __FILE__, __LINE__
    );
    throw_on_cuda_error(
      cudaMemcpy(wa_h.data(), wa_d, sizeof(WorkAssignmentCell) * num_workers *
          n_max, cudaMemcpyDeviceToHost),
      __FILE__, __LINE__
    );

    // process updated worker records
    bool all_done = true;
    bool any_done = false;
    int num_working = 0;
    int num_idle = 0;

    for (int id = 0; id < num_workers; ++id) {
      if (wi_h.at(id).done) {
        any_done = true;
        ++num_idle;
      } else {
        all_done = false;
        ++num_working;
      }

      MessageW2C msg;
      msg.worker_id = id;
      msg.count.assign(n_max + 1, 0);
      for (unsigned j = 0; j < n_max; ++j) {
        msg.count.at(j + 1) = wa_h.at(id * n_max + j).count;
        wa_h.at(id * n_max + j).count = 0;
      }
      msg.nnodes = wi_h.at(id).nnodes;
      wi_h.at(id).nnodes = 0;
      record_data_from_message(msg);
    }

    //std::cout << std::format("{} working, {} idle\n", num_working, num_idle);

    // handle any patterns in the buffer
    process_pattern_buffer(pb_d, graph, pattern_buffer_size);

    if (Coordinator::stopping || all_done)
      break;

    if (any_done) {
      assign_new_jobs(config, context, wi_h, wa_h, graph, n_max, num_workers);
    }
  }

  // free GPU memory
  cudaFree(pb_d);
  cudaFree(wi_d);
  cudaFree(wa_d);

  // move unfinished work assignments back into `context.assignments`
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).done) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph, n_max);
      context.assignments.push_back(wa);
    }
  }
}
*/


//------------------------------------------------------------------------------
// Helper functions (newly created)
//------------------------------------------------------------------------------

// Initialize CUDA device and check properties.

cudaDeviceProp Coordinator::initialize_cuda_device() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  printf("Device Number: %d\n", 0);
  printf("  Device name: %s\n", prop.name);
  printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
  printf("  Total global memory (bytes): %ld\n", prop.totalGlobalMem);
  printf("  Total constant memory (bytes): %ld\n", prop.totalConstMem);
  printf("  Shared memory per block (bytes): %ld\n", prop.sharedMemPerBlock);
  printf("  Shared memory per block, maximum opt-in (bytes): %ld\n",
         prop.sharedMemPerBlockOptin);

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

// Check if the graph and work data fit in GPU memory.

void Coordinator::check_memory_limits(const Graph& graph, unsigned alg,
        unsigned num_threadsperblock) {
  const unsigned graphcols =
      (alg == 0) ? graph.maxoutdegree : graph.maxoutdegree + 1;
  const size_t graph_buffer_size =
      graph.numstates * graphcols * sizeof(statenum_t);

  if (graph_buffer_size > sizeof(graphmatrix_d)) {
    throw std::runtime_error("CUDA error: Juggling graph too large");
  }

  shared_memory_size = ((num_threadsperblock + 31) / 32) * (
      128 * (graph.numstates + 1) +  // used[]
      256 * n_max                    // WorkAssignentCell[]
  );

  std::cout << "num_workers = " << num_workers << '\n'
            << "shared memory size = " << shared_memory_size << " bytes\n"
            << "steps per kernel call = " << num_steps << std::endl;
  if (shared_memory_size > 99 * 1024) {
    throw std::runtime_error("CUDA error: Not enough shared memory");
  }
}

// Set up CUDA shared memory configuration.

void Coordinator::configure_cuda_shared_memory(const Graph& graph,
                                               unsigned num_threadsperblock) {
  cudaFuncSetAttribute(cuda_gen_loops_normal,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);
}

// Copy graph data to GPU constant memory.

void Coordinator::copy_graph_to_gpu(const Graph& graph, unsigned alg) {
  const unsigned graphcols =
      (alg == 0) ? graph.maxoutdegree : graph.maxoutdegree + 1;
  const size_t graph_buffer_size =
      graph.numstates * graphcols * sizeof(statenum_t);

  std::vector<statenum_t> graph_buffer(graph_buffer_size, 0);

  for (unsigned i = 1; i <= graph.numstates; ++i) {
    for (unsigned j = 0; j < graph.outdegree.at(i); ++j) {
      graph_buffer.at((i - 1) * graphcols + j) = graph.outmatrix.at(i).at(j);
    }
    if (alg == 1) {
      graph_buffer.at((i - 1) * graphcols + graph.maxoutdegree) =
          graph.upstream_state(i);
    }
    if (alg == 2 || alg == 3) {
      graph_buffer.at((i - 1) * graphcols + graph.maxoutdegree) =
          graph.cyclenum.at(i);
    }
  }

  throw_on_cuda_error(
      cudaMemcpyToSymbol(graphmatrix_d, graph_buffer.data(),
                         sizeof(statenum_t) * graph_buffer.size()),
      __FILE__, __LINE__);
}

// Allocate GPU memory for patterns, WorkerInfo, and WorkAssignmentCells.

void Coordinator::allocate_gpu_memory() {
  throw_on_cuda_error(
      cudaMalloc(&pb_d, sizeof(statenum_t) * n_max * pattern_buffer_size),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMalloc(&wi_d, sizeof(WorkerInfo) * num_workers),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMalloc(&wa_d, sizeof(WorkAssignmentCell) * num_workers * n_max),
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

// Load initial work assignments.

void Coordinator::load_initial_work_assignments(const Graph& graph,
      std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h)  {
  for (int id = 0; id < num_workers; ++id) {
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, wi_h, wa_h, graph, config, n_max);

      if (config.verboseflag) {
        erase_status_output();
        jpout << std::format("worker {} given work:\n  ", id)
              << wa << std::endl;
        print_status_output();
      }
    } else {
      wi_h.at(id).done = 1;
    }
  }
}

// Launch the appropriate CUDA kernel.

void Coordinator::launch_cuda_kernel(unsigned alg, unsigned num_blocks,
      unsigned num_threadsperblock) {
  switch (alg) {
    case 0:
      cuda_gen_loops_normal
        <<<num_blocks, num_threadsperblock, shared_memory_size>>>
        (pb_d, wi_d, wa_d, config.n_min, n_max, num_steps, !config.countflag);
      break;
    default:
      throw std::runtime_error("CUDA error: algorithm not implemented");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::format("CUDA Error: {}",
        cudaGetErrorString(err)));
  }

  cudaDeviceSynchronize();
}

// Process worker results and handle pattern buffer.

void Coordinator::process_worker_results(const Graph& graph,
      std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h) {
  int num_working = 0;
  int num_idle = 0;

  for (int id = 0; id < num_workers; ++id) {
    if (wi_h.at(id).done) {
      ++num_idle;
    } else {
      ++num_working;
    }

    MessageW2C msg;
    msg.worker_id = id;
    msg.count.assign(n_max + 1, 0);
    for (unsigned j = 0; j < n_max; ++j) {
      msg.count.at(j + 1) = wa_h.at(id * n_max + j).count;
      wa_h.at(id * n_max + j).count = 0;
    }
    msg.nnodes = wi_h.at(id).nnodes;
    wi_h.at(id).nnodes = 0;
    record_data_from_message(msg);
  }

  process_pattern_buffer(pb_d, graph, pattern_buffer_size);
}

// Clean up GPU memory.

void Coordinator::cleanup_gpu_memory() {
  cudaFree(pb_d);
  cudaFree(wi_d);
  cudaFree(wa_d);
}

// Gather unfinished work assignments.

void Coordinator::gather_unfinished_work_assignments(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h) {
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).done) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph, n_max);
      context.assignments.push_back(wa);
    }
  }
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void Coordinator::run_cuda() {
  const unsigned num_blocks = 1;
  const unsigned num_threadsperblock = 64;
  num_workers = num_blocks * num_threadsperblock;
  num_steps = 20000;
  pattern_buffer_size = 100000;
  
  // 1. Initialization

  (void)initialize_cuda_device();
  Graph graph = build_and_reduce_graph();
  unsigned alg = select_CUDA_search_algorithm(graph);
  check_memory_limits(graph, alg, num_threadsperblock);
  configure_cuda_shared_memory(graph, num_threadsperblock);
  copy_graph_to_gpu(graph, alg);
  allocate_gpu_memory();
  copy_static_vars_to_gpu(graph);

  std::vector<WorkerInfo> wi_h(num_workers); // Initialize wi_h
  std::vector<WorkAssignmentCell> wa_h(num_workers * n_max); // Initialize wa_h
  load_initial_work_assignments(graph, wi_h, wa_h);

  // 2. Main Loop

  while (true) {
    throw_on_cuda_error(
        cudaMemcpy(wi_d, wi_h.data(), sizeof(WorkerInfo) * num_workers,
            cudaMemcpyHostToDevice),
        __FILE__, __LINE__);
    throw_on_cuda_error(
        cudaMemcpy(wa_d, wa_h.data(),
            sizeof(WorkAssignmentCell) * num_workers * n_max,
            cudaMemcpyHostToDevice),
        __FILE__, __LINE__);

    launch_cuda_kernel(alg, num_blocks, num_threadsperblock);

    throw_on_cuda_error(
        cudaMemcpy(wi_h.data(), wi_d, sizeof(WorkerInfo) * num_workers,
                   cudaMemcpyDeviceToHost),
        __FILE__, __LINE__);
    throw_on_cuda_error(
        cudaMemcpy(wa_h.data(), wa_d,
            sizeof(WorkAssignmentCell) * num_workers * n_max,
            cudaMemcpyDeviceToHost),
        __FILE__, __LINE__);

    process_worker_results(graph, wi_h, wa_h);

    // Termination condition
    bool all_done = true;
    for (const auto &wi : wi_h) {
      if (!wi.done) {
        all_done = false;
        break;
      }
    }
    if (Coordinator::stopping || all_done) break;

    bool any_done = false;
    for (const auto &wi : wi_h) {
      if (wi.done) {
        any_done = true;
        break;
      }
    }

    if (any_done) {
      assign_new_jobs(config, context, wi_h, wa_h, graph, n_max, num_workers);
    }
  }

  // 3. Cleanup

  cleanup_gpu_memory();
  gather_unfinished_work_assignments(graph, wi_h, wa_h);
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// choose a search algorithm to use

unsigned Coordinator::select_CUDA_search_algorithm(const Graph& graph) const {
  unsigned max_possible = (config.mode == SearchConfig::RunMode::SUPER_SEARCH)
      ? graph.superprime_period_bound(config.shiftlimit)
      : graph.prime_period_bound();

  unsigned alg = -1;
  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    if (config.graphmode == SearchConfig::GraphMode::FULL_GRAPH &&
        static_cast<double>(config.n_min) >
        0.66 * static_cast<double>(max_possible)) {
      // the overhead of marking is only worth it for long-period patterns
      alg = 0;  // 1;
    } else if (config.countflag) {
      alg = 0;
    } else {
      alg = 0;
    }
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    if (config.shiftlimit == 0) {
      alg = 3;
    } else {
      alg = 2;
    }
  }

  if (config.verboseflag) {
    static const std::vector<std::string> algs = {
      "cuda_gen_loops_normal()",
      "cuda_gen_loops_normal_marking()",
      "cuda_gen_loops_super()",
      "cuda_gen_loops_super0()",
    };
    jpout << std::format("selected algorithm {}", algs.at(alg)) << std::endl;
  }

  return alg;
}

// Handle CUDA errors by throwing a `std::runtime_error` exception with a
// relevant error message.

void throw_on_cuda_error(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << " in file "
       << file << " at line " << line;
    throw std::runtime_error(ss.str());
  }
}

// Load a work assignment into a worker's slot in the `WorkerInfo` and
// `WorkAssignmentCell` arrays.

void load_work_assignment(const unsigned id, const WorkAssignment& wa,
    std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h,
    const Graph& graph, const SearchConfig& config, unsigned n_max) {
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
  wi_h.at(id).done = 0;

  // set up WorkAssignmentCells

  for (unsigned i = 0; i < n_max; ++i) {
    wa_h.at(id * n_max + i).count = 0;
  }

  // default if `wa.partial_pattern` is empty
  wa_h.at(id * n_max).col = 0;
  wa_h.at(id * n_max).col_limit = static_cast<uint8_t>(graph.maxoutdegree);
  wa_h.at(id * n_max).from_state = start_state;

  unsigned from_state = start_state;

  for (unsigned i = 0; i < wa.partial_pattern.size(); ++i) {
    const unsigned tv = wa.partial_pattern.at(i);
    unsigned to_state = 0;

    for (unsigned j = 0; j < graph.outdegree.at(from_state); ++j) {
      if (graph.outthrowval.at(from_state).at(j) != tv)
        continue;

      to_state = graph.outmatrix.at(from_state).at(j);

      wa_h.at(id * n_max + i).col = static_cast<uint8_t>(j);
      wa_h.at(id * n_max + i).col_limit = (i < wa.root_pos ?
          static_cast<uint8_t>(j + 1) :
          static_cast<uint8_t>(graph.maxoutdegree));

      wa_h.at(id * n_max + i + 1).col = 0;
      wa_h.at(id * n_max + i + 1).col_limit =
          static_cast<uint8_t>(graph.maxoutdegree);
      wa_h.at(id * n_max + i + 1).from_state = to_state;
      break;
    }
    if (to_state == 0) {
      throw std::runtime_error("CUDA error: problem loading work assignment");
    }

    from_state = to_state;
  }

  // fix `col` and `col_limit` at position `root_pos`
  if (wa.root_throwval_options.size() > 0) {
    statenum_t from_state = wa_h.at(id * n_max + wa.root_pos).from_state;
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

    wa_h.at(id * n_max + wa.root_pos).col = col;
    wa_h.at(id * n_max + wa.root_pos).col_limit = col_limit;
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

WorkAssignment read_work_assignment(unsigned id, std::vector<WorkerInfo>& wi_h,
      std::vector<WorkAssignmentCell>& wa_h, const Graph& graph,
      unsigned n_max) {
  WorkAssignment wa;

  wa.start_state = wi_h.at(id).start_state;
  wa.end_state = wi_h.at(id).end_state;

  bool root_pos_found = false;

  for (unsigned i = 0; i <= wi_h.at(id).pos; ++i) {
    const unsigned from_state = wa_h.at(id * n_max + i).from_state;
    unsigned col = wa_h.at(id * n_max + i).col;
    const unsigned col_limit = std::min(graph.outdegree.at(from_state),
                  static_cast<unsigned>(wa_h.at(id * n_max + i).col_limit));

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

// Process the pattern buffer, copying any patterns to `context` and printing
// them to the console if needed. Then clear the buffer.
//
// In the event of a buffer overflow, throw a `std::runtime_error` exception
// with a relevant error message.

void Coordinator::process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size) {
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

// Assign new jobs to idle workers

void assign_new_jobs(const SearchConfig& config, SearchContext& context, 
    std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h,
    const Graph& graph, unsigned n_max, unsigned num_workers) {

  // sort the running work assignments to find the best ones to split
  std::vector<WorkAssignmentLine> sorted_assignments;
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).done) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph, n_max);
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
    if (!wi_h.at(id).done)
      continue;

    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();
      load_work_assignment(id, wa, wi_h, wa_h, graph, config, n_max);
      continue;
    }
  
    // split one of the running jobs and give it to worker `id`

    bool success = false;
    while (!success) {
      if (index == sorted_assignments.size())
        break;

      WorkAssignmentLine& wal = sorted_assignments.at(index);
      WorkAssignment wa = wal.wa;

      /*
      std::cout << std::format("worker {} went idle\n", id);
      std::cout << std::format("stealing from worker {}\n", wal.id);
      std::cout << "work before:\n" << wa << '\n';
      */
      try {
        WorkAssignment wa2 = wa.split(graph, config.split_alg);
        load_work_assignment(wal.id, wa, wi_h, wa_h, graph, config, n_max);
        load_work_assignment(id, wa2, wi_h, wa_h, graph, config, n_max);
        
        /*
        std::cout << "work after:\n" << wa << '\n';
        std::cout << std::format("new work for worker {}:\n", id) << wa2 << '\n';
        */
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
