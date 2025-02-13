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

#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>


// Helper function to handle CUDA errors.

void throw_on_cuda_error(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << " in file "
       << file << " at line " << line;
    throw std::runtime_error(ss.str());
  }
}


using statenum_t = uint16_t;


struct GraphInfo {
  uint16_t maxoutdegree;
  uint16_t numstates;
  uint32_t pattern_buffer_size;
};


struct WorkerInfo {
  statenum_t start_state = 0;  // current value of `start_state` (input/output)
  statenum_t end_state = 0;  // highest value of `start_state` (input)
  uint16_t pos = 0;  // position in WorkAssignmentCell array (input/output)
  uint64_t nnodes = 0;  // number of nodes completed (output)
  uint16_t done = 0;  // 1 if worker is done, 0 otherwise (output)
};


struct WorkAssignmentCell {
  uint16_t col = 0;
  uint16_t col_limit = 0;
  statenum_t from_state = 0;
  statenum_t to_state = 0;
  uint32_t count = 0;  // output
};


// GPU constant memory
//
// Every NVIDIA GPU from capability 5.0 through 12.0 has 64 KB of constant
// memory. This is where we place the juggling graph data.

__device__ __constant__ GraphInfo graphinfo_d;
__device__ __constant__ statenum_t graphmatrix_d[65535 / sizeof(statenum_t)
                                                 - sizeof(GraphInfo)];


// static global variables in device memory

__device__ uint32_t pattern_index_d = 0;



__global__ void cuda_gen_loops_normal(statenum_t* patterns_d, WorkerInfo* wi_d,
        WorkAssignmentCell* wa_d, unsigned n_min, unsigned n_max) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  /*
  printf("worker %d: numstates = %d\n", i, graphinfo_d.numstates);
  //statenum_t* gm = &graphmatrix_d;
  printf("worker %d: state 40 graphmatrix = ", i);
  for (unsigned j = 0; j < graphinfo_d.maxoutdegree + 1; ++j) {
    printf("%d ", graphmatrix_d[(40 - 1) * (graphinfo_d.maxoutdegree + 1) + j]);
  }
  printf("\n");
  printf("worker %d: pattern buffer size = %d\n", i, graphinfo_d.pattern_buffer_size);
  printf("worker %d: n_max = %d\n", i, n_max);
  */

  if (wi_d[i].done) {
    return;
  }
  int pos = wi_d[i].pos;
  uint64_t nnodes = 0;

  __shared__ uint8_t used[100];
  for (int j = 0; j < 100; ++j) {
    used[j] = 0;
  }
  for (int j = 0; j <= n_max; ++j) {
    wa_d[j].count = 0;
  }

  // Note that beat 0 is stored at index 1 in the `WorkAssignmentCell` array.
  // We do this to provide a guard since wa_d[0].col is modified at the end of
  // the search.
  WorkAssignmentCell* ss = &wa_d[pos + 1];

  statenum_t st_state = wi_d[i].start_state;

  // main search loop
  while (pos >= 0) {
    // begin with any necessary cleanup from previous marking operations
    if (ss->to_state != 0) {
      used[ss->to_state] = 0;
      ss->to_state = 0;
    }

    skip_unmarking:
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      --pos;
      --ss;
      ++ss->col;
      ++nnodes;
      continue;
    }

    const statenum_t to_state = graphmatrix_d[(ss->from_state - 1) *
          (graphinfo_d.maxoutdegree + 1) + ss->col];
    if (to_state < st_state) {
      --pos;
      --ss;
      ++ss->col;
      ++nnodes;
      continue;
    }
    
    if (to_state == st_state) {
      // found a valid pattern
      ++ss->count;
      if (pos + 1 >= n_min) {
        const uint32_t idx = atomicAdd(&pattern_index_d, 1);
        if (idx < graphinfo_d.pattern_buffer_size) {
          for (int j = 0; j <= pos; ++j) {
            patterns_d[idx * n_max + j] = wa_d[j + 1].from_state;
          }
          if (pos + 1 < n_max) {
            patterns_d[idx * n_max + pos + 1] = 0;
          }
        }
      }

      ++ss->col;
      goto skip_unmarking;
    }

    if (used[to_state]) {
      ++ss->col;
      goto skip_unmarking;
    }

    if (pos + 1 == n_max) {
      ++ss->col;
      goto skip_unmarking;
    }

    // current throw is valid, so advance to next beat
    used[to_state] = 1;
    ss->to_state = to_state;
    ++pos;
    ++ss;
    ss->col = 0;
    ss->col_limit = graphinfo_d.maxoutdegree;
    ss->from_state = to_state;
    ss->to_state = 0;
    goto skip_unmarking;
  }

  ++pos;
  wi_d[i].pos = pos;
  wi_d[i].nnodes = nnodes;
  wi_d[i].done = 1;
}


// jprime 3 8 -g -count
// runtime = 0.2116 sec (226.6M nodes/sec)
//
// jprime 3 8 -g -count -cuda
// runtime = 18.7906 sec



// in kernel call, pass in:
//  - pointer to WorkerInfo array (global memory)
//          WorkerInfo[NUM_WORKERS]
//  - pointer to WorkAssignment array (global memory)
//          WorkAssignmentCell[NUM_WORKERS * (n_max + 1)]
//  - pointer to PatternBuffer (global memory)
//          2 * uint32_t + statenum_t[PATTERN_BUFFER_SIZE * n_max]
//  - n_max

// graph data:
//  - maxoutdegree (uint16_t)
//  - numstates (uint16_t)
//  - array of statenum_t[numstates * (maxoutdegree + 1)]


// Run the search on a CUDA-enabled GPU.
//
// In the event of an error, throw a `std::runtime_error` exception with a
// relevant error message.

void Coordinator::run_cuda() {
  unsigned num_workers = 1;
  unsigned pattern_buffer_size = 100000;

  // build juggling graph

  Graph graph = {config.b, config.h, config.xarray,
      config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH ?
      config.n_min : 0};
  graph.build_graph();
  // TODO: call customize_graph() here
  graph.reduce_graph();

  // will graph fit into GPU constant memory?
  size_t graph_buffer_size = graph.numstates * (graph.maxoutdegree + 1);
  if (graph_buffer_size > sizeof(graphmatrix_d)) {
    throw std::runtime_error("CUDA error: Juggling graph too large");
  }

  // GraphInfo in GPU constant memory

  GraphInfo gi_h = {static_cast<uint16_t>(graph.maxoutdegree),
                    static_cast<uint16_t>(graph.numstates),
                    static_cast<uint32_t>(pattern_buffer_size)};
  throw_on_cuda_error(
    cudaMemcpyToSymbol(graphinfo_d, &gi_h, sizeof(GraphInfo)),
    __FILE__, __LINE__
  );
  jpout << "host: numstates = " << graph.numstates << "\n";

  // Graph matrix data in GPU constant memory

  std::vector<statenum_t> graph_buffer(graph_buffer_size, 0);
  for (unsigned i = 1; i <= graph.numstates; ++i) {
    for (unsigned j = 0; j < graph.outdegree.at(i); ++j) {
      graph_buffer.at((i - 1) * (graph.maxoutdegree + 1) + j) =
          graph.outmatrix.at(i).at(j);
    }
    graph_buffer.at((i - 1) * (graph.maxoutdegree + 1) + graph.maxoutdegree) =
        graph.upstream_state(i);
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

  // Arrays for WorkerInfo and WorkAssignmentCells (global memory)

  WorkerInfo* wi_d;
  WorkAssignmentCell* wa_d;
  throw_on_cuda_error(
    cudaMalloc(&wi_d, sizeof(WorkerInfo) * num_workers),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMalloc(&wa_d, sizeof(WorkAssignmentCell) * num_workers * (n_max + 1)),
    __FILE__, __LINE__
  );

  std::vector<WorkerInfo> wi_h(num_workers);
  std::vector<WorkAssignmentCell> wa_h(num_workers * (n_max + 1));
  
  // TODO: initialize WorkerInfo and WorkAssignmentCell arrays here
  // initialize a simple ground state search
  wi_h.at(0).start_state = 1;
  wi_h.at(0).end_state = 1;

  wa_h.at(1).col = 0;
  wa_h.at(1).col_limit = gi_h.maxoutdegree;
  wa_h.at(1).from_state = 1;
  wa_h.at(1).to_state = 0;
  wa_h.at(1).count = 0;

  throw_on_cuda_error(
    cudaMemcpy(wi_d, wi_h.data(), sizeof(WorkerInfo) * num_workers,
        cudaMemcpyHostToDevice),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMemcpy(wa_d, wa_h.data(), sizeof(WorkAssignmentCell) * num_workers *
        (n_max + 1), cudaMemcpyHostToDevice),
    __FILE__, __LINE__
  );
  
  // ----------------------- launch kernel -----------------------

  cuda_gen_loops_normal<<<1, num_workers>>>(pb_d, wi_d, wa_d, config.n_min, n_max);
  cudaDeviceSynchronize();

  // ----------------- copy results back to host -----------------

  throw_on_cuda_error(
    cudaMemcpy(wi_h.data(), wi_d, sizeof(WorkerInfo) * num_workers,
        cudaMemcpyDeviceToHost),
    __FILE__, __LINE__
  );
  throw_on_cuda_error(
    cudaMemcpy(wa_h.data(), wa_d, sizeof(WorkAssignmentCell) * num_workers *
        (n_max + 1), cudaMemcpyDeviceToHost),
    __FILE__, __LINE__
  );

  for (int i = 0; i < num_workers; ++i) {
    MessageW2C msg;
    msg.worker_id = i;
    msg.count.assign(n_max + 1, 0);
    for (unsigned j = 0; j <= n_max; ++j) {
      msg.count.at(j) = wa_h.at(i * (n_max + 1) + j).count;
    }
    msg.nnodes = wi_h.at(i).nnodes;
    record_data_from_message(msg);
  }

  // patterns in pattern buffer
  process_pattern_buffer(pb_d, graph, pattern_buffer_size);

  // free GPU memory
  cudaFree(pb_d);
  cudaFree(wi_d);
  cudaFree(wa_d);
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
  if (pattern_count > pattern_buffer_size) {
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

  throw_on_cuda_error(
    cudaMemset(&pattern_index_d, 0, sizeof(uint32_t)),
    __FILE__, __LINE__
  );
}
