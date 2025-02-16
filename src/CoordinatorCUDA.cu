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
#include <format>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>


// ----------------------- Data types -----------------------

using statenum_t = uint16_t;


struct WorkerInfo {
  statenum_t start_state = 0;  // current value of `start_state` (input/output)
  statenum_t end_state = 0;  // highest value of `start_state` (input)
  uint16_t pos = 0;  // position in WorkAssignmentCell array (input/output)
  uint64_t nnodes = 0;  // number of nodes completed (output)
  uint16_t done = 1;  // 1 if worker is done, 0 otherwise (output)
};


struct WorkAssignmentCell {
  uint8_t col = 0;
  uint8_t col_limit = 0;
  statenum_t from_state = 0;
  uint32_t count = 0;  // output
};


// ----------------------- Function prototypes -----------------------

void throw_on_cuda_error(cudaError_t code, const char *file, int line);

void load_work_assignment(const unsigned id, const WorkAssignment& wa,
  std::vector<WorkerInfo>& wi_h, std::vector<WorkAssignmentCell>& wa_h,
  const Graph& graph, const SearchConfig& config, unsigned n_max);

WorkAssignment read_work_assignment(unsigned id, std::vector<WorkerInfo>& wi_h,
  std::vector<WorkAssignmentCell>& wa_h, const Graph& graph);

// ----------------------- GPU memory layout -----------------------

// GPU constant memory
//
// Every NVIDIA GPU from capability 5.0 through 12.0 has 64 KB of constant
// memory. This is where we place the juggling graph data.

__device__ __constant__ statenum_t graphmatrix_d[65536 / sizeof(statenum_t)];


// GPU global memory

__device__ uint8_t maxoutdegree_d;
__device__ uint16_t numstates_d;
__device__ uint32_t pattern_buffer_size_d;
__device__ uint32_t pattern_index_d = 0;


// ----------------------- GPU kernels -----------------------

__global__ void cuda_gen_loops_normal(statenum_t* const patterns_d,
        WorkerInfo* const wi_d, WorkAssignmentCell* const wa_d,
        const unsigned n_min, const unsigned n_max, const unsigned steps,
        const bool report) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (wi_d[i].done) {
    return;
  }

  statenum_t st_state = wi_d[i].start_state;
  int pos = wi_d[i].pos;
  uint64_t nnodes = wi_d[i].nnodes;
  const uint8_t outdegree = maxoutdegree_d;
  WorkAssignmentCell* ss = &wa_d[pos];

  // set up shared memory
  __shared__ uint8_t used[100];
  for (int j = 0; j <= numstates_d; ++j) {
    used[j] = 0;
  }
  for (int j = 1; j <= pos; ++j) {
    used[wa_d[j].from_state] = 1;
  }

  for (unsigned step = 0; ; ++step) {
    if (ss->col == ss->col_limit) {
      // beat is finished, go back to previous one
      used[ss->from_state] = 0;
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[i].end_state) {
          wi_d[i].done = 1;
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
      used[ss->from_state] = 0;
      ++nnodes;

      if (pos == 0) {
        if (st_state == wi_d[i].end_state) {
          wi_d[i].done = 1;
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
            patterns_d[idx * n_max + j] = wa_d[j].from_state;
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

    if (used[to_state]) {
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
    used[to_state] = 1;
  }

  wi_d[i].start_state = st_state;
  wi_d[i].pos = pos;
  wi_d[i].nnodes = nnodes;
}


// jprime 3 8 -g -count
// runtime = 0.2116 sec (226.6M nodes/sec)
//
// jprime 3 8 -g -count -cuda
// runtime = 18.7906 sec

// ----------------------- Execution entry point -----------------------

// Run the search on a CUDA-enabled GPU.
//
// In the event of an error, throw a `std::runtime_error` exception with a
// relevant error message.

void Coordinator::run_cuda() {
  const unsigned num_blocks = 1;
  const unsigned num_threadsperblock = 1;
  const unsigned num_workers = num_blocks * num_threadsperblock;
  const unsigned pattern_buffer_size = 100000;

  // build juggling graph

  Graph graph = {config.b, config.h, config.xarray,
      config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH ?
      config.n_min : 0};
  graph.build_graph();
  // TODO: call customize_graph() here
  graph.reduce_graph();

  // choose which search algorithm to use
  const unsigned alg = select_CUDA_search_algorithm(graph);

  // will graph fit into GPU constant memory?
  const unsigned graphcols = (alg < 2) ? graph.maxoutdegree :
                    graph.maxoutdegree + 1;
  const size_t graph_buffer_size = graph.numstates * graphcols;
  if (graph_buffer_size > sizeof(graphmatrix_d)) {
    throw std::runtime_error("CUDA error: Juggling graph too large");
  }

  // Graph matrix data in GPU constant memory

  std::vector<statenum_t> graph_buffer(graph_buffer_size, 0);
  for (unsigned i = 1; i <= graph.numstates; ++i) {
    for (unsigned j = 0; j < graph.outdegree.at(i); ++j) {
      graph_buffer.at((i - 1) * graphcols + j) = graph.outmatrix.at(i).at(j);
    }
    if (alg == 2) {
      graph_buffer.at((i - 1) * graphcols + graph.maxoutdegree) =
          graph.upstream_state(i);
    }
    if (alg == 3 || alg == 4) {
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
    cudaMemset(&pattern_index_d, 0, sizeof(uint32_t)), __FILE__, __LINE__
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
        cuda_gen_loops_normal<<<num_blocks, num_threadsperblock>>>
            (pb_d, wi_d, wa_d, config.n_min, n_max, 100000, !config.countflag);
        break;
      default:
        throw std::runtime_error("CUDA error: algorithm not implemented");
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

    bool all_done = true;

    for (int id = 0; id < num_workers; ++id) {
      if (!wi_h.at(id).done) {
        all_done = false;
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

    // handle patterns that have accumulated in the buffer
    process_pattern_buffer(pb_d, graph, pattern_buffer_size);

    if (Coordinator::stopping || all_done) {
      break;
    }
  }

  // free GPU memory
  cudaFree(pb_d);
  cudaFree(wi_d);
  cudaFree(wa_d);

  // move unfinished work assignments back into `context.assignments`
  for (unsigned id = 0; id < num_workers; ++id) {
    if (!wi_h.at(id).done) {
      WorkAssignment wa = read_work_assignment(id, wi_h, wa_h, graph);
      context.assignments.push_back(wa);
    }
  }

}

/*
jprime 3 8 -count -cuda

prime search for period: 1- (bound 49)
graph: 56 states, 7 shift cycles, 0 short cycles
11906414 patterns in range (11906414 seen, 49962563 nodes)
runtime = 15.9088 sec (3.1M nodes/sec)
*/

// ----------------------- Helper functions -----------------------

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
      alg = 1;
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
  wi_h.at(id).pos = wa.partial_pattern.size() == 0 ? 0 :
      wa.partial_pattern.size() - 1;
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

  // fix `col_limit` at position `root_pos`
  if (wa.root_throwval_options.size() > 0) {
    wa_h.at(id * n_max + wa.root_pos).col_limit =
        wa_h.at(id * n_max + wa.root_pos).col + 1 +
        static_cast<uint8_t>(wa.root_throwval_options.size());
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
      std::vector<WorkAssignmentCell>& wa_h, const Graph& graph) {
  WorkAssignment wa;

  wa.start_state = wi_h.at(id).start_state;
  wa.end_state = wi_h.at(id).end_state;

  bool root_pos_found = false;

  for (unsigned i = 0; i <= wi_h.at(id).pos; ++i) {
    const unsigned from_state = wa_h.at(id * graph.maxoutdegree + i).from_state;
    unsigned col = wa_h.at(id * graph.maxoutdegree + i).col;
    const unsigned col_limit = wa_h.at(id * graph.maxoutdegree + i).col_limit;

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
