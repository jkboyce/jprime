//
// CoordinatorCUDA.cuh
//
// Defines data types and classes for executing the search on a CUDA GPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATORCUDA_CUH_
#define JPRIME_COORDINATORCUDA_CUH_

#include "Coordinator.h"
#include "Graph.h"
#include "SearchConfig.h"
#include "SearchContext.h"
#include "WorkAssignment.h"

#include <chrono>
#include <cuda_runtime.h>


//------------------------------------------------------------------------------
// Data types
//------------------------------------------------------------------------------

// Type for holding state numbers

using statenum_t = uint16_t;


// Type for holding information about the state of a single worker

struct WorkerInfo {  // 16 bytes
  statenum_t start_state = 0;  // current value of `start_state` (input/output)
  statenum_t end_state = 0;  // highest value of `start_state` (input)
  uint16_t pos = 0;  // position in WorkAssignmentCell array (input/output)
  uint64_t nnodes = 0;  // number of nodes completed (output)
  uint16_t status = 1;  // bit 0 = is worker done, other bits unused
};


// Type for sorting WorkAssignments during splitting

struct WorkAssignmentLine {
  unsigned id;
  WorkAssignment wa;
};

// Storage for used[] bitarray, for 32 threads = 32 bits per thread. Each state
// in the graph maps onto a single bit.
//
// Data layout gives each thread its own bank in shared memory

struct ThreadStorageUsed {  // 128 bytes
  uint32_t used;
  uint32_t unused[31];
};

// Storage for a single work cell (single value of `pos`), for 32 threads =
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
  unsigned num_blocks = 1;
  unsigned num_threadsperblock = 32;
  size_t pattern_buffer_size = 1;
  size_t shared_memory_size = 0;
  bool used_in_shared = true;
  unsigned window_lower = 0;
  unsigned window_upper = 0;
};


//------------------------------------------------------------------------------
// Coordinator subclass
//------------------------------------------------------------------------------

class CoordinatorCUDA : public Coordinator {
 public:
  CoordinatorCUDA(const SearchConfig& config, SearchContext& context,
    std::ostream& jpout);

 protected:
  // memory blocks in GPU global memory
  statenum_t* pb_d = nullptr;  // if needed
  WorkerInfo* wi_d = nullptr;
  ThreadStorageWorkCell* wa_d = nullptr;
  statenum_t* graphmatrix_d = nullptr;  // if needed
  ThreadStorageUsed* used_d = nullptr;  // if needed

  // GPU runtime parameters
  unsigned num_workers;
  double total_kernel_time = 0;
  double total_host_time = 0;

 protected:
  virtual void run_search() override;

  // setup
  cudaDeviceProp initialize_cuda_device();
  Graph build_and_reduce_graph();
  CudaAlgorithm select_cuda_search_algorithm(const Graph& graph);
  std::vector<statenum_t> make_graph_buffer(const Graph& graph,
    CudaAlgorithm alg);
  CudaRuntimeParams find_runtime_params(const cudaDeviceProp& prop,
    CudaAlgorithm alg, unsigned num_states);
  size_t calc_shared_memory_size(CudaAlgorithm alg, unsigned num_states,
    unsigned n_max, const CudaRuntimeParams& p);
  void configure_cuda_shared_memory(const CudaRuntimeParams& params);
  void allocate_gpu_device_memory(const CudaRuntimeParams& params,
    const std::vector<statenum_t>& graph_buffer, unsigned num_states);
  void copy_graph_to_gpu(const std::vector<statenum_t>& graph_buffer);
  void copy_static_vars_to_gpu(const CudaRuntimeParams& params,
    const Graph& graph);

  // main loop
  void copy_worker_data_to_gpu(std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h);
  void launch_cuda_kernel(const CudaRuntimeParams& params, CudaAlgorithm alg,
    unsigned cycles);
  void copy_worker_data_from_gpu(std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h);
  void process_worker_results(std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h);
  uint32_t process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles,
    std::chrono::time_point<std::chrono::system_clock> prev_after_kernel,
    std::chrono::time_point<std::chrono::system_clock> before_kernel,
    std::chrono::time_point<std::chrono::system_clock> after_kernel,
    unsigned num_done, uint32_t pattern_count, CudaRuntimeParams params);
    
  // cleanup
  void cleanup_gpu_memory();
  void gather_unfinished_work_assignments(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h);

  // manage work assignments
  void load_initial_work_assignments(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h);
  void load_work_assignment(const unsigned id, const WorkAssignment& wa,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h,
    const Graph& graph);
  WorkAssignment read_work_assignment(unsigned id,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h,
    const Graph& graph);
  void assign_new_jobs(const Graph& graph, std::vector<WorkerInfo>& wi_h,
    std::vector<ThreadStorageWorkCell>& wa_h);
    
  // helper
  void throw_on_cuda_error(cudaError_t code, const char *file, int line);
};

#endif  // JPRIME_COORDINATORCUDA_CUH_
