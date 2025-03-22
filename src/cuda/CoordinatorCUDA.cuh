//
// CoordinatorCUDA.cuh
//
// Coordinator that executes the search on a CUDA GPU.
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
#include "CudaTypes.h"

#include <cuda_runtime.h>


class CoordinatorCUDA : public Coordinator {
 public:
  CoordinatorCUDA(SearchConfig& config, SearchContext& context,
    std::ostream& jpout);

 protected:
  // memory blocks in GPU global memory
  statenum_t* pb_d = nullptr;  // if needed
  WorkerInfo* wi_d = nullptr;
  ThreadStorageWorkCell* wc_d = nullptr;
  statenum_t* graphmatrix_d = nullptr;  // if needed
  uint32_t* used_d = nullptr;  // if needed

  // memory blocks in host memory
  WorkerInfo* wi_h = nullptr;
  ThreadStorageWorkCell* wc_h = nullptr;
  unsigned max_active_idx = 0;

  // timing parameters specific to GPU
  double total_kernel_time = 0;
  double total_host_time = 0;

 protected:
  // live status display
  std::vector<unsigned> longest_by_startstate_ever;
  std::vector<unsigned> longest_by_startstate_current;

 protected:
  virtual void run_search() override;

  // setup
  cudaDeviceProp initialize_cuda_device();
  Graph build_and_reduce_graph();
  CudaAlgorithm select_cuda_search_algorithm(const Graph& graph);
  std::vector<statenum_t> make_graph_buffer(const Graph& graph,
    CudaAlgorithm alg);
  CudaRuntimeParams find_runtime_params(const cudaDeviceProp& prop,
    CudaAlgorithm alg, const Graph& graph);
  size_t calc_shared_memory_size(CudaAlgorithm alg, const Graph& graph,
    unsigned n_max, const CudaRuntimeParams& p);
  void configure_cuda_shared_memory(const CudaRuntimeParams& params);
  void allocate_memory(CudaAlgorithm alg, const CudaRuntimeParams& params,
    const std::vector<statenum_t>& graph_buffer, const Graph& graph);
  void copy_graph_to_gpu(const std::vector<statenum_t>& graph_buffer);
  void copy_static_vars_to_gpu(const CudaRuntimeParams& params,
    const Graph& graph);

  // main loop
  void copy_worker_data_to_gpu(bool startup, unsigned max_idx);
  void launch_cuda_kernel(const CudaRuntimeParams& params, CudaAlgorithm alg,
    unsigned cycles);
  void copy_worker_data_from_gpu(unsigned max_idx);
  void process_worker_counters();
  uint32_t process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size);
  void record_working_time(double host_time, double kernel_time,
    unsigned idle_before, unsigned idle_after);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles, double host_time,
    double kernel_time, unsigned idle_start, unsigned idle_end,
    uint32_t pattern_count, CudaRuntimeParams p);

  // cleanup
  void cleanup_memory();
  void gather_unfinished_work_assignments(const Graph& graph);

  // manage work assignments
  void load_initial_work_assignments(const Graph& graph);
  void load_work_assignment(const unsigned id, const WorkAssignment& wa,
    const Graph& graph);
  WorkAssignment read_work_assignment(unsigned id, const Graph& graph);
  unsigned assign_new_jobs(const CudaWorkerSummary& summary,
    const Graph& graph);

  // summarization and status display
  CudaWorkerSummary summarize_worker_status(const Graph& graph);
  void do_status_display(const CudaWorkerSummary& summary,
    const CudaWorkerSummary& last_summary, double host_time,
    double kernel_time);

  // helper
  ThreadStorageWorkCell& workcell(unsigned id, unsigned pos);
  void throw_on_cuda_error(cudaError_t code, const char *file, int line);
};

#endif
