//
// CoordinatorCUDA.h
//
// Coordinator that executes the search on a CUDA GPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATORCUDA_H_
#define JPRIME_COORDINATORCUDA_H_

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
  // memory blocks in host memory
  WorkerInfo* wi_h[2] = { nullptr, nullptr };
  ThreadStorageWorkCell* wc_h[2] = { nullptr, nullptr };
  unsigned max_active_idx[2] = { 0, 0 };

  // timing parameters specific to GPU
  double total_kernel_time = 0;
  double total_host_time = 0;

  // live status display
  std::vector<unsigned> longest_by_startstate_ever;
  std::vector<unsigned> longest_by_startstate_current;
  jptimer_t last_display_time;

  // CUDA streams
  cudaStream_t stream[2];

  // timing
  jptimer_t before_kernel[2];
  jptimer_t after_kernel[2];
  jptimer_t after_host[2];

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
  void allocate_memory(CudaAlgorithm alg, const CudaRuntimeParams& params,
    const std::vector<statenum_t>& graph_buffer, const Graph& graph,
    CudaMemoryPointers& ptrs);
  void copy_graph_to_gpu(const std::vector<statenum_t>& graph_buffer,
    const CudaMemoryPointers& ptrs);
  void copy_static_vars_to_gpu(const CudaRuntimeParams& params,
    const Graph& graph, const CudaMemoryPointers& ptrs);

  // main loop
  void copy_worker_data_to_gpu(unsigned bank, const CudaMemoryPointers& ptrs,
    unsigned max_idx, bool startup);
  void launch_cuda_kernel(const CudaRuntimeParams& params,
    const CudaMemoryPointers& ptrs, CudaAlgorithm alg, unsigned bank,
    uint64_t cycles);
  void copy_worker_data_from_gpu(unsigned bank, const CudaMemoryPointers& ptrs,
    unsigned max_idx);
  void process_worker_counters(unsigned bank);
  uint32_t process_pattern_buffer(unsigned bank, const CudaMemoryPointers& ptrs,
    const Graph& graph,const uint32_t pattern_buffer_size);
  void record_working_time(double kernel_time, double host_time,
    unsigned idle_before, unsigned idle_after, uint64_t cycles_startup,
    uint64_t cycles_run);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles,
    uint64_t last_cycles_startup, double kernel_time, double host_time,
    unsigned idle_start, unsigned idle_after, unsigned next_idle_start,
    uint32_t pattern_count, CudaRuntimeParams p);

  // cleanup
  void gather_unfinished_work_assignments(const Graph& graph);
  void cleanup(CudaMemoryPointers& ptrs);

  // manage work assignments
  void load_initial_work_assignments(const Graph& graph);
  void load_work_assignment(unsigned bank, const unsigned id,
    const WorkAssignment& wa, const Graph& graph);
  WorkAssignment read_work_assignment(unsigned bank, unsigned id,
    const Graph& graph);
  unsigned assign_new_jobs(unsigned bank, const CudaWorkerSummary& summary,
    const Graph& graph);

  // summarization and status display
  CudaWorkerSummary summarize_worker_status(unsigned bank, const Graph& graph);
  CudaWorkerSummary summarize_all_jobs(const CudaWorkerSummary& a,
    const CudaWorkerSummary& b);
  void do_status_display(const CudaWorkerSummary& summary,
    const CudaWorkerSummary& last_summary, double kernel_time,
    double host_time);

  // helper
  ThreadStorageWorkCell& workcell(unsigned bank, unsigned id, unsigned pos);
  void throw_on_cuda_error(cudaError_t code, const char *file, int line);
};

void CUDART_CB record_kernel_completion(void* data);

#endif
