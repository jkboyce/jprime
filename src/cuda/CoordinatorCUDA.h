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
#include "WorkSpace.h"
#include "CudaTypes.h"

#include <cuda_runtime.h>


class CoordinatorCUDA : public Coordinator, public WorkSpace {
 public:
  CoordinatorCUDA(SearchConfig& config, SearchContext& context,
    std::ostream& jpout);

 protected:
  Graph graph;

  // pinned memory blocks in host
  WorkerInfo* wi_h[2] = { nullptr, nullptr };
  ThreadStorageWorkCell* wc_h[2] = { nullptr, nullptr };
  uint32_t* pattern_count_h = nullptr;  // if needed
  statenum_t* pb_h = nullptr;  // if needed

  // optimizing memory copies during startup
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
  void build_and_reduce_graph();
  CudaAlgorithm select_cuda_search_algorithm();
  std::vector<statenum_t> make_graph_buffer(CudaAlgorithm alg);
  CudaRuntimeParams find_runtime_params(const cudaDeviceProp& prop,
    CudaAlgorithm alg);
  size_t calc_shared_memory_size(CudaAlgorithm alg, unsigned n_max,
    const CudaRuntimeParams& p);
  void allocate_memory(CudaAlgorithm alg, const CudaRuntimeParams& params,
    const std::vector<statenum_t>& graph_buffer, CudaMemoryPointers& ptrs);
  void copy_graph_to_gpu(const std::vector<statenum_t>& graph_buffer,
    const CudaMemoryPointers& ptrs);
  void copy_static_vars_to_gpu(const CudaRuntimeParams& params,
    const CudaMemoryPointers& ptrs);

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
    const uint32_t pattern_buffer_size);
  void record_working_time(double kernel_time, double host_time,
    unsigned idle_before, unsigned idle_after, uint64_t cycles_startup,
    uint64_t cycles_run);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles,
    uint64_t last_cycles_startup, double kernel_time, double host_time,
    unsigned idle_start, unsigned idle_after, unsigned next_idle_start,
    uint32_t pattern_count, CudaRuntimeParams p);

  // cleanup
  void gather_unfinished_work_assignments();
  void cleanup(CudaMemoryPointers& ptrs);

  // manage work assignments
  void load_initial_work_assignments();
  void load_work_assignment(unsigned bank, const unsigned id,
    WorkAssignment& wa);
  WorkAssignment read_work_assignment(unsigned bank, unsigned id);
  unsigned assign_new_jobs(unsigned bank, unsigned idle_before_b,
    const CudaWorkerSummary& summary, unsigned idle_before_a);

  // summarization and status display
  CudaWorkerSummary summarize_worker_status(unsigned bank);
  CudaWorkerSummary summarize_all_jobs(const CudaWorkerSummary& a,
    const CudaWorkerSummary& b);
  void do_status_display(const CudaWorkerSummary& summary,
    const CudaWorkerSummary& last_summary, double kernel_time,
    double host_time);

  // helper
  ThreadStorageWorkCell& workcell(unsigned bank, unsigned id, unsigned pos);
  const ThreadStorageWorkCell& workcell(unsigned bank, unsigned id,
    unsigned pos) const;

  void throw_on_cuda_error(cudaError_t code, const char *file, int line);

  // WorkSpace methods
  virtual const Graph& get_graph() const override;
  virtual void set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state) override;
  virtual std::tuple<unsigned, unsigned, unsigned> get_cell(unsigned slot,
    unsigned index) const override;
  virtual void set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos) override;
  virtual std::tuple<unsigned, unsigned, int> get_info(unsigned slot) const
    override;
};

void CUDART_CB record_kernel_completion(void* data);

#endif
