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


/*
Job splitting must be done by the host, so we execute for a defined period of
time in the GPU and then pause to give new jobs to idle workers.

The basic loop is:
- run the GPU kernel
- download the data from the GPU
- process the data (patterns found, pattern counts, etc.)
- split jobs and give jobs to idle workers
- copy the data back to the GPU

To increase efficiency, we use two separate banks of worker jobs and alternate
between them, so that while bankA is executing we do host processing of bankB
concurrently. Most newer GPUs support asynchronous copying of data between
device and host while kernels are executing.
*/

class CoordinatorCUDA : public Coordinator, public WorkSpace {
 public:
  CoordinatorCUDA(SearchConfig& config, SearchContext& context,
    std::ostream& jpout);

 protected:
  // set up during initialization
  cudaDeviceProp prop;
  cudaStream_t stream[2];  // distinct CUDA stream for each job bank
  CudaAlgorithm alg;
  std::vector<statenum_t> graph_buffer;
  CudaRuntimeParams params;
  CudaMemoryPointers ptrs;

  // pinned memory blocks in host
  WorkerInfo* wi_h[2] = { nullptr, nullptr };  // workerinfo arrays
  ThreadStorageWorkCell* wc_h[2] = { nullptr, nullptr };  // workcell arrays
  uint32_t* pattern_count_h = nullptr;  // pattern count, if needed
  statenum_t* pb_h = nullptr;  // pattern buffer, if needed

  // worker summaries for two banks of jobs
  CudaWorkerSummary summary_before[2];
  CudaWorkerSummary summary_after[2];
  unsigned max_active_idx[2] = { 0, 0 };
  bool warmed_up[2] = { false, false };

  // timing
  jptimer_t before_kernel[2];
  jptimer_t after_kernel[2];
  jptimer_t after_host[2];
  double total_kernel_time = 0;
  double total_host_time = 0;

  // live status display
  std::vector<unsigned> longest_by_startstate_ever;
  std::vector<unsigned> longest_by_startstate_current;
  jptimer_t last_status_time;
  uint64_t njobs = 0;
  uint64_t last_njobs = 0;
  uint64_t last_nnodes = 0;
  uint64_t last_ntotal = 0;

 protected:
  virtual void run_search() override;

  // setup
  void initialize();
  cudaDeviceProp initialize_cuda_device();
  CudaAlgorithm select_cuda_search_algorithm();
  std::vector<statenum_t> make_graph_buffer();
  CudaRuntimeParams find_runtime_params();
  size_t calc_shared_memory_size(unsigned n_max, const CudaRuntimeParams& p);
  void allocate_memory();
  void copy_graph_to_gpu();
  void copy_static_vars_to_gpu();

  // main loop
  void skip_unusable_startstates(unsigned bank);
  void copy_worker_data_to_gpu(unsigned bank, bool startup = false);
  void launch_cuda_kernel(unsigned bank, uint64_t cycles);
  void copy_worker_data_from_gpu(unsigned bank);
  void process_worker_counters(unsigned bank);
  uint32_t process_pattern_buffer(unsigned bank);
  void record_working_time(unsigned bank, double kernel_time, double host_time,
    uint64_t cycles_run);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles, unsigned bank,
    double kernel_time, double host_time, unsigned idle_start,
    uint32_t pattern_count);

  // cleanup
  void gather_unfinished_work_assignments();
  void cleanup();

  // summarization and status display
  CudaWorkerSummary summarize_worker_status(unsigned bank);
  CudaWorkerSummary summarize_all_jobs(const CudaWorkerSummary& a,
    const CudaWorkerSummary& b);
  void do_status_display(unsigned bankB, double kernel_time, double host_time);

  // manage work assignments
  void load_initial_work_assignments();
  void load_work_assignment(unsigned bank, const unsigned id,
    const WorkAssignment& wa);
  WorkAssignment read_work_assignment(unsigned bank, unsigned id);
  void assign_new_jobs(unsigned bankB);

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

void CUDART_CB record_kernel_completion_time(void* data);

#endif
