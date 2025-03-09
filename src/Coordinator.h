//
// Coordinator.h
//
// Coordinator that manages the overall search.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATOR_H_
#define JPRIME_COORDINATOR_H_

#include "Messages.h"
#include "SearchConfig.h"
#include "SearchContext.h"
#include "Worker.h"
#include "Graph.h"

//#ifdef CUDA_ENABLED
#include "CoordinatorCUDA.cuh"
//#endif

#include <queue>
#include <mutex>
#include <list>
#include <set>
#include <vector>
#include <string>
#include <thread>
#include <memory>
#include <csignal>


class Coordinator {
 public:
  Coordinator(const SearchConfig& config, SearchContext& context,
      std::ostream& jpout);
  Coordinator() = delete;

 public:
  std::queue<MessageW2C> inbox;
  std::mutex inbox_lock;

 private:
  const SearchConfig& config;
  SearchContext& context;
  std::ostream& jpout;  // all console output goes here
  unsigned n_max = 0;  // max pattern period to find

  // workers
  std::vector<std::unique_ptr<Worker>> worker;
  std::vector<std::unique_ptr<std::thread>> worker_thread;
  std::set<unsigned> workers_idle;
  std::set<unsigned> workers_splitting;
  std::vector<unsigned> worker_startstate;
  std::vector<unsigned> worker_endstate;
  std::vector<unsigned> worker_rootpos;

  static volatile sig_atomic_t stopping;
  static constexpr unsigned MAX_STATES = 1000000u;  // memory limit

  // check inbox 10x more often than workers do
  static constexpr double NANOSECS_PER_INBOX_CHECK =
      1e8 * Worker::SECS_PER_INBOX_CHECK_TARGET;

  // live status display
  static constexpr double SECS_PER_STATUS = 1;
  static constexpr int WAITS_PER_STATUS = static_cast<int>(1e9 *
      SECS_PER_STATUS / NANOSECS_PER_INBOX_CHECK);
  static constexpr int STATUS_WIDTH = 55;
  unsigned stats_counter = 0;
  unsigned stats_received = 0;
  bool stats_printed = false;
  std::vector<std::string> worker_status;
  std::vector<std::vector<unsigned>> worker_options_left_start;
  std::vector<std::vector<unsigned>> worker_options_left_last;
  std::vector<unsigned> worker_longest_start;
  std::vector<unsigned> worker_longest_last;

 public:
  bool run();

 private:
  void calc_graph_size();
  bool passes_prechecks();
  double expected_patterns_at_maxperiod();
  static void signal_handler(int signum);
  void process_search_result(const MessageW2C& msg);
  void print_search_description() const;
  void print_results() const;
  void erase_status_output() const;
  void print_status_output();
  static std::string current_time_string();
  void record_data_from_message(const MessageW2C& msg);
  std::string make_worker_status(const MessageW2C& msg);

 // defined in CoordinatorCPU.cc

 private:
  void run_cpu();
  void message_worker(const MessageC2W& msg, unsigned worker_id) const;
  void give_assignments();
  void steal_work();
  unsigned find_stealing_target_mostremaining() const;
  void collect_stats();
  void process_inbox();
  void process_worker_idle(const MessageW2C& msg);
  void process_returned_work(const MessageW2C& msg);
  void process_returned_stats(const MessageW2C& msg);
  void process_worker_update(const MessageW2C& msg);
  void start_workers();
  void stop_workers();
  bool is_worker_idle(const unsigned id) const;
  bool is_worker_splitting(const unsigned id) const;


//#ifdef CUDA_ENABLED

 private:
  // memory blocks in GPU global memory
  statenum_t* pb_d = nullptr;  // if needed
  WorkerInfo* wi_d = nullptr;
  ThreadStorageWorkCell* wa_d = nullptr;
  statenum_t* graphmatrix_d = nullptr;  // if needed

  // GPU runtime parameters
  unsigned num_workers;
  double total_kernel_time = 0;
  double total_host_time = 0;

 private:
  void run_cuda();

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
    const std::vector<statenum_t>& graph_buffer);
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
  void process_worker_results(const Graph& graph,
    std::vector<WorkerInfo>& wi_h, std::vector<ThreadStorageWorkCell>& wa_h);
  void process_pattern_buffer(statenum_t* const pb_d,
    const Graph& graph, const uint32_t pattern_buffer_size);
  uint64_t calc_next_kernel_cycles(uint64_t last_cycles,
    std::chrono::time_point<std::chrono::system_clock> prev_after_kernel,
    std::chrono::time_point<std::chrono::system_clock> before_kernel,
    std::chrono::time_point<std::chrono::system_clock> after_kernel,
    unsigned num_done);
    
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

//#endif
};

#endif
