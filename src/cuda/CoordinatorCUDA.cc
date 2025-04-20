//
// CoordinatorCUDA.cc
//
// Coordinator that executes the search on a CUDA GPU.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "CoordinatorCUDA.h"

#include <iostream>
#include <vector>
#include <format>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cassert>

// defined in CudaKernels.cu
void configure_cuda_shared_memory(const CudaRuntimeParams& p);
void get_gpu_static_pointers(CudaMemoryPointers& ptr);
void launch_kernel(const CudaRuntimeParams& p, const CudaMemoryPointers& ptrs,
  CudaAlgorithm alg, unsigned bank, uint64_t cycles, cudaStream_t stream);



CoordinatorCUDA::CoordinatorCUDA(SearchConfig& a, SearchContext& b,
    std::ostream& c) : Coordinator(a, b, c) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void CoordinatorCUDA::run_search() {
  const auto prop = initialize_cuda_device();
  build_and_reduce_graph();
  const auto alg = select_cuda_search_algorithm();
  const auto graph_buffer = make_graph_buffer(alg);
  const auto params = find_runtime_params(prop, alg);

  CudaMemoryPointers ptrs;
  get_gpu_static_pointers(ptrs);
  allocate_memory(alg, params, graph_buffer, ptrs);
  copy_graph_to_gpu(graph_buffer, ptrs);
  copy_static_vars_to_gpu(params, ptrs);
  configure_cuda_shared_memory(params);

  // timing setup
  const auto now = std::chrono::high_resolution_clock::now();
  uint64_t cycles[2];
  for (unsigned bank = 0; bank < 2; ++bank) {
    before_kernel[bank] = now;
    after_kernel[bank] = now;
    after_host[bank] = now;
    cycles[bank] = 1000000;
  }
  last_display_time = now;

  // worker setup
  load_initial_work_assignments();
  CudaWorkerSummary summary_before[2];
  CudaWorkerSummary summary_after[2];
  unsigned idle_before[2];
  for (unsigned bank = 0; bank < 2; ++bank) {
    summary_before[bank] = summarize_worker_status(bank);
    idle_before[bank] = summary_before[bank].workers_idle.size();
    copy_worker_data_to_gpu(bank, ptrs, max_active_idx[bank], true);
  }
  cudaDeviceSynchronize();

  for (unsigned run = 0; ; ++run) {
    const unsigned bankA = run % 2;
    const unsigned bankB = (run + 1) % 2;

    // start the workers for bankA
    before_kernel[bankA] = std::chrono::high_resolution_clock::now();
    if (idle_before[bankA] < config.num_threads) {
      launch_cuda_kernel(params, ptrs, alg, bankA, cycles[bankA]);
    }
    cudaLaunchHostFunc(stream[bankA], record_kernel_completion,
        &after_kernel[bankA]);

    // process results from the other bank (bankB)
    copy_worker_data_from_gpu(bankB, ptrs, max_active_idx[bankB]);
    cudaStreamSynchronize(stream[bankB]);
    process_worker_counters(bankB);
    const auto pattern_count = process_pattern_buffer(bankB, ptrs,
        params.pattern_buffer_size);
    summary_after[bankB] = summarize_worker_status(bankB);

    const auto kernel_time = calc_duration_secs(before_kernel[bankB],
        after_kernel[bankB]);
    const auto host_time = calc_duration_secs(after_kernel[bankB],
        after_host[bankA]);
    record_working_time(kernel_time, host_time, idle_before[bankB],
        summary_after[bankB].workers_idle.size(),
        summary_after[bankB].cycles_startup, cycles[bankB]);
    do_status_display(summary_after[bankB], summary_before[bankA], kernel_time,
        host_time);

    if (Coordinator::stopping) {
      process_worker_counters(bankA);  // node counts from splitting
      break;
    }
    if (summary_after[bankB].workers_idle.size() == config.num_threads &&
        idle_before[bankA] == config.num_threads &&
        context.assignments.size() == 0) {
      break;
    }

    // prepare bankB for its next run
    const auto prev_idle_before = idle_before[bankB];
    idle_before[bankB] = assign_new_jobs(bankB, idle_before[bankB],
        summary_after[bankB], idle_before[bankA]);
    copy_worker_data_to_gpu(bankB, ptrs, max_active_idx[bankB], false);
    if (prev_idle_before < config.num_threads) {
      cycles[bankB] = calc_next_kernel_cycles(cycles[bankB],
          summary_after[bankB].cycles_startup, kernel_time, host_time,
          prev_idle_before, summary_after[bankB].workers_idle.size(),
          idle_before[bankB], pattern_count, params);
    }
    summary_before[bankB] = summarize_worker_status(bankB);
    cudaStreamSynchronize(stream[bankB]);
    after_host[bankB] = std::chrono::high_resolution_clock::now();

    // wait for bankA workers to finish
    cudaStreamSynchronize(stream[bankA]);
  }

  gather_unfinished_work_assignments();
  cleanup(ptrs);

  if (config.verboseflag) {
    erase_status_output();
    jpout << "total kernel time = " << total_kernel_time
          << "\ntotal host time = " << total_host_time
          << '\n';
  }
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

// Initialize CUDA device and check properties.

cudaDeviceProp CoordinatorCUDA::initialize_cuda_device() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  for (int i = 0; i < 2; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  if (config.verboseflag) {
    jpout << "Device Number: " << 0
          << "\n  device name: " << prop.name
          << "\n  multiprocessor (MP) count: " << prop.multiProcessorCount
          << "\n  max threads per MP: " << prop.maxThreadsPerMultiProcessor
          << "\n  max threads per block: " << prop.maxThreadsPerBlock
          << "\n  async engine count: " << prop.asyncEngineCount
          << "\n  total global memory (bytes): " << prop.totalGlobalMem
          << "\n  total constant memory (bytes): " << prop.totalConstMem
          << "\n  shared memory per block (bytes): " << prop.sharedMemPerBlock
          << "\n  shared memory per block, maximum opt-in (bytes): "
          << prop.sharedMemPerBlockOptin << std::endl;
  }

  return prop;
}

// Build and reduce the juggling graph.

void CoordinatorCUDA::build_and_reduce_graph() {
  graph = {
    config.b,
    config.h,
    config.xarray,
    config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH
                     ? config.n_min : 0
  };
  graph.build_graph();
  customize_graph(graph);
  graph.reduce_graph();
}

// Choose a search algorithm to use.

CudaAlgorithm CoordinatorCUDA::select_cuda_search_algorithm() {
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
      alg = CudaAlgorithm::NORMAL;
    } else {
      alg = CudaAlgorithm::NORMAL;
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

std::vector<statenum_t> CoordinatorCUDA::make_graph_buffer(CudaAlgorithm alg) {
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
    if (alg == CudaAlgorithm::SUPER || alg == CudaAlgorithm::SUPER0) {
      graph_buffer.push_back(graph.cyclenum.at(i));
    }
  }

  return graph_buffer;
}

// Speedup as a function of number of warps per block, when the workcell[] array
// is placed in shared memory or global memory
//
// columns are {warps, shared memory speedup, global memory speedup}

const double throughput[33][3] = {
  {  0,  0.000, 0.000 },  // unused
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

// Determine an optimal runtime configuration for the GPU hardware available.

CudaRuntimeParams CoordinatorCUDA::find_runtime_params(
      const cudaDeviceProp& prop, CudaAlgorithm alg) {
  CudaRuntimeParams params;
  params.num_blocks = prop.multiProcessorCount;
  params.pattern_buffer_size = (config.countflag ? 0 :
    (prop.totalGlobalMem / 16) / sizeof(statenum_t) / n_max);

  // heuristic: see if used[] arrays for 10 warps will fit into shared memory;
  // if not then put into device memory
  params.num_threadsperblock = 32 * 10;
  params.used_in_shared = true;
  params.window_lower = params.window_upper = 0;
  size_t shared_mem = calc_shared_memory_size(alg, n_max, params);
  if (shared_mem > prop.sharedMemPerBlockOptin) {
    params.used_in_shared = false;
  }

  const auto access_fraction = build_access_model(graph.numstates);

  // consider each warp value in turn, and estimate throughput for each
  unsigned best_warps = 1;
  unsigned best_lower = 0;
  unsigned best_upper = 0;
  double best_throughput = -1;
  const int max_warps = prop.maxThreadsPerBlock / 32;

  for (int warps = 1; warps <= max_warps; ++warps) {
    // jpout << "calculating for " << warps << " warps:\n";

    // find the maximum window size that fits into shared memory
    unsigned lower = 0;
    unsigned upper = n_max;
    for (; upper != 0; --upper) {
      params.num_threadsperblock = 32 * warps;
      params.window_lower = lower;
      params.window_upper = upper;
      shared_mem = calc_shared_memory_size(alg, n_max, params);
      // jpout << "  upper = " << upper << ", mem = " << shared_mem << '\n';

      if (shared_mem <= prop.sharedMemPerBlockOptin)
        break;
    }
    if (upper == 0) {
      // used[] array is in shared, and too big for this many warps
      break;
    }

    // slide the window to maximize the area contained by it
    while (upper < n_max &&
          access_fraction.at(lower) < access_fraction.at(upper)) {
      ++lower;
      ++upper;
    }
    double S = 0;
    for (unsigned i = lower; i < upper; ++i) {
      S += access_fraction.at(i);
    }

    // estimate throughput
    const double throughput_est = (warps <= 32 ?
        throughput[warps][1] * S + throughput[warps][2] * (1 - S) :
        0.5 * warps * S + 0.25 * warps * (1 - S));
    /* jpout << "  window [" << lower << ',' << upper
          << "), S = " << S << ", T = " << throughput_avg << '\n'; */

    if (throughput_est > best_throughput + 0.25) {
      // jpout << "  new best warps: " << warps << '\n';
      best_throughput = throughput_est;
      best_warps = warps;
      best_lower = lower;
      best_upper = upper;
    }

    /*
    jpout << std::format("warps {}: window [{},{}) S = {}, throughput = {}\n",
          warps, lower, upper, S, throughput_avg);*/
  }

  params.num_threadsperblock = 32 * best_warps;
  params.window_lower = best_lower;
  params.window_upper = best_upper;
  params.shared_memory_size = calc_shared_memory_size(alg, n_max, params);
  config.num_threads = params.num_blocks * params.num_threadsperblock;

  params.n_min = config.n_min;
  params.n_max = n_max;
  params.report = !config.countflag;
  params.shiftlimit = config.shiftlimit;

  if (config.verboseflag) {
    jpout << "Execution parameters:\n"
          << "  algorithm: " << cuda_algs[static_cast<int>(alg)]
          << "\n  blocks: " << params.num_blocks
          << "\n  warps per block: " << best_warps
          << "\n  threads per block: " << params.num_threadsperblock
          << "\n  worker count: " << config.num_threads
          << "\n  pattern buffer size: " << params.pattern_buffer_size
          << " patterns ("
          << (sizeof(statenum_t) * n_max * params.pattern_buffer_size)
          << " bytes)"
          << "\n  shared memory used: " << params.shared_memory_size << " bytes"
          << std::format("\n  placing used[] into {} memory",
                params.used_in_shared ? "shared" : "device")
          << "\n  workcell[] window in shared memory = ["
          << params.window_lower << ',' << params.window_upper << ')'
          << std::endl;
  }

  return params;
}

// Return the amount of shared memory needed per block, in bytes, to support a
// set of runtime parameters.

size_t CoordinatorCUDA::calc_shared_memory_size(CudaAlgorithm alg,
        unsigned n_max, const CudaRuntimeParams& p) {
  size_t shared_bytes = 0;

  switch (alg) {
    case CudaAlgorithm::NORMAL:
      if (p.used_in_shared) {
        // used[] as bitfields in shared memory
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
      }
      break;
    case CudaAlgorithm::SUPER:
      if (p.used_in_shared) {
        // used[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
      }
      [[fallthrough]];
    case CudaAlgorithm::SUPER0:
      if (p.used_in_shared) {
        // cycleused[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
        // isexitcycle[]
        shared_bytes += ((p.num_threadsperblock + 31) / 32) *
            sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
      }
      break;
    default:
      break;
  }

  if (p.window_lower < p.window_upper && p.window_lower < n_max) {
    // workcell[] partially in shared memory
    const unsigned upper = std::min(n_max, p.window_upper);
    shared_bytes += ((p.num_threadsperblock + 31) / 32) *
        sizeof(ThreadStorageWorkCell) * (upper - p.window_lower);
  }

  return shared_bytes;
}

// Allocate memory in the GPU and the host, and initialize host memory.

void CoordinatorCUDA::allocate_memory(CudaAlgorithm alg,
      const CudaRuntimeParams& params,
      const std::vector<statenum_t>& graph_buffer, CudaMemoryPointers& ptrs) {
  // GPU memory
  for (unsigned bank = 0; bank < 2; ++bank) {
    throw_on_cuda_error(
        cudaMalloc(&(ptrs.wi_d[bank]), sizeof(WorkerInfo) * config.num_threads),
        __FILE__, __LINE__);
    throw_on_cuda_error(
        cudaMalloc(&(ptrs.wc_d[bank]), sizeof(ThreadStorageWorkCell) * n_max *
            ((config.num_threads + 31) / 32)),
        __FILE__, __LINE__);
    if (!config.countflag) {
      throw_on_cuda_error(
          cudaMalloc(&(ptrs.pb_d[bank]), sizeof(statenum_t) * n_max *
              params.pattern_buffer_size),
          __FILE__, __LINE__);
    }
  }
  if (graph_buffer.size() * sizeof(statenum_t) > 65536) {
    // graph doesn't fit in constant memory
    throw_on_cuda_error(
        cudaMalloc(&(ptrs.graphmatrix_d),
        graph_buffer.size() * sizeof(statenum_t)), __FILE__, __LINE__);
  }
  if (!params.used_in_shared) {
    // put used[], cycleused[], and isexitcycle[] arrays in device memory
    size_t used_size = 0;
    switch (alg) {
      case CudaAlgorithm::NORMAL:
        // used[] only
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            (((graph.numstates + 1) + 31) / 32) * sizeof(ThreadStorageUsed);
        break;
      case CudaAlgorithm::SUPER:
        // used[] not needed for SUPER0
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            (((graph.numstates + 1) + 31) / 32) * sizeof(ThreadStorageUsed);
        [[fallthrough]];
      case CudaAlgorithm::SUPER0:
        // cycleused[] and isexitcycle[]
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            ((graph.numcycles + 31) / 32) * sizeof(ThreadStorageUsed);
        used_size += params.num_blocks *
            ((params.num_threadsperblock + 31) / 32) *
            ((graph.numcycles + 31) / 32) * sizeof(ThreadStorageUsed);
        break;
      default:
        break;
    }
    if (used_size != 0) {
      if (config.verboseflag) {
        jpout << "  allocating used[] in device memory (" << used_size
              << " bytes)\n";
      }
      throw_on_cuda_error(
        cudaMalloc(&(ptrs.used_d), used_size), __FILE__, __LINE__);
    }
  }

  // Host memory (pinned)
  for (unsigned bank = 0; bank < 2; ++bank) {
    throw_on_cuda_error(
      cudaHostAlloc(&wi_h[bank], sizeof(WorkerInfo) * config.num_threads,
          cudaHostAllocDefault),
      __FILE__, __LINE__);
    throw_on_cuda_error(
      cudaHostAlloc(&wc_h[bank], sizeof(ThreadStorageWorkCell) * n_max *
          ((config.num_threads + 31) / 32), cudaHostAllocDefault),
      __FILE__, __LINE__);

    for (unsigned id = 0; id < config.num_threads; ++id) {
      wi_h[bank][id].start_state = 0;
      wi_h[bank][id].end_state = 0;
      wi_h[bank][id].pos = 0;
      wi_h[bank][id].nnodes = 0;
      wi_h[bank][id].status = 1;  // done
      for (unsigned i = 0; i < n_max; ++i) {
        auto& cell = workcell(bank, id, i);
        cell.col = 0;
        cell.col_limit = 0;
        cell.from_state = 0;
        cell.count = 0;
      }
    }
  }
  if (!config.countflag) {
    // pattern count
    throw_on_cuda_error(
      cudaHostAlloc(&pattern_count_h, sizeof(uint32_t), cudaHostAllocDefault),
      __FILE__, __LINE__);
    // pattern buffer
    throw_on_cuda_error(
      cudaHostAlloc(&pb_h, sizeof(statenum_t) * n_max *
          params.pattern_buffer_size, cudaHostAllocDefault),
      __FILE__, __LINE__);

    *pattern_count_h = 0;
  }
}

// Copy graph data to GPU.

void CoordinatorCUDA::copy_graph_to_gpu(
      const std::vector<statenum_t>& graph_buffer,
      const CudaMemoryPointers& ptrs) {
  if (config.verboseflag) {
    erase_status_output();
    jpout << std::format("  placing graph into {} memory ({} bytes)\n",
               (ptrs.graphmatrix_d != nullptr ? "device" : "constant"),
               sizeof(statenum_t) * graph_buffer.size());
    print_status_output();
  }
  throw_on_cuda_error(
      cudaMemcpy(ptrs.graphmatrix_d != nullptr ? ptrs.graphmatrix_d :
          ptrs.graphmatrix_c, graph_buffer.data(),
          sizeof(statenum_t) * graph_buffer.size(), cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

// Copy static global variables to GPU global memory.

void CoordinatorCUDA::copy_static_vars_to_gpu(const CudaRuntimeParams& params,
      const CudaMemoryPointers& ptrs) {
  uint8_t maxoutdegree_h = static_cast<uint8_t>(graph.maxoutdegree);
  uint16_t numstates_h = static_cast<uint16_t>(graph.numstates);
  uint16_t numcycles_h = static_cast<uint16_t>(graph.numcycles);
  uint32_t pattern_buffer_size_h = params.pattern_buffer_size;
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
      cudaMemcpy(ptrs.maxoutdegree_d, &maxoutdegree_h, sizeof(uint8_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(ptrs.numstates_d, &numstates_h, sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(ptrs.numcycles_d, &numcycles_h, sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(ptrs.pattern_buffer_size_d, &pattern_buffer_size_h,
          sizeof(uint32_t), cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(ptrs.pattern_index_d[0], &pattern_index_h, sizeof(uint32_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(ptrs.pattern_index_d[1], &pattern_index_h, sizeof(uint32_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

//------------------------------------------------------------------------------
// Main loop
//------------------------------------------------------------------------------

// Copy worker data to the GPU.
//
// This copies WorkerInfo and WorkCells for threads [0, max_idx]. If `startup`
// is true then all WorkerInfo data is copied.

void CoordinatorCUDA::copy_worker_data_to_gpu(unsigned bank,
    const CudaMemoryPointers& ptrs, unsigned max_idx, bool startup) {
  throw_on_cuda_error(
      cudaMemcpyAsync(ptrs.wi_d[bank], wi_h[bank], sizeof(WorkerInfo) *
          (startup ? config.num_threads : max_idx + 1),
          cudaMemcpyHostToDevice, stream[bank]),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyAsync(ptrs.wc_d[bank], wc_h[bank],
          sizeof(ThreadStorageWorkCell) * (max_idx / 32 + 1) * n_max,
          cudaMemcpyHostToDevice, stream[bank]),
      __FILE__, __LINE__);
}

// Launch the appropriate CUDA kernel.
//
// In the event of an error, throw a `std::runtime_error` exception with an
// appropriate error message.

void CoordinatorCUDA::launch_cuda_kernel(const CudaRuntimeParams& params,
    const CudaMemoryPointers& ptrs, CudaAlgorithm alg, unsigned bank,
    uint64_t cycles) {
  launch_kernel(params, ptrs, alg, bank, cycles, stream[bank]);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::format("CUDA Error in kernel: {}",
        cudaGetErrorString(err)));
  }
}

// Copy worker data from the GPU. Copy only the worker data for threads
// [0, max_idx].

void CoordinatorCUDA::copy_worker_data_from_gpu(unsigned bank,
    const CudaMemoryPointers& ptrs, unsigned max_idx) {
  throw_on_cuda_error(
      cudaMemcpyAsync(wi_h[bank], ptrs.wi_d[bank],
          sizeof(WorkerInfo) * (max_idx + 1), cudaMemcpyDeviceToHost,
          stream[bank]),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyAsync(wc_h[bank], ptrs.wc_d[bank],
          sizeof(ThreadStorageWorkCell) * (max_idx / 32 + 1) * n_max,
          cudaMemcpyDeviceToHost, stream[bank]),
      __FILE__, __LINE__);
}

// Process the worker counters after a kernel run, and reset to initial values.

void CoordinatorCUDA::process_worker_counters(unsigned bank) {
  if (longest_by_startstate_ever.size() > 0) {
    longest_by_startstate_current.assign(longest_by_startstate_ever.size(), 0);
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    context.nnodes += wi_h[bank][id].nnodes;
    wi_h[bank][id].nnodes = 0;

    const statenum_t st_state = wi_h[bank][id].start_state;
    if (st_state >= longest_by_startstate_ever.size()) {
      longest_by_startstate_ever.resize(st_state + 1, 0);
      longest_by_startstate_current.resize(st_state + 1, 0);
    }

    for (size_t i = 0; i < n_max; ++i) {
      auto& cell = workcell(bank, id, i);
      if (cell.count == 0)
        continue;

      context.count.at(i + 1) += cell.count;
      context.ntotal += cell.count;
      if (i + 1 >= config.n_min && i + 1 <= n_max) {
        context.npatterns += cell.count;
      }
      if (i + 1 > longest_by_startstate_current.at(st_state)) {
        longest_by_startstate_current.at(st_state) = i + 1;
        if (i + 1 > longest_by_startstate_ever.at(st_state)) {
          longest_by_startstate_ever.at(st_state) = i + 1;
        }
      }
      cell.count = 0;
    }
  }
}

// Process the pattern buffer. Copy any patterns in the buffer to `context`, and
// print them to the console if needed. Then clear the buffer.
//
// Returns the count of patterns retrieved from the buffer.
//
// In the event of a pattern buffer overflow, throw a `std::runtime_error`
// exception with a relevant error message.

uint32_t CoordinatorCUDA::process_pattern_buffer(unsigned bank,
    const CudaMemoryPointers& ptrs, const uint32_t pattern_buffer_size) {
  if (ptrs.pb_d[bank] == nullptr) {
    return 0;
  }

  // get the number of patterns in the buffer
  throw_on_cuda_error(
    cudaMemcpyAsync(pattern_count_h, ptrs.pattern_index_d[bank],
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[bank]),
    __FILE__, __LINE__
  );
  cudaStreamSynchronize(stream[bank]);

  if (*pattern_count_h == 0) {
    return 0;
  } else if (*pattern_count_h > pattern_buffer_size) {
    throw std::runtime_error("CUDA error: pattern buffer overflow");
  }

  // copy pattern data to host
  throw_on_cuda_error(
    cudaMemcpyAsync(pb_h, ptrs.pb_d[bank], sizeof(statenum_t) * n_max *
        (*pattern_count_h), cudaMemcpyDeviceToHost, stream[bank]),
    __FILE__, __LINE__
  );
  cudaStreamSynchronize(stream[bank]);

  // work out each pattern's throw values from the list of state numbers
  // traversed, and process them
  std::vector<int> pattern_throws(n_max + 1);

  for (unsigned i = 0; i < *pattern_count_h; ++i) {
    const statenum_t start_state = pb_h[i * n_max];
    statenum_t from_state = start_state;

    for (unsigned j = 0; j < n_max; ++j) {
      statenum_t to_state = (j == n_max - 1) ? start_state :
                              pb_h[i * n_max + j + 1];
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
        std::cerr << "pattern count = " << *pattern_count_h << '\n';
        std::cerr << "i = " << i << '\n';
        std::cerr << "j = " << j << '\n';
        for (unsigned k = 0; k < n_max; ++k) {
          statenum_t st = pb_h[i * n_max + k];
          if (st == 0)
            break;
          std::cerr << "state(" << k << ") = " << graph.state.at(st) << '\n';
        }
        std::cerr << "from_state = " << from_state << " ("
                  << graph.state.at(from_state) << ")\n"
                  << "to_state = " << to_state << '\n'
                  << "outdegree(from_state) = "
                  << graph.outdegree.at(from_state) << '\n';
        for (unsigned k = 0; k < graph.outdegree.at(from_state); ++k) {
          std::cerr << "outmatrix(from_state)[" << k << "] = "
                    << graph.outmatrix.at(from_state).at(k)
                    << " ("
                    << graph.state.at(graph.outmatrix.at(from_state).at(k))
                    << ")\n";
        }
        throw std::runtime_error("CUDA error: invalid pattern");
      }
      pattern_throws.at(j) = throwval;

      if (to_state == start_state) {
        pattern_throws.at(j + 1) = -1;  // signals end of the pattern
        break;
      }
      from_state = to_state;
    }

    const std::string pattern =
        pattern_output_format(pattern_throws, start_state);
    process_search_result(pattern);
  }

  // reset the pattern buffer index
  const uint32_t pattern_count = *pattern_count_h;
  *pattern_count_h = 0;
  throw_on_cuda_error(
    cudaMemcpyAsync(ptrs.pattern_index_d[bank], pattern_count_h,
        sizeof(uint32_t), cudaMemcpyHostToDevice, stream[bank]),
    __FILE__, __LINE__
  );

  return pattern_count;
}

// Update the global time counters.

void CoordinatorCUDA::record_working_time(double kernel_time, double host_time,
    unsigned idle_before, unsigned idle_after, uint64_t cycles_startup,
    uint64_t cycles_run) {
  // negative host_time means that host processing finished before other bank's
  // kernel run; only count host_time > 0 when GPU was idle
  total_kernel_time += kernel_time;
  total_host_time += std::max(host_time, 0.0);

  // deduct kernel time spent doing initialization
  const double working_time = (cycles_startup + cycles_run == 0 ?
      kernel_time :
      kernel_time * static_cast<double>(cycles_run) /
      static_cast<double>(cycles_startup + cycles_run));

  // assume that the workers that went idle during the kernel run did so
  // at a uniform rate
  assert(idle_after >= idle_before);
  context.secs_working += working_time *
      (config.num_threads - idle_before / 2 - idle_after / 2);
}

// Calculate the next number of kernel cycles to run, based on timing and
// progress.

uint64_t CoordinatorCUDA::calc_next_kernel_cycles(uint64_t last_cycles,
      uint64_t last_cycles_startup, double kernel_time, double host_time,
      unsigned idle_start, unsigned idle_after, unsigned next_idle_start,
      uint32_t pattern_count, CudaRuntimeParams p) {
  const uint64_t min_cycles = 100000ul;

  assert(idle_after >= idle_start);
  last_cycles = std::max(last_cycles, min_cycles);

  // cycles per second
  const double cps = (kernel_time > 0 ?
      static_cast<double>(last_cycles + last_cycles_startup) / kernel_time :
      1.0e9);
  // jobs completed per cycle
  const double beta = static_cast<double>(idle_after - idle_start) /
      static_cast<double>(last_cycles);
  // predicted cycles to initialize the workers (i.e., non-useful time)
  const double startup = static_cast<double>(last_cycles_startup);
  // number of threads with jobs at the start of the next run
  const double workers = static_cast<double>(config.num_threads -
      next_idle_start);

  // optimal value
  double target_cycles = (beta > 0.0 ?
      sqrt(startup * startup + 2 * startup * workers / beta) - startup :
      static_cast<double>(2 * last_cycles));

  // apply constraints
  target_cycles = std::max(target_cycles, static_cast<double>(min_cycles));
  double target_time =
      (static_cast<double>(last_cycles_startup) + target_cycles) / cps;
  // note that `kernel_time + host_time` is the real host processing time
  target_time = std::max(target_time, 1.0 * (kernel_time + host_time));
  target_time = std::min(target_time, 2.0);  // max of 2 seconds
  target_cycles = target_time * cps - static_cast<double>(last_cycles_startup);

  // try to keep the pattern buffer from overflowing
  if (pattern_count > p.pattern_buffer_size / 3) {
    const auto frac = static_cast<double>(p.pattern_buffer_size / 3) /
        static_cast<double>(pattern_count);
    const auto max_cycles = static_cast<double>(last_cycles) * frac;
    target_cycles = std::min(target_cycles, max_cycles);
  }
  /*std::cerr << "cps = " << cps << ", last_cycles = " << last_cycles
            << ", target_cycles = " << target_cycles << '\n';*/
  return static_cast<uint64_t>(target_cycles);
}

//------------------------------------------------------------------------------
// Cleanup
//------------------------------------------------------------------------------

// Gather unfinished work assignments.

void CoordinatorCUDA::gather_unfinished_work_assignments() {
  for (unsigned bank = 0; bank < 2; ++bank) {
    for (unsigned id = 0; id < config.num_threads; ++id) {
      if ((wi_h[bank][id].status & 1) == 0) {
        WorkAssignment wa = read_work_assignment(bank, id);
        context.assignments.push_back(wa);
      }
      wi_h[bank][id].status = 1;
    }
  }
}

// Destroy CUDA streams and free allocated GPU and host memory.

void CoordinatorCUDA::cleanup(CudaMemoryPointers& ptrs) {
  for (int i = 0; i < 2; ++i) {
    cudaStreamDestroy(stream[i]);
    stream[i] = nullptr;
  }

  // GPU
  for (unsigned bank = 0; bank < 2; ++bank) {
    if (ptrs.wi_d[bank] != nullptr) {
      cudaFree(ptrs.wi_d[bank]);
      ptrs.wi_d[bank] = nullptr;
    }
    if (ptrs.wc_d[bank] != nullptr) {
      cudaFree(ptrs.wc_d[bank]);
      ptrs.wc_d[bank] = nullptr;
    }
    if (ptrs.pb_d[bank] != nullptr) {
      cudaFree(ptrs.pb_d[bank]);
      ptrs.pb_d[bank] = nullptr;
    }
  }
  if (ptrs.graphmatrix_d != nullptr) {
    cudaFree(ptrs.graphmatrix_d);
    ptrs.graphmatrix_d = nullptr;
  }
  if (ptrs.used_d != nullptr) {
    cudaFree(ptrs.used_d);
    ptrs.used_d = nullptr;
  }

  // Host
  for (unsigned bank = 0; bank < 2; ++bank) {
    if (wi_h[bank] != nullptr) {
      cudaFreeHost(wi_h[bank]);
      wi_h[bank] = nullptr;
    }
    if (wc_h[bank] != nullptr) {
      cudaFreeHost(wc_h[bank]);
      wc_h[bank] = nullptr;
    }
  }
  if (pattern_count_h != nullptr) {
    cudaFreeHost(pattern_count_h);
    pattern_count_h = nullptr;
  }
  if (pb_h != nullptr) {
    cudaFreeHost(pb_h);
    pb_h = nullptr;
  }
}

//------------------------------------------------------------------------------
// Manage work assignments
//------------------------------------------------------------------------------

// Load initial work assignments into bank 0, which will execute first.

void CoordinatorCUDA::load_initial_work_assignments() {
  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (context.assignments.size() > 0) {
      WorkAssignment wa = context.assignments.front();
      context.assignments.pop_front();

      // if it's a STARTUP assignment then initialize
      if (wa.start_state == 0) {
        wa.start_state = (config.groundmode ==
            SearchConfig::GroundMode::EXCITED_SEARCH ? 2 : 1);
      }
      if (wa.end_state == 0) {
        wa.end_state = (config.groundmode ==
            SearchConfig::GroundMode::GROUND_SEARCH ? 1 : graph.numstates);
      }

      load_work_assignment(0, id, wa);
      wi_h[0][id].status = 0;
      max_active_idx[0] = id;
    } else {
      wi_h[0][id].status = 1;
    }
    wi_h[1][id].status = 1;
  }
  max_active_idx[1] = 0;
}

// Load a work assignment into a worker's slot in the `WorkerInfo` and
// `ThreadStorageWorkCell` arrays.

void CoordinatorCUDA::load_work_assignment(unsigned bank, const unsigned id,
    WorkAssignment& wa) {
  wa.to_workspace(this, bank * config.num_threads + id);
  wi_h[bank][id].nnodes = 0;
  wi_h[bank][id].status = 0;

  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, bank * config.num_threads + id);
  assert(wa == wa2);
}

// Read out the work assignment for worker `id` from the workcells in bank
// `bank`.
//
// This is a non-destructive read, i.e., the workcells are unchanged.

WorkAssignment CoordinatorCUDA::read_work_assignment(unsigned bank,
    unsigned id) {
  WorkAssignment wa;
  wa.from_workspace(this, bank * config.num_threads + id);
  return wa;
}

// Assign new jobs to idle workers.
//
// Returns the number of idle workers with no jobs assigned.

unsigned CoordinatorCUDA::assign_new_jobs(unsigned bank, unsigned idle_before_b,
    const CudaWorkerSummary& summary, unsigned idle_before_a) {
  const unsigned idle_after_b = summary.workers_idle.size();
  if (idle_after_b == 0)
    return 0;

  // split working jobs and add them to the pool, until all workers are
  // processed or pool size >= target_job_count
  //
  // jobs needed:
  //   this bank (bankB):  `idle_after_b`
  //   other bank (bankA): `idle_before_a` + `idle_after_b - idle_before_b`
  //
  // where the last term is an estimate of how many jobs will complete during
  // bankA's kernel execution (happening when this code runs)

  const unsigned target_job_count = idle_after_b + idle_before_a +
      idle_after_b - idle_before_b;

  std::vector<int> has_split(config.num_threads, 0);
  auto it = summary.workers_multiple_start_states.begin();

  while (context.assignments.size() < target_job_count) {
    if (it == summary.workers_multiple_start_states.end()) {
      it = summary.workers_rpm_plus0.begin();
    }
    if (it == summary.workers_rpm_plus0.end()) {
      it = summary.workers_rpm_plus1.begin();
    }
    if (it == summary.workers_rpm_plus1.end()) {
      it = summary.workers_rpm_plus2.begin();
    }
    if (it == summary.workers_rpm_plus2.end()) {
      it = summary.workers_rpm_plus3.begin();
    }
    if (it == summary.workers_rpm_plus3.end()) {
      it = summary.workers_rpm_plus4p.begin();
    }
    if (it == summary.workers_rpm_plus4p.end()) {
      break;
    }

    if (has_split.at(*it)) {
      ++it;
      continue;
    }
    has_split.at(*it) = 1;

    WorkAssignment wa = read_work_assignment(bank, *it);
    if (wa.is_splittable()) {
      WorkAssignment wa2 = wa.split(graph, config.split_alg);
      load_work_assignment(bank, *it, wa);
      context.assignments.push_back(wa2);
      ++context.splits_total;

      // Avoid double counting nodes: Each of the nodes in the partial path for
      // the new assignment will be reported twice to the coordinator: by the
      // worker doing the original job `wa`, and by the worker that does `wa2`.
      if (wa.start_state == wa2.start_state) {
        wi_h[bank][*it].nnodes -= wa2.partial_pattern.size();
      }
    }
    ++it;
  }

  // assign to idle workers from the pool

  unsigned idle_remaining = idle_after_b;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if ((wi_h[bank][id].status & 1) == 0)
      continue;
    if (context.assignments.size() == 0)
      break;

    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();
    load_work_assignment(bank, id, wa);
    max_active_idx[bank] = std::max(max_active_idx[bank], id);
    --idle_remaining;
  }

  return idle_remaining;
}

//------------------------------------------------------------------------------
// Summarization and status display
//------------------------------------------------------------------------------

// Produce a summary of the current worker status.

CudaWorkerSummary CoordinatorCUDA::summarize_worker_status(unsigned bank) {
  unsigned root_pos_min = -1u;
  statenum_t max_start_state = 0;
  max_active_idx[bank] = 0;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if ((wi_h[bank][id].status & 1) != 0) {
      continue;
    }

    max_active_idx[bank] = id;
    max_start_state = std::max(max_start_state, wi_h[bank][id].start_state);

    for (int i = 0; i <= wi_h[bank][id].pos; ++i) {
      const auto& cell = workcell(bank, id, i);
      unsigned col = cell.col;
      const unsigned from_state = cell.from_state;
      const unsigned col_limit = std::min(graph.outdegree.at(from_state),
          static_cast<unsigned>(cell.col_limit));

      if (col < col_limit - 1) {
        // `root_pos` == i for this worker
        if (static_cast<unsigned>(i) < root_pos_min || root_pos_min == -1u) {
          root_pos_min = i;
        }
        break;
      }
    }
  }

  CudaWorkerSummary summary;
  summary.root_pos_min = root_pos_min;
  summary.max_start_state = max_start_state;
  summary.cycles_startup = 0;
  summary.count_rpm_plus0.assign(max_start_state + 1, 0);
  summary.count_rpm_plus1.assign(max_start_state + 1, 0);
  summary.count_rpm_plus2.assign(max_start_state + 1, 0);
  summary.count_rpm_plus3.assign(max_start_state + 1, 0);
  summary.count_rpm_plus4p.assign(max_start_state + 1, 0);
  summary.npatterns = context.npatterns;
  summary.nnodes = context.nnodes;
  summary.ntotal = context.ntotal;

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if ((wi_h[bank][id].status & 1) != 0) {
      summary.workers_idle.push_back(id);
      continue;
    }

    summary.cycles_startup += static_cast<uint64_t>(
        wi_h[bank][id].cycles_startup);

    if (wi_h[bank][id].start_state != wi_h[bank][id].end_state) {
      summary.workers_multiple_start_states.push_back(id);
    }

    for (int i = 0; i <= wi_h[bank][id].pos; ++i) {
      const auto& cell = workcell(bank, id, i);
      unsigned col = cell.col;
      const unsigned from_state = cell.from_state;
      const unsigned col_limit = std::min(graph.outdegree.at(from_state),
          static_cast<unsigned>(cell.col_limit));

      if (col < col_limit - 1) {
        switch (static_cast<unsigned>(i) - root_pos_min) {
          case 0:
            summary.workers_rpm_plus0.push_back(id);
            summary.count_rpm_plus0.at(wi_h[bank][id].start_state) += 1;
            break;
          case 1:
            summary.workers_rpm_plus1.push_back(id);
            summary.count_rpm_plus1.at(wi_h[bank][id].start_state) += 1;
            break;
          case 2:
            summary.workers_rpm_plus2.push_back(id);
            summary.count_rpm_plus2.at(wi_h[bank][id].start_state) += 1;
            break;
          case 3:
            summary.workers_rpm_plus3.push_back(id);
            summary.count_rpm_plus3.at(wi_h[bank][id].start_state) += 1;
            break;
          default:
            summary.workers_rpm_plus4p.push_back(id);
            summary.count_rpm_plus4p.at(wi_h[bank][id].start_state) += 1;
            break;
        }
        break;
      }
    }
  }

  if (config.num_threads > summary.workers_idle.size()) {
    summary.cycles_startup /=
        (config.num_threads - summary.workers_idle.size());
  } else {
    summary.cycles_startup = 0;
  }
  return summary;
}

// Summarize all active jobs in the worker banks and the assignments queue.

CudaWorkerSummary CoordinatorCUDA::summarize_all_jobs(
    const CudaWorkerSummary& last, const CudaWorkerSummary& prev) {
  CudaWorkerSummary summary;

  summary.root_pos_min = std::min(last.root_pos_min, prev.root_pos_min);
  summary.max_start_state = std::max(last.max_start_state,
      prev.max_start_state);
  for (const WorkAssignment& wa : context.assignments) {
    summary.root_pos_min = std::min(summary.root_pos_min, wa.root_pos);
    summary.max_start_state = std::max(summary.max_start_state,
        static_cast<statenum_t>(wa.start_state));
  }

  std::vector<std::vector<unsigned>> counts(summary.root_pos_min + 5);
  for (unsigned i = 0; i < summary.root_pos_min + 5; ++i) {
    counts.at(i).assign(summary.max_start_state + 1, 0);
  }

  auto trunc = [&](unsigned a) {
    return std::min(a, summary.root_pos_min + 4);
  };
  for (const WorkAssignment& wa : context.assignments) {
    counts.at(trunc(wa.root_pos)).at(wa.start_state) += 1;
  }

  for (statenum_t j = 1; j < summary.max_start_state + 1; ++j) {
    if (j <= last.max_start_state) {
      counts.at(trunc(last.root_pos_min)).at(j) += last.count_rpm_plus0.at(j);
      counts.at(trunc(last.root_pos_min + 1)).at(j) +=
          last.count_rpm_plus1.at(j);
      counts.at(trunc(last.root_pos_min + 2)).at(j) +=
          last.count_rpm_plus2.at(j);
      counts.at(trunc(last.root_pos_min + 3)).at(j) +=
          last.count_rpm_plus3.at(j);
      counts.at(trunc(last.root_pos_min + 4)).at(j) +=
          last.count_rpm_plus4p.at(j);
    }
    if (j <= prev.max_start_state) {
      counts.at(trunc(prev.root_pos_min)).at(j) += prev.count_rpm_plus0.at(j);
      counts.at(trunc(prev.root_pos_min + 1)).at(j) +=
          prev.count_rpm_plus1.at(j);
      counts.at(trunc(prev.root_pos_min + 2)).at(j) +=
          prev.count_rpm_plus2.at(j);
      counts.at(trunc(prev.root_pos_min + 3)).at(j) +=
          prev.count_rpm_plus3.at(j);
      counts.at(trunc(prev.root_pos_min + 4)).at(j) +=
          prev.count_rpm_plus4p.at(j);
    }
  }

  summary.count_rpm_plus0.assign(summary.max_start_state + 1, 0);
  summary.count_rpm_plus1.assign(summary.max_start_state + 1, 0);
  summary.count_rpm_plus2.assign(summary.max_start_state + 1, 0);
  summary.count_rpm_plus3.assign(summary.max_start_state + 1, 0);
  summary.count_rpm_plus4p.assign(summary.max_start_state + 1, 0);
  summary.count_total = 0;

  for (unsigned i = 0; i < summary.root_pos_min + 5; ++i) {
    for (statenum_t j = 1; j < summary.max_start_state + 1; ++j) {
      if (i < summary.root_pos_min) {
        assert(counts.at(i).at(j) == 0);
      } else if (i == summary.root_pos_min) {
        summary.count_rpm_plus0.at(j) += counts.at(i).at(j);
      } else if (i == summary.root_pos_min + 1) {
        summary.count_rpm_plus1.at(j) += counts.at(i).at(j);
      } else if (i == summary.root_pos_min + 2) {
        summary.count_rpm_plus2.at(j) += counts.at(i).at(j);
      } else if (i == summary.root_pos_min + 3) {
        summary.count_rpm_plus3.at(j) += counts.at(i).at(j);
      } else {
        summary.count_rpm_plus4p.at(j) += counts.at(i).at(j);
      }
      summary.count_total += counts.at(i).at(j);
    }
  }

  summary.npatterns = last.npatterns;
  summary.nnodes = last.nnodes;
  return summary;
}

// Create and display the live status indicator, if needed.

void CoordinatorCUDA::do_status_display(const CudaWorkerSummary& summary_afterB,
      const CudaWorkerSummary& summary_beforeA, double kernel_time,
      double host_time) {
  if (config.verboseflag) {
    erase_status_output();
    jpout << std::format(
        "kernel = {:.3}, host = {:.3}, startup = {}, busy = {}\n",
        kernel_time, host_time, summary_afterB.cycles_startup,
        config.num_threads - summary_afterB.workers_idle.size());
    print_status_output();
  }

  if (!config.statusflag)
    return;
  auto now = std::chrono::system_clock::now();
  if (calc_duration_secs(last_display_time, now) < 1.0) {
    return;
  }
  last_display_time = now;

  status_lines.clear();
  status_lines.push_back("Status on: " + current_time_string());
  auto summary = summarize_all_jobs(summary_afterB, summary_beforeA);

  if (summary.root_pos_min != -1u) {
    std::string workers_str = std::to_string(config.num_threads);
    std::string jobs_str = std::to_string(summary.count_total);
    std::string period_str = "          period";
    if (workers_str.length() + jobs_str.length() < 14) {
      period_str.insert(0, 14 - workers_str.length() - jobs_str.length(), ' ');
    }
    status_lines.push_back(std::format(
      " state  root_pos and job count ({} workers, {} jobs) {}", workers_str,
      jobs_str, period_str));

    auto format1 = [](unsigned a, unsigned b, bool plus) {
      if (b == 0) {
        if (plus) {
          return std::string("             ");
        } else {
          return std::string("            ");
        }
      }
      if (plus) {
        return std::format("{:4}+: {: <6}", a, b);
      }
      return std::format("{:4}: {: <6}", a, b);
    };

    const statenum_t MAX_START = 9;
    for (unsigned start_state = 1; start_state <=
          std::min(MAX_START, summary.max_start_state); ++start_state) {
      unsigned num_rpm_plus0 = summary.count_rpm_plus0.at(start_state);
      unsigned num_rpm_plus1 = summary.count_rpm_plus1.at(start_state);
      unsigned num_rpm_plus2 = summary.count_rpm_plus2.at(start_state);
      unsigned num_rpm_plus3 = summary.count_rpm_plus3.at(start_state);
      unsigned num_rpm_plus4p = summary.count_rpm_plus4p.at(start_state);
      unsigned longest_now = longest_by_startstate_current.at(start_state);
      unsigned longest_ever = longest_by_startstate_ever.at(start_state);

      char ch = ' ';
      if (start_state == MAX_START && summary.max_start_state > MAX_START) {
        ch = '+';
        for (unsigned st = start_state + 1;
              st < longest_by_startstate_ever.size(); ++st) {
          if (st <= summary.max_start_state) {
            num_rpm_plus0 += summary.count_rpm_plus0.at(st);
            num_rpm_plus1 += summary.count_rpm_plus1.at(st);
            num_rpm_plus2 += summary.count_rpm_plus2.at(st);
            num_rpm_plus3 += summary.count_rpm_plus3.at(st);
            num_rpm_plus4p += summary.count_rpm_plus4p.at(st);
          }
          longest_now = std::max(longest_now,
              longest_by_startstate_current.at(st));
          longest_ever = std::max(longest_ever,
              longest_by_startstate_ever.at(st));
        }
      }

      status_lines.push_back(std::format(
        "{:4}{}  {}{}{}{}{}  {:4} {:4}",
        start_state,
        ch,
        format1(summary.root_pos_min, num_rpm_plus0, false),
        format1(summary.root_pos_min + 1, num_rpm_plus1, false),
        format1(summary.root_pos_min + 2, num_rpm_plus2, false),
        format1(summary.root_pos_min + 3, num_rpm_plus3, false),
        format1(summary.root_pos_min + 4, num_rpm_plus4p, true),
        longest_now,
        longest_ever
      ));
    }
  }

  auto format2 = [](double a) {
    if (a < 1 || a > 9999000000000) {
      return std::string("-----");
    }
    if (a < 99999) {
      auto result = std::format("{:5g}", a);
      return result.substr(0, 5);
    } else if (a < 1000000) {
      auto result = std::format("{:4g}", a / 1000);
      return result.substr(0, 4) + "K";
    } else if (a < 1000000000) {
      auto result = std::format("{:4g}", a / 1000000);
      return result.substr(0, 4) + "M";
    } else {
      auto result = std::format("{:4g}", a / 1000000000);
      return result.substr(0, 4) + "B";
    }
  };

  const double nodespersec =
      static_cast<double>(summary_afterB.nnodes - summary_beforeA.nnodes) /
          (std::max(host_time, 0.0) + kernel_time);
  const double patspersec =
      static_cast<double>(summary_afterB.ntotal - summary_beforeA.ntotal) /
          (std::max(host_time, 0.0) + kernel_time);

  status_lines.push_back(std::format(
    "idled:{:7}, nodes/s: {}, pats/s: {}, pats in range:{:19}",
    summary_afterB.workers_idle.size(),
    format2(nodespersec),
    format2(patspersec),
    context.npatterns
  ));

  erase_status_output();
  print_status_output();
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Return a reference to the workcell for thread `id`, position `pos`.

ThreadStorageWorkCell& CoordinatorCUDA::workcell(unsigned bank, unsigned id,
    unsigned pos) {
  ThreadStorageWorkCell* start_warp = &wc_h[bank][(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}

const ThreadStorageWorkCell& CoordinatorCUDA::workcell(unsigned bank,
    unsigned id, unsigned pos) const {
  ThreadStorageWorkCell* start_warp = &wc_h[bank][(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}

// Handle CUDA errors by throwing a `std::runtime_error` exception with a
// relevant error message.

void CoordinatorCUDA::throw_on_cuda_error(cudaError_t code, const char *file,
      int line) {
  if (code != cudaSuccess) {
    std::ostringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << " in file "
       << file << " at line " << line;
    throw std::runtime_error(ss.str());
  }
}

//------------------------------------------------------------------------------
// WorkSpace methods
//------------------------------------------------------------------------------

const Graph& CoordinatorCUDA::get_graph() const {
  return graph;
}

void CoordinatorCUDA::set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state) {
  assert(slot < 2 * config.num_threads);
  assert(index < n_max);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  ThreadStorageWorkCell& wc = workcell(bank, id, index);
  wc.col = col;
  wc.col_limit = col_limit;
  wc.from_state = from_state;
}

std::tuple<unsigned, unsigned, unsigned> CoordinatorCUDA::get_cell(
    unsigned slot, unsigned index) const {
  assert(slot < 2 * config.num_threads);
  assert(index < n_max);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  const ThreadStorageWorkCell& wc = workcell(bank, id, index);
  return std::make_tuple(wc.col, wc.col_limit, wc.from_state);
}

void CoordinatorCUDA::set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos) {
  assert(slot < 2 * config.num_threads);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  WorkerInfo& wi = wi_h[bank][id];
  wi.start_state = new_start_state;
  wi.end_state = new_end_state;
  wi.pos = new_pos;
}

std::tuple<unsigned, unsigned, int> CoordinatorCUDA::get_info(unsigned slot)
    const {
  assert(slot < 2 * config.num_threads);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  const WorkerInfo& wi = wi_h[bank][id];
  return std::make_tuple(wi.start_state, wi.end_state, wi.pos);
}

//------------------------------------------------------------------------------
// Free functions
//------------------------------------------------------------------------------

// Record kernel completion time (CUDA callback function)

void CUDART_CB record_kernel_completion(void* data) {
  jptimer_t* ptr = (jptimer_t*)data;
  *ptr = std::chrono::high_resolution_clock::now();
}
