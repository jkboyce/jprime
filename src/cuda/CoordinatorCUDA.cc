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
#include <array>
#include <format>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cassert>

// defined in CudaKernels.cu
void configure_cuda_shared_memory(const CudaRuntimeParams& p);
CudaGlobalPointers get_gpu_static_pointers();
void launch_kernel(const CudaGlobalPointers& gptrs, unsigned bank,
  Coordinator::SearchAlgorithm alg, const CudaRuntimeParams& p,
  uint64_t cycles, cudaStream_t& stream);


CoordinatorCUDA::CoordinatorCUDA(SearchConfig& a, SearchContext& b,
    std::ostream& c)
    : Coordinator(a, b, c)
{}

CoordinatorCUDA::~CoordinatorCUDA()
{
  cleanup();  // does nothing if search exits cleanly
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

void CoordinatorCUDA::run_search()
{
  initialize();

  // set up workers
  load_initial_work_assignments();
  uint64_t cycles[2] = { MINCYCLES, MINCYCLES };
  cudaDeviceSynchronize();

  for (unsigned run = 0; ; ++run) {
    const unsigned bankA = run % 2;
    const unsigned bankB = (run + 1) % 2;

    // start the workers for bankA
    before_kernel[bankA] = std::chrono::high_resolution_clock::now();
    launch_cuda_kernel(bankA, cycles[bankA]);
    cudaLaunchHostFunc(stream[bankA], record_kernel_completion_time,
        &after_kernel[bankA]);

    // process results from the other bank (bankB)
    copy_worker_data_from_gpu(bankB);
    cudaStreamSynchronize(stream[bankB]);
    process_worker_counters(bankB);
    const auto pattern_count = process_pattern_buffer(bankB);
    summary_after[bankB] = summarize_worker_status(bankB);

    const auto kernel_time = calc_duration_secs(before_kernel[bankB],
        after_kernel[bankB]);
    const auto host_time = calc_duration_secs(after_kernel[bankB],
        after_host[bankA]);
    record_working_time(bankB, kernel_time, host_time, cycles[bankB]);
    do_status_display(bankB, kernel_time, host_time, run);

    if (Coordinator::stopping) {
      process_worker_counters(bankA);  // node count changes from splitting
      gather_unfinished_work_assignments();
      break;
    }
    if (summary_after[bankB].workers_idle.size() == config.num_threads &&
        summary_before[bankA].workers_idle.size() == config.num_threads &&
        context.assignments.empty()) {
      break;
    }

    // prepare bankB for its next run
    assign_new_jobs(bankB);
    skip_unusable_startstates(bankB);
    copy_worker_data_to_gpu(bankB);
    const auto prev_idle_before =
        static_cast<unsigned>(summary_before[bankB].workers_idle.size());
    summary_before[bankB] = summarize_worker_status(bankB);
    cycles[bankB] = calc_next_kernel_cycles(cycles[bankB], bankB, kernel_time,
        host_time, prev_idle_before, pattern_count);
    cudaStreamSynchronize(stream[bankB]);
    after_host[bankB] = std::chrono::high_resolution_clock::now();

    // wait for bankA workers to finish
    cudaStreamSynchronize(stream[bankA]);
  }

  cleanup();

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

// Initialize data structures and copy non-worker data to the GPU.

void CoordinatorCUDA::initialize()
{
  prop = initialize_cuda_device();
  graph_buffer = make_graph_buffer();
  params = find_runtime_params();
  gptrs = get_gpu_static_pointers();

  allocate_memory();
  copy_graph_to_gpu();
  copy_static_vars_to_gpu();
  configure_cuda_shared_memory(params);

  // timing setup
  const auto now = std::chrono::high_resolution_clock::now();
  for (unsigned bank = 0; bank < 2; ++bank) {
    before_kernel[bank] = now;
    after_kernel[bank] = now;
    after_host[bank] = now;
  }
  last_status_time = now;
  last_nnodes = context.nnodes;
  last_ntotal = context.ntotal;
}

// Initialize CUDA device and check properties.

cudaDeviceProp CoordinatorCUDA::initialize_cuda_device()
{
  cudaDeviceProp pr;
  cudaGetDeviceProperties(&pr, 0);

  for (int i = 0; i < 2; ++i) {
    cudaStreamCreate(&stream[i]);
  }

  if (config.verboseflag) {
    jpout << "Device Number: " << 0
          << "\n  device name: " << pr.name
          << "\n  multiprocessor (MP) count: " << pr.multiProcessorCount
          << "\n  max threads per MP: " << pr.maxThreadsPerMultiProcessor
          << "\n  max threads per block: " << pr.maxThreadsPerBlock
          << "\n  async engine count: " << pr.asyncEngineCount
          << "\n  total global memory (bytes): " << pr.totalGlobalMem
          << "\n  total constant memory (bytes): " << pr.totalConstMem
          << "\n  shared memory per block (bytes): " << pr.sharedMemPerBlock
          << "\n  shared memory per block, maximum opt-in (bytes): "
          << pr.sharedMemPerBlockOptin << std::endl;
  }

  return pr;
}

// Return a version of the graph for the GPU.

std::vector<statenum_t> CoordinatorCUDA::make_graph_buffer()
{
  std::vector<statenum_t> buffer;

  // in MARKING mode, append a list of excluded states to the graph
  std::vector<statenum_t> exclude_buffer;
  std::vector<std::vector<unsigned>> excludestates_tail;
  std::vector<std::vector<unsigned>> excludestates_head;
  if (alg == SearchAlgorithm::NORMAL_MARKING) {
    std::tie(excludestates_tail, excludestates_head) =
        graph.get_exclude_states();
  }
  uint32_t exclude_offset = (graph.numstates + 1) * (graph.maxoutdegree + 5);

  for (unsigned i = 0; i <= graph.numstates; ++i) {
    // include unused state 0

    // state numbers for outgoing links
    for (unsigned j = 0; j < graph.maxoutdegree; ++j) {
      if (j < graph.outdegree.at(i)) {
        buffer.push_back(static_cast<statenum_t>(graph.outmatrix.at(i).at(j)));
      } else {
        buffer.push_back(0);
      }
    }

    // cycle number, for modes that need it
    if (alg == SearchAlgorithm::SUPER || alg == SearchAlgorithm::SUPER0 ||
        alg == SearchAlgorithm::NORMAL_MARKING) {
      buffer.push_back(static_cast<statenum_t>(graph.cyclenum.at(i)));
    }

    // pointers to lists of excluded states, in MARKING mode
    if (alg == SearchAlgorithm::NORMAL_MARKING) {
      if (excludestates_tail.at(i).at(0) == 0) {
        buffer.push_back(0);
        buffer.push_back(0);
      } else {
        buffer.push_back(static_cast<statenum_t>(exclude_offset & 0xFFFF));
        buffer.push_back(static_cast<statenum_t>
            ((exclude_offset >> 16) & 0xFFFF));
        for (auto s : excludestates_tail.at(i)) {
          if (s == 0)
            break;
          exclude_buffer.push_back(static_cast<statenum_t>(s));
          ++exclude_offset;
        }
        exclude_buffer.push_back(0);
        ++exclude_offset;
      }
      if (excludestates_head.at(i).at(0) == 0) {
        buffer.push_back(0);
        buffer.push_back(0);
      } else {
        buffer.push_back(static_cast<statenum_t>(exclude_offset & 0xFFFF));
        buffer.push_back(static_cast<statenum_t>
            ((exclude_offset >> 16) & 0xFFFF));
        for (auto s : excludestates_head.at(i)) {
          if (s == 0)
            break;
          exclude_buffer.push_back(static_cast<statenum_t>(s));
          ++exclude_offset;
        }
        exclude_buffer.push_back(0);
        ++exclude_offset;
      }
    }
  }

  // append the exclude buffer, in MARKING mode
  if (alg == SearchAlgorithm::NORMAL_MARKING) {
    assert(buffer.size() == (graph.numstates + 1) * (graph.maxoutdegree + 5));
    buffer.insert(buffer.end(), exclude_buffer.begin(),
        exclude_buffer.end());
  }

  return buffer;
}

// Speedup as a function of number of warps per block, when the workcell[] array
// is placed in shared memory or global memory
//
// columns are {warps, shared memory speedup, global memory speedup}

static const double throughput[33][3] = {
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

CudaRuntimeParams CoordinatorCUDA::find_runtime_params()
{
  CudaRuntimeParams p;
  p.num_blocks = prop.multiProcessorCount;
  p.pattern_buffer_size = (config.countflag ? 0 :
    (prop.totalGlobalMem / 16) / sizeof(statenum_t) / n_max);

  // heuristic: see if used[] arrays for 10 warps will fit into shared memory;
  // if not then put into device memory
  p.num_threadsperblock = 32 * 10;
  p.used_in_shared = true;
  p.window_lower = p.window_upper = 0;
  size_t shared_mem = calc_shared_memory_size(n_max, p);
  if (shared_mem > prop.sharedMemPerBlockOptin) {
    p.used_in_shared = false;
  }

  const auto access_fraction = build_access_model(graph.numstates);

  // consider each warp value in turn, and estimate throughput for each
  unsigned best_warps = 1;
  unsigned best_lower = 0;
  unsigned best_upper = 0;
  double best_throughput = -1;
  const int max_warps = prop.maxThreadsPerBlock / 32;

  constexpr bool set_warps = false;
  constexpr int warps_target = 9;
  constexpr bool print_info = false;

  for (int warps = 1; warps <= max_warps; ++warps) {
    if constexpr (print_info) {
      jpout << "calculating for " << warps << " warps:\n";
    }

    // find the maximum window size that fits into shared memory
    unsigned lower = 0;
    unsigned upper = n_max;
    for (; upper != 0; --upper) {
      p.num_threadsperblock = 32 * warps;
      p.window_lower = lower;
      p.window_upper = upper;
      shared_mem = calc_shared_memory_size(n_max, p);
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
    if constexpr (print_info) {
      jpout << "  window [" << lower << ',' << upper
            << "), S = " << S << ", T = " << throughput_est << '\n';
    }

    if ((!set_warps && throughput_est > best_throughput + 0.25) ||
        (set_warps && warps == warps_target)) {
      if constexpr (print_info) {
        jpout << "  new best warps: " << warps << '\n';
      }
      best_throughput = throughput_est;
      best_warps = warps;
      best_lower = lower;
      best_upper = upper;
    }

    if constexpr (print_info) {
      jpout << std::format("warps {}: window [{},{}) S = {}, throughput = {}\n",
          warps, lower, upper, S, throughput_est);
    }
  }

  p.num_threadsperblock = 32 * best_warps;
  p.window_lower = best_lower;
  p.window_upper = best_upper;
  p.shared_memory_used = calc_shared_memory_size(n_max, p);
  config.num_threads = p.num_blocks * p.num_threadsperblock;

  p.n_min = config.n_min;
  p.n_max = n_max;
  p.report = !config.countflag;
  p.shiftlimit = config.shiftlimit;

  if (config.verboseflag) {
    unsigned algnum = 0;
    switch (alg) {
      case Coordinator::SearchAlgorithm::NONE:
        algnum = 0;
        break;
      case Coordinator::SearchAlgorithm::NORMAL:
        algnum = 1;
        break;
      case Coordinator::SearchAlgorithm::NORMAL_MARKING:
        algnum = 2;
        break;
      case Coordinator::SearchAlgorithm::SUPER:
      case Coordinator::SearchAlgorithm::SUPER0:
        algnum = 3;
        break;
    }

    static constexpr std::array cuda_algs = {
      "no_algorithm",
      "cuda_gen_loops_normal()",
      "cuda_gen_loops_normal_marking()",
      "cuda_gen_loops_super()",
    };

    jpout << "Execution parameters:\n"
          << "  algorithm: " << cuda_algs.at(algnum)
          << "\n  blocks: " << p.num_blocks
          << "\n  warps per block: " << best_warps
          << "\n  threads per block: " << p.num_threadsperblock
          << "\n  worker count: " << config.num_threads
          << "\n  pattern buffer size: " << p.pattern_buffer_size
          << " patterns ("
          << (sizeof(statenum_t) * n_max * p.pattern_buffer_size)
          << " bytes)"
          << "\n  shared memory used: " << p.shared_memory_used << " bytes"
          << std::format("\n  placing used[] into {} memory",
                p.used_in_shared ? "shared" : "device")
          << "\n  workcell[] window in shared memory = ["
          << p.window_lower << ',' << p.window_upper << ')'
          << std::endl;
  }

  return p;
}

// Return the amount of shared memory needed per block, in bytes, to support a
// set of runtime parameters.

size_t CoordinatorCUDA::calc_shared_memory_size(unsigned nmax,
    const CudaRuntimeParams& p)
{
  assert(alg != SearchAlgorithm::NONE);
  size_t shared_bytes = 0;

  if (p.used_in_shared) {
    if (alg != SearchAlgorithm::SUPER0) {
      // used[] (1 bit/state)
      shared_bytes += ((p.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
    }

    if (alg == SearchAlgorithm::NORMAL_MARKING) {
      // deadstates[] (8 bits/cycle)
      shared_bytes += ((p.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 3) / 4);
    }

    if (alg == SearchAlgorithm::SUPER || alg == SearchAlgorithm::SUPER0) {
      // cycleused[] (1 bit/cycle)
      shared_bytes += ((p.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
      // isexitcycle[] (1 bit/cycle)
      shared_bytes += ((p.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
    }
  }

  if (p.window_lower < p.window_upper && p.window_lower < nmax) {
    // workcell[] partially in shared memory
    const unsigned upper = std::min(nmax, p.window_upper);
    shared_bytes += ((p.num_threadsperblock + 31) / 32) *
        sizeof(ThreadStorageWorkCell) * (upper - p.window_lower);
  }

  return shared_bytes;
}

// Allocate memory in the GPU and the host, and initialize host memory.

void CoordinatorCUDA::allocate_memory()
{
  // GPU memory
  for (unsigned bank = 0; bank < 2; ++bank) {
    throw_on_cuda_error(
        cudaMalloc(&(gptrs.wi_d[bank]), sizeof(WorkerInfo) *
          config.num_threads),
        __FILE__, __LINE__);
    throw_on_cuda_error(
        cudaMalloc(&(gptrs.wc_d[bank]), sizeof(ThreadStorageWorkCell) * n_max *
            ((config.num_threads + 31) / 32)),
        __FILE__, __LINE__);
    if (!config.countflag) {
      throw_on_cuda_error(
          cudaMalloc(&(gptrs.pb_d[bank]), sizeof(statenum_t) * n_max *
              params.pattern_buffer_size),
          __FILE__, __LINE__);
    }
  }

  if (graph_buffer.size() * sizeof(statenum_t) > 65536) {
    // graph doesn't fit in constant memory
    throw_on_cuda_error(
        cudaMalloc(&(gptrs.graphmatrix_d),
        graph_buffer.size() * sizeof(statenum_t)), __FILE__, __LINE__);
  }

  if (!params.used_in_shared) {
    // used[], deadstates[], cycleused[], and isexitcycle[] in device memory
    size_t used_size = 0;
    if (alg != SearchAlgorithm::SUPER0) {
      // used[] (1 bit/state)
      used_size += params.num_blocks *
          ((params.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * (((graph.numstates + 1) + 31) / 32);
    }

    if (alg == SearchAlgorithm::NORMAL_MARKING) {
      // deadstates[] (8 bits/cycle)
      used_size += params.num_blocks *
          ((params.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 3) / 4);
    }

    if (alg == SearchAlgorithm::SUPER || alg == SearchAlgorithm::SUPER0) {
      // cycleused[] (1 bit/cycle)
      used_size += params.num_blocks *
          ((params.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
      // isexitcycle[] (1 bit/cycle)
      used_size += params.num_blocks *
          ((params.num_threadsperblock + 31) / 32) *
          sizeof(ThreadStorageUsed) * ((graph.numcycles + 31) / 32);
    }

    if (used_size != 0) {
      if (config.verboseflag) {
        jpout << "  allocating used[] in device memory (" << used_size
              << " bytes)\n";
      }
      throw_on_cuda_error(
        cudaMalloc(&(gptrs.used_d), used_size), __FILE__, __LINE__);
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

void CoordinatorCUDA::copy_graph_to_gpu()
{
  if (config.verboseflag) {
    print_string(std::format("  placing graph into {} memory ({} bytes)",
               (gptrs.graphmatrix_d != nullptr ? "device" : "constant"),
               sizeof(statenum_t) * graph_buffer.size()));
  }
  throw_on_cuda_error(
      cudaMemcpy(gptrs.graphmatrix_d != nullptr ? gptrs.graphmatrix_d :
          gptrs.graphmatrix_c, graph_buffer.data(),
          sizeof(statenum_t) * graph_buffer.size(), cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

// Copy static global variables to GPU global memory.

void CoordinatorCUDA::copy_static_vars_to_gpu()
{
  uint8_t maxoutdegree_h = static_cast<uint8_t>(graph.maxoutdegree);
  uint16_t numstates_h = static_cast<uint16_t>(graph.numstates);
  uint16_t numcycles_h = static_cast<uint16_t>(graph.numcycles);
  uint32_t pattern_buffer_size_h =
      static_cast<uint32_t>(params.pattern_buffer_size);
  uint32_t pattern_index_h = 0;
  throw_on_cuda_error(
      cudaMemcpy(gptrs.maxoutdegree_d, &maxoutdegree_h, sizeof(uint8_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(gptrs.numstates_d, &numstates_h, sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(gptrs.numcycles_d, &numcycles_h, sizeof(uint16_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(gptrs.pattern_buffer_size_d, &pattern_buffer_size_h,
          sizeof(uint32_t), cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(gptrs.pattern_index_d[0], &pattern_index_h, sizeof(uint32_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpy(gptrs.pattern_index_d[1], &pattern_index_h, sizeof(uint32_t),
          cudaMemcpyHostToDevice),
      __FILE__, __LINE__);
}

//------------------------------------------------------------------------------
// Main loop
//------------------------------------------------------------------------------

// Handle work assignments that can be (partially) skipped, by adjusting and
// reloading them.
//
// This occurs when either `start_state` is unusable, or the maximum pattern
// length is too short. This should exactly mirror the logic in
// Worker::do_work_assignment() to keep operation identical between CPU and GPU.

void CoordinatorCUDA::skip_unusable_startstates(unsigned bank)
{
  for (unsigned id = 0; id < config.num_threads; ++id) {
    auto& wi = wi_h[bank][id];
    if (wi.pos != -1)
      continue;  // not a new search at `start_state`, skip
    if ((wi.status & 1) != 0)
      continue;

    while (true) {
      if (wi.start_state > wi.end_state) {
        wi.status |= 1;
        break;
      }

      const auto max_possible = get_max_length(wi.start_state);
      if (max_possible == -1) {  // current start_state is unusable
        ++wi.start_state;
        continue;
      }

      if (max_possible < static_cast<int>(config.n_min)) {
        wi.status |= 1;
      }
      break;
    }

    if ((wi.status & 1) == 0) {
      WorkAssignment wa;
      wa.start_state = wi.start_state;
      wa.end_state = wi.end_state;
      load_work_assignment(bank, id, wa);
    }
  }
}

// Copy worker data to the GPU.
//
// This copies WorkerInfo and WorkCells for threads [0, max_active_idx[bank]].
// If optional `startup` is true then all WorkerInfo data is copied.

void CoordinatorCUDA::copy_worker_data_to_gpu(unsigned bank, bool startup)
{
  auto idx_count = startup ? config.num_threads : (max_active_idx[bank] + 1);

  throw_on_cuda_error(
      cudaMemcpyAsync(gptrs.wi_d[bank], wi_h[bank], sizeof(WorkerInfo) *
          idx_count, cudaMemcpyHostToDevice, stream[bank]),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyAsync(gptrs.wc_d[bank], wc_h[bank],
          sizeof(ThreadStorageWorkCell) * (max_active_idx[bank] / 32 + 1) *
          n_max, cudaMemcpyHostToDevice, stream[bank]),
      __FILE__, __LINE__);
}

// Launch the appropriate CUDA kernel.
//
// In the event of an error, throw a `std::runtime_error` exception with an
// appropriate error message.

void CoordinatorCUDA::launch_cuda_kernel(unsigned bank, uint64_t cycles)
{
  if (summary_before[bank].workers_idle.size() == config.num_threads)
    return;

  launch_kernel(gptrs, bank, alg, params, cycles, stream[bank]);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::format("CUDA Error in kernel: {}",
        cudaGetErrorString(err)));
  }
}

// Copy worker data from the GPU. Copy only the worker data for threads
// [0, max_active_idx[bank]].

void CoordinatorCUDA::copy_worker_data_from_gpu(unsigned bank)
{
  throw_on_cuda_error(
      cudaMemcpyAsync(wi_h[bank], gptrs.wi_d[bank],
          sizeof(WorkerInfo) * (max_active_idx[bank] + 1),
          cudaMemcpyDeviceToHost, stream[bank]),
      __FILE__, __LINE__);
  throw_on_cuda_error(
      cudaMemcpyAsync(wc_h[bank], gptrs.wc_d[bank],
          sizeof(ThreadStorageWorkCell) * (max_active_idx[bank] / 32 + 1) *
          n_max, cudaMemcpyDeviceToHost, stream[bank]),
      __FILE__, __LINE__);
}

// Process the worker counters after a kernel run, and reset to initial values.
// Also raise an exception if an error was encountered during kernel execution.

void CoordinatorCUDA::process_worker_counters(unsigned bank)
{
  if (longest_by_startstate_ever.size() > 0) {
    longest_by_startstate_current.assign(longest_by_startstate_ever.size(), 0);
  }

  for (unsigned id = 0; id < config.num_threads; ++id) {
    const auto errorcode = wi_h[bank][id].status & 6u;
    if (errorcode != 0) {
      std::ostringstream ss;
      ss << "bank " << bank << ", worker " << id << ": "
         << read_work_assignment(bank, id);
      std::string errorstring = ss.str();
      if (errorcode == 2) {
        throw std::runtime_error("Error during initialization: " + errorstring);
      } else if (errorcode == 4) {
        throw std::runtime_error("Error during runtime: " + errorstring);
      } else {
        throw std::runtime_error("Error (unknown): " + errorstring);
      }
    }

    context.nnodes += wi_h[bank][id].nnodes;
    wi_h[bank][id].nnodes = 0;

    const statenum_t st_state = wi_h[bank][id].start_state;
    if (st_state >= longest_by_startstate_ever.size()) {
      longest_by_startstate_ever.resize(st_state + 1, 0);
      longest_by_startstate_current.resize(st_state + 1, 0);
    }

    for (unsigned i = 0; i < n_max; ++i) {
      auto& cell = workcell(bank, id, i);
      if (cell.count == 0)
        continue;

      context.count.at(i + 1) += cell.count;
      context.ntotal += cell.count;
      if (i + 1 >= config.n_min && i + 1 <= n_max) {
        context.npatterns += cell.count;
      }
      if (i + 1 > longest_by_startstate_current.at(st_state)) {
        longest_by_startstate_current.at(st_state) =
            static_cast<unsigned>(i + 1);
        if (i + 1 > longest_by_startstate_ever.at(st_state)) {
          longest_by_startstate_ever.at(st_state) =
              static_cast<unsigned>(i + 1);
        }
      }
      cell.count = 0;
    }
  }
}

// Process the pattern buffer. Copy any patterns in the buffer to `context`,
// print them to the console if needed, then clear the buffer.
//
// Returns the count of patterns retrieved from the buffer.
//
// In the event of a pattern buffer overflow, throw a `std::runtime_error`
// exception with a relevant error message.

uint32_t CoordinatorCUDA::process_pattern_buffer(unsigned bank)
{
  if (gptrs.pb_d[bank] == nullptr) {
    return 0;
  }

  // get the number of patterns in the buffer
  throw_on_cuda_error(
    cudaMemcpyAsync(pattern_count_h, gptrs.pattern_index_d[bank],
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream[bank]),
    __FILE__, __LINE__
  );
  cudaStreamSynchronize(stream[bank]);

  if (*pattern_count_h == 0) {
    return 0;
  } else if (*pattern_count_h > params.pattern_buffer_size) {
    throw std::runtime_error("CUDA error: pattern buffer overflow");
  }

  // copy pattern data to host
  throw_on_cuda_error(
    cudaMemcpyAsync(pb_h, gptrs.pb_d[bank], sizeof(statenum_t) * n_max *
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
    cudaMemcpyAsync(gptrs.pattern_index_d[bank], pattern_count_h,
        sizeof(uint32_t), cudaMemcpyHostToDevice, stream[bank]),
    __FILE__, __LINE__
  );

  return pattern_count;
}

// Update the global time counters.

void CoordinatorCUDA::record_working_time(unsigned bank, double kernel_time,
    double host_time, uint64_t cycles_run)
{
  const auto idle_before =
      static_cast<unsigned>(summary_before[bank].workers_idle.size());
  const auto idle_after =
      static_cast<unsigned>(summary_after[bank].workers_idle.size());
  assert(idle_after >= idle_before);
  const uint64_t cycles_startup = summary_after[bank].cycles_startup;

  // negative host_time means that host processing finished before other bank's
  // kernel run completed; only count host_time > 0, when GPU was idle
  total_kernel_time += kernel_time;
  total_host_time += std::max(host_time, 0.0);

  // deduct kernel time spent doing initialization
  const double working_time = (cycles_startup + cycles_run == 0 ?
      kernel_time :
      kernel_time * static_cast<double>(cycles_run) /
      static_cast<double>(cycles_startup + cycles_run));

  // assume the workers that went idle did so at a uniform rate
  context.secs_working += working_time *
      (config.num_threads - idle_before / 2 - idle_after / 2);
}

// Calculate the next number of kernel cycles to run, based on timing and
// progress.

uint64_t CoordinatorCUDA::calc_next_kernel_cycles(uint64_t last_cycles,
      unsigned bank, double kernel_time, double host_time, unsigned idle_start,
      uint32_t pattern_count)
{
  if (idle_start == config.num_threads) {
    return last_cycles;  // no workers ran last time
  }
  if (idle_start > config.num_threads / 2) {
    if (!warmed_up[bank]) {
      return MINCYCLES;  // run for a short time so we can split
    }
  } else {
    warmed_up[bank] = true;
  }

  const auto next_idle_start =
      static_cast<unsigned>(summary_before[bank].workers_idle.size());
  const auto idle_after =
      static_cast<unsigned>(summary_after[bank].workers_idle.size());
  const uint64_t last_cycles_startup = summary_after[bank].cycles_startup;
  assert(idle_after >= idle_start);

  // GPU cycles per second
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

  // Solve for x == target_cycles, the number of cycles to run next.
  //
  // We approach this as an optimization problem, maximizing the useful work
  // completed per unit (wall clock) time.
  //
  // The average number of running workers will be:
  //     workers - beta * x / 2
  // and the workers will be doing useful work for duration `x`. Including the
  // (non-useful) startup time s == startup, we express the useful work in terms
  // of worker equivalents averaged over the total time:
  //     (workers - beta * x / 2) * x / (x + s)
  //
  // Maximizing this with respect to x gives:
  //     x = sqrt(s^2 + 2 * s * workers / beta) - s

  double target_cycles = (beta > 0.0 ?
      sqrt(startup * startup + 2 * startup * workers / beta) - startup :
      static_cast<double>(2 * last_cycles));

  // apply constraints
  target_cycles = std::max(target_cycles, static_cast<double>(MINCYCLES));
  double target_time = (startup + target_cycles) / cps;
  // ensure the kernel won't finish before host processing of the other bank is
  // done; note that `kernel_time + host_time` is the real host processing time
  target_time = std::max(target_time, 1.0 * (kernel_time + host_time));
  target_time = std::min(target_time, 2.0);  // max of 2 seconds
  target_cycles = target_time * cps - startup;

  // try to keep the pattern buffer from overflowing
  if (pattern_count > 0) {
    const auto max_cycles = static_cast<double>(last_cycles) *
        static_cast<double>(params.pattern_buffer_size / 3) /
        static_cast<double>(pattern_count);
    target_cycles = std::min(target_cycles, max_cycles);
  }

  return static_cast<uint64_t>(target_cycles);
}

//------------------------------------------------------------------------------
// Cleanup
//------------------------------------------------------------------------------

// Gather unfinished work assignments.

void CoordinatorCUDA::gather_unfinished_work_assignments()
{
  for (unsigned bank = 0; bank < 2; ++bank) {
    for (unsigned id = 0; id < config.num_threads; ++id) {
      if ((wi_h[bank][id].status & 1) == 0) {
        WorkAssignment wa = read_work_assignment(bank, id);
        context.assignments.push_back(wa);
      }
      wi_h[bank][id].status |= 1;
    }
  }
}

// Destroy CUDA streams and free allocated GPU and host memory.

void CoordinatorCUDA::cleanup()
{
  // CUDA streams
  for (int i = 0; i < 2; ++i) {
    if (stream[i] != nullptr) {
      cudaStreamDestroy(stream[i]);
      stream[i] = nullptr;
    }
  }

  // GPU memory
  for (unsigned bank = 0; bank < 2; ++bank) {
    if (gptrs.wi_d[bank] != nullptr) {
      cudaFree(gptrs.wi_d[bank]);
      gptrs.wi_d[bank] = nullptr;
    }
    if (gptrs.wc_d[bank] != nullptr) {
      cudaFree(gptrs.wc_d[bank]);
      gptrs.wc_d[bank] = nullptr;
    }
    if (gptrs.pb_d[bank] != nullptr) {
      cudaFree(gptrs.pb_d[bank]);
      gptrs.pb_d[bank] = nullptr;
    }
  }
  if (gptrs.graphmatrix_d != nullptr) {
    cudaFree(gptrs.graphmatrix_d);
    gptrs.graphmatrix_d = nullptr;
  }
  if (gptrs.used_d != nullptr) {
    cudaFree(gptrs.used_d);
    gptrs.used_d = nullptr;
  }

  // Host memory
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
// Summarization and status display
//------------------------------------------------------------------------------

// Produce a summary of the current worker status.

CudaWorkerSummary CoordinatorCUDA::summarize_worker_status(unsigned bank)
{
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
        static_cast<uint32_t>(config.num_threads - summary.workers_idle.size());
  } else {
    summary.cycles_startup = 0;
  }
  return summary;
}

// Summarize all active jobs in the worker banks and the assignments queue.

CudaWorkerSummary CoordinatorCUDA::summarize_all_jobs(
    const CudaWorkerSummary& last, const CudaWorkerSummary& prev)
{
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

void CoordinatorCUDA::do_status_display(unsigned bankB, double kernel_time,
      double host_time, int run)
{
  // some housekeeping before the status display
  const auto& summary_afterB = summary_after[bankB];
  const auto& summary_beforeA = summary_before[1 - bankB];
  njobs += (summary_afterB.workers_idle.size() -
      summary_before[bankB].workers_idle.size());

  if (config.verboseflag) {
    print_string(std::format(
        "run = {}, kernel = {:.3}, host = {:.3}, startup = {}, busy = {}",
        run, kernel_time, host_time, summary_afterB.cycles_startup,
        config.num_threads - summary_afterB.workers_idle.size()));
  }

  if (!config.statusflag)
    return;
  const auto now = std::chrono::high_resolution_clock::now();
  if (calc_duration_secs(last_status_time, now) < 1.0) {
    return;
  }

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

  const double elapsed = calc_duration_secs(last_status_time, now);
  const double jobspersec = static_cast<double>(njobs - last_njobs) / elapsed;
  const double nodespersec =
      static_cast<double>(summary_afterB.nnodes - last_nnodes) / elapsed;
  const double patspersec =
      static_cast<double>(summary_afterB.ntotal - last_ntotal) / elapsed;

  last_status_time = now;
  last_njobs = njobs;
  last_nnodes = summary_afterB.nnodes;
  last_ntotal = summary_afterB.ntotal;

  status_lines.push_back(std::format(
    "jobs/s: {}, nodes/s: {}, pats/s: {}, pats in range:{:19}",
    format2(jobspersec), format2(nodespersec), format2(patspersec),
    context.npatterns
  ));

  erase_status_output();
  print_status_output();
}

//------------------------------------------------------------------------------
// Manage work assignments
//------------------------------------------------------------------------------

// Load initial work assignments and copy worker data to the GPU.

void CoordinatorCUDA::load_initial_work_assignments()
{
  for (unsigned id = 0; id < config.num_threads; ++id) {
    if (context.assignments.empty())
      break;

    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();

    if (wa.get_type() == WorkAssignment::Type::STARTUP) {
      wa.start_state = (config.groundmode ==
          SearchConfig::GroundMode::EXCITED_SEARCH ? 2 : 1);
      wa.end_state = (config.groundmode ==
          SearchConfig::GroundMode::GROUND_SEARCH ? 1 : graph.numstates);
    }

    load_work_assignment(0, id, wa);
    max_active_idx[0] = id;
  }

  for (unsigned bank = 0; bank < 2; ++bank) {
    skip_unusable_startstates(bank);
    copy_worker_data_to_gpu(bank, true);
    summary_before[bank] = summarize_worker_status(bank);
  }
}

// Load a work assignment into a worker's slot in the `WorkerInfo` and
// `ThreadStorageWorkCell` arrays.

void CoordinatorCUDA::load_work_assignment(unsigned bank, const unsigned id,
    const WorkAssignment& wa)
{
  wa.to_workspace(this, bank * config.num_threads + id);
  wi_h[bank][id].nnodes = 0;
  wi_h[bank][id].status = 0;

#ifndef NDEBUG
  // verify the assignment is unchanged by round trip through the workspace
  WorkAssignment wa2;
  wa2.from_workspace(this, bank * config.num_threads + id);
  assert(wa == wa2);
#endif
}

// Read out the work assignment for worker `id` from the workcells in bank
// `bank`.
//
// This is a non-destructive read, i.e., the workcells are unchanged.

WorkAssignment CoordinatorCUDA::read_work_assignment(unsigned bank,
    unsigned id)
{
  WorkAssignment wa;
  wa.from_workspace(this, bank * config.num_threads + id);
  return wa;
}

// Assign new jobs to idle workers in bankB.

void CoordinatorCUDA::assign_new_jobs(unsigned bankB)
{
  const CudaWorkerSummary& summary = summary_after[bankB];
  const auto idle_before_a =
      static_cast<unsigned>(summary_before[1 - bankB].workers_idle.size());
  const auto idle_after_b =
      static_cast<unsigned>(summary.workers_idle.size());

  // split working jobs and add them to the pool, until all workers are
  // processed or pool size >= target_job_count
  //
  // jobs needed:
  //   this bank (bankB):  `idle_after_b`
  //   other bank (bankA): `idle_before_a`

  const unsigned target_job_count = idle_after_b + idle_before_a;

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

    WorkAssignment wa = read_work_assignment(bankB, *it);
    if (wa.is_splittable()) {
      WorkAssignment wa2 = wa.split(graph, config.split_alg);
      load_work_assignment(bankB, *it, wa);
      context.assignments.push_back(wa2);
      ++context.splits_total;

      // Avoid double counting nodes: Each of the nodes in the partial path for
      // the new assignment will be reported twice to the coordinator: by the
      // worker doing the original job `wa`, and by the worker that does `wa2`.
      if (wa.start_state == wa2.start_state) {
        wi_h[bankB][*it].nnodes -= wa2.partial_pattern.size();
      }
    }
    ++it;
  }

  // assign to idle workers from the pool

  for (unsigned id = 0; id < config.num_threads; ++id) {
    if ((wi_h[bankB][id].status & 1) == 0)
      continue;
    if (context.assignments.empty())
      break;

    WorkAssignment wa = context.assignments.front();
    context.assignments.pop_front();
    load_work_assignment(bankB, id, wa);
    max_active_idx[bankB] = std::max(max_active_idx[bankB], id);
  }
}

//------------------------------------------------------------------------------
// Helper methods
//------------------------------------------------------------------------------

// Return a reference to the workcell for thread `id`, position `pos`.

ThreadStorageWorkCell& CoordinatorCUDA::workcell(unsigned bank, unsigned id,
    unsigned pos)
{
  ThreadStorageWorkCell* start_warp = &wc_h[bank][(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}

const ThreadStorageWorkCell& CoordinatorCUDA::workcell(unsigned bank,
    unsigned id, unsigned pos) const
{
  ThreadStorageWorkCell* start_warp = &wc_h[bank][(id / 32) * n_max];
  uint32_t* start_warp_u32 = reinterpret_cast<uint32_t*>(start_warp);
  ThreadStorageWorkCell* start_thread =
      reinterpret_cast<ThreadStorageWorkCell*>(&start_warp_u32[id & 31]);
  return start_thread[pos];
}

// Handle CUDA errors by throwing a `std::runtime_error` exception with a
// relevant error message.

void CoordinatorCUDA::throw_on_cuda_error(cudaError_t code, const char *file,
      int line)
{
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

const Graph& CoordinatorCUDA::get_graph() const
{
  return graph;
}

void CoordinatorCUDA::set_cell(unsigned slot, unsigned index, unsigned col,
    unsigned col_limit, unsigned from_state)
{
  assert(slot < 2 * config.num_threads);
  assert(index < n_max);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  ThreadStorageWorkCell& wc = workcell(bank, id, index);
  wc.col = static_cast<uint8_t>(col);
  wc.col_limit = static_cast<uint8_t>(col_limit);
  wc.from_state = static_cast<statenum_t>(from_state);
}

std::tuple<unsigned, unsigned, unsigned> CoordinatorCUDA::get_cell(
    unsigned slot, unsigned index) const
{
  assert(slot < 2 * config.num_threads);
  assert(index < n_max);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  const ThreadStorageWorkCell& wc = workcell(bank, id, index);
  return std::make_tuple(wc.col, wc.col_limit, wc.from_state);
}

void CoordinatorCUDA::set_info(unsigned slot, unsigned new_start_state,
    unsigned new_end_state, int new_pos)
{
  assert(slot < 2 * config.num_threads);
  unsigned bank = slot < config.num_threads ? 0 : 1;
  unsigned id = bank == 0 ? slot : slot - config.num_threads;

  WorkerInfo& wi = wi_h[bank][id];
  wi.start_state = static_cast<statenum_t>(new_start_state);
  wi.end_state = static_cast<statenum_t>(new_end_state);
  wi.pos = static_cast<int16_t>(new_pos);
}

std::tuple<unsigned, unsigned, int> CoordinatorCUDA::get_info(unsigned slot)
    const
{
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

void CUDART_CB record_kernel_completion_time(void* data)
{
  jptimer_t* ptr = (jptimer_t*)data;
  *ptr = std::chrono::high_resolution_clock::now();
}
