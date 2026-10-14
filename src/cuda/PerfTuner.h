//
// PerfTuner.h
//
// Service that optimizes performance on a CUDA GPU by varying runtime
// parameters.
//
// Copyright (C) 2025-2026 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_PERFTUNER_H_
#define JPRIME_PERFTUNER_H_

#include "CudaTypes.h"

#include <deque>
#include <vector>


class PerfTuner {
 public:
  PerfTuner() = default;
  PerfTuner(unsigned max_w, RunConfig start_config = {
    .graph_in_shared = true,
    .used_in_shared = true,
    .warps = 1
  });

  void report_run_perf(unsigned bank, double last_perf,
      bool all_workers_assigned);
  RunConfig get_next_config(unsigned bank, bool retry);
  bool last_was_experiment(unsigned bank) const;
  RunConfig get_best_config() const;

 private:
  struct BankState {
    RunConfig last_config {};
    bool has_config = false;
    bool is_experiment = false;
    bool warmed_up = false;
  };

  unsigned max_warps = 0;
  unsigned runs_between_experiments = 10;
  unsigned exploitation_run_count = 0;

  // for each of the four combinations of { used_in_shared, graph_in_shared },
  // performance for each warp count
  std::vector<double> perf[4];

  // memory-infeasible combinations learned via retry
  std::vector<bool> infeasible[4];

  RunConfig start_config {};
  RunConfig best_config {};
  double best_perf = 0.0;

  // bank state for double-buffered execution
  BankState banks[2];

  // experiments queue
  std::deque<RunConfig> experiment_queue;

  static unsigned combo_index(const RunConfig& config);
  static RunConfig make_config(unsigned combo, unsigned warps);
  unsigned get_best_warps_for_combo(unsigned combo) const;
  void start_experiment_round();
  void backoff();
};

#endif
