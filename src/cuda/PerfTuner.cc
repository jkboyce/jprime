//
// PerfTuner.cc
//
// Service that optimizes performance on a CUDA GPU by adjusting runtime
// parameters.
//
// It works by alternating between an "experiment" phase where candidate
// RunConfigs are tested, and an "exploit" phase where the RunConfig with the
// best reported performance is consistently used. An exponential backoff makes
// experiment phases less frequent as the calculation proceeds.
//
// Copyright (C) 2025-2026 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "PerfTuner.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>


PerfTuner::PerfTuner(unsigned max_w, RunConfig start)
    : max_warps(max_w),
      exploitation_run_count(runs_between_experiments),
      start_config(start),
      best_config(start)
{
  for (unsigned c = 0; c < 4; ++c) {
    perf[c].assign(max_warps + 1, 0.0);
    infeasible[c].assign(max_warps + 1, false);
  }
}

//------------------------------------------------------------------------------
// Methods for Coordinator to interact with PerfTuner
//------------------------------------------------------------------------------

// Receive the measured performance of a just-completed run.
//
// This is generally called just before get_next_config() below.

void PerfTuner::report_run_perf(unsigned bank, double last_perf,
    bool all_workers_assigned)
{
  assert(bank < 2);
  assert(last_perf >= 0.0 && std::isfinite(last_perf));
  banks[bank].warmed_up |= all_workers_assigned;

  // If valid perf measurement, then record results
  if (banks[bank].has_config && all_workers_assigned) {
    const auto& last = banks[bank].last_config;
    const unsigned c = combo_index(last);
    perf[c][last.warps] = last_perf;

    if (last_perf > best_perf) {
      best_perf = last_perf;

      if (last != best_config) {
        best_config = last;
        // new optimum; immediately start a new round of experiments
        start_experiment_round();
        return;
      }
    }
  }

  if (banks[bank].is_experiment && experiment_queue.empty()) {
    const bool other_in_flight = banks[1 - bank].is_experiment;
    if (!other_in_flight) {
      backoff();
    }
  }
}

// Get the next RunConfig for bank `bank`.
//
// If `retry` is true then this call is an immediate re-try of a previous
// call that returned an infeasible RunConfig. We keep track of infeasible
// RunConfigs so that we don't return them again in the future.

RunConfig PerfTuner::get_next_config(unsigned bank, bool retry)
{
  assert(bank < 2);

  if (retry) {
    // The previous configuration returned for this bank was infeasible
    assert(banks[bank].has_config);
    const auto& last = banks[bank].last_config;
    const unsigned c = combo_index(last);
    infeasible[c][last.warps] = true;

    if (last == best_config) {
      // can only happen at startup, when `start_config` is infeasible; fall
      // back to a value that is feasible on any CUDA device
      best_config = RunConfig {
        .graph_in_shared = false,
        .used_in_shared = false,
        .warps = 1
      };
    }
  }

  RunConfig next_config = best_config;
  bool next_is_experiment = false;

  if (banks[bank].warmed_up) {
    if (experiment_queue.empty() && !banks[1 - bank].is_experiment && !retry) {
      ++exploitation_run_count;
      if (exploitation_run_count >= runs_between_experiments) {
        start_experiment_round();
      }
    }

    while (!experiment_queue.empty()) {
      const auto candidate = experiment_queue.front();
      experiment_queue.pop_front();
      const unsigned c = combo_index(candidate);
      if (!infeasible[c][candidate.warps]) {
        next_config = candidate;
        next_is_experiment = true;
        break;
      }
    }
  }

  assert(!infeasible[combo_index(next_config)][next_config.warps]);

  banks[bank].last_config = next_config;
  banks[bank].is_experiment = next_is_experiment;
  banks[bank].has_config = true;

  return next_config;
}

//------------------------------------------------------------------------------
// For logging output
//------------------------------------------------------------------------------

bool PerfTuner::last_was_experiment(unsigned bank) const
{
  assert(bank < 2);
  return banks[bank].is_experiment;
}

RunConfig PerfTuner::get_best_config() const
{
  return best_config;
}

//------------------------------------------------------------------------------
// Private methods
//------------------------------------------------------------------------------

unsigned PerfTuner::combo_index(const RunConfig& config)
{
  return (config.graph_in_shared ? 2U : 0U) | (config.used_in_shared ? 1U : 0U);
}


RunConfig PerfTuner::make_config(unsigned combo, unsigned warps)
{
  return RunConfig {
    .graph_in_shared = (combo & 2U) != 0,
    .used_in_shared = (combo & 1U) != 0,
    .warps = warps
  };
}


unsigned PerfTuner::get_best_warps_for_combo(unsigned combo) const
{
  unsigned best_w = 0;
  double max_p = 0.0;

  for (unsigned w = 1; w <= max_warps; ++w) {
    if (perf[combo][w] > max_p) {
      max_p = perf[combo][w];
      best_w = w;
    }
  }

  if (best_w == 0) {
    best_w = std::clamp(best_config.warps, 1U, max_warps);
  }

  return best_w;
}


void PerfTuner::start_experiment_round()
{
  experiment_queue.clear();
  exploitation_run_count = 0;

  if (best_perf == 0.0) {
    const unsigned c = combo_index(start_config);
    if (!infeasible[c][start_config.warps]) {
      experiment_queue.push_back(start_config);
    }
  }

  const unsigned best_combo = combo_index(best_config);

  for (unsigned i = 0; i < 4; ++i) {
    const unsigned combo = (best_combo + i) % 4;
    const unsigned best_w = get_best_warps_for_combo(combo);

    constexpr int OFFSETS[] = { 0, 1, -1 };
    for (int offset : OFFSETS) {
      const int w = static_cast<int>(best_w) + offset;
      if (w < 1) {
        continue;
      }
      const auto warps = static_cast<unsigned>(w);
      if (warps > max_warps || infeasible[combo][warps]) {
        continue;
      }

      const RunConfig candidate = make_config(combo, warps);

      // Skip current best RunConfig
      if (best_perf > 0.0 && candidate == best_config) {
        continue;
      }

      // Avoid duplicate entries in experiment queue
      if (std::ranges::find(experiment_queue, candidate) ==
          experiment_queue.end()) {
        experiment_queue.push_back(candidate);
      }
    }
  }

  if (experiment_queue.empty()) {
    backoff();
  }
}


void PerfTuner::backoff()
{
  if (runs_between_experiments <= std::numeric_limits<unsigned>::max() / 2) {
    runs_between_experiments *= 2;
  } else {
    runs_between_experiments = std::numeric_limits<unsigned>::max();
  }
  exploitation_run_count = 0;
}

