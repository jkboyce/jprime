//
// Coordinator.cc
//
// Coordinator that manages the overall search.
//
// The computation is depth first search on multiple worker threads using work
// stealing to keep the workers busy. The business of the coordinator is to
// interact with the workers to distribute work, and also to manage output.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Coordinator.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cassert>
#include <format>
#include <stdexcept>


Coordinator::Coordinator(const SearchConfig& a, SearchContext& b,
    std::ostream& c) : config(a), context(b), jpout(c) {}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the calculation specified in `config`, storing results in `context`,
// and sending console output to `jpout`.
//
// Returns true on success, false on failure.

bool Coordinator::run() {
  try {
    calc_graph_size();
  } catch (const std::overflow_error& oe) {
    jpout << "Overflow occurred computing graph size\n";
    return false;
  }
  if (!passes_prechecks()) {
    return false;
  }

  // the search is a go and `n_bound` fits into an unsigned int
  n_max = (config.n_max > 0) ? config.n_max
      : static_cast<unsigned>(context.n_bound);
  context.count.resize(n_max + 1, 0);

  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  const auto start = std::chrono::high_resolution_clock::now();
  #ifdef CUDA_ENABLED
  if (config.cudaflag) {
    try {
      run_cuda();
    } catch (const std::runtime_error& re) {
      jpout << re.what() << '\n';
      return false;
    }
  } else {
    run_cpu();
  }
  #else
  run_cpu();
  #endif
  const auto end = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double> diff = end - start;
  const double runtime = diff.count();
  context.secs_elapsed += runtime;
  context.secs_available += runtime * config.num_threads;

  erase_status_output();
  if (config.verboseflag) {
    jpout << "Finished on: " << current_time_string();
  }
  if (context.assignments.size() > 0) {
    jpout << "\nPARTIAL RESULTS:\n";
  }
  print_search_description();
  print_results();
  return true;
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

// Determine as much as possible about the size of the computation before
// starting up the workers, saving results in `context`.
//
// Note these quantities may be very large so we use uint64s for all of them.
//
// In the event of a math overflow error, throw a `std::overflow_error`
// exception with a relevant error message.

void Coordinator::calc_graph_size() {
  // size of the full graph
  context.full_numstates = Graph::combinations(config.h, config.b);
  context.full_numcycles = 0;
  context.full_numshortcycles = 0;
  unsigned max_cycle_period = 0;

  for (unsigned p = 1; p <= config.h; ++p) {
    const auto cycles = Graph::shift_cycle_count(config.b, config.h, p);
    context.full_numcycles += cycles;
    if (p < config.h) {
      context.full_numshortcycles += cycles;
    }
    if (cycles > 0) {
      max_cycle_period = p;
    }
  }

  // largest period possible of the pattern type selected, if all states are
  // active
  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    // two possibilities: Stay on a single cycle, or use multiple cycles
    context.n_bound = std::max(static_cast<std::uint64_t>(max_cycle_period),
        context.full_numstates - context.full_numcycles);
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    context.n_bound = (context.full_numcycles > 1 ?
        context.full_numcycles + config.shiftlimit : 0);
  }

  // number of states that will be resident in memory if we build the graph
  if (config.graphmode == SearchConfig::GraphMode::FULL_GRAPH) {
    context.memory_numstates = context.full_numstates;
  } else if (config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH) {
    context.memory_numstates =
        Graph::ordered_partitions(config.b, config.h, config.n_min);
  }
}

// Perform checks before starting the workers.
//
// Returns true if the search is cleared to proceed.

bool Coordinator::passes_prechecks() {
  const auto n_requested = std::max(config.n_min, config.n_max);
  const bool period_error = (n_requested > context.n_bound);
  const bool memory_error = (context.memory_numstates > MAX_STATES);

  if (!config.infoflag && !period_error && !memory_error) {
    return true;
  }

  print_search_description();
  if (period_error) {
    jpout << std::format(
               "ERROR: Requested period {} is greater than bound of {}\n",
               n_requested, context.n_bound);
  }
  if (memory_error) {
    jpout << std::format(
               "ERROR: Number of states {} exceeds memory limit of {}\n",
               context.memory_numstates, MAX_STATES);
  }
  return false;
}

// Use the distribution of patterns found so far to extrapolate the expected
// number of patterns at period `n_bound`. This may be a useful signal of the
// degree of search completion.
//
// The distribution of patterns by period is observed to closely follow a
// Gaussian (normal) shape, so we fit the logarithm to a parabola and use that
// to extrapolate.

double Coordinator::expected_patterns_at_maxperiod() {
  size_t mode = 0;
  size_t max = 0;
  std::uint64_t modeval = 0;

  for (size_t i = 0; i < context.count.size(); ++i) {
    if (context.count.at(i) > modeval) {
      mode = i;
      modeval = context.count.at(i);
    }
    if (context.count.at(i) > 0) {
      max = i;
    }
  }

  // fit a parabola to the log of pattern count
  double s1 = 0, sx = 0, sx2 = 0, sx3 = 0, sx4 = 0;
  double sy = 0, sxy = 0, sx2y = 0;
  const size_t xstart = std::max(max - 10, mode);

  for (size_t i = xstart; i < context.count.size(); ++i) {
    if (context.count.at(i) < 5)
      continue;

    const double x = static_cast<double>(i);
    const double y = log(static_cast<double>(context.count.at(i)));
    s1 += 1;
    sx += x;
    sx2 += x * x;
    sx3 += x * x * x;
    sx4 += x * x * x * x;
    sy += y;
    sxy += x * y;
    sx2y += x * x * y;
  }

  // Solve this 3x3 linear system for A, B, C, the coefficients in the parabola
  // of best fit y = Ax^2 + Bx + C:
  //
  // | sx4  sx3  sx2  | | A |   | sx2y |
  // | sx3  sx2  sx   | | B | = | sxy  |
  // | sx2  sx   s1   | | C |   | sy   |

  // Find matrix inverse
  const double det = sx4 * (sx2 * s1 - sx * sx) - sx3 * (sx3 * s1 - sx * sx2) +
      sx2 * (sx3 * sx - sx2 * sx2);
  const double M11 = (sx2 * s1 - sx * sx) / det;
  const double M12 = (sx2 * sx - sx3 * s1) / det;
  const double M13 = (sx3 * sx - sx2 * sx2) / det;
  const double M21 = M12;
  const double M22 = (sx4 * s1 - sx2 * sx2) / det;
  const double M23 = (sx2 * sx3 - sx4 * sx) / det;
  const double M31 = M13;
  const double M32 = M23;
  const double M33 = (sx4 * sx2 - sx3 * sx3) / det;

  auto is_close = [](double a, double b) {
    double epsilon = 1e-3;
    return (b > a - epsilon && b < a + epsilon);
  };
  assert(is_close(M11 * sx4 + M12 * sx3 + M13 * sx2, 1));
  assert(is_close(M11 * sx3 + M12 * sx2 + M13 * sx, 0));
  assert(is_close(M11 * sx2 + M12 * sx + M13 * s1, 0));
  assert(is_close(M21 * sx4 + M22 * sx3 + M23 * sx2, 0));
  assert(is_close(M21 * sx3 + M22 * sx2 + M23 * sx, 1));
  assert(is_close(M21 * sx2 + M22 * sx + M23 * s1, 0));
  assert(is_close(M31 * sx4 + M32 * sx3 + M33 * sx2, 0));
  assert(is_close(M31 * sx3 + M32 * sx2 + M33 * sx, 0));
  assert(is_close(M31 * sx2 + M32 * sx + M33 * s1, 1));

  const double A = M11 * sx2y + M12 * sxy + M13 * sy;
  const double B = M21 * sx2y + M22 * sxy + M23 * sy;
  const double C = M31 * sx2y + M32 * sxy + M33 * sy;

  // evaluate the expected number of patterns found at x = n_bound
  const double x = static_cast<double>(context.n_bound);
  const double lny = A * x * x + B * x + C;
  return exp(lny);
}

// Static variable for indicating the user has interrupted execution.

volatile sig_atomic_t Coordinator::stopping = 0;

// Respond to a SIGINT (ctrl-c) interrupt during execution.

void Coordinator::signal_handler(int signum) {
  (void)signum;
  stopping = true;
}

//------------------------------------------------------------------------------
// Handle terminal output
//------------------------------------------------------------------------------

// Handle a pattern sent to us by a worker. We store it and optionally print it
// to the terminal.

void Coordinator::process_search_result(const MessageW2C& msg) {
  // workers only send patterns in the target period range
  context.patterns.push_back(msg.pattern);

  if (config.printflag) {
    erase_status_output();
    if (config.verboseflag) {
      jpout << msg.worker_id << ": " << msg.pattern << std::endl;
    } else {
      jpout << msg.pattern << std::endl;
    }
    print_status_output();
  }
}

void Coordinator::print_search_description() const {
  jpout << std::format("objects: {}, max throw: {}\n",
      (config.dualflag ? config.h - config.b : config.b), config.h);

  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    jpout << "prime ";
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    jpout << "superprime ";
    if (config.shiftlimit == 1) {
      jpout << "(+1 shift) ";
    } else {
      jpout << std::format("(+{} shifts) ", config.shiftlimit);
    }
  }
  jpout << "search for period: " << config.n_min;
  if (config.n_max != config.n_min) {
    if (config.n_max == 0) {
      jpout << '-';
    } else {
      jpout << '-' << config.n_max;
    }
  }
  jpout << std::format(" (bound {})", context.n_bound);
  if (config.groundmode == SearchConfig::GroundMode::GROUND_SEARCH) {
    jpout << ", ground state only\n";
  } else if (config.groundmode == SearchConfig::GroundMode::EXCITED_SEARCH) {
    jpout << ", excited states only\n";
  } else {
    jpout << '\n';
  }

  jpout << std::format("graph: {} states, {} shift cycles, {} short cycles",
             context.full_numstates, context.full_numcycles,
             context.full_numshortcycles)
        << std::endl;

  if (config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH) {
    jpout << std::format("period-{} subgraph: {} {}", config.n_min,
               context.memory_numstates,
               context.memory_numstates == 1 ? "state" : "states")
          << std::endl;
  }
}

void Coordinator::print_results() const {
  jpout << std::format("{} {} in range ({} seen, {} {})\n",
             context.npatterns,
             context.npatterns == 1 ? "pattern" : "patterns",
             context.ntotal, context.nnodes,
             context.nnodes == 1 ? "node" : "nodes");

  jpout << std::format("runtime = {:.4f} sec ({:.1f}M nodes/sec",
             context.secs_elapsed, static_cast<double>(context.nnodes) /
             context.secs_elapsed / 1000000);
  if (config.num_threads > 1) {
    jpout << std::format(", {:.1f} % util, {} {})\n",
               (context.secs_working / context.secs_available) * 100,
               context.splits_total,
               context.splits_total == 1 ? "split" : "splits");
  } else {
    jpout << ")\n";
  }

  if (config.countflag || n_max > config.n_min) {
    jpout << "\nPattern count by period:\n";
    for (unsigned i = config.n_min; i <= n_max; ++i) {
      jpout << i << ", " << context.count.at(i) << '\n';
    }
  }
}

void Coordinator::erase_status_output() const {
  if (!config.statusflag || !stats_printed)
    return;
  for (unsigned i = 0; i < config.num_threads + 2; ++i) {
    std::cout << '\x1B' << "[1A"
              << '\x1B' << "[2K";
  }
}

void Coordinator::print_status_output() {
  if (!config.statusflag)
    return;

  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && n_max > 3 * STATUS_WIDTH);
  std::cout << "Status on: " << current_time_string();
  std::cout << " cur/ end  rp options remaining at position";
  if (compressed) {
    std::cout << " (compressed view)";
    for (int i = 47; i < STATUS_WIDTH; ++i) {
      std::cout << ' ';
    }
  } else {
    for (int i = 29; i < STATUS_WIDTH; ++i) {
      std::cout << ' ';
    }
  }
  std::cout << "    period\n";
  for (unsigned i = 0; i < config.num_threads; ++i) {
    std::cout << worker_status.at(i) << std::endl;
  }

  stats_printed = true;
}

std::string Coordinator::current_time_string() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_timet = std::chrono::system_clock::to_time_t(now);
  return std::ctime(&now_timet);
}

//------------------------------------------------------------------------------
// Manage worker status
//------------------------------------------------------------------------------

// Copy status data out of the worker message, into appropriate data structures
// in the coordinator.

void Coordinator::record_data_from_message(const MessageW2C& msg) {
  context.nnodes += msg.nnodes;
  context.secs_working += msg.secs_working;

  // pattern counts by period
  assert(msg.count.size() == n_max + 1);
  assert(context.count.size() == n_max + 1);

  for (size_t i = 1; i < msg.count.size(); ++i) {
    context.count.at(i) += msg.count.at(i);
    context.ntotal += msg.count.at(i);
    if (i >= config.n_min && i <= n_max) {
      context.npatterns += msg.count.at(i);
    }
    if (config.statusflag && msg.count.at(i) > 0) {
      worker_longest_start.at(msg.worker_id) = std::max(
          worker_longest_start.at(msg.worker_id), static_cast<unsigned>(i));
      worker_longest_last.at(msg.worker_id) = std::max(
          worker_longest_last.at(msg.worker_id), static_cast<unsigned>(i));
    }
  }
}

// Create a status display for a worker, showing its current state in the
// search.

std::string Coordinator::make_worker_status(const MessageW2C& msg) {
  std::ostringstream buffer;

  if (!msg.running) {
    buffer << "   -/   -   - IDLE";
    for (int i = 1; i < STATUS_WIDTH; ++i) {
      buffer << ' ';
    }
    buffer << "-    -";
    return buffer.str();
  }

  const unsigned id = msg.worker_id;
  const unsigned root_pos = worker_rootpos.at(id);
  const std::vector<unsigned>& ops = msg.worker_options_left;
  const std::vector<unsigned>& ds_extra = msg.worker_deadstates_extra;
  std::vector<unsigned>& ops_start = worker_options_left_start.at(id);
  std::vector<unsigned>& ops_last = worker_options_left_last.at(id);

  buffer << std::setw(4) << std::min(worker_startstate.at(id), 9999u) << '/';
  buffer << std::setw(4) << std::min(worker_endstate.at(id), 9999u) << ' ';
  buffer << std::setw(3) << std::min(worker_rootpos.at(id), 999u) << ' ';

  const bool compressed = (config.mode == SearchConfig::RunMode::NORMAL_SEARCH
      && n_max > 3 * STATUS_WIDTH);
  const bool show_deadstates =
      (config.mode == SearchConfig::RunMode::NORMAL_SEARCH &&
      config.graphmode == SearchConfig::GraphMode::FULL_GRAPH);
  const bool show_shifts = (config.mode == SearchConfig::RunMode::SUPER_SEARCH);

  unsigned printed = 0;
  bool hl_start = false;
  bool did_hl_start = false;
  bool hl_last = false;
  bool did_hl_last = false;
  bool hl_deadstate = false;
  bool hl_shift = false;

  assert(ops.size() == msg.worker_throw.size());
  assert(ops.size() == ds_extra.size());

  for (size_t i = 0; i < ops.size(); ++i) {
    const unsigned throwval = msg.worker_throw.at(i);

    if (!hl_start && !did_hl_start && i < ops_start.size() &&
        ops.at(i) != ops_start.at(i)) {
      hl_start = did_hl_start = true;
    }
    if (!hl_last && !did_hl_last && i < ops_last.size() &&
        ops.at(i) != ops_last.at(i)) {
      hl_last = did_hl_last = true;
    }
    if (show_deadstates && i < ds_extra.size() && ds_extra.at(i) > 0) {
      hl_deadstate = true;
    }
    if (show_shifts && i < msg.worker_throw.size() && (throwval == 0 ||
        throwval == config.h)) {
      hl_shift = true;
    }

    if (i < root_pos)
      continue;

    char ch = '\0';

    if (compressed) {
      if (i == root_pos) {
        ch = '0' + ops.at(i);
      } else if (throwval == 0 || throwval == config.h) {
        // skip
      } else {
        ch = '0' + ops.at(i);
      }
    } else {
      ch = '0' + ops.at(i);
    }

    if (ch == '\0')
      continue;

    // use ANSI terminal codes to do inverse, bolding, and color
    const bool escape = (hl_start || hl_last || hl_deadstate || hl_shift);
    if (escape) { buffer << '\x1B' << '['; }
    if (hl_start) { buffer << "7;"; }
    if (hl_last) { buffer << "1;"; }
    if (hl_deadstate || hl_shift) { buffer << "32"; }
    if (escape) { buffer << 'm'; }
    buffer << ch;
    if (escape) { buffer << '\x1B' << "[0m"; }
    hl_start = hl_last = hl_deadstate = hl_shift = false;

    if (++printed >= STATUS_WIDTH)
      break;
  }

  while (printed < STATUS_WIDTH) {
    buffer << ' ';
    ++printed;
  }

  buffer << std::setw(5) << worker_longest_last.at(id)
         << std::setw(5) << worker_longest_start.at(id);

  ops_last = ops;
  if (ops_start.size() == 0) {
    ops_start = ops;
  }
  worker_longest_last.at(id) = 0;

  return buffer.str();
}
