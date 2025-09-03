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
#include "CoordinatorCPU.h"
#ifdef CUDA_ENABLED
#include "CoordinatorCUDA.h"
#endif
#include "Graph.h"
#include "Pattern.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cassert>
#include <format>
#include <cstring>
#include <stdexcept>


Coordinator::Coordinator(SearchConfig& a, SearchContext& b, std::ostream& c)
    : config(a), context(b), jpout(c)
{}

Coordinator::~Coordinator()
{}

// Factory method to return the correct type of Coordinator for the search
// requested.

std::unique_ptr<Coordinator> Coordinator::make_coordinator(
    SearchConfig& config, SearchContext& context, std::ostream& jpout)
{
  if (config.cudaflag) {
#ifdef CUDA_ENABLED
    return make_unique<CoordinatorCUDA>(config, context, jpout);
#else
    throw std::runtime_error("CUDA support not enabled");
#endif
  } else {
    return make_unique<CoordinatorCPU>(config, context, jpout);
  }
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

// Execute the calculation specified in `config`, storing results in `context`,
// and sending console output to `jpout`.
//
// Returns true on success, false on failure.

bool Coordinator::run()
{
  try {
    calc_graph_size();
  } catch (const std::overflow_error& oe) {
    (void)oe;
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
  initialize_graph();
  select_search_algorithm();

  // register signal handler for ctrl-c interrupt
  signal(SIGINT, Coordinator::signal_handler);

  const auto start = std::chrono::high_resolution_clock::now();
  run_search();
  const auto end = std::chrono::high_resolution_clock::now();

  const double runtime = calc_duration_secs(start, end);
  context.secs_elapsed += runtime;
  context.secs_available += runtime * config.num_threads;

  erase_status_output();
  std::flush(std::cout);
  if (config.verboseflag) {
    jpout << "Finished on: " << current_time_string() << '\n';
  }
  if (!context.assignments.empty()) {
    jpout << "\nPARTIAL RESULTS:\n";
  }
  print_search_description();
  print_results();
  return true;
}

// Empty method; subclasses override this

void Coordinator::run_search()
{}

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

void Coordinator::calc_graph_size()
{
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
    if (context.full_numcycles < 2) {
      context.n_bound = 0;
    } else if (config.shiftlimit == -1U) {
      context.n_bound = context.full_numstates - context.full_numcycles;
    } else {
      context.n_bound =
          std::min(context.full_numstates - context.full_numcycles,
              context.full_numcycles + config.shiftlimit);
    }
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

bool Coordinator::passes_prechecks()
{
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

// Build the `graph` and `max_length` objects.

void Coordinator::initialize_graph()
{
  graph = {
    config.b,
    config.h,
    config.xarray,
    config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH
                      ? config.n_min : 0
  };
  customize_graph(graph);
  graph.validate_graph();

  // build table of maximum pattern length by start_state
  max_length.push_back(-1);
  for (unsigned start_state = 1; start_state <= graph.numstates;
      ++start_state) {
    if (start_state > graph.max_startstate_usable.at(start_state)) {
      max_length.push_back(-1);
      continue;
    }

    unsigned max_possible = 0;
    if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
      max_possible = graph.prime_period_bound(start_state);
    } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
      if (config.shiftlimit == -1U) {
        max_possible = graph.superprime_period_bound(start_state);
      } else {
        max_possible = graph.superprime_period_bound(start_state,
            config.shiftlimit);
      }
    }
    max_length.push_back(static_cast<int>(max_possible));
  }
}

// Edit the graph after its initial construction. This is an opportunity to
// apply optimizations depending on search mode, etc. This should be executed
// once, immediately after the graph is built.
//
// Note this routine should never set states as active!

void Coordinator::customize_graph(Graph& g) const
{
  std::vector<bool> state_active(g.numstates + 1, true);

  // (1) In SUPER mode we are not allowed to make link throws within a single
  // shift cycle, so remove them.

  if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= g.numstates; ++i) {
      unsigned outthrownum = 0;

      for (size_t j = 0; j < g.outdegree.at(i); ++j) {
        const auto tv = g.outthrowval.at(i).at(j);
        if (g.cyclenum.at(g.outmatrix.at(i).at(j)) ==
            g.cyclenum.at(i) && tv != 0 && tv != config.h) {
          continue;
        }

        // throw is allowed
        if (outthrownum != j) {
          g.outmatrix.at(i).at(outthrownum) = g.outmatrix.at(i).at(j);
          g.outthrowval.at(i).at(outthrownum) = tv;
        }
        ++outthrownum;
      }

      g.outdegree.at(i) = outthrownum;
    }
  }

  // (2) In SUPER mode, number of consecutive '-'s at the start of the state,
  // plus the number of consecutive 'x's at the end of the state, cannot exceed
  // `shiftlimit`.

  if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    for (size_t i = 1; i <= g.numstates; ++i) {
      unsigned start0s = 0;
      while (start0s < g.h && g.state.at(i).slot(start0s) == 0) {
        ++start0s;
      }
      unsigned end1s = 0;
      while (end1s < g.h &&
          g.state.at(i).slot(g.h - end1s - 1) != 0) {
        ++end1s;
      }
      if (start0s + end1s > config.shiftlimit) {
        state_active.at(i) = false;
      }
    }
  }

  // (3) In SUPER mode, apply some special cases for (b,h) = (b,2b) due to the
  // properties of the period-2 shift cycle generated by state (x-)^b.

  if (config.mode == SearchConfig::RunMode::SUPER_SEARCH &&
        config.h == (2 * config.b) && config.n_min > 2) {
    State period2_state(config.h);
    for (size_t i = 0; i < config.h; i += 2) {
      period2_state.slot(i) = 1;
    }
    const auto k = g.get_statenum(period2_state);
    assert(k != 0);

    if (config.shiftlimit == 0) {
      // in this case state (x-)^b is excluded
      state_active.at(k) = false;
    } else if (config.shiftlimit == 1 && config.n_min == g.numcycles + 1) {
      // in this case state (x-)^b is required to be in the pattern, and the one
      // shift throw can only be in the cycle immediately preceding or following
      // state (x-)^b in the pattern

      for (unsigned i = 1; i <= g.numstates; ++i) {
        bool allowed_to_shift_throw = false;

        // does i's downstream state have a throw to (x-)^b ?
        auto s = g.downstream_state(i);
        if (s != 0) {
          for (size_t j = 0; j < g.outdegree.at(s); ++j) {
            if (g.outmatrix.at(s).at(j) == k) {
              allowed_to_shift_throw = true;
            }
          }
        }

        // does (x-)^b have a throw into i ?
        for (size_t j = 0; j < g.outdegree.at(k); ++j) {
          if (g.outmatrix.at(k).at(j) == i) {
            allowed_to_shift_throw = true;
          }
        }

        // if either of the above is true, keep the shift throw out of i
        if (allowed_to_shift_throw) {
          continue;
        }

        // otherwise remove it
        unsigned outthrownum = 0;
        for (size_t j = 0; j < g.outdegree.at(i); ++j) {
          const auto tv = g.outthrowval.at(i).at(j);
          if (tv == 0 || tv == config.h) {
            continue;
          }

          g.outmatrix.at(i).at(outthrownum) = g.outmatrix.at(i).at(j);
          g.outthrowval.at(i).at(outthrownum) = tv;
          ++outthrownum;
        }
        g.outdegree.at(i) = outthrownum;
      }
    }
  }

  for (size_t i = 0; i <= g.numstates; ++i) {
    if (!state_active.at(i)) {
      g.outdegree.at(i) = 0;
    }
  }
}

// Choose a search algorithm to use.

void Coordinator::select_search_algorithm()
{
  alg = SearchAlgorithm::NONE;

  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    if (static_cast<uint64_t>(graph.numstates) == context.full_numstates &&
        static_cast<double>(config.n_min) >
        0.66 * static_cast<double>(get_max_length(1))) {
      // the overhead of marking is only worth it for long-period patterns, and
      // the algorithm requires all states to be present in the graph
      alg = SearchAlgorithm::NORMAL_MARKING;
    } else {
      alg = SearchAlgorithm::NORMAL;
    }
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    if (config.shiftlimit == 0) {
      alg = SearchAlgorithm::SUPER0;
    } else {
      alg = SearchAlgorithm::SUPER;
    }
  }
}

// Create a model for the fraction of memory accesses that will occur at each
// position in the pattern during DFS.
//
// Returns a normalized std::vector<double> with `n_max` elements.

std::vector<double> Coordinator::build_access_model(unsigned num_states) const
{
  const double pos_mean = 0.48 * static_cast<double>(num_states) - 1;
  const double pos_fwhm = sqrt(pos_mean + 1) * (config.b == 2 ? 3.25 : 2.26);
  const double pos_sigma = pos_fwhm / (2 * sqrt(2 * log(2)));

  std::vector<double> access_fraction(n_max, 0);
  double maxval = 0;
  for (unsigned i = 0; i < n_max; ++i) {
    const auto x = static_cast<double>(i);
    // be careful to avoid underflowing exp(-x^2)
    const double val = -(x - pos_mean) * (x - pos_mean) /
        (2 * pos_sigma * pos_sigma);
    access_fraction.at(i) = val;
    if (i == 0 || val > maxval) {
      maxval = val;
    }
  }
  double sum = 0;
  for (unsigned i = 0; i < n_max; ++i) {
    access_fraction.at(i) = exp(access_fraction.at(i) - maxval);
    sum += access_fraction.at(i);
  }
  for (unsigned i = 0; i < n_max; ++i) {
    access_fraction.at(i) /= sum;
  }
  return access_fraction;
}

// Use the distribution of patterns found so far to extrapolate the expected
// number of patterns at period `n_bound`. This may be a useful signal of the
// degree of search completion.
//
// The distribution of patterns by period is observed to closely follow a
// Gaussian (normal) shape, so we fit the logarithm to a parabola and use that
// to extrapolate.

double Coordinator::expected_patterns_at_maxperiod()
{
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
  double s1 = 0;
  double sx = 0;
  double sx2 = 0;
  double sx3 = 0;
  double sx4 = 0;
  double sy = 0;
  double sxy = 0;
  double sx2y = 0;
  const size_t xstart = std::max(max - 10, mode);

  for (size_t i = xstart; i < context.count.size(); ++i) {
    if (context.count.at(i) < 5) {
      continue;
    }

    const auto x = static_cast<double>(i);
    const auto y = log(static_cast<double>(context.count.at(i)));
    s1 += 1;
    sx += x;
    sx2 += x * x;
    sx3 += x * x * x;
    sx4 += x * x * x * x;
    sy += y;
    sxy += x * y;
    sx2y += x * x * y;
  }

  // Solve this 3x3 linear system for a, b, c, the coefficients in the parabola
  // of best fit y = ax^2 + bx + c:
  //
  // | sx4  sx3  sx2  | | a |   | sx2y |
  // | sx3  sx2  sx   | | b | = | sxy  |
  // | sx2  sx   s1   | | c |   | sy   |

  // Find matrix inverse
  const double det = sx4 * (sx2 * s1 - sx * sx) - sx3 * (sx3 * s1 - sx * sx2) +
      sx2 * (sx3 * sx - sx2 * sx2);
  const double m11 = (sx2 * s1 - sx * sx) / det;
  const double m12 = (sx2 * sx - sx3 * s1) / det;
  const double m13 = (sx3 * sx - sx2 * sx2) / det;
  const double m21 = m12;
  const double m22 = (sx4 * s1 - sx2 * sx2) / det;
  const double m23 = (sx2 * sx3 - sx4 * sx) / det;
  const double m31 = m13;
  const double m32 = m23;
  const double m33 = (sx4 * sx2 - sx3 * sx3) / det;

  auto is_close = [](double a, double b) {
    double epsilon = 1e-3;
    return (b > a - epsilon && b < a + epsilon);
  };
  (void)is_close;
  assert(is_close(m11 * sx4 + m12 * sx3 + m13 * sx2, 1));
  assert(is_close(m11 * sx3 + m12 * sx2 + m13 * sx, 0));
  assert(is_close(m11 * sx2 + m12 * sx + m13 * s1, 0));
  assert(is_close(m21 * sx4 + m22 * sx3 + m23 * sx2, 0));
  assert(is_close(m21 * sx3 + m22 * sx2 + m23 * sx, 1));
  assert(is_close(m21 * sx2 + m22 * sx + m23 * s1, 0));
  assert(is_close(m31 * sx4 + m32 * sx3 + m33 * sx2, 0));
  assert(is_close(m31 * sx3 + m32 * sx2 + m33 * sx, 0));
  assert(is_close(m31 * sx2 + m32 * sx + m33 * s1, 1));

  const double a = m11 * sx2y + m12 * sxy + m13 * sy;
  const double b = m21 * sx2y + m22 * sxy + m23 * sy;
  const double c = m31 * sx2y + m32 * sxy + m33 * sy;

  // evaluate the expected number of patterns found at x = n_bound
  const auto x = static_cast<double>(context.n_bound);
  const auto lny = a * x * x + b * x + c;
  return exp(lny);
}

// Static variable for indicating the user has interrupted execution.

volatile sig_atomic_t Coordinator::stopping = 0;

// Respond to a SIGINT (ctrl-c) interrupt during execution.

void Coordinator::signal_handler(int signum)
{
  (void)signum;
  stopping = 1;
}

//------------------------------------------------------------------------------
// Handle terminal output
//------------------------------------------------------------------------------

void Coordinator::print_search_description() const
{
  jpout << std::format("objects: {}, max throw: {}\n",
      (config.dualflag ? config.h - config.b : config.b), config.h);

  if (config.mode == SearchConfig::RunMode::NORMAL_SEARCH) {
    jpout << "prime";
  } else if (config.mode == SearchConfig::RunMode::SUPER_SEARCH) {
    jpout << "superprime";
    if (config.shiftlimit == 1) {
      jpout << " (+1 shift)";
    } else if (config.shiftlimit != -1U) {
      jpout << std::format(" (+{} shifts)", config.shiftlimit);
    }
  }
  jpout << " search for period: " << config.n_min;
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
        << '\n';

  if (config.graphmode == SearchConfig::GraphMode::SINGLE_PERIOD_GRAPH) {
    jpout << std::format("period-{} subgraph: {} {}", config.n_min,
               context.memory_numstates,
               context.memory_numstates == 1 ? "state" : "states")
          << '\n';
  }
}

void Coordinator::print_results() const
{
  jpout << std::format("{} {} in range ({} seen, {} {})\n",
             context.npatterns,
             context.npatterns == 1 ? "pattern" : "patterns",
             context.ntotal, context.nnodes,
             context.nnodes == 1 ? "node" : "nodes");

  jpout << std::format("runtime = {:.4f} sec ({:.1f}M nodes/sec",
             context.secs_elapsed, static_cast<double>(context.nnodes) /
             context.secs_elapsed / 1000000);
  if (config.num_threads > 1 || config.cudaflag) {
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

// Send a string to the terminal output.

void Coordinator::print_string(const std::string& s)
{
  if (config.statusflag && status_printed &&
      jpout.rdbuf() == std::cout.rdbuf()) {
    erase_status_output();
    jpout << s << '\n';
    print_status_output();
  } else {
    jpout << s << '\n' << std::flush;
  }
}

void Coordinator::erase_status_output()
{
  if (!config.statusflag || !status_printed) {
    return;
  }

  for (int i = 0; i < status_lines_displayed; ++i) {
    std::cout << '\x1B' << "[1A"
              << '\x1B' << "[2K";
  }
  status_lines_displayed = 0;
}

void Coordinator::print_status_output()
{
  if (!config.statusflag) {
    return;
  }

  assert(status_lines_displayed == 0);
  status_printed = true;
  for (const std::string& line : status_lines) {
    std::cout << line << '\n';
    ++status_lines_displayed;
  }
  std::flush(std::cout);
}

std::string Coordinator::current_time_string()
{
  const auto now = std::chrono::system_clock::now();
  const auto now_timet = std::chrono::system_clock::to_time_t(now);
  char* now_str = std::ctime(&now_timet);
  now_str[strlen(now_str) - 1] = '\0';  // remove trailing carriage return
  return now_str;
}

// Handle a pattern found during the search. We store it and optionally print
// to the terminal.

void Coordinator::process_search_result(const std::string& pattern)
{
  // workers only send patterns in the target period range
  context.patterns.push_back(pattern);

  if (config.printflag) {
    print_string(pattern);
  }
}

//------------------------------------------------------------------------------
// Utility methods
//------------------------------------------------------------------------------

// Format a pattern for output.

std::string Coordinator::pattern_output_format(const std::vector<int>& pattern,
    const unsigned start_state) const
{
  std::ostringstream buffer;

  if (config.groundmode != SearchConfig::GroundMode::GROUND_SEARCH) {
    if (start_state == 1) {
      buffer << "  ";
    } else {
      buffer << "* ";
    }
  }

  Pattern pat(pattern, static_cast<int>(config.h));
  if (config.dualflag) {
    buffer << pat.dual().to_string(static_cast<int>(config.throwdigits),
        !config.noblockflag);
  } else {
    buffer << pat.to_string(static_cast<int>(config.throwdigits),
        !config.noblockflag);
  }

  if (start_state != 1) {
    buffer << " *";
  }

  if (config.invertflag) {
    Pattern inverse = pat.inverse();

    if ((inverse.period() != 0) != pat.is_superprime()) {
      std::cerr << "error with inverse of:\n"
                << "  " << pat << " :\n"
                << "  " << inverse << '\n'
                << "inverse.period() = " << inverse.period() << '\n'
                << "pat.is_superprime() = " << pat.is_superprime()
                << '\n';
    }
    if (pat.is_superprime() != inverse.is_superprime()) {
      std::cerr << "error with inverse of:\n"
                << "  " << pat << " :\n"
                << "  " << inverse << '\n'
                << "pat.is_superprime() = " << pat.is_superprime() << '\n'
                << "inverse.is_superprime() = " << inverse.is_superprime()
                << '\n';
    }
    assert((inverse.period() != 0) == pat.is_superprime());
    assert(pat.is_superprime() == inverse.is_superprime());

    if (inverse.is_valid()) {
      if (config.groundmode != SearchConfig::GroundMode::GROUND_SEARCH &&
          start_state == 1) {
        buffer << "  ";
      }
      if (config.dualflag) {
        buffer << " : " << inverse.dual().to_string(
            static_cast<int>(config.throwdigits), !config.noblockflag);
      } else {
        buffer << " : " << inverse.to_string(
            static_cast<int>(config.throwdigits), !config.noblockflag);
      }
    }
  }

  return buffer.str();
}

// Return the duration between two time points, in seconds.

double Coordinator::calc_duration_secs(const jptimer_t& before,
    const jptimer_t& after)
{
  const std::chrono::duration<double> diff = after - before;
  return diff.count();
}

// Return the search algorithm to use.

Coordinator::SearchAlgorithm Coordinator::get_search_algorithm() const
{
  return alg;
}

// Return the maximum pattern length for a given `start_state`.

int Coordinator::get_max_length(unsigned start_state) const
{
  return max_length.at(start_state);
}
