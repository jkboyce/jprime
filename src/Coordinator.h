//
// Coordinator.h
//
// Coordinator that manages the overall search. This is a base class that needs
// to be overridden to define the `run_search()` method.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#ifndef JPRIME_COORDINATOR_H_
#define JPRIME_COORDINATOR_H_

#include "SearchConfig.h"
#include "SearchContext.h"
#include "Graph.h"

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <csignal>


using jptimer_t = std::chrono::time_point<std::chrono::high_resolution_clock>;


class Coordinator {
 public:
  Coordinator(SearchConfig& config, SearchContext& context,
    std::ostream& jpout);
  Coordinator() = delete;
  virtual ~Coordinator();

  // factory method
  static std::unique_ptr<Coordinator> make_coordinator(
    SearchConfig& config, SearchContext& context, std::ostream& jpout);

 protected:
  SearchConfig& config;
  SearchContext& context;
  std::ostream& jpout;  // all console output goes here except status display
  Graph graph;
  std::vector<int> max_length;
  unsigned n_max = 0;  // max pattern period to find

  // live status display
  std::vector<std::string> status_lines;
  bool status_printed = false;
  int status_line_count_last = 0;

  static volatile sig_atomic_t stopping;  // to handle ctrl-c
  static constexpr unsigned MAX_STATES = 1000000u;  // memory limit

 public:
  bool run();

 protected:
  virtual void run_search();  // subclasses define this

  // helper functions
  void calc_graph_size();
  bool passes_prechecks();
  void initialize_graph();
  void customize_graph(Graph& graph);
  std::vector<double> build_access_model(unsigned num_states);
  double expected_patterns_at_maxperiod();
  static void signal_handler(int signum);

  // handle terminal output
  void print_search_description() const;
  void print_results() const;
  void erase_status_output();
  void print_status_output();
  static std::string current_time_string();
  void process_search_result(const std::string& pattern);

 public:
  // utility methods
  std::string pattern_output_format(const std::vector<int>& pattern,
    const unsigned start_state);
  static double calc_duration_secs(const jptimer_t& before,
    const jptimer_t& after);
  int get_max_length(unsigned start_state) const;
};

#endif
