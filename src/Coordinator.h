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
#include <csignal>


class Coordinator {
 public:
  Coordinator(const SearchConfig& config, SearchContext& context,
      std::ostream& jpout);
  Coordinator() = delete;
  static std::unique_ptr<Coordinator> make_coordinator(
    const SearchConfig& config, SearchContext& context, std::ostream& jpout);

 protected:
  const SearchConfig& config;
  SearchContext& context;
  std::ostream& jpout;  // all console output goes here except status display
  unsigned n_max = 0;  // max pattern period to find

  // live status display
  std::vector<std::string> status_lines;
  bool status_printed = false;
  int status_line_count_last = 0;

  static volatile sig_atomic_t stopping;
  static constexpr unsigned MAX_STATES = 1000000u;  // memory limit

 public:
  bool run();

 protected:
  virtual void run_search();

  // helper functions
  void calc_graph_size();
  bool passes_prechecks();
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
  void customize_graph(Graph& graph);
  std::string pattern_output_format(const std::vector<int>& pattern,
    const unsigned start_state);
};

#endif
