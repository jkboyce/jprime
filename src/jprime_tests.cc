//
// jprime_tests.cc
//
// Collection of tests for jprime functionality.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"

#include <iostream>
#include <sstream>
#include <cstdint>
#include <format>


// Individual record for a test case

struct TestCase {
  // execution engines to test:
  //   1 = interative
  //   2 = recursive
  //   4 = CUDA (if available)
  //   8 = CUDA, where node counts do not need to match
  uint32_t engines;

  // command line input
  std::string input;

  // expected results
  std::uint64_t npatterns;
  std::uint64_t ntotal;
  std::uint64_t nnodes;
};

const std::vector<TestCase> tests {
  {  7, "jprime 2 60 30 -count",           1391049900, 2591724915, 6595287598 },
  {  3, "jprime 3 16 524",                         30,   11920253,  291293062 },
  {  7, "jprime 5 13 10 -super 1 -count",      532478,     685522,    7032405 },
  {  3, "jprime 5 10 225",                        838,   29762799, 1458188812 },
  {  7, "jprime 3 18 47 -super 1",                 50,  128149175,  307570492 },
  {  7, "jprime 3 19 51- -super 1 -count",        222,  535200936, 1419055003 },
  {  7, "jprime 3 21 64 -super 0",                  1,  388339361,  876591490 },
  {  7, "jprime 3 9 -super -count",         133410514,  133410514,  810573685 },

  {  7, "jprime 3 8 -g -count",              11578732,   11578732,   47941320 },
  {  7, "jprime 3 21 -super 0 -g -count",   388339361,  388339361,  876591490 },
  {  7, "jprime 4 11 -super 1 -g -count",    69797298,   69797298,  334414789 },
  {  7, "jprime 3 9 -super -g -count",      123417152,  123417152,  732374333 },
  {  7, "jprime 4 56 14 -g -count",          66129382,   84454394,  497746428 },
  {  7, "jprime 5 15 1-12 -g -count",        17996072,   17996072,  229780787 },
  {  7, "jprime 5 15 1-12 -super 1 -g -count", 8519730,   8519730,   76552560 },

  {  4, "jprime 3 9 -count",           30513071763, 30513071763, 141933045309 },
  {  7, "jprime 5 15 1-12 -super 0 -count",   6411338,    6411338,   70254546 },
  {  7, "jprime 5 15 1-12 -super 1 -count",  23826278,   23826278,  370793129 },
};

// Run a single test case and compare against known values, outputting results
// to the console.
//
// Return true on test pass, false on failure.

bool run_one_test(const TestCase& tc)
{
  std::cout << std::format("Executing: {}\n", tc.input)
            << "               patterns,         seen,"
            << "        nodes,  time (sec)\n"
            << std::format("target     {:12}, {:12}, {:12}", tc.npatterns,
                 tc.ntotal, tc.nnodes)
            << std::endl;

  int run_limit = 2;
  #ifdef CUDA_ENABLED
  run_limit = 3;
  #endif

  bool success = true;

  for (int run = 0; run < run_limit; ++run) {
    if ((tc.engines & (1u << run)) == 0)
      continue;

    // prep config
    SearchConfig config;
    try {
      if (run == 0) {
        config.from_args(tc.input + " -status -threads 4");
      } else if (run == 1) {
        config.from_args(tc.input + " -status -threads 4 -recursive");
      } else {
        config.from_args(tc.input + " -status -cuda");
      }
    } catch (const std::invalid_argument& ie) {
      std::cout << "Error parsing test input: " << ie.what()
          << "\nTEST FAILED ################################################"
          << std::endl;
      return false;
    }

    // prep context
    SearchContext context;
    WorkAssignment wa;
    context.assignments.push_back(wa);

    std::ostringstream buffer;
    auto coordinator = Coordinator::make_coordinator(config, context, buffer);

    // run test
    if (!coordinator->run()) {
      std::cout
          << "TEST FAILED TO EXECUTE #####################################\n"
          << std::endl;
      success = false;
      continue;
    }

    if (run == 0) {
      std::cout << "iterative";
    } else if (run == 1) {
      std::cout << "recursive";
    } else {
      std::cout << "CUDA     ";
    }
    std::cout << std::format("  {:12}, {:12}, {:12},  {:.4f}",
                   context.npatterns, context.ntotal, context.nnodes,
                   context.secs_elapsed)
              << std::endl;

    if (context.npatterns != tc.npatterns) {
      success = false;
    }
    if (run < 3) {
      if (context.ntotal != tc.ntotal || context.nnodes != tc.nnodes) {
        success = false;
      }
    }
  }

  if (success) {
    std::cout << "Test succeeded\n" << std::endl;
  } else {
    std::cout
        << "TEST FAILED ################################################\n"
        << std::endl;
  }

  return success;
}

// Execute all test cases and report on results.

void do_tests()
{
  int runs = 0;
  int passes = 0;

  for (const auto& testcase : tests) {
    #ifndef CUDA_ENABLED
    if ((testcase.engines & 3) == 0)
      continue;
    #endif

    ++runs;
    std::cout << std::format("\nStarting test {}:\n\n", runs);
    if (run_one_test(testcase)) {
      ++passes;
    }
  }

  std::cout << "------------------------------------------------------------\n"
            << std::format("Passed {} out of {} tests", passes, runs)
            << std::endl;
}
