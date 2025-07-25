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
  uint32_t engines;

  // command line input
  std::string input;

  // expected results
  std::uint64_t npatterns;
  std::uint64_t ntotal;
  std::uint64_t nnodes;
};

static const std::vector<TestCase> tests {
  // NORMAL mode
  {  7, "jprime 3 8 -count",                 11906414,   11906414,   49961711 },
  {  7, "jprime 5 15 1-12 -g -count",        17996072,   17996072,  229780787 },
  // NORMAL_MARKING mode
  {  7, "jprime 3 16 524",                         30,    8045966,  291293062 },
  {  7, "jprime 5 10 225",                        838,   23754590, 1458188812 },
  {  7, "jprime 4 8 41- -ng -count",           298662,    7084550,   43179517 },
  // SUPER mode
  {  7, "jprime 5 13 10 -super 1 -count",      532478,     685522,    7032405 },
  {  7, "jprime 3 18 47 -super 1",                 50,  128149175,  307570492 },
  {  7, "jprime 4 11 -super 1 -g -count",    69797298,   69797298,  334414789 },
  {  7, "jprime 5 15 1-12 -super 1 -g -count", 8519730,   8519730,   76552560 },
  {  7, "jprime 5 15 1-12 -super 1 -count",  23826278,   23826278,  370793129 },
  {  7, "jprime 3 19 51- -super 1 -count",        222,  535200936, 1419055003 },
  {  7, "jprime 3 9 -super -count",         133410514,  133410514,  810573685 },
  // SUPER0 mode
  {  7, "jprime 3 21 -super 0 -g -count",   388339361,  388339361,  876591490 },
  {  7, "jprime 3 21 64 -super 0",                  1,  388339361,  876591490 },
  {  7, "jprime 5 15 1-12 -super 0 -count",   6411338,    6411338,   70254546 },
  // single-period graph mode
  {  7, "jprime 4 56 14 -g -count",          66129382,   84454394,  497746428 },
  {  7, "jprime 2 50 25 -count",             42451179,   78584312,  198939672 },
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
    if (Coordinator::stopping) {
      std::cout << "TEST ABORTED\n" << std::endl;
      return false;
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

    if (context.npatterns != tc.npatterns || context.ntotal != tc.ntotal ||
        context.nnodes != tc.nnodes) {
      success = false;
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

// Execute test cases and report on results.
//
// If `testnum` == -1 then execute all test cases sequentially.
//
// Returns EXIT_SUCCESS on success, EXIT_FAILURE on failure.

int do_tests(int testnum)
{
  int casenum = 0;
  int runs = 0;
  int passes = 0;

  for (const auto& testcase : tests) {
    ++casenum;

#ifndef CUDA_ENABLED
    if ((testcase.engines & 3) == 0)
      continue;
#endif

    if (testnum != -1 && casenum != testnum)
      continue;

    std::cout << std::format("\nStarting test {}:\n\n", casenum);
    if (run_one_test(testcase)) {
      ++passes;
    } else if (Coordinator::stopping) {
      break;
    }
    ++runs;
  }

  std::cout << "------------------------------------------------------------\n"
            << std::format("Passed {} out of {} tests", passes, runs)
            << std::endl;
  return (passes == runs) ? EXIT_SUCCESS : EXIT_FAILURE;
}
