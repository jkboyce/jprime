//
// jprime_tests.cc
//
// Collection of tests for jprime functionality.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
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
  // command line input
  std::string input;

  // expected results
  std::uint64_t npatterns;
  std::uint64_t ntotal;
  std::uint64_t nnodes;
};

const std::vector<TestCase> tests {
  { "jprime 2 60 30 -threads 4 -count", 1391049900, 2591724915, 6595287598 },
  { "jprime 3 16 524",                          30,   11920253,  291293062 },
  { "jprime 5 13 10 -super 1 -count",       532478,     685522,    6778559 },
  { "jprime 5 10 225",                         838,   29762799, 1458188812 },
  { "jprime 3 18 47 -super 1",                  50,  128149175,  307570492 },
  { "jprime 3 20 58 -super 1 -threads 4",       92, 2661959187, 6400527070 },
  { "jprime 3 21 64 -super 0",                   1,  388339361,  876591490 },
};

// Run a single test case and compare against known values, outputting results
// to the console.
//
// Return true on test pass, false on failure.

bool run_one_test(const TestCase& tc) {
  std::cout << std::format("Executing: {}\n", tc.input);

  // prep config
  SearchConfig config;
  try {
    config.from_args(tc.input);
  } catch (const std::invalid_argument& ie) {
    std::cout << "Error parsing test input: " << ie.what() << '\n'
              << "TEST FAILED" << std::endl;
    return false;
  }

  // prep context
  SearchContext context;
  WorkAssignment wa;
  context.assignments.push_back(wa);

  std::ostringstream buffer;
  Coordinator coordinator(config, context, buffer);

  // run test
  std::cout << "            patterns,         seen,        nodes\n"
            << std::format("target: {:12}, {:12}, {:12}", tc.npatterns,
                 tc.ntotal, tc.nnodes)
            << std::endl;
  if (!coordinator.run()) {
    std::cout << "TEST FAILED\n" << std::endl;
    return false;
  }
  std::cout << std::format("got:    {:12}, {:12}, {:12}", context.npatterns,
                 context.ntotal, context.nnodes)
            << std::endl;

  if (context.npatterns != tc.npatterns || context.ntotal != tc.ntotal ||
      context.nnodes != tc.nnodes) {
    std::cout << "TEST FAILED\n" << std::endl;
    return false;
  } else {
    std::cout << "Test succeeded\n" << std::endl;
    return true;
  }
}

// Execute all test cases and report on results.

void do_tests() {
  int runs = 0;
  int passes = 0;

  for (const auto& testcase : tests) {
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
