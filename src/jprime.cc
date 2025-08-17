//
// jprime.cc
//
// This program finds juggling patterns in siteswap notation, in particular
// async siteswaps that are prime. A prime siteswap is one that has no
// repeatable subpatterns; in a corresponding graph search problem they
// correspond to cycles in the graph that visit no vertex more than once.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//
//------------------------------------------------------------------------------
// Version history:
//
// 1998.06     Version 1.0
// 1998.07.29  Version 2.0 adds the ability to search for block patterns (an
//             idea due to Johannes Waldmann), and more command line options.
// 1998.08.02  Finds patterns in dual graph, if faster.
// 1999.01.02  Version 3.0 implements a new algorithm for graph trimming. My
//             machine now solves (3,9) in 10 seconds (vs 24 hours with V2.0)!
// 1999.02.14  Version 5.0 can find superprime (and nearly superprime) patterns.
//             It also implements an improved algorithm for the standard mode,
//             which uses shift cycles to speed the search.
// 1999.02.17  Version 5.1 adds -inverse option to print inverses of patterns
//             found in -super mode.
// 2023.09.17  Version 6.0 implements parallel depth first search in C++, to run
//             faster on modern multicore machines.
// 2023.10.03  Version 6.1 enables -inverse option for all modes.
// 2023.12.21  Version 6.2 adds many performance improvements.
// 2024.01.28  Version 6.3 adds pattern counting by length, and search status
//             display in -verbose mode.
// 2024.03.04  Version 6.4 changes command line interface, improves performance,
//             and adds non-recursive search.
// 2024.03.14  Version 6.5 adds single-period mode and support for throws > 35.
// 2024.04.03  Version 6.6 adds efficiency improvements.
// 2024.04.15  Version 6.7 code refactoring.
// 2024.06.16  Version 6.8 adds analyzer and efficiency improvements for (b,2b).
// 2025.02.20  Version 6.9 makes <shifts> optional in -super mode.
// 2025.03.22  Version 7.0 adds support for running on a CUDA GPU.
//

#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"
#include "Pattern.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <stdexcept>
#include <format>
#include <tuple>


int do_tests(int testnum = -1);  // defined in jprime_tests.cc

//------------------------------------------------------------------------------
// Help message
//------------------------------------------------------------------------------

void print_help()
{
  static const std::string help_string =
    "jprime version 7.0 (2025.03.22)\n"
    "Copyright (C) 1998-2025 Jack Boyce <jboyce@gmail.com>\n"
    "\n"
    "This program searches for prime siteswap juggling patterns. For an explanation\n"
    "of these terms see http://wikipedia.org/wiki/Siteswap\n"
    "\n"
    "Recognized command line formats:\n"
    "   jprime <# objects> <max. throw> [<period(s)>] [options]\n"
    "   jprime -analyze <pattern> [/<h>]\n"
    "   jprime -test [<testnum>]\n"
    "\n"
    "where:\n"
    "   <# objects>        = number of objects\n"
    "   <max. throw>       = largest allowed throw value\n"
    "   <period(s)>        = pattern period(s) to find, assumed all if omitted\n"
    "   <pattern> [/<h>]   = pattern to analyze, with optional suffix indicating\n"
    "                          beats per state\n"
    "\n"
    "Recognized search options:\n"
    "   -super [<shifts>]  find superprime patterns, optionally restricting to at\n"
    "                         most the given number of shift throws\n"
    "   -g                 find ground-state patterns only\n"
    "   -ng                find excited-state patterns only\n"
    "   -x <throw1 throw2 ...>\n"
    "                      exclude listed throw values\n"
    "   -inverse           print/save inverse pattern, if it exists\n"
    "   -noblock           print/save without using +, - for h and 0\n"
    "   -count             print/save pattern counts only\n"
    "   -info              print info without executing search\n"
    "   -recursive         use recursive search algorithms instead of iterative\n"
    "   -file <name>       use the named file for checkpointing (when jprime is\n"
    "                         interrupted via ctrl-c), resuming, and final output\n"
    "   -noprint           do not print patterns to console *\n"
    "   -status            display live search status (needs ANSI terminal) *\n"
    "   -verbose           print worker diagnostic information during search *\n"
    "   -threads <num>     run search on CPU using <num> threads (default 1) *\n"
    "   -cuda              run search on CUDA-enabled GPU (if available) *\n"
    "\n"
    "When resuming a calculation from a checkpoint file, the other parts of the\n"
    "input are restored and can be omitted. For example: jprime -file testrun\n"
    "Also, options marked (*) can be used to override the original invocation.\n\n"
    "Examples:\n"
    "   jprime 5 7 -noblock\n"
    "   jprime 6 10 187-188\n"
    "   jprime 4 9 14- -super 0 -inverse\n"
    "   jprime 7 42 6 -g -file 7_42_6_g\n"
    "   jprime 2 60 30 -count -threads 4 -status\n"
    "   jprime -analyze ++3--+-+1+--+-6-+--+4--++-6---\n\n";

  std::cout << help_string;
}

//------------------------------------------------------------------------------
// Pattern analysis
//------------------------------------------------------------------------------

void print_analysis(int argc, char** argv)
{
  if (argc < 3) {
    return;
  }

  std::string input;
  for (int i = 2; i < argc; ++i) {
    input += argv[i];
  }

  try {
    Pattern pat(input);
    std::cout << pat.make_analysis();
  } catch (const std::invalid_argument& ie) {
    std::cout << std::format("Error analyzing input: {}\n{}\n", input,
                   ie.what());
  }
}

//------------------------------------------------------------------------------
// Search: Prepare `config` and `context` data structures
//------------------------------------------------------------------------------

std::tuple<SearchConfig, SearchContext> prepare_calculation(int argc,
      char** argv)
{
  SearchConfig config;
  SearchContext context;

  // first check if the user wants file output mode
  std::string outfile;
  for (int i = 1; i < argc - 1; ++i) {
    if (strcmp(argv[i], "-file") == 0) {
      outfile = std::string(argv[i + 1]);
      break;
    }
  }

  if (!outfile.empty()) {
    std::ifstream myfile(outfile);
    if (myfile.good()) {
      try {
        // file exists; try resuming calculation
        std::cout << std::format("Reading checkpoint file '{}'\n", outfile);
        context.from_file(outfile);

        if (context.assignments.empty()) {
          std::cout << "Calculation is finished\n";
          std::exit(EXIT_SUCCESS);
        }

        // get any potential overrides in current arguments
        const auto overrides = SearchConfig::get_overrides(argc, argv);

        // initialize from the original arguments, plus overrides
        config.from_args(context.arglist + overrides);

        std::cout << std::format("Resuming calculation: {}\n"
                       "with overrides:{}\n"
                       "Loaded {} patterns and {} work assignments\n",
                       context.arglist, overrides, context.npatterns,
                       context.assignments.size());
        return std::make_tuple(config, context);
      } catch (const std::invalid_argument& ie) {
        std::cerr << ie.what() << '\n';
        std::exit(EXIT_FAILURE);
      }
    }
  }

  // if not resuming, then get config from args
  try {
    config.from_args(argc, argv);
  } catch (const std::invalid_argument& ie) {
    std::cerr << ie.what() << '\n';
    std::exit(EXIT_FAILURE);
  }

  // save original argument list
  for (int i = 0; i < argc; ++i) {
    if (i != 0) {
      context.arglist += " ";
    }
    context.arglist += argv[i];
  }

  // default work assignment does entire calculation
  WorkAssignment wa;
  context.assignments.push_back(wa);

  return std::make_tuple(config, context);
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
  if (argc > 1 && strcmp(argv[1], "-test") == 0) {
    if (argc > 2) {
      return do_tests(std::stoi(argv[2]));
    }
    return do_tests();
  }

  if (argc < 3) {
    print_help();
    return EXIT_SUCCESS;
  }

  if (strcmp(argv[1], "-analyze") == 0) {
    print_analysis(argc, argv);
    return EXIT_SUCCESS;
  }

  auto [config, context] = prepare_calculation(argc, argv);
  auto coordinator = Coordinator::make_coordinator(config, context, std::cout);
  if (!coordinator->run()) {
    return EXIT_SUCCESS;
  }

  std::cout << "------------------------------------------------------------\n";
  if (config.fileoutputflag) {
    std::cout << std::format("Saving checkpoint file '{}'\n", config.outfile);
    context.to_file(config.outfile);
  }
  return EXIT_SUCCESS;
}
