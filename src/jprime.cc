//
// jprime.cc
//
// This program finds juggling patterns in siteswap notation, in particular
// async siteswaps that are prime. A prime siteswap is one that has no
// repeatable subpatterns; in a corresponding graph search problem they
// correspond to cycles in the graph that visit no vertex more than once.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
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
// 2024.03.14  Version 6.5 adds single-period mode, support for throws > 35.
// 2024.04.03  Version 6.6 adds efficiency improvements.
// 2024.04.15  Version 6.7 code refactoring.


#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <stdexcept>


//------------------------------------------------------------------------------
// Help message
//------------------------------------------------------------------------------

void print_help() {
  const std::string helpString =
    "jprime version 6.7 (2024.04.15)\n"
    "Copyright (C) 1998-2024 Jack Boyce <jboyce@gmail.com>\n"
    "\n"
    "This program searches for long prime async siteswap patterns. For an\n"
    "explanation of these terms, consult the page:\n"
    "   http://www.juggling.org/help/siteswap/\n"
    "\n"
    "Command line format:\n"
    "   jprime <# objects> <max. throw> [<length>] [options]\n"
    "\n"
    "where:\n"
    "   <# objects>       = number of objects\n"
    "   <max. throw>      = largest allowed throw value\n"
    "   <length>          = pattern length(s) to find\n"
    "\n"
    "Recognized options:\n"
    "   -super <shifts>   find (nearly) superprime patterns, allowing the\n"
    "                        specified number of shift throws\n"
    "   -g                find ground-state patterns only\n"
    "   -ng               find excited-state patterns only\n"
    "   -x <throw1 throw2 ...>\n"
    "                     exclude listed throw values\n"
    "   -inverse          print/save inverse pattern, if it exists\n"
    "   -noplus           print/save without using +, - for h and 0\n"
    "   -info             print info without executing search\n"
    "   -noprint          do not print patterns\n"
    "   -count            print/save pattern counts only\n"
    "   -status           display live search status (needs ANSI terminal)\n"
    "   -verbose          print worker diagnostic information during search\n"
    "   -threads <num>    run with the given number of worker threads (default 1)\n"
    "   -file <name>      use the named file for checkpointing (when jprime is\n"
    "                        interrupted via ctrl-c), resuming, and final output\n"
    "\n"
    "When resuming a calculation from a checkpoint file, the other parts of the\n"
    "input are ignored and can be omitted. For example: jprime -file testrun\n"
    "\n"
    "Examples:\n"
    "   jprime 6 10 187-\n"
    "   jprime 4 9 14- -super 0 -inverse\n"
    "   jprime 7 42 6 -g\n"
    "   jprime 5 7 -noplus -file 5_7_all\n"
    "   jprime 2 60 30 -count -threads 4 -status";

  std::cout << helpString << std::endl;
}

//------------------------------------------------------------------------------
// Prep `config` and `context` data structures for calculation
//------------------------------------------------------------------------------

void prepare_calculation(int argc, char** argv, SearchConfig& config,
      SearchContext& context) {
  // first check if the user wants file output mode
  std::string outfile;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-file") && (i + 1) < argc) {
      outfile = std::string(argv[i + 1]);
      break;
    }
  }

  if (outfile.size() > 0) {
    std::ifstream myfile(outfile);
    if (myfile.good()) {
      try {
        // file exists; try resuming calculation
        std::cout << "reading checkpoint file '" << outfile << "'\n";
        context.from_file(outfile);

        if (context.assignments.size() == 0) {
          std::cout << "calculation is finished" << std::endl;
          std::exit(0);
        }

        // get config from the original arguments
        config.from_args(context.arglist);

        // in case the user renamed the checkpoint file since the original
        // invocation, use the current filename
        config.outfile = outfile;

        std::cout << "resuming calculation: " << context.arglist << '\n'
                  << "loaded " << context.npatterns
                  << " patterns and " << context.assignments.size()
                  << " work assignments" << std::endl;
        return;
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
    if (i != 0)
      context.arglist += " ";
    context.arglist += argv[i];
  }

  // set initial work assignment; default value does entire calculation
  WorkAssignment wa;
  context.assignments.push_back(wa);
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc < 3) {
    print_help();
    return 0;
  }

  SearchConfig config;
  SearchContext context;
  prepare_calculation(argc, argv, config, context);
  Coordinator coordinator(config, context);
  const bool success = coordinator.run();

  std::cout << "------------------------------------------------------------"
            << std::endl;

  if (success && config.fileoutputflag) {
    std::cout << "saving checkpoint file '" << config.outfile << "'\n";
    context.to_file(config.outfile);
  }
  return 0;
}
