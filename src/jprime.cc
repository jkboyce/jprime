//
// jprime.cc
//
// This program finds juggling patterns in siteswap notation, in particular
// async siteswaps that are prime. A prime siteswap is one that has no
// repeatable subpatterns; in a corresponding graph search problem they
// correspond to cycles in the graph that visit no vertex more than once.
//
// Copyright (C) 1998-2023 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//
//------------------------------------------------------------------------------
// Version history:
//
//    06/98  Version 1.0
// 07/29/98  Version 2.0 adds the ability to search for block patterns (an idea
//           due to Johannes Waldmann), and more command line options.
// 08/02/98  Finds patterns in dual graph, if faster.
// 01/02/99  Version 3.0 implements a new algorithm for graph trimming. My
//           machine now solves (3,9) in 10 seconds (vs 24 hours with V2.0)!
// 02/14/99  Version 5.0 can find superprime (and nearly superprime) patterns.
//           It also implements an improved algorithm for the standard mode,
//           which uses shift cycles to speed the search.
// 02/17/99  Version 5.1 adds -inverse option to print inverses of patterns
//           found in -super mode.
// 09/17/23  Version 6.0 implements parallel depth first search in C++, to run
//           faster on modern multicore machines.
// 10/03/23  Version 6.1 enables -inverse option for all modes.


#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <string>


void print_help() {
  const std::string helpString =
    "jprime version 6.1 (2023.10.03)\n"
    "Copyright (C) 1998-2023 Jack Boyce\n"
    "\n"
    "This program searches for long prime async siteswap patterns. For an\n"
    "explanation of these terms, consult the page:\n"
    "   http://www.juggling.org/help/siteswap/\n"
    "\n"
    "Command line format:\n"
    "   jprime <# objects> <max. throw> [<min. length>] [options]\n"
    "\n"
    "where:\n"
    "   <# objects>       = number of objects\n"
    "   <max. throw>      = largest allowed throw value\n"
    "   <min. length>     = shortest patterns to find (optional, speeds search)\n"
    "\n"
    "Recognized options:\n"
    "   -block <skips>    find patterns in block form, allowing the specified\n"
    "                        number of skips\n"
    "   -super <shifts>   find (nearly) superprime patterns, allowing the\n"
    "                        specified number of shift throws\n"
    "   -g                find ground-state patterns only\n"
    "   -ng               find excited-state patterns only\n"
    "   -x <throw1 throw2 ...>\n"
    "                     exclude listed throws (speeds search)\n"
    "   -all              print all patterns; otherwise only patterns as long\n"
    "                        currently-longest one found are printed\n"
    "   -noplus           print without using +, - for h and 0 respectively\n"
    "   -exact            print all patterns of the exact length specified\n"
    "   -inverse          print inverse pattern, if it exists\n"
    "   -noprint          suppress printing of patterns\n"
    "   -threads <num>    run with the given number of worker threads (default 1)\n"
    "   -verbose          print worker status information\n"
    "   -steal_alg <num>  algorithm for selecting a worker to take work from\n"
    "   -split_alg <num>  algorithm for splitting a stolen work assignment\n"
    "   -file <name>      use the named file for checkpointing (when jprime is\n"
    "                        interrupted via ctrl-c), resuming, and final output\n"
    "\n"
    "When resuming a calculation from a checkpoint file, the other parts of the\n"
    "input are ignored and can be omitted. For example: jprime -file testrun\n"
    "\n"
    "Examples:\n"
    "   jprime 4 7\n"
    "   jprime 5 7 15 -noplus -exact\n"
    "   jprime 5 7 -noplus -all -file 5_7_all\n"
    "   jprime 6 10 -super 0 -g -file 6_10_s0\n";

  std::cout << helpString << std::endl;
}

//------------------------------------------------------------------------------
// Parsing command line arguments
//------------------------------------------------------------------------------

void parse_args(int argc, char** argv, SearchConfig* const config,
      SearchContext* const context) {
  if (config != nullptr) {
    config->n = atoi(argv[1]);
    if (config->n < 1) {
      std::cerr << "Must have at least 1 object" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    config->h = atoi(argv[2]);
    if (config->h < config->n) {
      std::cerr << "Max. throw value must equal or exceed number of objects"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // excluded self-throws
    config->xarray.resize(config->h + 1, false);

    // defaults for using dual graph
    if (config->h > (2 * config->n)) {
      config->dualflag = true;
      config->n = config->h - config->n;
    }
  }

  bool fullflag = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-noprint")) {
      if (config != nullptr)
        config->printflag = false;
    } else if (!strcmp(argv[i], "-inverse")) {
      if (config != nullptr)
        config->invertflag = true;
    } else if (!strcmp(argv[i], "-g")) {
      if (config != nullptr)
        config->groundmode = 1;
    } else if (!strcmp(argv[i], "-ng")) {
      if (config != nullptr)
        config->groundmode = 2;
    } else if (!strcmp(argv[i], "-all")) {
      if (config != nullptr) {
        fullflag = true;
        config->longestflag = false;
      }
    } else if (!strcmp(argv[i], "-noplus")) {
      if (config != nullptr) {
        config->noplusminusflag = true;
      }
    } else if (!strcmp(argv[i], "-exact")) {
      if (config != nullptr) {
        config->exactflag = true;
        config->longestflag = false;
        if (config->l < 2) {
          std::cerr << "Must specify a length > 1 when using -exact flag"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        if (config != nullptr && config->mode != RunMode::NORMAL_SEARCH) {
          std::cerr << "Can only select one mode at a time" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        ++i;
        if (config != nullptr) {
          config->mode = RunMode::SUPER_SEARCH;
          config->shiftlimit = atoi(argv[i]);
          if (config->shiftlimit == 0)
            config->xarray[0] = config->xarray[config->h] = true;
        }
      } else {
        std::cerr << "Must provide shift limit in -super mode" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-file")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr) {
          context->fileoutputflag = true;
          context->outfile = std::string(argv[i]);
        }
      } else {
        std::cerr << "No filename provided after -file" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-steal_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->steal_alg = atoi(argv[i]);
      } else {
        std::cerr << "No number provided after -steal_alg" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-split_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->split_alg = atoi(argv[i]);
      } else {
        std::cerr << "No number provided after -split_alg" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-block")) {
      if ((i + 1) < argc) {
        if (config != nullptr && config->mode != RunMode::NORMAL_SEARCH) {
          std::cerr << "Can only select one mode at a time" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        ++i;
        if (config != nullptr) {
          config->mode = RunMode::BLOCK_SEARCH;
          config->skiplimit = atoi(argv[i]);
        }
      } else {
        std::cerr << "Must provide skip limit in -block mode" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        int j = atoi(argv[i]);
        if (config != nullptr && j >= 0 && j <= config->h)
          config->xarray[config->dualflag ? (config->h - j) : j] = true;
        ++i;
      }
      --i;
    } else if (!strcmp(argv[i], "-verbose")) {
      if (config != nullptr)
        config->verboseflag = true;
    } else if (!strcmp(argv[i], "-threads")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->num_threads = static_cast<int>(atoi(argv[i]));
      } else {
        std::cerr << "Missing number of threads after -threads" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (i > 2) {
      char* p;
      long temp = strtol(argv[i], &p, 10);
      if (*p || i != 3) {
        std::cerr << "unrecognized input: " << argv[i] << std::endl;
        std::exit(EXIT_FAILURE);
      } else if (config != nullptr) {
        config->l = static_cast<int>(temp);
      }
    }
  }

  // consistency checks
  if (config != nullptr && fullflag && config->exactflag) {
    std::cerr << "-all and -exact flags cannot be used together" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void parse_args(std::string str, SearchConfig* const config,
      SearchContext* const context) {
  // tokenize the argslist string
  std::stringstream ss(str);
  std::string s;
  std::vector<std::string> args;
  while (std::getline(ss, s, ' '))
    args.push_back(s);

  const int argc = args.size();
  char** argv = new char*[argc];
  for (int i = 0; i < argc; ++i)
    argv[i] = &(args[i][0]);

  parse_args(argc, argv, config, context);

  delete[] argv;
}

//------------------------------------------------------------------------------
// Saving and loading checkpoint files
//------------------------------------------------------------------------------

void save_context(const SearchContext& context) {
  std::ofstream myfile;
  myfile.open(context.outfile, std::ios::out | std::ios::trunc);
  if (!myfile || !myfile.is_open())
    return;

  myfile << "version           6.1" << std::endl
         << "command line      " << context.arglist << std::endl
         << "length            " << context.l_current << std::endl
         << "length limit      " << context.maxlength << std::endl
         << "states            " << context.numstates << std::endl
         << "shift cycles      " << context.numcycles << std::endl
         << "short cycles      " << context.numshortcycles << std::endl
         << "patterns          " << context.npatterns << std::endl
         << "patterns (seen)   " << context.ntotal << std::endl
         << "nodes visited     " << context.nnodes << std::endl
         << "threads           " << context.num_threads << std::endl
         << "hardware threads  " << std::thread::hardware_concurrency()
                                 << std::endl
         << "seconds elapsed   " << std::fixed << std::setprecision(4)
                                 << context.secs_elapsed << std::endl
         << "seconds working   " << context.secs_working << std::endl
         << "seconds avail     " << context.secs_available << std::endl
         << std::endl;

  myfile << "patterns" << std::endl;
  for (const std::string& str : context.patterns)
    myfile << str << std::endl;

  if (context.assignments.size() > 0) {
    myfile << std::endl << "work assignments remaining" << std::endl;
    for (const WorkAssignment& wa : context.assignments)
      myfile << "  " << wa << std::endl;
  }

  myfile.close();
}

bool pattern_compare(const std::string& pat1, const std::string& pat2) {
  if (pat2.size() == 0)
    return false;
  if (pat1.size() == 0)
    return true;

  int pat1_start = (pat1[0] == ' ' || pat1[0] == '*') ? 2 : 0;
  int pat1_end = pat1_start;
  while (pat1_end != (int)pat1.size() && pat1[pat1_end] != ' ')
    ++pat1_end;

  int pat2_start = (pat2[0] == ' ' || pat2[0] == '*') ? 2 : 0;
  int pat2_end = pat2_start;
  while (pat2_end != (int)pat2.size() && pat2[pat2_end] != ' ')
    ++pat2_end;

  // shorter patterns sort earlier
  if ((pat1_end - pat1_start) < (pat2_end - pat2_start))
    return true;
  if ((pat1_end - pat1_start) > (pat2_end - pat2_start))
    return false;

  // ground state before excited state patterns
  if (pat1[0] == ' ' && pat2[0] == '*')
    return true;
  if (pat1[0] == '*' && pat2[0] == ' ')
    return false;

  // ascii order, except '+' is higher than any other character
  for (int i = pat1_start; i < pat1_end; ++i) {
    if (pat1[i] == pat2[i])
      continue;
    if (pat1[i] == '+' && pat2[i] != '+')
      return false;
    if (pat1[i] != '+' && pat2[i] == '+')
      return true;
    return pat1[i] < pat2[i];
  }
  return false;
}

static inline void trim(std::string& s) {
  // trim left, then trim right
  s.erase(
      s.begin(),
      std::find_if(s.begin(), s.end(),
          [](unsigned char ch) { return !std::isspace(ch); })
  );
  s.erase(
      std::find_if(s.rbegin(), s.rend(),
          [](unsigned char ch) { return !std::isspace(ch); }).base(),
      s.end()
  );
}

// return true on success

bool load_context(std::string file, SearchContext& context) {
  std::ifstream myfile;
  myfile.open(file, std::ios::in);
  if (!myfile || !myfile.is_open()) {
    std::cerr << "error reading file: could not open" << std::endl;
    return false;
  }

  std::string s;
  int linenum = 0;
  bool reading_assignments = false;
  const int column_start = 17;

  while (myfile) {
    std::getline(myfile, s);
    std::string val;
    std::string error;

    switch (linenum) {
      case 0:
        if (s.rfind("version", 0) != 0) {
          error = "syntax in line 1";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        if (val != "6.1") {
          error = "file version is not 6.1";
          break;
        }
        break;
      case 1:
        if (s.rfind("command", 0) != 0) {
          error = "syntax in line 2";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.arglist = val;
        break;
      case 2:
        if (s.rfind("length", 0) != 0) {
          error = "syntax in line 3";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.l_current = std::stoi(val);
        break;
      case 3:
        if (s.rfind("length", 0) != 0) {
          error = "syntax in line 4";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.maxlength = std::stoi(val);
        break;
      case 4:
        if (s.rfind("states", 0) != 0) {
          error = "syntax in line 5";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numstates = std::stoi(val);
        break;
      case 5:
        if (s.rfind("shift cycles", 0) != 0) {
          error = "syntax in line 6";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numcycles = std::stoi(val);
        break;
      case 6:
        if (s.rfind("short cycles", 0) != 0) {
          error = "syntax in line 7";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numshortcycles = std::stoi(val);
        break;
      case 7:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 8";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.npatterns = std::stol(val);
        break;
      case 8:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 9";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.ntotal = std::stol(val);
        break;
      case 9:
        if (s.rfind("nodes", 0) != 0) {
          error = "syntax in line 10";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.nnodes = std::stol(val);
        break;
      case 10:
      case 11:
        break;
      case 12:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 13";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_elapsed = std::stod(val);
        break;
      case 13:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 14";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_working = std::stod(val);
        break;
      case 14:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 15";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_available = std::stod(val);
        break;
      case 15:
        break;
      case 16:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 17";
          break;
        }
        break;
    }

    if (error.size() > 0) {
      std::cerr << "error reading file: " << error << std::endl;
      myfile.close();
      return false;
    }

    val = s;
    trim(val);
    if (reading_assignments) {
      WorkAssignment wa;
      if (val.size() == 0) {
        // ignore empty line
      } else if (wa.from_string(val)) {
        context.assignments.push_back(wa);
      } else {
        std::cerr << "error reading work assignment in line " << (linenum + 1)
                  << std::endl;
        myfile.close();
        return false;
      }
    } else if (linenum > 16) {
      if (s.rfind("work", 0) == 0) {
        reading_assignments = true;
      } else if (val.size() > 0) {
        context.patterns.push_back(s);
      }
    }

    ++linenum;
  }
  myfile.close();
  return true;
}

//------------------------------------------------------------------------------
// Prep `config` and `context` data structures for calculation
//------------------------------------------------------------------------------

void prepare_calculation(int argc, char** argv, SearchConfig& config,
      SearchContext& context) {
  // first check if the user wants file output mode
  SearchContext args_context;
  parse_args(argc, argv, nullptr, &args_context);

  if (args_context.fileoutputflag) {
    std::ifstream myfile(args_context.outfile);
    if (myfile.good()) {
      // file exists; try resuming calculation
      std::cout << "reading checkpoint file '" << args_context.outfile << "'"
                << std::endl;

      if (!load_context(args_context.outfile, context))
        std::exit(EXIT_FAILURE);
      if (context.assignments.size() == 0) {
        std::cout << "calculation is finished" << std::endl;
        std::exit(0);
      }

      // parse the loaded argument list (from the original invocation) to get
      // the config, plus fill in the elements of context that weren't loaded
      parse_args(context.arglist, &config, &context);

      // in case the user has renamed the checkpoint file since the original
      // invocation, use current filename
      context.outfile = args_context.outfile;

      std::cout << "resuming calculation: " << context.arglist << std::endl
                << "loaded " << context.npatterns
                << " patterns (length " << context.l_current
                << ") and " << context.assignments.size()
                << " work assignments" << std::endl;
      return;
    }
  }

  // if not resuming, then get config and context from args
  parse_args(argc, argv, &config, &context);

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
  coordinator.run();

  if (context.fileoutputflag) {
    std::cout << "saving checkpoint file '" << context.outfile << "'"
              << std::endl;
    std::sort(context.patterns.begin(), context.patterns.end(),
        pattern_compare);
    save_context(context);
  }
  return 0;
}
