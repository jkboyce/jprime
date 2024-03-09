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
// 12/21/23  Version 6.2 adds many performance improvements.
// 01/28/24  Version 6.3 adds pattern counting by length, and search status
//           display in -verbose mode.
// 03/04/24  Version 6.4 changes command line interface, improves performance,
//           and adds non-recursive NORMAL mode search


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
#include <stdexcept>


void print_help() {
  const std::string helpString =
    "jprime version 6.4 (2024.03.04)\n"
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
    "   jprime 4 7\n"
    "   jprime 5 7 14-15\n"
    "   jprime 5 7 -noplus -file 5_7_all\n"
    "   jprime 4 9 112\n"
    "   jprime 4 9 14- -super 0 -inverse\n";

  std::cout << helpString << std::endl;
}

//------------------------------------------------------------------------------
// Parsing command line arguments
//------------------------------------------------------------------------------

void parse_args(size_t argc, char** argv, SearchConfig* const config,
      SearchContext* const context) {
  // defaults for length
  int l_min = 1;
  int l_max = -1;

  if (config != nullptr) {
    config->n = atoi(argv[1]);
    if (config->n < 1) {
      std::cerr << "Must have at least 1 object\n";
      std::exit(EXIT_FAILURE);
    }
    config->h = atoi(argv[2]);
    if (config->h < config->n) {
      std::cerr << "Max. throw value must equal or exceed number of objects\n";
      std::exit(EXIT_FAILURE);
    }

    // defaults for excluded self-throws
    config->xarray.resize(config->h + 1, false);

    // defaults for using dual graph
    if (config->h > (2 * config->n)) {
      config->dualflag = true;
      config->n = config->h - config->n;
    }
  }

  bool stealalg_given = false;

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
    } else if (!strcmp(argv[i], "-noplus")) {
      if (config != nullptr) {
        config->noplusminusflag = true;
      }
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        if (config != nullptr && config->mode != RunMode::NORMAL_SEARCH) {
          std::cerr << "Can only select one mode at a time\n";
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
        std::cerr << "Must provide shift limit in -super mode\n";
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
        std::cerr << "No filename provided after -file\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-steal_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr) {
          context->steal_alg = atoi(argv[i]);
          stealalg_given = true;
        }
      } else {
        std::cerr << "No number provided after -steal_alg\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-split_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->split_alg = atoi(argv[i]);
      } else {
        std::cerr << "No number provided after -split_alg\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        int j = atoi(argv[i]);
        if (config != nullptr && j == config->h) {
          std::cerr << "Cannot exclude max. throw value with -x\n";
          std::exit(EXIT_FAILURE);
        }
        if (config != nullptr && j >= 0 && j < config->h)
          config->xarray[config->dualflag ? (config->h - j) : j] = true;
        ++i;
      }
      --i;
    } else if (!strcmp(argv[i], "-verbose")) {
      if (config != nullptr)
        config->verboseflag = true;
    } else if (!strcmp(argv[i], "-count")) {
      if (config != nullptr)
        config->countflag = true;
    } else if (!strcmp(argv[i], "-info")) {
      if (config != nullptr)
        config->infoflag = true;
    } else if (!strcmp(argv[i], "-status")) {
      if (config != nullptr)
        config->statusflag = true;
    } else if (!strcmp(argv[i], "-threads")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->num_threads = static_cast<int>(atoi(argv[i]));
      } else {
        std::cerr << "Missing number of threads after -threads\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (i == 3) {
      // try to parse argument as a length or length range
      bool success = false;
      std::string s{argv[i]};
      int hyphens = std::count(s.begin(), s.end(), '-');
      if (hyphens == 0) {
        int num = atoi(argv[i]);
        if (num > 0) {
          l_min = l_max = num;
          success = true;
        }
      } else if (hyphens == 1) {
        success = true;
        auto hpos = s.find('-');
        std::string s1 = s.substr(0, hpos);
        std::string s2 = s.substr(hpos + 1);
        if (s1.size() > 0) {
          int num = atoi(s1.c_str());
          if (num > 0)
            l_min = num;
          else
            success = false;
        }
        if (s2.size() > 0) {
          int num = atoi(s2.c_str());
          if (num > 0)
            l_max = num;
          else
            success = false;
        }
      }

      if (!success) {
        std::cerr << "Error parsing length: " << argv[i] << '\n';
        std::exit(EXIT_FAILURE);
      }
    } else if (i > 3) {
      std::cerr << "Unrecognized input: " << argv[i] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  if (config != nullptr) {
    config->l_min = l_min;
    config->l_max = l_max;
  }

  // defaults
  if (context != nullptr && !stealalg_given) {
    context->steal_alg = 3;  // steal from lowest `root_pos`
  }

  // consistency checks
  if (context != nullptr && context->num_threads < 1) {
    std::cerr << "Must have at least one worker thread\n";
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

  const size_t argc = args.size();
  char** argv = new char*[argc];
  for (size_t i = 0; i < argc; ++i)
    argv[i] = &(args[i][0]);

  parse_args(argc, argv, config, context);

  delete[] argv;
}

//------------------------------------------------------------------------------
// Saving and loading checkpoint files
//------------------------------------------------------------------------------

void save_context(const SearchConfig& config, const SearchContext& context) {
  std::ofstream myfile;
  myfile.open(context.outfile, std::ios::out | std::ios::trunc);
  if (!myfile || !myfile.is_open())
    return;

  myfile << "version           6.4\n"
         << "command line      " << context.arglist << '\n'
         << "length bound      " << context.l_bound << '\n'
         << "states            " << context.numstates << '\n'
         << "shift cycles      " << context.numcycles << '\n'
         << "short cycles      " << context.numshortcycles << '\n'
         << "patterns          " << context.npatterns << '\n'
         << "patterns (seen)   " << context.ntotal << '\n'
         << "nodes visited     " << context.nnodes << '\n'
         << "threads           " << context.num_threads << '\n'
         << "hardware threads  " << std::thread::hardware_concurrency() << '\n'
         << "seconds elapsed   " << std::fixed << std::setprecision(4)
                                 << context.secs_elapsed << '\n'
         << "seconds working   " << context.secs_working << '\n'
         << "seconds avail     " << context.secs_available << '\n';

  myfile << "\npatterns\n";
  for (const std::string& str : context.patterns)
    myfile << str << '\n';

  myfile << "\ncounts\n";
  for (int i = 1; i <= (config.l_max > 0 ? config.l_max: context.l_bound); ++i)
    myfile << i << ", " << context.count[i] << '\n';

  if (context.assignments.size() > 0) {
    myfile << "\nwork assignments remaining\n";
    for (const WorkAssignment& wa : context.assignments)
      myfile << "  " << wa << '\n';
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
  while (pat1_end != static_cast<int>(pat1.size()) && pat1[pat1_end] != ' ')
    ++pat1_end;

  int pat2_start = (pat2[0] == ' ' || pat2[0] == '*') ? 2 : 0;
  int pat2_end = pat2_start;
  while (pat2_end != static_cast<int>(pat2.size()) && pat2[pat2_end] != ' ')
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

bool load_context(const std::string& file, SearchContext& context) {
  std::ifstream myfile;
  myfile.open(file, std::ios::in);
  if (!myfile || !myfile.is_open()) {
    std::cerr << "error reading file: could not open\n";
    return false;
  }

  std::string s;
  int linenum = 0;
  int section = 1;
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
        if (val != "6.4") {
          error = "file version is not 6.4";
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
        if (s.rfind("length bound", 0) != 0) {
          error = "syntax in line 3";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.l_bound = std::stoi(val);
        context.count.resize(context.l_bound + 1, 0);
        break;
      case 3:
        if (s.rfind("states", 0) != 0) {
          error = "syntax in line 4";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numstates = std::stoi(val);
        break;
      case 4:
        if (s.rfind("shift cycles", 0) != 0) {
          error = "syntax in line 5";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numcycles = std::stoi(val);
        break;
      case 5:
        if (s.rfind("short cycles", 0) != 0) {
          error = "syntax in line 6";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.numshortcycles = std::stoi(val);
        break;
      case 6:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 7";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.npatterns = std::stol(val);
        break;
      case 7:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 8";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.ntotal = std::stol(val);
        break;
      case 8:
        if (s.rfind("nodes", 0) != 0) {
          error = "syntax in line 9";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.nnodes = std::stol(val);
        break;
      case 9:
      case 10:
        break;
      case 11:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 12";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_elapsed = std::stod(val);
        break;
      case 12:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 13";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_working = std::stod(val);
        break;
      case 13:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 14";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_available = std::stod(val);
        break;
      case 14:
        break;
      case 15:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 16";
          break;
        }
        section = 2;
        break;
    }

    if (error.size() > 0) {
      std::cerr << "error reading file: " << error << '\n';
      myfile.close();
      return false;
    }

    if (linenum < 16) {
      ++linenum;
      continue;
    }

    val = s;
    trim(val);

    if (val.size() == 0) {
      // ignore empty lines
    } else if (s.rfind("count", 0) == 0) {
      section = 3;
    } else if (s.rfind("work", 0) == 0) {
      section = 4;
    } else if (section == 2) {
      context.patterns.push_back(s);
    } else if (section == 3) {
      // read counts
      size_t commapos = s.find(",");
      if (commapos == std::string::npos) {
        std::cerr << "error reading count in line " << (linenum + 1) << '\n';
        myfile.close();
        return false;
      }
      std::string field1 = s.substr(0, commapos);
      std::string field2 = s.substr(commapos + 1, s.size());
      int i = std::stoi(field1);
      context.count[i] = static_cast<std::uint64_t>(std::stoull(field2));
    } else if (section == 4) {
      // read work assignments
      WorkAssignment wa;
      if (wa.from_string(val)) {
        context.assignments.push_back(wa);
      } else {
        std::cerr << "error reading work assignment in line " << (linenum + 1)
                  << '\n';
        myfile.close();
        return false;
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
      std::cout << "reading checkpoint file '" << args_context.outfile << "'\n";

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

      std::cout << "resuming calculation: " << context.arglist << '\n'
                << "loaded " << context.npatterns
                << " patterns and " << context.assignments.size()
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
    save_context(config, context);
  }
  return 0;
}
