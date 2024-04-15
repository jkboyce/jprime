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


#include "SearchConfig.h"
#include "SearchContext.h"
#include "Coordinator.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string>
#include <exception>


//------------------------------------------------------------------------------
// Help message
//------------------------------------------------------------------------------

void print_help() {
  const std::string helpString =
    "jprime version 6.6 (2024.04.03)\n"
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
// Loading checkpoint files
//------------------------------------------------------------------------------

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

// Load a checkpoint file from disk into the SearchContext structure.
//
// Returns true on success. In case of any error, print an error message to
// std::cerr and return false.

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
        if (val != "6.6") {
          error = "file version is not 6.6";
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
      case 3:
      case 4:
      case 5:
      case 6:
        break;
      case 7:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 8";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.npatterns = std::stoull(val);
        break;
      case 8:
        if (s.rfind("patterns (", 0) != 0) {
          error = "syntax in line 9";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.ntotal = std::stoull(val);
        break;
      case 9:
        if (s.rfind("nodes", 0) != 0) {
          error = "syntax in line 10";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.nnodes = std::stoull(val);
        break;
      case 10:
      case 11:
        break;
      case 12:
        if (s.rfind("seconds elapsed", 0) != 0) {
          error = "syntax in line 13";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_elapsed = std::stod(val);
        break;
      case 13:
        if (s.rfind("seconds working", 0) != 0) {
          error = "syntax in line 14";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        context.secs_working = std::stod(val);
        break;
      case 14:
        if (s.rfind("seconds avail", 0) != 0) {
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
        section = 2;
        break;
    }

    if (error.size() > 0) {
      std::cerr << "error reading file: " << error << '\n';
      myfile.close();
      return false;
    }

    if (linenum < 17) {
      ++linenum;
      continue;
    }

    val = s;
    trim(val);

    if (val.size() == 0) {
      // ignore empty lines
    } else if (s.rfind("counts", 0) == 0) {
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
      context.count.resize(i + 1, 0);
      context.count.at(i) = std::stoull(field2);
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
// Saving checkpoint files
//------------------------------------------------------------------------------

void save_context(const SearchConfig& config, const SearchContext& context) {
  std::ofstream myfile;
  myfile.open(config.outfile, std::ios::out | std::ios::trunc);
  if (!myfile || !myfile.is_open())
    return;

  myfile << "version           6.6\n"
         << "command line      " << context.arglist << '\n'
         << "states            " << context.full_numstates << '\n'
         << "shift cycles      " << context.full_numcycles << '\n'
         << "short cycles      " << context.full_numshortcycles << '\n'
         << "length bound      " << context.l_bound << '\n'
         << "states (in mem)   " << context.memory_numstates << '\n'
         << "patterns          " << context.npatterns << '\n'
         << "patterns (total)  " << context.ntotal << '\n'
         << "nodes completed   " << context.nnodes << '\n'
         << "threads           " << config.num_threads << '\n'
         << "cores avail       " << std::thread::hardware_concurrency() << '\n'
         << "seconds elapsed   " << std::fixed << std::setprecision(4)
                                 << context.secs_elapsed << '\n'
         << "seconds working   " << context.secs_working << '\n'
         << "seconds avail     " << context.secs_available << '\n';

  myfile << "\npatterns\n";
  for (const std::string& str : context.patterns)
    myfile << str << '\n';

  myfile << "\ncounts\n";
  for (size_t i = 1; i <= (config.l_max > 0 ? config.l_max : context.l_bound);
      ++i) {
    myfile << i << ", " << context.count[i] << '\n';
  }

  if (context.assignments.size() > 0) {
    myfile << "\nwork assignments remaining\n";
    for (const WorkAssignment& wa : context.assignments)
      myfile << "  " << wa << '\n';
  }

  myfile.close();
}

//------------------------------------------------------------------------------
// Sorting patterns
//------------------------------------------------------------------------------

// Convert a pattern line printed as comma-separated integers into a vector of
// ints.

std::vector<int> parse_pattern_line(const std::string& p) {
  std::string pat;
  std::vector<int> result;

  // remove the first colon and anything beyond
  pat = {p.begin(), std::find(p.begin(), p.end(), ':')};

  // extract part between first and second '*'s if present
  auto star = std::find(pat.begin(), pat.end(), '*');
  if (star != pat.end()) {
    pat = {star + 1, std::find(star + 1, pat.end(), '*')};
  }

  auto x = pat.begin();
  while (true) {
    auto y = std::find(x, pat.end(), ',');
    std::string s{x, y};
    result.push_back(atoi(s.c_str()));
    if (y == pat.end())
      break;
    x = y + 1;
  }

  return result;
}

// Compare patterns printed as comma-separated integers
//
// Test case: jprime 7 42 6 -file test

bool pattern_compare_ints(const std::string& pat1, const std::string& pat2) {
  std::vector<int> vec1 = parse_pattern_line(pat1);
  std::vector<int> vec2 = parse_pattern_line(pat2);

  if (vec2.size() == 0)
    return false;
  if (vec1.size() == 0)
    return true;

  // shorter patterns sort earlier
  if (vec1.size() < vec2.size())
    return true;
  if (vec1.size() > vec2.size())
    return false;

  // ground state before excited state patterns
  if (pat1[0] == ' ' && pat2[0] == '*')
    return true;
  if (pat1[0] == '*' && pat2[0] == ' ')
    return false;

  // sort lower leading throws first
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] == vec2[i])
      continue;
    return vec1[i] < vec2[i];
  }
  return false;
}

// Compare patterns printed with letters (10='a', etc.)

bool pattern_compare_letters(const std::string& pat1, const std::string& pat2) {
  if (pat2.size() == 0)
    return false;
  if (pat1.size() == 0)
    return true;

  unsigned int pat1_start = (pat1[0] == ' ' || pat1[0] == '*') ? 2 : 0;
  unsigned int pat1_end = pat1_start;
  while (pat1_end != pat1.size() && pat1[pat1_end] != ' ')
    ++pat1_end;

  unsigned int pat2_start = (pat2[0] == ' ' || pat2[0] == '*') ? 2 : 0;
  unsigned int pat2_end = pat2_start;
  while (pat2_end != pat2.size() && pat2[pat2_end] != ' ')
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
  for (size_t i = pat1_start; i < pat1_end; ++i) {
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

// Standard library compliant Compare relation for patterns.
//
// Returns true if the first argument appears before the second in a strict
// weak ordering, and false otherwise.

bool pattern_compare(const std::string& pat1, const std::string& pat2) {
  bool has_comma = (std::find(pat1.begin(), pat1.end(), ',') != pat1.end() ||
      std::find(pat2.begin(), pat2.end(), ',') != pat2.end());

  if (has_comma)
    return pattern_compare_ints(pat1, pat2);
  else
    return pattern_compare_letters(pat1, pat2);
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
      // file exists; try resuming calculation
      std::cout << "reading checkpoint file '" << outfile << "'\n";

      if (!load_context(outfile, context)) {
        std::exit(EXIT_FAILURE);
      }
      if (context.assignments.size() == 0) {
        std::cout << "calculation is finished" << std::endl;
        std::exit(0);
      }

      // get config from the original arguments
      try {
        config.from_args(context.arglist);
      } catch (const std::invalid_argument& ie) {
        std::cerr << ie.what() << '\n';
        std::exit(EXIT_FAILURE);
      }

      // in case the user renamed the checkpoint file since the original
      // invocation, use the current filename
      config.outfile = outfile;

      std::cout << "resuming calculation: " << context.arglist << '\n'
                << "loaded " << context.npatterns
                << " patterns and " << context.assignments.size()
                << " work assignments" << std::endl;
      return;
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
    std::sort(context.patterns.begin(), context.patterns.end(),
        pattern_compare);
    save_context(config, context);
  }
  return 0;
}
