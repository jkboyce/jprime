
/************************************************************************/
/*   jdeep version 6.0             by Jack Boyce        3/11/2023       */
/*                                 jboyce@gmail.com                     */
/*                                                                      */
/*   This is a modification of the original j.c, optimized for speed.   */
/*   It finds prime (no subpatterns) async siteswaps, using several     */
/*   tricks to speed up the search.                                     */
/*   Try the cases:                                                     */
/*       jdeep 4 7                                                      */
/*       jdeep 6 10                                                     */
/************************************************************************/

/*------------------------------------------------------------------------------
Version history:

 6/98     Version 1.0
 7/29/98  Version 2.0 adds the ability to search for block patterns (an idea
          from Johannes Waldmann), and more command line options.
 8/2/98   Finds patterns in dual graph, if faster.
 1/2/99   Version 3.0 implements a new algorithm for graph trimming. My machine
          now finds (3,9) in 10 seconds (vs 24 hours with V2.0)!
 2/14/99  Version 5.0 can find superprime (and nearly superprime) patterns.
          It also implements an improved algorithm for the standard mode,
          which uses shift cycles to speed the search.
 2/17/99  Version 5.1 adds -inverse option to print inverses of patterns found
          in -super mode.
------------------------------------------------------------------------------*/

#include "SearchConfig.hpp"
#include "SearchContext.hpp"
#include "Coordinator.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <ctime>

void print_help() {
  const std::string helpString =
    "jdeep version 6.0           by Jack Boyce\n"
    "   (04/14/23)                  jboyce@gmail.com\n"
    "\n"
    "The purpose of this program is to search for long prime async siteswap\n"
    "patterns. For an explanation of these terms, consult the page:\n"
    "    http://www.juggling.org/help/siteswap/\n"
    "\n"
    "Command-line format is:\n"
    "   jdeep <# objects> <max. throw> [<min. length>] [options]\n"
    "\n"
    "where:\n"
    "    <# objects>   = number of objects\n"
    "    <max. throw>  = largest allowed throw value\n"
    "    <min. length> = shortest patterns to find (optional, speeds search)\n"
    "\n"
    "Recognized options:\n"
    "    -block <skips>    find patterns in block form, allowing the specified\n"
    "                         number of skips\n"
    "    -super <shifts>   find (nearly) superprime patterns, allowing the\n"
    "                         specified number of shift throws\n"
    "    -inverse          print inverse also, in -super mode\n"
    "    -g                find ground-state patterns only\n"
    "    -ng               find excited-state patterns only\n"
    "    -full             print all patterns; otherwise only patterns as long\n"
    "                         currently-longest one found are printed\n"
    "    -noprint          suppress printing of patterns\n"
    "    -exact            prints patterns of exact length specified (no longer)\n"
    "    -x <throw1 throw2 ...>\n"
    "                      exclude listed throws (speeds search)\n"
    "    -trim             turn graph trimming on\n"
    "    -notrim           turn graph trimming off\n"
    "    -file <name>      use the named file for checkpointing (when jdeep is\n"
    "                         interrupted), resuming, and final output\n"
    "    -threads <num>    run with the given number of worker threads\n"
    "    -verbose          print worker status information\n"
    "    -steal_alg <num>  algorithm for selecting a worker to take work from\n"
    "    -split_alg <num>  algorithm for splitting a stolen work assignment\n";

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
      std::cout << "Must have at least 1 object" << std::endl;
      std::exit(0);
    }
    config->h = atoi(argv[2]);
    if (config->h < config->n) {
      std::cout << "Max. throw value must equal or exceed number of objects"
                << std::endl;
      std::exit(0);
    }

    // excluded self-throws
    config->xarray.resize(config->n, false);

    // defaults for using dual graph
    if (config->h > (2 * config->n)) {
      config->dualflag = true;
      config->n = config->h - config->n;
    }
  }

  bool trimspecified = false;

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
    } else if (!strcmp(argv[i], "-trim")) {
      trimspecified = true;
      if (config != nullptr)
        config->trimflag = true;
    } else if (!strcmp(argv[i], "-notrim")) {
      trimspecified = true;
      if (config != nullptr)
        config->trimflag = false;
    } else if (!strcmp(argv[i], "-full")) {
      if (config != nullptr)
        config->longestflag = false;
    } else if (!strcmp(argv[i], "-exact")) {
      if (config != nullptr) {
        config->exactflag = 1;
        config->longestflag = false;
        if (config->l < 2) {
          std::cout << "Must specify a length > 1 when using -exact flag"
                    << std::endl;
          std::exit(0);
        }
      }
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        if (config != nullptr && config->mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        ++i;
        if (config != nullptr) {
          config->mode = SUPER_MODE;
          config->shiftlimit = atoi(argv[i]);
          if (config->shiftlimit == 0)
            config->xarray[0] = config->xarray[config->h] = 1;
        }
      } else {
        std::cout << "Must provide shift limit in -super mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-file")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr) {
          context->fileoutputflag = true;
          context->outfile = std::string(argv[i]);
        }
      } else {
        std::cout << "No filename provided after -file" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-steal_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->steal_alg = atoi(argv[i]);
      } else {
        std::cout << "No number provided after -steal_alg" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-split_alg")) {
      if ((i + 1) < argc) {
        ++i;
        if (context != nullptr)
          context->split_alg = atoi(argv[i]);
      } else {
        std::cout << "No number provided after -split_alg" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-block")) {
      if ((i + 1) < argc) {
        if (config != nullptr && config->mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        ++i;
        if (config != nullptr) {
          config->mode = BLOCK_MODE;
          config->skiplimit = atoi(argv[i]);
        }
      } else {
        std::cout << "Must provide skip limit in -block mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        int j = atoi(argv[i]);
        if (config != nullptr && j >= 0 && j <= config->h)
          config->xarray[config->dualflag ? (config->h - j) : j] = 1;
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
        std::cout << "Missing number of threads after -threads" << std::endl;
        std::exit(0);
      }
    } else if (i > 2) {
      char* p;
      long temp = strtol(argv[i], &p, 10);
      if (*p || i != 3) {
        std::cout << "unrecognized input: " << argv[i] << std::endl;
        std::exit(0);
      } else if (config != nullptr) {
        config->l = static_cast<int>(temp);
      }
    }
  }

  // consistency checks
  if (config != nullptr && config->invertflag && config->mode != SUPER_MODE) {
    std::cout << "-inverse flag can only be used in -super mode" << std::endl;
    std::exit(0);
  }

  // defaults for when to trim the graph
  if (config != nullptr && !trimspecified) {
    if (config->mode == BLOCK_MODE)
      config->trimflag = false;
    else if (config->mode == SUPER_MODE)
      config->trimflag = false;
    else if (config->longestflag)
      config->trimflag = true;
    else
      config->trimflag = false;
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

  myfile << "version           6.0" << std::endl
         << "command line      " << context.arglist << std::endl
         << "length            " << context.l_current << std::endl
         << "length limit      " << context.maxlength << std::endl
         << "states            " << context.numstates << std::endl
         << "patterns          " << context.npatterns << std::endl
         << "patterns (seen)   " << context.ntotal << std::endl
         << "nodes visited     " << context.nnodes << std::endl
         << "threads           " << context.num_threads << std::endl
         << "hardware threads  " << std::thread::hardware_concurrency() << std::endl
         << "seconds elapsed   " << context.secs_elapsed << std::endl
         << "seconds working   " << context.secs_elapsed_working << std::endl
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
  if (!myfile || !myfile.is_open())
    return false;

  std::string s;
  int linenum = 0;
  bool reading_assignments = false;

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
        val = s.substr(17, s.size());
        trim(val);
        if (val != "6.0") {
          error = "file version is not 6.0";
          break;
        }
        break;
      case 1:
        if (s.rfind("command", 0) != 0) {
          error = "syntax in line 2";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.arglist = val;
        break;
      case 2:
        if (s.rfind("length", 0) != 0) {
          error = "syntax in line 3";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.l_current = std::stoi(val);
        break;
      case 3:
        if (s.rfind("length", 0) != 0) {
          error = "syntax in line 4";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.maxlength = std::stoi(val);
        break;
      case 4:
        if (s.rfind("states", 0) != 0) {
          error = "syntax in line 5";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.numstates = std::stoi(val);
        break;
      case 5:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 6";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.npatterns = std::stol(val);
        break;
      case 6:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 7";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.ntotal = std::stoi(val);
        break;
      case 7:
        if (s.rfind("nodes", 0) != 0) {
          error = "syntax in line 8";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.nnodes = std::stol(val);
        break;
      case 8:
      case 9:
        break;
      case 10:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 11";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.secs_elapsed = std::stod(val);
        break;
      case 11:
        if (s.rfind("seconds", 0) != 0) {
          error = "syntax in line 12";
          break;
        }
        val = s.substr(17, s.size());
        trim(val);
        context.secs_elapsed_working = std::stod(val);
        break;
      case 12:
        break;
      case 13:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 14";
          break;
        }
        break;
    }

    if (error.size() > 0) {
      std::cout << "error reading file: " << error << std::endl;
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
        std::cout << "error reading work assignment in line " << (linenum + 1)
                  << std::endl;
        myfile.close();
        return false;
      }
    } else if (linenum > 13) {
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
      std::cout << "reading checkpoint file '" << args_context.outfile
                << "'" << std::endl;

      if (load_context(args_context.outfile, context)) {
        if (context.assignments.size() == 0) {
          std::cout << "calculation is finished" << std::endl;
          std::exit(0);
        }

        // parse the loaded argument list (from the original invocation) to get
        // the config, plus fill in the elements of context that aren't loaded
        parse_args(context.arglist, &config, &context);

        std::cout << "resuming calculation: " << context.arglist << std::endl
                  << "loaded " << context.npatterns
                  << " patterns (length " << context.l_current
                  << ") and " << context.assignments.size()
                  << " work assignments" << std::endl;
        for (const std::string& s : context.patterns)
          std::cout << s << std::endl;
        return;
      } else {
        // error in load_context()
        std::exit(0);
      }
    }
  }

  // if not resuming, then get config and context from args
  parse_args(argc, argv, &config, &context);

  // save original argument list
  if (context.fileoutputflag) {
    for (int i = 0; i < argc; ++i) {
      if (i != 0)
        context.arglist += " ";
      context.arglist += argv[i];
    }
  }

  // set initial work assignment
  WorkAssignment wa;
  wa.start_state = -1;
  wa.end_state = -1;
  wa.root_pos = 0;
  for (int i = 0; i <= config.h; ++i) {
    if (!config.xarray[i])
      wa.root_throwval_options.push_back(i);
  }
  context.assignments.push_back(wa);
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc == 1) {
    print_help();
    std::exit(0);
  }

  timespec start_ts, end_ts;
  SearchConfig config;
  SearchContext context;
  prepare_calculation(argc, argv, config, context);
  Coordinator coordinator(config, context);

  timespec_get(&start_ts, TIME_UTC);
  coordinator.run();
  timespec_get(&end_ts, TIME_UTC);

  double runtime = ((double)end_ts.tv_sec + 1.0e-9 * end_ts.tv_nsec) -
      ((double)start_ts.tv_sec + 1.0e-9 * start_ts.tv_nsec);
  context.secs_elapsed += runtime;
  std::cout << "running time = "
            << std::setprecision(5) << context.secs_elapsed
            << " sec" << std::endl;
  if (context.num_threads > 1) {
    std::cout << "worker utilization = "
              << ((context.secs_elapsed_working / context.num_threads) /
                     context.secs_elapsed) * 100
              << " %" << std::endl;
  }

  if (context.fileoutputflag) {
    save_context(context);
    std::cout << "saved to checkpoint file '" << context.outfile
              << "'" << std::endl;
  }

  return 0;
}
