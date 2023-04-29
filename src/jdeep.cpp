
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
    "    -block <skips>  find patterns in block form, allowing the specified\n"
    "                       number of skips\n"
    "    -super <shifts> find (nearly) superprime patterns, allowing the specified\n"
    "                       number of shift throws\n"
    "    -inverse        print inverse also, in -super mode\n"
    "    -g              find ground-state patterns only\n"
    "    -ng             find excited-state patterns only\n"
    "    -full           print all patterns; otherwise only patterns as long\n"
    "                       currently-longest one found are printed\n"
    "    -noprint        suppress printing of patterns\n"
    "    -exact          prints patterns of exact length specified (no longer)\n"
    "    -x <throw1 throw2 ...>\n"
    "                    exclude listed throws (speeds search)\n"
    "    -trim           turn graph trimming on\n"
    "    -notrim         turn graph trimming off\n"
    "    -file <name>    use the named file for checkpointing (when jdeep is\n"
    "                       interrupted), resuming, and final output\n"
    "    -threads <num>  run with the given number of worker threads\n"
    "    -verbose        print worker status information\n";

  std::cout << helpString << std::endl;
}

bool try_file_resume(char* file, SearchConfig& config, SearchContext& context) {
  return false;
}

void configure_search(int argc, char** argv, SearchConfig& config,
      SearchContext& context) {
  config.n = atoi(argv[1]);
  if (config.n < 1) {
    std::cout << "Must have at least 1 object" << std::endl;
    std::exit(0);
  }
  config.h = atoi(argv[2]);
  if (config.h < config.n) {
    std::cout << "Max. throw value must equal or exceed number of objects"
              << std::endl;
    std::exit(0);
  }

  // excluded self-throws
  config.xarray.resize(config.n, false);

  // defaults for going into dual space
  if (config.h > (2 * config.n)) {
    config.dualflag = true;
    config.n = config.h - config.n;
  }

  bool trimspecified = false;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "-noprint")) {
      config.printflag = false;
    } else if (!strcmp(argv[i], "-inverse")) {
      config.invertflag = true;
    } else if (!strcmp(argv[i], "-g")) {
      config.groundmode = 1;
    } else if (!strcmp(argv[i], "-ng")) {
      config.groundmode = 2;
    } else if (!strcmp(argv[i], "-trim")) {
      trimspecified = true;
      config.trimflag = true;
    } else if (!strcmp(argv[i], "-notrim")) {
      trimspecified = true;
      config.trimflag = false;
    } else if (!strcmp(argv[i], "-full")) {
      config.longestflag = false;
    } else if (!strcmp(argv[i], "-exact")) {
      config.exactflag = 1;
      config.longestflag = false;
      if (config.l < 2) {
        std::cout << "Must specify a length > 1 when using -exact flag"
                  << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        if (config.mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        config.mode = SUPER_MODE;
        config.shiftlimit = (double)atoi(argv[++i]);
        if (config.shiftlimit == 0)
          config.xarray[0] = config.xarray[config.h] = 1;
      } else {
        std::cout << "Must provide shift limit in -super mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-file")) {
      if ((i + 1) < argc) {
        context.fileoutputflag = true;
        context.outfile = argv[i + 1];
        ++i;
      } else {
        std::cout << "No filename provided after -file" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-block")) {
      if ((i + 1) < argc) {
        if (config.mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        config.mode = BLOCK_MODE;
        config.skiplimit = (double)atoi(argv[++i]);
      } else {
        std::cout << "Must provide skip limit in -block mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        int j = atoi(argv[i]);
        if (j >= 0 && j <= config.h)
          config.xarray[config.dualflag ? (config.h - j) : j] = 1;
        ++i;
      }
      --i;
    } else if (!strcmp(argv[i], "-verbose")) {
      config.verboseflag = true;
    } else if (!strcmp(argv[i], "-threads")) {
      if ((i + 1) < argc)
        context.num_threads = static_cast<int>(atoi(argv[++i]));
      else {
        std::cout << "Missing number of threads after -threads" << std::endl;
        std::exit(0);
      }
    } else {
      char* p;
      long temp = strtol(argv[i], &p, 10);
      if (*p) {
        std::cout << "unrecognized input: " << argv[i] << std::endl;
        std::exit(0);
      } else
        config.l = temp;
    }
  }

  // consistency checks
  if (config.invertflag && config.mode != SUPER_MODE) {
    std::cout << "-inverse flag can only be used in -super mode" << std::endl;
    std::exit(0);
  }

  // defaults for when to trim and when not to
  if (!trimspecified) {
    if (config.mode == BLOCK_MODE)
      config.trimflag = false;
    else if (config.mode == SUPER_MODE)
      config.trimflag = false;
    else if (config.longestflag)
      config.trimflag = true;
    else
      config.trimflag = false;
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

void save_output_file(const SearchContext& context) {
  std::ofstream myfile;
  myfile.open(context.outfile, std::ios::out | std::ios::trunc);

  myfile << "version          6.0" << std::endl
         << "command line     " << context.arglist << std::endl
         << "length           " << context.l_current << std::endl
         << "length limit     " << context.maxlength << std::endl
         << "num states       " << context.numstates << std::endl
         << "patterns (max)   " << context.npatterns << std::endl
         << "patterns (seen)  " << context.ntotal << std::endl
         << "hw threads       " << std::thread::hardware_concurrency() << std::endl
         << "secs elapsed     " << context.secs_elapsed << std::endl
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

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  if (argc == 1) {
    print_help();
    std::exit(0);
  }

  SearchConfig config;
  SearchContext context;
  configure_search(argc, argv, config, context);

  Coordinator coordinator(config, context);

  timespec start_ts, end_ts;
  timespec_get(&start_ts, TIME_UTC);
  coordinator.run();
  timespec_get(&end_ts, TIME_UTC);

  double runtime = ((double)end_ts.tv_sec + 1.0e-9 * end_ts.tv_nsec) -
      ((double)start_ts.tv_sec + 1.0e-9 * start_ts.tv_nsec);
  context.secs_elapsed += runtime;
  std::cout << "running time = "
            << std::setprecision(5) << context.secs_elapsed
            << " sec" << std::endl;

  if (context.fileoutputflag) {
    if (context.arglist.empty()) {
      for (int i = 0; i < argc; ++i) {
        if (i != 0)
          context.arglist += " ";
        context.arglist += argv[i];
      }
    }

    save_output_file(context);
  }

  return 0;
}
