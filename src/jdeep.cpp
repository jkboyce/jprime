
/************************************************************************/
/*   jdeep.c version 5.2           by Jack Boyce        3/11/2023       */
/*                                 jboyce@gmail.com                     */
/*                                                                      */
/*   This is a modification of the original j.c, optimized for speed.   */
/*   It is used to find simple (no subpatterns) asynch siteswaps.       */
/*   Basically it works by finding cycles in the state transition       */
/*   matrix.  Try the cases:                                            */
/*       jdeep 4 7                                                      */
/*       jdeep 5 8                                                      */
/************************************************************************/

/*
-------------------------------------------------------------------------
Version history:
   6/98    Version 1.0
   7/29/98 Version 2.0 adds the ability to search for block patterns (an
           idea brazenly scammed from Johannes Waldmann), as well as provides
           more command line options to configure program behavior.
   8/2/98  Automatically finds patterns in dual graph, if that's faster.
   1/2/99  Version 3.0 implements a new algorithm (see below).  My machine
           now finds (3,9) in 10 seconds (rather than 24 hours with V2.0)!
   2/14/99 Version 5.0 can find supersimple (and nearly supersimple)
           patterns.  It also implements an improved algorithm for the
           standard mode, one which is aware of the shift cycles.
   2/17/99 Version 5.1 adds -inverse option to print inverses of patterns
           found in -super mode.
-------------------------------------------------------------------------
Documentation:

jdeep version 5.1           by Jack Boyce
   (02/17/99)                  jboyce7@yahoo.com

The purpose of this program is to search for long simple asynch siteswap
patterns.  For an explanation of these terms, consult the page:
    http://www.juggling.org/help/siteswap/

Command-line format is:
    <# objects> <max. throw> [<min. length>] [options]

where:
    <# objects>   = number of objects in the patterns found
    <max. throw>  = largest throw value to use
    <min. length> = shortest patterns to find (optional, speeds search)

The various command-line options are:
    -block <skips>  find patterns in block form, allowing the specified
                       number of skips
    -super <shifts> find (nearly) supersimple patterns, allowing the specified
                       number of shift throws
    -inverse        print inverse also, in -super mode
    -g              find ground-state patterns only
    -ng             find excited-state patterns only
    -full           print all patterns; otherwise only patterns as long
                       currently-longest one found are printed
    -noprint        suppress printing of patterns
    -exact          prints patterns of exact length specified (no longer)
    -x <throw1 throw2 ...>
                    exclude listed throws (speeds search)
    -trim           force graph trimming algorithm on
    -notrim         force graph trimming algorithm off
    -file           run in file output mode
    -time <secs>    run for specified time before stopping

File Output Mode:

The program can be run in "file output mode", where output is sent
to disk and the calculation can be interrupted and restarted.  (This
mode was added to facilitate very large, time-consuming searches.)

When you execute the program with file output mode on, a file called
"jdeep5.core" is created in the same directory as the application.
When the program stops (either as a result of setting a time limit
with the -time option, or pressing a key under MacOS, or finishing
execution) it outputs the current state of the calculation to
"jdeep5.core".  Executing the program again (no command lines input
needed this time) will load "jdeep.core" and resume the calculation,
again stopping when time runs out.  On successive runs, program output
is sent to the files "jdeep5.out.001", "jdeep5.out.002", etc.

VERY IMPORTANT: You must manually delete the "jdeep5.core" file when
a file output run is finished!  When the program sees this file at
startup it thinks it's in the middle of a calculation and ignores
the command line options.  (Thus making the program appear to behave
oddly.)

New Algorithm:

The new faster algorithm in version 3.0 was inspired by an algorithm
for finding Hamiltonian circuits in a directed graph due to Nicos
Christofides (see "Graph Theory: An Algorithmic Approach" by N.
Christofides, 1975 (Academic Press), ISBN 0-12-174350-0, Ch. 10).
My adaptation is quite simple and is roughly described as follows:

There is a new variable max_possible which starts at ns (the number of
states) and records the longest possible pattern that could come out of
the current partial path.

When a link is added to the path (from a to b),
    1)  delete all other links going out of a, to {c,d,...}, and
        "inupdate" {c,d,...}
    2)  delete all other links going into b, from {e,f,...}, and
        "outupdate" {e,f,...}

"inupdating" a vertex i consists of:

    If i has indegree 0:
        a)  decrement max_possible
        b)  if (max_possible < l) then backtrack
        c)  delete all links going out of i, to {j,k,...}
        d)  inupdate {j,k,...}

"outupdating" a vertex i consists of:

    If i has outdegree 0:
        a)  decrement max_possible
        b)  if (max_possible < l) then backtrack
        c)  delete all links from {j,k,...} going into i
        d)  outupdate {j,k,...}

This new procedure is not activated when searching for patterns in
block form (the additional overhead makes it slower than ordinary
searching), or when doing a -full search.  In the latter case there
is certainly a cutoff in the min_length parameter above which the
new algorithm would be more efficient, but I haven't determined
this cutoff.
-------------------------------------------------------------------------
*/

#include "jdeep.hpp"
#include "Coordinator.hpp"

#include <iostream>
#include <iomanip>
#include <ctime>

void print_help() {
  const std::string helpString =
    "jdeep version 6.0           by Jack Boyce\n"
    "   (04/14/23)                  jboyce@gmail.com\n"
    "\n"
    "The purpose of this program is to search for long simple async siteswap\n"
    "patterns. For an explanation of these terms, consult the page:\n"
    "    http://www.juggling.org/help/siteswap/\n"
    "\n"
    "Command-line format is:\n"
    "   jdeep <# objects> <max. throw> [<min. length>] [options]\n"
    "\n"
    "where:\n"
    "    <# objects>   = number of objects in the patterns found\n"
    "    <max. throw>  = largest throw value to use\n"
    "    <min. length> = shortest patterns to find (optional, speeds search)\n"
    "\n"
    "The recognized options are:\n"
    "    -block <skips>  find patterns in block form, allowing the specified\n"
    "                       number of skips\n"
    "    -super <shifts> find (nearly) supersimple patterns, allowing the specified\n"
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
    "    -trim           force state graph trimming algorithm on\n"
    "    -notrim         force state graph trimming algorithm off\n"
    "    -file           run in file output mode\n"
    "    -time <secs>    run for specified time before stopping\n"
    "    -threads <num>  run with the given number of worker threads\n"
    "    -verbose        print worker status information\n";

  std::cout << helpString << std::endl;
}

JdeepConfig parse_CLI_parameters(int argc, char** argv) {
  JdeepConfig config;

  config.sc.n = atoi(argv[1]);
  if (config.sc.n < 1) {
    std::cout << "Must have at least 1 object" << std::endl;
    std::exit(0);
  }
  config.sc.h = atoi(argv[2]);
  if (config.sc.h < config.sc.n) {
    std::cout << "Max. throw value must equal or exceed number of objects"
              << std::endl;
    std::exit(0);
  }

  // excluded self-throws
  config.sc.xarray.resize(config.sc.n, false);

  // defaults for going into dual space
  if (config.sc.h > (2 * config.sc.n)) {
    config.sc.dualflag = true;
    config.sc.n = config.sc.h - config.sc.n;
  }

  bool trimspecified = false;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "-noprint")) {
      config.sc.printflag = false;
    } else if (!strcmp(argv[i], "-inverse")) {
      config.sc.invertflag = true;
    } else if (!strcmp(argv[i], "-g")) {
      config.sc.groundmode = 1;
    } else if (!strcmp(argv[i], "-ng")) {
      config.sc.groundmode = 2;
    } else if (!strcmp(argv[i], "-trim")) {
      trimspecified = true;
      config.sc.trimflag = true;
    } else if (!strcmp(argv[i], "-notrim")) {
      trimspecified = true;
      config.sc.trimflag = false;
    } else if (!strcmp(argv[i], "-full")) {
      config.sc.longestflag = false;
    } else if (!strcmp(argv[i], "-exact")) {
      config.sc.exactflag = 1;
      config.sc.longestflag = false;
      if (config.sc.l < 2) {
        std::cout << "Must specify a length > 1 when using -exact flag"
                  << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-super")) {
      if ((i + 1) < argc) {
        if (config.sc.mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        config.sc.mode = SUPER_MODE;
        config.sc.shiftlimit = (double)atoi(argv[++i]);
        if (config.sc.shiftlimit == 0)
          config.sc.xarray[0] = config.sc.xarray[config.sc.h] = 1;
      } else {
        std::cout << "Must provide shift limit in -super mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-file")) {
      // config.sc.fileoutputflag = 1;
    } else if (!strcmp(argv[i], "-time")) {
      if ((i + 1) < argc) {
        // timelimiton = 1;
        // timelimit = (double)atoi(argv[++i]);
      }
    } else if (!strcmp(argv[i], "-block")) {
      if ((i + 1) < argc) {
        if (config.sc.mode != NORMAL_MODE) {
          std::cout << "Can only select one mode at a time" << std::endl;
          std::exit(0);
        }
        config.sc.mode = BLOCK_MODE;
        config.sc.skiplimit = (double)atoi(argv[++i]);
      } else {
        std::cout << "Must provide skip limit in -block mode" << std::endl;
        std::exit(0);
      }
    } else if (!strcmp(argv[i], "-x")) {
      ++i;
      while (i < argc && argv[i][0] != '-') {
        int j = atoi(argv[i]);
        if (j >= 0 && j <= config.sc.h)
          config.sc.xarray[config.sc.dualflag ? (config.sc.h - j) : j] = 1;
        ++i;
      }
      --i;
    } else if (!strcmp(argv[i], "-verbose")) {
      config.sc.verboseflag = true;
    } else if (!strcmp(argv[i], "-threads")) {
      if ((i + 1) < argc)
        config.num_threads = static_cast<int>(atoi(argv[++i]));
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
        config.sc.l = temp;
    }
  }

  // consistency checks
  if (config.sc.invertflag && config.sc.mode != SUPER_MODE) {
    std::cout << "-inverse flag can only be used in -super mode" << std::endl;
    std::exit(0);
  }

  // defaults for when to trim and when not to
  if (!trimspecified) {
    if (config.sc.mode == BLOCK_MODE)
      config.sc.trimflag = false;
    else if (config.sc.mode == SUPER_MODE)
      config.sc.trimflag = false;
    else if (config.sc.longestflag)
      config.sc.trimflag = true;
    else
      config.sc.trimflag = false;
  }

  return config;
}

//------------------------------------------------------------------------------
// Execution entry point
//------------------------------------------------------------------------------

int main(int argc, char** argv) {
  // char filename[80];
  // FILE *fpcore;

  /* if ((fpcore = fopen("jdeep5.core", "r")) != 0) {
      jcore.readcore(fpcore);
      fclose(fpcore);
      if (jcore.finished)
          std::exit(0);
      jcore.reloading = true;
      jcore.reloadcount = 0;
  } else { */

  if (argc < 3) {
    print_help();
    std::exit(0);
  }

  JdeepConfig config = parse_CLI_parameters(argc, argv);

  /*
  if (jcore.fileoutputflag) {
      snprintf(filename, 80, "jdeep5.out.%.3d", jcore.nextfilenum);
      if ((jcore.fpout = fopen(filename, "w")) == 0) {
          fprintf(stderr, "Error: Can't open file: %s\n", filename);
          std::exit(0);
      }
  } else
      jcore.fpout = stdout;
  */

  timespec start_ts, end_ts;
  timespec_get(&start_ts, TIME_UTC);
  Coordinator coordinator(config.sc);
  coordinator.run(config.num_threads);
  timespec_get(&end_ts, TIME_UTC);

  double runtime = ((double)end_ts.tv_sec + 1.0e-9 * end_ts.tv_nsec) -
      ((double)start_ts.tv_sec + 1.0e-9 * start_ts.tv_nsec);

  std::cout << "running time = " << std::setprecision(5) << runtime << " sec"
            << std::endl;

  /*
  if (jcore.fileoutputflag) {
      fclose(jcore.fpout);
      jcore.finished = 1;  // mark finished in case we try to restart
      if ((fpcore = fopen("jdeep5.core", "w")) != 0) {
          jcore.writecore(fpcore);
          fclose(fpcore);
      }
  }
  */

  return 0;
}
