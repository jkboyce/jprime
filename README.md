# jprime
Parallel depth first search (DFS) to find prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. Prime patterns then correpond to *cycles* in the graph, i.e. circuits that visit each state in the pattern only once.

`jprime` searches the juggling state graph for prime patterns, using efficiency tricks to speed up the search. The search is done in parallel using a work-stealing scheme to distribute the search across multiple execution threads. `jprime` can also run on a CUDA GPU, scaling to 50,000 or more concurrent workers.

## Description and results

The following notes discuss the theory of prime juggling patterns, some details about algorithms used by `jprime`, and search results to date:

* [Notes on Prime Juggling Patterns (2025.05.09)](papers/prime%20juggling_2025.pdf)

## Building and running jprime

The CMake build tool is used to build the `jprime` executable on Linux, Windows, and macOS.

Prerequisites:
* Cloned repository
  * `git clone https://github.com/jkboyce/jprime.git`
* C++ build tools for your platform
  * Linux: `gcc` or `clang` recent enough to support C++20
  * Windows: `Visual Studio 2022 Community Edition` includes everything needed, and is free; use Tools-->Command Line-->Developer Command Prompt to run CMake
  * macOS: `Xcode` and `Xcode Command Line Tools` from Apple
* `CMake` for your platform
  * Version 3.21 or later
  * For Linux and Windows this is included in the above build tools; for macOS it can be installed via Homebrew
* Optional: CUDA Toolkit from NVIDIA
  * Enables the CUDA build option, which allows searches to be run on the GPU
  * Linux and Windows installers are available from NVIDIA (NVIDIA no longer supports the CUDA Toolkit on macOS)

How to build:
* See instructions in `CMakeLists.txt` for how to configure and build from the command line

How to run:
* After building, find the `jprime` executable in the `build` folder of the repository
* Execute `jprime` with no arguments to get a help message (it has no GUI and runs only from the command line)
* `jprime -test` runs a series of tests to confirm a successful build
