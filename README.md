# jprime
Parallel depth first search (DFS) to find prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. Prime patterns then correpond to *cycles* in the graph, i.e. circuits that visit each state in the pattern only once.

`jprime` searches the juggling state graph for prime patterns, using efficiency tricks to speed up the search. The search is done in parallel using a work-stealing scheme to distribute the search across multiple execution threads. `jprime` can also run on a CUDA GPU, scaling to 50,000 or more concurrent execution threads.

## Description and results

The following notes discuss the theory of prime juggling patterns, some details about algorithms used by `jprime`, and search results to date:

- [Notes on Prime Juggling Patterns (2025.05.09)](papers/prime%20juggling_2025.pdf)

## Running jprime

After cloning the repository, on a Unix system run `make` to build the `jprime` binary using the included makefile. `jprime` requires the following compiler versions or later: GCC 12, Clang 15, MSVC 19.29, Apple Clang 14.0.3. Install the Nvidia CUDA Toolkit if you would like to build the CUDA-enabled target using `make cuda`. Run the binary with no arguments to get a help message.
