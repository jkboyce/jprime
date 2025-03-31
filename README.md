# jprime
Parallel depth first search (DFS) to find prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state graph for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state graph. Prime patterns then correpond to *cycles* in the graph, i.e. circuits that visit each state in the pattern only once.

`jprime` searches the juggling state graph for prime patterns, using efficiency tricks to speed up the search. The search is done in parallel using a work-stealing scheme to distribute the search across multiple execution threads.

## Description and results

I have written these notes to discuss the theory of prime patterns, how `jprime` works, and search results to date.

- [Notes on Prime Juggling Patterns](papers/prime juggling_2025.pdf)

## Running jprime

After cloning the repository, on a Unix system run `make` to build the `jprime` binary using the included makefile. `jprime` requires the following compiler versions or later: GCC 12, Clang 15, MSVC 19.29, Apple Clang 14.0.3. Run the binary with no arguments to get a help message.
