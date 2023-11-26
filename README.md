# jprime
Parallel depth first search (DFS) to find extremely long prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as a composition (concatenation) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state diagram for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state diagram. Prime patterns correpond to *cycles* in the graph, which visit each state in the pattern only once.

Because the graph for $N$ objects and maximum throw $H$ is of finite size, there exists a longest prime siteswap pattern(s) for that case. The theory behind these longest prime patterns and how to find them is discussed in this 1999 [paper](https://github.com/jkboyce/jprime/blob/main/longest_prime_siteswaps_1999.pdf). Here we update the table of results in the paper to include results discovered since then.

`jprime` searches the juggling state graph to find patterns, and it exploits the structure of the graph to speed up the search. In addition the search is done in parallel over $T$ separate threads, by using a work-stealing scheme to balance workloads across threads.
