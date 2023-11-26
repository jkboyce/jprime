# jprime
Parallel depth first search (DFS) to find extremely long prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as a composition (concatenation) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state diagram for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state diagram. Prime patterns correpond to *cycles* in the graph, which visit each state in the pattern exactly only once during the traversal of the pattern.

Because the graph for $N$ objects and maximum throw $H$ is of finite size, there exists a longest prime siteswap pattern(s) for that case. Finding those longest prime patterns is the primary goal of `jprime`. The theory behind this, and how the automated search works, is described in this 1999 (paper)[https://github.com/jkboyce/jprime/blob/main/longest_prime_siteswaps_1999.pdf]. Note the table of known results in that paper does not include findings since then.
