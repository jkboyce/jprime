# jprime
Parallel depth first search (DFS) to find extremely long prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state graph for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state graph. Prime patterns then correpond to *cycles* in the graph, circuits that visit each state in the pattern only once.

Because the graph for $N$ objects and maximum throw $H$ is of finite size equal to the number of states ($H$ choose $N$), there exists a longest prime siteswap pattern(s) for that case. The theory behind these longest prime patterns and how to find them is discussed in this 1999 [paper](https://github.com/jkboyce/jprime/blob/main/longest_prime_siteswaps_1999.pdf). Here we update the table of results in the paper to include results discovered since then.

`jprime` searches the juggling state graph to find patterns, and it exploits the structure of the graph to speed up the search. In addition the search is done in parallel over $T$ threads, using a work-stealing scheme to balance the work across threads.

# Results

| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns {complete, incomplete} |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  2  |  3  |   3    |     3       |   3 |      1                          |
|  2  |  4  |   6    |     4       |   4 |      2                          |
|  2  |  5  |   10    |     8       |   8 |      1                          |
|  2  |  6  |   15    |     12       |   12 |      1                          |
|  2  |  7  |   21    |     18       |   18 |      1                          |
|  2  |  8  |   28    |     24       |   24 |      1                          |
|  2  |  9  |   36    |     32       |   32 |      1                          |
|  2  |  10  |   45    |     40       |   40 |      1                          |
|  2  |  11  |   55    |     50       |   50 |      1                          |
|  2  |  12  |   66    |     60       |   60 |      1                          |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  3  |  4  |  4    |    4       |  4 |     1                      |
|  3  |  5  |  10    |    8       |  8 |     1                      |
|  3  |  6  |  20    |    16       |  15 |     {6, 0}                      |
|  3  |  7  |  35    |    30       |  30 |     1                      |
|  3  |  8  |  56    |    49       |  49 |     3                      |
|  3  |  9  |  84    |    74       |  74 |     1                      |
|  3  |  10  |  120    |    108       |  108 |     1                      |
|  3  |  11  |  165    |    150       |  149 |     {18, 0}                      |
|  3  |  12  |  220    |    201       |  200 |     {28, 2}                      |
|  3  |  13  |  286    |    264       |  263 |     {4, 4}                      |
|  3  |  14  |  364    |    338       |  337 |     {38, 0}                      |
|  3  |  15  |  455    |    424       |  424 |     1                      |
|  3  |  16  |  560    |    525       |  524 |     {20, 10}                      |
|  3  |  17  |  680    |    640       |  639 |     {34, 4}                      |
|  3  |  18  |  816    |    770       |  769 |     {50, 7}                      |
|  3  |  19  |  969    |    918       |  917 |     {0, 4}                      |
|  3  |  20  |  1140    |    1083       |  1082 |     {92, 4}                      |
|  3  |  21  |  1330    |    1266       |  1266 |     1                      |
|  3  |  22  |  1540    |    1470       |  1469 |     {[18](https://github.com/jkboyce/jprime/blob/main/runs/3_22_s1), ?} |
|  3  |  23  |  1771    |    1694       |  1693 |     {[56](https://github.com/jkboyce/jprime/blob/main/runs/3_23_s1), ?} |
|  3  |  24  |  2024    |    1939       |  1938 |     {[44](https://github.com/jkboyce/jprime/blob/main/runs/3_24_s1), ?} |
|  3  |  25  |  2300    |    2208       |  < 2208 |   {[?](https://github.com/jkboyce/jprime/blob/main/runs/3_25_s1), ?} |
|  3  |  26  |  2600    |    2500       |  2499 |     {[180](https://github.com/jkboyce/jprime/blob/main/runs/3_26_s1), ?} |
|  3  |  27  |  2925    |    2816       |  2816 |     [1](https://github.com/jkboyce/jprime/blob/main/runs/3_27_s0) |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  4  |  5  |  5    |    5       |  5 |     1                      |
|  4  |  6  |  15    |    12       |  12 |     1                      |
|  4  |  7  |  35    |    30       |  30 |     1                      |
|  4  |  8  |  70    |    60       |  58 |     {26, 18}                      |
|  4  |  9  |  126    |    112       |  112 |     1                      |
|  4  |  10  |  210    |    188       |  188 |     9                      |
|  4  |  11  |  330    |    300       |  300 |     144                      |
|  4  |  12  |  495    |    452       |  452 |     45                      |
|  4  |  13  |  715    |    660       |  660 |     >= 245                      |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  5  |  6  |  6    |    6       |  6 |     1                      |
|  5  |  7  |  21    |    18       |  18 |     1                      |
|  5  |  8  |  56    |    49       |  49 |     3                      |
|  5  |  9  |  126    |    112       |  112 |     5                      |
|  5  |  10  |  252    |    226       |  225 |     {752, 86}                      |
|  5  |  11  |  462    |    420       |  420 |     59346                      |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  6  |  7  |  7    |    7       |  7 |     1                      |
|  6  |  8  |  28    |    24       |  7 |     1                      |
|  6  |  9  |  84    |    74       |  7 |     1                      |
|  6  |  10  |  210    |    188       |  7 |     9                      |
|  6  |  11  |  462    |    420       |  7 |     59346                      |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  7  |  8  |  8    |    8       |  8 |     1                      |
|  7  |  9  |  36    |    32       |  32 |     1                      |
|  7  |  10  |  120    |    108       |  108 |     1                      |
|  7  |  11  |  330    |    300       |  300 |     144                      |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  8  |  9  |  9    |    9       |  9 |     1                      |
|  8  |  10  |  45    |    40       |  40 |     1                      |
|  8  |  11  |  165    |    150       |  149 |     {18, 0}                      |
|  8  |  12  |  495    |    452       |  452 |     45                      |
| --- | --- | ------ | ----------- | --- | ------------------------------- |
|  9  |  10  |  10    |    10       |  10 |     1                      |
|  9  |  11  |  55    |    50       |  50 |     1                      |
|  9  |  12  |  220    |    201       |  200 |     {28, 2}                      |
|  9  |  13  |  715    |    660       |  660 |     >= 245                      |

# Technical details

(Info about parallel DFS goes here.)
