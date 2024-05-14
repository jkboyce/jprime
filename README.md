# jprime
Parallel depth first search (DFS) to find prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state graph for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state graph. Prime patterns then correpond to *cycles* in the graph, i.e. circuits that visit each state in the pattern only once.

`jprime` searches the juggling state graph for patterns, using efficiency tricks to speed up the search. The search is done in parallel using a work-stealing scheme to distribute the search across multiple execution threads.

Using `jprime` I have carried out some investigations into prime siteswap patterns, and summarize my findings here.

## Finding the longest patterns in $(N, H)$

The first investigation is to find the longest prime patterns for a given number of objects ($N$) and maximum throwing value ($H$). Since the graph for $N$ objects and maximum throw $H$ is finite in size – with a number of vertices equal to the number of states ($H$ choose $N$) – we know there must exist a longest prime siteswap pattern(s) for that case. (Recall again that states may not be revisited in a prime pattern, so the order of the graph acts as an upper bound on the length of prime patterns.)

The theory behind these longest prime patterns and how to find them is discussed in this 1999 [paper](https://github.com/jkboyce/jprime/blob/main/longest_prime_siteswaps_1999.pdf).

The table below summarizes everything known about the longest prime siteswap patterns. $L$ is the length of the longest prime pattern for the given $(N, H)$, and $L_{bound}$ is the theoretical upper bound on that length.

Table notes:
- When $L < L_{bound}$, this means there are no *complete* prime patterns for that case. (Consult the 1999 paper; in short a complete prime pattern is the maximum length possible, missing exactly one state on each shift cycle. Every complete prime pattern has a superprime inverse, and vice versa.) When there are no complete patterns, the count is listed as "{Type I patterns, Type II patterns}" where Type I patterns are those having an inverse.
- There is an isomorphism between the juggling graphs for $(N, H)$ and $(H-N, H)$. So for example $(5,11)$ and $(6,11)$ have identical results below. A *duality transform* maps a siteswap in $(N,H)$ to its equivalent in $(H-N,H)$: You reverse the throws and subtract each from $H$. E.g., `868671` in $(6,9)$ maps to `823131` in $(3,9)$. Primality is preserved under this transform.
- The table for $N=2$ is truncated; the observed pattern appears to continue.

| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
| --- | --- | ------ | ------ | ------ | -------- |
|  2  |  3  |  3  |  3  |  3  |  1  |
|  2  |  4  |  6  |  4  |  4  |  2  |
|  2  |  5  |  10  |  8  |  8  |  1  |
|  2  |  6  |  15  |  12  |  12  |  1  |
|  2  |  7  |  21  |  18  |  18  |  1  |
|  2  |  8  |  28  |  24  |  24  |  1  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  3  |  4  |  4  |  4  |  4  |  1  |
|  3  |  5  |  10  |  8  |  8  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_5_8)  |
|  3  |  6  |  20  |  16  |  15  |  [{6, 0}](https://github.com/jkboyce/jprime/blob/main/runs/3_6_15)  |
|  3  |  7  |  35  |  30  |  30  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_7_30)  |
|  3  |  8  |  56  |  49  |  49  |  [3](https://github.com/jkboyce/jprime/blob/main/runs/3_8_49)  |
|  3  |  9  |  84  |  74  |  74  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_9_74)  |
|  3  |  10  |  120  |  108  |  108  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_10_108)  |
|  3  |  11  |  165  |  150  |  149  |  [{18, 0}](https://github.com/jkboyce/jprime/blob/main/runs/3_11_149)  |
|  3  |  12  |  220  |  201  |  200  |  [{28, 2}](https://github.com/jkboyce/jprime/blob/main/runs/3_12_200)  |
|  3  |  13  |  286  |  264  |  263  |  [{4, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_13_263)  |
|  3  |  14  |  364  |  338  |  337  |  [{38, 0}](https://github.com/jkboyce/jprime/blob/main/runs/3_14_337)  |
|  3  |  15  |  455  |  424  |  424  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_15_s0)  |
|  3  |  16  |  560  |  525  |  524  |  [{20, 10}](https://github.com/jkboyce/jprime/blob/main/runs/3_16_524)  |
|  3  |  17  |  680  |  640  |  639  |  [{34, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_17_639)  |
|  3  |  18  |  816  |  770  |  769  |  [{50, 7}](https://github.com/jkboyce/jprime/blob/main/runs/3_18_769)  |
|  3  |  19  |  969  |  918  |  917  |  [{0, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_19_917)  |
|  3  |  20  |  1140  |  1083  |  1082  |  [{92, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_20_1082)  |
|  3  |  21  |  1330  |  1266  |  1266  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_21_s0)  |
|  3  |  22  |  1540  |  1470  |  1469  |  [{18, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_22_1469)  |
|  3  |  23  |  1771  |  1694  |  1693  |  [{56, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_23_1693)  |
|  3  |  24  |  2024  |  1939  |  1938  |  [{44, 3}](https://github.com/jkboyce/jprime/blob/main/runs/3_24_1938)  |
|  3  |  25  |  2300  |  2208  |  2207  |  [{0, 4}](https://github.com/jkboyce/jprime/blob/main/runs/3_25_2207)  |
|  3  |  26  |  2600  |  2500  |  2499  |  {[180](https://github.com/jkboyce/jprime/blob/main/runs/3_26_s1_g), ?}  |
|  3  |  27  |  2925  |  2816  |  2816  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/3_27_s0_g)  |
|  3  |  28  |  3276  |  3159  |  [< 3159](https://github.com/jkboyce/jprime/blob/main/runs/3_28_s0_g)  |  {?, ?}  |
|  3  |  29  |  3654  |  3528  |  [< 3528](https://github.com/jkboyce/jprime/blob/main/runs/3_29_s0)  |  {?, ?}  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  4  |  5  |  5  |  5  |  5  |  1  |
|  4  |  6  |  15  |  12  |  12  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/4_6_12)  |
|  4  |  7  |  35  |  30  |  30  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/4_7_30)  |
|  4  |  8  |  70  |  60  |  58  |  [{28, 16}](https://github.com/jkboyce/jprime/blob/main/runs/4_8_58)  |
|  4  |  9  |  126  |  112  |  112  |  [5](https://github.com/jkboyce/jprime/blob/main/runs/4_9_112)  |
|  4  |  10  |  210  |  188  |  188  |  [9](https://github.com/jkboyce/jprime/blob/main/runs/4_10_188)  |
|  4  |  11  |  330  |  300  |  300  |  [144](https://github.com/jkboyce/jprime/blob/main/runs/4_11_300)  |
|  4  |  12  |  495  |  452  |  452  |  [45](https://github.com/jkboyce/jprime/blob/main/runs/4_12_s0)  |
|  4  |  13  |  715  |  660  |  660  |  [16317](https://github.com/jkboyce/jprime/blob/main/runs/4_13_s0_g)  |
|  4  |  14  |  1001  |  928  |  928  |  [>= 2054](https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/4_14_s0_g)  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  5  |  6  |  6  |  6  |  6  |  1  |
|  5  |  7  |  21  |  18  |  18  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/5_7_18)  |
|  5  |  8  |  56  |  49  |  49  |  [3](https://github.com/jkboyce/jprime/blob/main/runs/5_8_49)  |
|  5  |  9  |  126  |  112  |  112  |  [5](https://github.com/jkboyce/jprime/blob/main/runs/5_9_112)  |
|  5  |  10  |  252  |  226  |  225  |  [{752, 86}](https://github.com/jkboyce/jprime/blob/main/runs/5_10_225)  |
|  5  |  11  |  462  |  420  |  420  |  [59346](https://github.com/jkboyce/jprime/blob/main/runs/5_11_s0)  |
|  5  |  12  |  792  |  726  |  726  |  [>= 81870](https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/5_12_s0_g)  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  6  |  7  |  7  |  7  |  7  |  1  |
|  6  |  8  |  28  |  24  |  24  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/6_8_24)  |
|  6  |  9  |  84  |  74  |  74  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/6_9_74)  |
|  6  |  10  |  210  |  188  |  188  |  [9](https://github.com/jkboyce/jprime/blob/main/runs/6_10_s0)  |
|  6  |  11  |  462  |  420  |  420  |  [59346](https://github.com/jkboyce/jprime/blob/main/runs/5_11_s0)  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  7  |  8  |  8  |  8  |  8  |  1  |
|  7  |  9  |  36  |  32  |  32  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/7_9_32)  |
|  7  |  10  |  120  |  108  |  108  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/7_10_s0)  |
|  7  |  11  |  330  |  300  |  300  |  [144](https://github.com/jkboyce/jprime/blob/main/runs/7_11_s0)  |
|  7  |  12  |  792  |  726  |  726  |  [>= 81870](https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/5_12_s0_g)  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  8  |  9  |  9  |  9  |  9  |  1  |
|  8  |  10  |  45  |  40  |  40  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/8_10_40)  |
|  8  |  11  |  165  |  150  |  149  |  [{18, 0}](https://github.com/jkboyce/jprime/blob/main/runs/8_11_149)  |
|  8  |  12  |  495  |  452  |  452  |  [45](https://github.com/jkboyce/jprime/blob/main/runs/8_12_s0)  |
| --- | --- | ------ | ------ | ------ | -------- |
| $N$ | $H$ | States | $L_{bound}$ | $L$ | Patterns |
|  9  |  10  |  10  |  10  |  10  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/9_10_10)  |
|  9  |  11  |  55  |  50  |  50  |  [1](https://github.com/jkboyce/jprime/blob/main/runs/9_11_50)  |
|  9  |  12  |  220  |  201  |  200  |  [{28, 2}](https://github.com/jkboyce/jprime/blob/main/runs/9_12_200)  |
|  9  |  13  |  715  |  660  |  660  |  [16317](https://github.com/jkboyce/jprime/blob/main/runs/4_13_s0)  |

## Counting prime patterns

As a second investigation, we use `jprime` to count the total number of prime patterns for a given number of objects ($N$) and pattern period ($L$).

The counting problem has been solved analytically for siteswap patterns in general. In [_Juggling Drops and Descents_](https://mathweb.ucsd.edu/~ronspubs/94_01_juggling.pdf) (Buhler, Eisenbud, Graham, and Wright, 1994) the following formula is derived for the total number of periodic siteswap patterns with $N$ objects and period (length) $L$: $(N+1)^L - N^L$.

Note this formula treats rotated versions of the same pattern as distinct, e.g., `531`, `315`, and `153` are counted separately. Since this formula includes all prime patterns (as well as non-prime ones), and because each prime pattern occurs exactly $L$ times in the count, it follows that an upper bound on the number of prime patterns of length $L$, where we do not treat rotations as distinct, is $P(N,L) <= [(N+1)^L - N^L] / L$.

Counting prime patterns specifically is a more difficult problem than the general case. One set of results comes from [_Counting prime juggling patterns_](https://arxiv.org/abs/1508.05296) (Banaian, Butler, Cox, Davis, Landgraf, and Ponce, 2015). They find an exact formula for $P(N,L)$ for the case $N=2$, and also establish the lower bound $P(N,L) >= N^{L-1}$.

The table below shows exact counts for the number of prime patterns at each period. For the cases $N = 2, 3, 4, 5$ these are OEIS sequences [A260744](https://oeis.org/A260744), [A260745](https://oeis.org/A260745), [A260746](https://oeis.org/A260746), and [A260752](https://oeis.org/A260752) respectively.
<pre>
2 OBJECTS
1, 1
2, 2
3, 5
4, 10
5, 23
6, 48
7, 105
8, 216
9, 467
10, 958
11, 2021
12, 4146
13, 8631
14, 17604
15, 36377
16, 73876
17, 151379
18, 306882
19, 625149
20, 1263294
21, 2563895
22, 5169544
23, 10454105
24, 21046800
25, 42451179
26, 85334982
27, 171799853
28, 344952010
29, 693368423
30, 1391049900
31, 2792734257
32, 5598733260
33, 11230441259
34, 22501784458
35, 45103949933
36, 90335055318
37, 180975948735
38, 362339965776
39, 725616088097
40, 1452416238568

3 OBJECTS
1, 1
2, 3
3, 11
4, 36
5, 127
6, 405
7, 1409
8, 4561
9, 15559
10, 50294
11, 169537
12, 551001
13, 1835073
14, 5947516
15, 19717181
16, 63697526
17, 209422033
18, 676831026
19, 2208923853
20, 7112963260
21, 23127536979
22, 74225466424
23, 239962004807
24, 768695233371
25, 2473092566267
26, 7896286237030
27, 25316008015581
28, 80572339461372

4 OBJECTS
1, 1
2, 4
3, 19
4, 83
5, 391
6, 1663
7, 7739
8, 33812
9, 153575
10, 677901
11, 3075879
12, 13586581
13, 61458267
14, 272367077
15, 1228519987
16, 5456053443
17, 24547516939
18, 109153816505
19, 490067180301
20, 2180061001275
21, 9772927018285

5 OBJECTS
1, 1
2, 5
3, 29
4, 157
5, 901
6, 4822
7, 27447
8, 149393
9, 836527
10, 4610088
11, 25846123
12, 142296551
13, 799268609
14, 4426204933
15, 24808065829
16, 137945151360
17, 773962487261
18, 4310815784117

6 OBJECTS
1, 1
2, 6
3, 41
4, 264
5, 1777
6, 11378
7, 76191
8, 493550
9, 3263843
10, 21373408
11, 141617885
12, 926949128
13, 6157491321
14, 40536148951
15, 268893316363
16, 1777319903383

7 OBJECTS
1, 1
2, 7
3, 55
4, 410
5, 3163
6, 23511
7, 180477
8, 1353935
9, 10297769
10, 77849603
11, 593258483
12, 4486556303
13, 34267633327
14, 260349728028
15, 1987331381633

8 OBJECTS
1, 1
2, 8
3, 71
4, 601
5, 5227
6, 44181
7, 382221
8, 3254240
9, 27976325
10, 239491149
11, 2061545183
12, 17664073336
13, 152326783983
14, 1309746576182

9 OBJECTS
1, 1
2, 9
3, 89
4, 843
5, 8161
6, 77248
7, 743823
8, 7081367
9, 67880511
10, 648866014
11, 6225810713
12, 59574064361
13, 572427402861
</pre>

## Running jprime

After cloning the repository, on a Unix system run `make` to build the `jprime` binary using the included makefile. Run the binary with no arguments to get a help message.

`jprime` has two modes of operation, intended to search for prime patterns in complementary ways:

- Normal mode, the default. This finds patterns by searching the juggling graph directly for cycles.
- Super mode (faster). This searches the juggling graph for _superprime_ patterns, which visit no _shift cycle_ more than once. The significance here is that many of the longest prime patterns have inverses that are superprime, or nearly superprime, for reasons described in the paper. So a quick way to find these patterns is to search for long superprime patterns, and find their inverses. Invoke super mode with `-super <shifts>` on the command line, where `<shifts>` is how many shift throws to allow (e.g., `shifts = 0` corresponds to true superprime patterns). The `-inverse` option prints the inverse of each pattern found, if the inverse exists. The limitation of this method of finding patterns is that it cannot find Type II incomplete patterns as defined in the paper.
