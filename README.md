# jprime
Parallel depth first search (DFS) to find prime siteswap juggling patterns.

Prime [siteswap](https://en.wikipedia.org/wiki/Siteswap) patterns are those which cannot be expressed as compositions (concatenations) of shorter patterns. This is most easily understood by looking at siteswaps as paths on an associated "state diagram" graph. (The Wikipedia article shows an example state graph for 3 objects and maximum throw value of 5.) Valid siteswaps are closed paths (circuits) in the associated state graph. Prime patterns then correpond to *cycles* in the graph, i.e. circuits that visit each state in the pattern only once.

`jprime` searches the juggling state graph for prime patterns, using efficiency tricks to speed up the search. The search is done in parallel using a work-stealing scheme to distribute the search across multiple execution threads.

## Counting prime patterns by period $n$

Using `jprime` I have carried out some research into prime siteswap patterns. The first is to use `jprime` to count the total number of prime patterns for a given number of objects $b$ and pattern period $n$.

The counting problem has been solved analytically for siteswap patterns in general. In [_Juggling Drops and Descents_](https://mathweb.ucsd.edu/~ronspubs/94_01_juggling.pdf) (Buhler, Eisenbud, Graham, and Wright, 1994) the following formula is derived for the total number of periodic siteswap patterns with $b$ objects and period (length) $n$: $(b+1)^n - b^n$.

Note this formula treats rotated versions of the same pattern as distinct, e.g., `531`, `315`, and `153` are counted separately. Since this formula includes all prime patterns (as well as non-prime ones), and because each prime pattern occurs exactly $n$ times in the count, it follows that an upper bound on the number of prime patterns of length $n$, where we do not treat rotations as distinct, is $P(n,b) <= [(b+1)^n - b^n] / n$.

Counting prime patterns specifically is a more difficult problem than the general case. One set of results comes from [_Counting prime juggling patterns_](https://arxiv.org/abs/1508.05296) (Banaian, Butler, Cox, Davis, Landgraf, and Ponce, 2015). They find an exact formula for $P(n,b)$ for the case $b=2$, and also establish the lower bound $P(n,b) >= b^{n-1}$.

The table below shows exact counts for the number of prime patterns at each period. For the cases $b = 2, 3, 4, 5$ these are OEIS sequences [A260744](https://oeis.org/A260744), [A260745](https://oeis.org/A260745), [A260746](https://oeis.org/A260746), and [A260752](https://oeis.org/A260752) respectively.
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
22, 43467641569472
         
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
19, 24208263855765

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

## Counting prime patterns by height $h$

For a real juggler there is a physical limit to how high a person can throw. In fact many of the patterns found above (prime patterns of a given period $n$) are not readily juggleable because they contain very large throw values.

Here we ask a slightly different question: If we restrict ourselves to siteswap throws no greater than some value $h$, how many prime juggling patterns are there? To answer this, we construct the state graph $(b, h)$ for $b$ objects and maximum throw $h$.

The table below shows exact counts for the number of prime patterns in state graph $(b, h)$, for each value of $h$. We can see that the prime pattern count increases very quickly as $h$ increases, because the number of states ($h$ choose $b$) increases very rapidly.

<pre>
2 OBJECTS
3, 3
4, 8
5, 26
6, 79
7, 337
8, 1398
9, 7848
10, 42749
11, 297887
12, 2009956
13, 16660918
14, 133895979
15, 1284371565
16, 11970256082
17, 130310396228
18, 1381323285721

3 OBJECTS
4, 4
5, 26
6, 349
7, 29693
8, 11906414
9, 30513071763

4 OBJECTS
5, 5
6, 79
7, 29693
8, 1505718865         
</pre>

## Finding the longest prime patterns in $(b, h)$

As a final investigation, we aim to identify the longest prime patterns for a given number of objects $b$ and maximum throw value $h$.

Since the state graph $(b, h)$ for $b$ objects and maximum throw $h$ is finite in size – with a number of vertices equal to the number of states ($h$ choose $b$) – we know there must exist a longest prime siteswap pattern(s) for that case. (Recall that states may not be revisited in a prime pattern, so the order of the graph acts as an upper bound on its length.) The theory behind these longest prime patterns and how to find them is discussed in this 1999 [paper](https://github.com/jkboyce/jprime/blob/main/longest_prime_siteswaps_1999.pdf).

The table below summarizes everything known about the longest prime siteswap patterns. $n$ is the length of the longest prime pattern for the given $(b, h)$, and $n_{bound}$ is the theoretical upper bound on that length.

Table notes:
- When $n < n_{bound}$, this means there are no *complete* prime patterns for that case. (Consult the 1999 paper; in short a complete prime pattern is the maximum length possible, missing exactly one state on each shift cycle. Every complete prime pattern is superprime, and has a superprime inverse.) When $n < n_{bound}$ the pattern count is shown as `{superprime, non-superprime}` patterns. The superprime patterns are faster to find than the non-superprime ones, and in some cases only the former have been tabulated.
- There is an isomorphism between the juggling graphs for $(b, h)$ and $(h-b, h)$. So for example $(5,11)$ and $(6,11)$ have identical results below. A *duality transform* maps a siteswap in $(b, h)$ to its equivalent in $(h-b, h)$: You reverse the throws and subtract each from $h$. E.g., `868671` in $(6,9)$ maps to `823131` in $(3,9)$. Primality is preserved under this transform.
- The table for $b=2$ is truncated; the observed pattern appears to continue.

The current record holder is $(3,28)$ which has prime patterns with 3158 throws. If juggled at a normal pace it would take over 10 minutes to complete a single cycle of this pattern!
<pre>
         2 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
3,    3    (3),          1
4,    4    (4),          2
5,    8    (8),          1
6,    12   (12),         1
7,    18   (18),         1
8,    24   (24),         1
.
.
.

         3 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
4,    4    (4),          1
5,    8    (8),          <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_5_8">1</a>
6,    15   (16),       <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_6_15">{6, 0}</a>
7,    30   (30),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_7_30">1</a>
8,    49   (49),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_8_49">3</a>
9,    74   (74),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_9_74">1</a>
10,   108  (108),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_10_108">1</a>
11,   149  (150),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_11_149">{18, 0}</a>
12,   200  (201),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_12_200">{28, 2}</a>
13,   263  (264),      <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_13_263">{4, 4}</a>
14,   337  (338),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_14_337">{38, 0}</a>
15,   424  (424),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_15_s0">1</a>
16,   524  (525),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_16_524">{20, 10}</a>
17,   639  (640),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_17_639">{34, 4}</a>
18,   769  (770),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_18_769">{50, 7}</a>
19,   917  (918),      <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_19_917">{0, 4}</a>
20,   1082 (1083),    <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_20_1082">{92, 4}</a>
21,   1266 (1266),       <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_21_s0">1</a>
22,   1469 (1470),    <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_22_1469">{18, 4}</a>
23,   1693 (1694),    <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_23_1693">{56, 4}</a>
24,   1938 (1939),    <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_24_1938">{44, 3}</a>
25,   2207 (2208),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_25_2207">{0, 4}</a>
26,   2499 (2500),   <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_26_s1_g">{180, ?}</a>
27,   2816 (2816),       <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_27_s0_g">1</a>
28,   3158 (3159),  <a href="https://github.com/jkboyce/jprime/blob/main/runs (in progress)/3_28_118_s1">{>=14, ?}</a>
29,  <a href="https://github.com/jkboyce/jprime/blob/main/runs/3_29_s0"><3528</a> (3528),     {?, ?}

  
         4 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
5,    5    (5),          1
6,    12   (12),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_6_12">1</a>
7,    30   (30),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_7_30">1</a>
8,    58   (60),      <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_8_58">{28, 16}</a>
9,    112  (112),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_9_112">5</a>
10,   188  (188),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_10_188">9</a>
11,   300  (300),       <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_11_300">144</a>
12,   452  (452),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_12_s0">45</a>
13,   660  (660),      <a href="https://github.com/jkboyce/jprime/blob/main/runs/4_13_55_s0">16317</a>
14,   928  (928),     <a href="https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/4_14_s0_g">>=18911</a>

  
         5 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
6,    6    (6),          1
7,    18   (18),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/5_7_18">1</a>
8,    49   (49),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/5_8_49">3</a>
9,    112  (112),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/5_9_112">5</a>
10,   225  (226),    <a href="https://github.com/jkboyce/jprime/blob/main/runs/5_10_225">{752, 86}</a>
11,   420  (420),      <a href="https://github.com/jkboyce/jprime/blob/main/runs/5_11_s0">59346</a>
12,   726  (726),    <a href="https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/5_12_s0_g">>=309585</a>

  
         6 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
7,    7    (7),          1
8,    24   (24),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/6_8_24">1</a>
9,    74   (74),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/6_9_74">1</a>
10,   188  (188),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/6_10_s0">9</a>
11,   420  (420),      59346
12,   843  (844), {>=(<a href="https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/6_12_79_s0">104</a>+<a href="https://github.com/jkboyce/jprime/blob/main/runs%20(in%20progress)/6_12_81_s1">263</a>), ?} (see note)

  
         7 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
8,    8    (8),          1
9,    32   (32),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/7_9_32">1</a>
10,   108  (108),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/7_10_s0">1</a>
11,   300  (300),       <a href="https://github.com/jkboyce/jprime/blob/main/runs/7_11_s0">144</a>
12,   726  (726),    >=309585

  
         8 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
9,    9    (9),          1
10,   40   (40),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/8_10_40">1</a>
11,   149  (150),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/8_11_149">{18, 0}</a>
12,   452  (452),        <a href="https://github.com/jkboyce/jprime/blob/main/runs/8_12_s0">45</a>

  
         9 OBJECTS
H     N (N_bound)  Pattern count
--------------------------------
10,   10   (10),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/9_10_10">1</a>
11,   50   (50),         <a href="https://github.com/jkboyce/jprime/blob/main/runs/9_11_50">1</a>
12,   200  (201),     <a href="https://github.com/jkboyce/jprime/blob/main/runs/9_12_200">{28, 2}</a>
13,   660  (660),      16317
</pre>

**Note for case $(6,12)$.** Dietrich Kuske proved in 1999 that no graph $(b,2b)$ has a complete prime pattern when $b > 2$, because of the period-2 shift cycle generated by the state `(x-)^b`. In the case $(6,12)$ we do find patterns of length 843, which are one shorter than $n_{bound}$. The superprime ones are the inverses of two different kinds of (shorter) superprime patterns: (a) patterns that miss the `(x-)^b` shift cycle entirely, and use exactly one state on each other shift cycle, and (b) patterns that use state `(x-)^b` and every other shift cycle, including exactly one shift throw on one of these other cycles. The numbers shown are the quantities found of each, respectively. Similarly, the 752 maximal superprime patterns in $(5,10)$ can be found by combining results from `jprime 5 10 25 -super 0 -inverse` (308 patterns) and `jprime 5 10 27 -super 1 -inverse` (444 patterns).

## Running jprime

After cloning the repository, on a Unix system run `make` to build the `jprime` binary using the included makefile. `jprime` requires the following compiler versions or later: GCC 12, Clang 15, MSVC 19.29, Apple Clang 14.0.3. Run the binary with no arguments to get a help message.

`jprime` has two modes of operation, intended to search for prime patterns in complementary ways:

- Normal mode, the default. This finds prime patterns by searching the juggling graph directly for cycles.
- Super mode (faster). This searches the juggling graph for _superprime_ patterns, which visit no _shift cycle_ more than once. Invoke super mode with `-super <shifts>` on the command line, where `<shifts>` is how many shift throws in total to allow. The `-inverse` option prints the inverse of each pattern, if one exists. The significance here is that many of the longest prime patterns are superprime for reasons described in the paper. So a quick way to find these patterns is to search for long superprime patterns having a small number of shift throws, and find their inverses to get the desired long (superprime) patterns. The limitation of this method is that it cannot find non-superprime patterns.
