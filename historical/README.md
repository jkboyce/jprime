# Historical information

This folder contains some early juggling pattern generators I wrote, interesting now only for historical
reasons. They show some of the evolution of ideas behind `jprime` and juggling pattern generators in general.
These programs have been lightly edited to compile on a modern ANSI C99 compiler but in other respects they
are unchanged.

The first juggling pattern generator was apparently written by Don Hatch of Santa Cruz California,
sometime between 1981 and 1984. To my knowledge this code has never been made public, but pattern list printouts
dated July 1, 1984 are [shown in Sean Gandini's "Siteswaps" documentary](https://vimeo.com/497788314) from 2006.

At Caltech there were two early pattern generators. The first was written by Joel Hamkins in 1985 or 1986, using
the programming language APL. Code and printouts unfortunately no longer exist. Sometime later in the 1980s
Bengt Magnusson wrote a generator in Fortran, which in 1991 he refined and rewrote in C when he was a graduate
student at UCSB.

I have no information about whether the Cambridge group of siteswap inventors (Mike Day,
Colin Wright, Adam Chalcraft) also wrote software in that timeframe, but it seems likely.

## `j.c` (Nov 1990)

This was the first program to generate prime juggling patterns, and to generate patterns using a juggling state graph,
which were new ideas at the time.

For reasons I now forget, the program did not generate patterns containing zero throws. Versions immediately
after this fixed that problem. Also this code used a method to store and notate juggling states that isn't as
efficient as what soon emerged. The ideas took a while to gel.

## `j2.c` (Dec 1991)

`j2` was a rewrite of `j.c` and was I believe the first program to generate synchronous juggling patterns
like `(6x,4)(4,6x)` and multiplexed patterns like `24[54]`. It also introduced the notation of using letters a, b, c, ... to denote throw values above 9. The notation introduced in `j2` stuck with the juggling community and is in wide use today.

In 1995, `j2` evolved into the siteswap generator component of JuggleAnim, a Java juggling applet, which
evolved into [Juggling Lab](https://jugglinglab.org) which is still in use today. The siteswap generator in Juggling Lab
remains my best attempt at a fully-featured juggling pattern generator.

The files `j2.txt` and `3person` are for use with `j2`.

## `jdeep.c` (Jun 1998)

When Johannes Waldmann started looking at long prime patterns in 1998, I became interested and started adapting my
pattern-finding software to be as efficient as possible at the task.

`jdeep.c` was the initial result, and it continued to evolve rapidly. Eventually `jdeep` became `jprime` so in terms of code lineage this represents version 1 of `jprime`.

## Longest prime siteswaps paper (1999)

This was my first writeup of things I'd learned in the course of developing and running `jdeep/jprime`.
Due to a software bug there were two errors in the tabulated results for 4 balls.

Some of the ideas weren't yet mature here. For example the definition of a *superprime* pattern was overly
restrictive and missed the main idea.
