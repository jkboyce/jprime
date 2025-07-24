# Historical information

This folder contains some early siteswap pattern generators that I wrote, interesting now only for historical
reasons. They show some of the evolution of ideas behind `jprime` and juggling pattern generators in general.

The first juggling pattern generator was written by Don Hatch and Paul Klimek of Santa Cruz California,
sometime between 1981 and 1984. To my knowledge this code has never been made public, but pattern list printouts
dated July 1, 1984 are [shown in Sean Gandini's "Siteswaps" documentary](https://vimeo.com/497788314) from 2006.
Later in the mid-late 1980s
Bengt Magnusson at Caltech wrote a siteswap generator in Fortran, which he rewrote in C in 1991. I have no
information about whether the Cambridge group of siteswap inventors (Mike Day, Colin Wright, Adam Chalcraft)
wrote software in that timeframe, but it seems likely.

The programs here have been lightly edited to compile on a modern ANSI C99 compiler. In other respects they are unchanged.

## `j.c` (Nov 1990)

This was the first program to generate prime juggling patterns, and to generate patterns using a juggling state graph,
which were new ideas at the time.

For reasons I now forget, the program did not generate patterns containing zero throws. Versions immediately
after this fixed that problem. Also this code used a method to store and notate juggling states that isn't as
efficient as what soon emerged. The ideas took a while to gel.

## `j2.c` (Dec 1991)

`j2` was a rewrite of `j.c` and was I believe the first program to generate synchronous juggling patterns
like `(6x,4)(4,6x)`, and multiplexed patterns like `24[54]`. In any case the notation I introduced in `j2` for
synchronous and multiplexed juggling stuck with the juggling community and is in wide use today.

In the mid-1990s, `j2` evolved into the siteswap generator component of JuggleAnim, a Java juggling applet, which
evolved into [Juggling Lab](https://jugglinglab.org) which is still in use today. The siteswap generator in Juggling Lab
remains my best attempt at a fully-featured juggling pattern generator.

The files `j2.txt` and `3person` are for use with `j2`.

## `jdeep.c` (Jun 1998)

When Johannes Waldmann started looking at long prime patterns in 1998, I became interested and began adapting my
pattern-finding software to be as efficient as possible at the task.

`jdeep.c` was the initial result, and it continued to evolve rapidly through subsequent versions.
Eventually `jdeep` became `jprime` so in terms of code lineage this represents v1 of `jprime`.

## Longest prime siteswaps paper (1999)

This was my first writeup of things I'd learned in the course of developing and running `jdeep/jprime`.
Unfortunately, due to a software bug there are a couple of errors in the tabulated results for 4 balls!

Some of the ideas weren't yet mature here. For example the definition of a *superprime* pattern was overly
restrictive and missed the main idea.
