#
# makefile
#
# Simple makefile to build `jprime`, for use with GNU Make on Unix systems.
# Run `make` to build the executable, `make clean` to clean the intermediate
# object files in the `build` directory.
#
# Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
#
# This file is distributed under the MIT License.
#

CC = g++
CFLAGS = -Wall -Wextra -std=c++20 -O3
SDIR = src
ODIR = build
OBJ = jprime.o jprime_tests.o Graph.o State.o Worker.o GenLoopsRecursive.o \
      GenLoopsIterative.o Coordinator.o WorkAssignment.o SearchConfig.o \
      SearchContext.o Pattern.o
DEP = Graph.h State.h Worker.h Coordinator.h WorkAssignment.h SearchConfig.h \
      SearchContext.h Pattern.h

_OBJ = $(patsubst %,$(ODIR)/%,$(OBJ))
_DEP = $(patsubst %,$(SDIR)/%,$(DEP))

jprime: $(_OBJ)
	$(CC) -o jprime $(_OBJ) $(CFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cc $(_DEP) | builddir
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: builddir clean

builddir:
	mkdir -p $(ODIR)

clean:
	rm -rf $(ODIR)
