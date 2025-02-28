#
# makefile
#
# Makefile to build `jprime`, for use with GNU Make on Unix-like systems.
#
# Run `make` to build the executable, `make clean` to clean the intermediate
# object files in the `build` directory.
#
# Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
#
# This file is distributed under the MIT License.
#

CC = g++
CFLAGS = -Wall -Wextra -std=c++20 -O3
SDIR = src
ODIR = build
OBJ = jprime.o jprime_tests.o Graph.o State.o Worker.o GenLoopsRecursive.o \
      GenLoopsIterative.o Coordinator.o CoordinatorCPU.o WorkAssignment.o \
			SearchConfig.o SearchContext.o Pattern.o
DEP = Graph.h State.h Worker.h Coordinator.h WorkAssignment.h SearchConfig.h \
      SearchContext.h Pattern.h

_OBJ = $(patsubst %,$(ODIR)/%,$(OBJ))
_DEP = $(patsubst %,$(SDIR)/%,$(DEP))

jprime: $(_OBJ)
	$(CC) -o jprime $(_OBJ) $(CFLAGS)

$(ODIR)/%.o: $(SDIR)/%.cc $(_DEP) | builddir
	$(CC) -c -o $@ $< $(CFLAGS)

builddir:
	mkdir -p $(ODIR)

.PHONY: builddir clean

clean:
	rm -rf $(ODIR)

# Optional support for CUDA target. Build with `make cuda`.
#
# This requires the `nvcc` compiler, part of the CUDA Toolkit from NVIDIA.

CFLAGS_CUDA = -Wall -Wextra -std=c++20 -O3 -I/usr/local/cuda/include
_OBJ_CUDA = $(patsubst %,$(ODIR)/cuda/%,$(OBJ))
NVCCFLAGS = -std=c++20 -O3 -Xcudafe --diag_suppress=68 \
            -gencode arch=compute_60,code=compute_60 \
            -gencode arch=compute_89,code=sm_89 \
            -Wno-deprecated-gpu-targets

cuda: $(SDIR)/CoordinatorCUDA.cu $(_OBJ_CUDA)
	nvcc $(NVCCFLAGS) -o jprime src/CoordinatorCUDA.cu $(_OBJ_CUDA)

$(ODIR)/cuda/%.o: $(SDIR)/%.cc $(_DEP) | builddir_cuda
	$(CC) -DCUDA_ENABLED -c -o $@ $< $(CFLAGS_CUDA)

builddir_cuda:
	mkdir -p $(ODIR)/cuda
