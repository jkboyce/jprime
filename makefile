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
CFLAGS = -Wall -Wextra -std=c++20 -O3 -Isrc
SDIR = src
ODIR = build
OBJ = jprime.o jprime_tests.o Graph.o State.o Worker.o GenLoopsRecursive.o \
	  GenLoopsIterative.o Coordinator.o CoordinatorCPU.o WorkAssignment.o \
	  SearchConfig.o SearchContext.o Pattern.o
DEP = Coordinator.h CoordinatorCPU.h Graph.h Messages.h Pattern.h \
	  SearchConfig.h SearchContext.h State.h WorkAssignment.h WorkCell.h \
	  Worker.h

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
# This requires the `nvcc` compiler, part of the CUDA Toolkit from Nvidia.

CFLAGS_CUDA = -Wall -Wextra -std=c++20 -O3 -I/usr/local/cuda/include \
			  -Isrc -Isrc/cuda
_OBJ_CUDA = $(patsubst %,$(ODIR)/cuda/%,$(OBJ))
NVCCFLAGS = -std=c++20 -O3 -lineinfo -Xcudafe --diag_suppress=68 \
			-gencode arch=compute_60,code=compute_60 \
			-gencode arch=compute_89,code=sm_89 \
			-Wno-deprecated-gpu-targets -Isrc -Isrc/cuda

cuda: $(SDIR)/cuda/CudaKernels.cu $(_OBJ_CUDA) $(ODIR)/cuda/CoordinatorCUDA.o
	nvcc $(NVCCFLAGS) -o jprime src/cuda/CudaKernels.cu $(_OBJ_CUDA) \
	  $(ODIR)/cuda/CoordinatorCUDA.o

$(ODIR)/cuda/CoordinatorCUDA.o: $(SDIR)/cuda/CoordinatorCUDA.cc $(_DEP) \
  src/cuda/CoordinatorCUDA.h | builddir_cuda
	$(CC) -DCUDA_ENABLED -c -o $@ $< $(CFLAGS_CUDA)

$(ODIR)/cuda/%.o: $(SDIR)/%.cc $(_DEP) src/cuda/CoordinatorCUDA.h \
  | builddir_cuda
	$(CC) -DCUDA_ENABLED -c -o $@ $< $(CFLAGS_CUDA)

builddir_cuda:
	mkdir -p $(ODIR)/cuda
