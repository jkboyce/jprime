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
CFLAGS = -Wall -Wextra -Werror -std=c++20 -O3 -Isrc -Isrc/cpu
SDIR = src
ODIR = build
OBJ = jprime.o jprime_tests.o Coordinator.o Graph.o Pattern.o SearchConfig.o \
    SearchContext.o State.o WorkAssignment.o cpu/CoordinatorCPU.o cpu/Worker.o \
    cpu/GenLoopsRecursive.o cpu/GenLoopsIterative.o

DEP = Coordinator.h Graph.h Pattern.h SearchConfig.h SearchContext.h \
	  State.h WorkAssignment.h WorkSpace.h \
	  cpu/CoordinatorCPU.h cpu/Messages.h cpu/WorkCell.h cpu/Worker.h

_OBJ = $(patsubst %,$(ODIR)/build_cpu/%,$(OBJ))
_DEP = $(patsubst %,$(SDIR)/%,$(DEP))

jprime: $(_OBJ)
	$(CC) -o jprime $(_OBJ) $(CFLAGS)

$(ODIR)/build_cpu/%.o: $(SDIR)/%.cc $(_DEP) | builddir
	$(CC) -c -o $@ $< $(CFLAGS)

builddir:
	mkdir -p $(ODIR)/build_cpu/cpu

.PHONY: builddir clean

clean:
	rm -rf $(ODIR)/build_cpu
	rm -rf $(ODIR)/build_cuda

# Optional support for CUDA target. Build with `make cuda`.
#
# This requires the `nvcc` compiler, part of the CUDA Toolkit from Nvidia.

CFLAGS_CUDA = -Wall -Wextra -Werror -std=c++20 -O3 -I/usr/local/cuda/include \
    -Isrc -Isrc/cpu -Isrc/cuda
_OBJ_CUDA = $(patsubst %,$(ODIR)/build_cuda/%,$(OBJ))
NVCCFLAGS = -std=c++20 -O3 -lineinfo -Xcudafe --diag_suppress=68 \
    -gencode arch=compute_60,code=compute_60 \
    -gencode arch=compute_89,code=sm_89 \
    -Wno-deprecated-gpu-targets \
    -I/usr/local/cuda/include -Isrc -Isrc/cpu -Isrc/cuda

cuda: $(SDIR)/cuda/CudaKernels.cu $(_OBJ_CUDA) \
    $(ODIR)/build_cuda/cuda/CoordinatorCUDA.o
	nvcc $(NVCCFLAGS) -o jprime src/cuda/CudaKernels.cu $(_OBJ_CUDA) \
	    $(ODIR)/build_cuda/cuda/CoordinatorCUDA.o

$(ODIR)/build_cuda/%.o: $(SDIR)/%.cc $(_DEP) src/cuda/CoordinatorCUDA.h \
    | builddir_cuda
	$(CC) -DCUDA_ENABLED -c -o $@ $< $(CFLAGS_CUDA)

builddir_cuda:
	mkdir -p $(ODIR)/build_cuda/cpu $(ODIR)/build_cuda/cuda
