//
// CoordinatorCuda.cu
//
// Core graph search routines, implemented as iterative functions that are drop-
// in replacements for recursive versions in GenLoopsRecursive.cc. These
// routines are by far the most performance-critical portions of jprime.
//
// Copyright (C) 1998-2025 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "Coordinator.h"

#include <iostream>
#include <vector>


__global__ void helloCUDA(int* a, int* b, int* c) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}


bool Coordinator::run_cuda() {
  std::vector<int> a, b, c;

  for (int i = 0; i < 100; ++i) {
    a.push_back(i);
    b.push_back(i);
  }
  c.assign(a.size(), 0);

  helloCUDA<<<1, 100>>>(a.data(), b.data(), c.data());
  cudaDeviceSynchronize();

  for (int i = 0; i < c.size(); ++i) {
    std::cout << c[i] << "\n";
  }

  return true;
}
