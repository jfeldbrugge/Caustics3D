#!/usr/bin/python

import numpy as np
import sys
import textwrap

A = np.array([[float(i) for i in line.split()]
        for line in open('matrix.txt', 'r')
        if line[0] != '#'])

M = np.linalg.inv(A)

idx = np.indices(M.shape)
fragments = [ f"({i}, {j}, {M[i, j]})"
  for (i, j) in zip(idx[0].flat, idx[1].flat)
  if M[i,j] != 0.0 ]

print(f"const COEFFICIENTS: [(usize, usize, f64);{len(fragments)}] = [")
print("    " + ",\n    ".join(fragments) + "]")

