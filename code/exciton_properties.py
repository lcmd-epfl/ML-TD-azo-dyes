#!/usr/bin/env python3

import sys
import numpy as np
from utils import readmol,r2_c
import qm_config

def main():
  xyzfile = sys.argv[1]
  hole    = np.loadtxt(sys.argv[2])
  part    = np.loadtxt(sys.argv[3])
  mol     = readmol(xyzfile, qm_config.basis2)

  hole_N, hole_r, hole_r2 = r2_c(hole, mol)
  part_N, part_r, part_r2 = r2_c(part, mol)

  dist = np.linalg.norm(hole_r-part_r)
  hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
  part_extent = np.sqrt(part_r2-part_r@part_r)

  print(xyzfile, "\tdist = %e  hole_size = %e  part_size = %e" %(dist, hole_extent, part_extent))


if __name__ == "__main__":
  main()

