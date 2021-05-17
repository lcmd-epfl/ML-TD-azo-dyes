#!/usr/bin/env python3

import sys
import numpy as np
from utils import readmol
import qm_config


def main():
  xyzfile = sys.argv[1]
  hole_D  = np.loadtxt(sys.argv[2])
  part_D  = np.loadtxt(sys.argv[3])
  mol     = readmol(xyzfile, qm_config.basis)

  with mol.with_common_orig((0,0,0)):
    ao_r = mol.intor_symmetric('int1e_r', comp=3)
  ao_r2 = mol.intor_symmetric('int1e_r2')
  ao_u = mol.intor_symmetric('int1e_ovlp')

  hole_r = np.einsum('xij,ji->x', ao_r, hole_D)
  part_r = np.einsum('xij,ji->x', ao_r, part_D)
  hole_r2 = np.einsum('ij,ji', ao_r2, hole_D)
  part_r2 = np.einsum('ij,ji', ao_r2, part_D)

  dist = np.linalg.norm(hole_r-part_r)
  hole_extent = np.sqrt(hole_r2-hole_r@hole_r)
  part_extent = np.sqrt(part_r2-part_r@part_r)

  print(xyzfile, "\tdist = %e  hole_size = %e  part_size = %e" %(dist, hole_extent, part_extent))


if __name__ == "__main__":
  main()
