#!/usr/bin/env python3

import sys
import numpy as np
from utils import readmol
import qm_config


def main():
  xyzfile = sys.argv[1]
  state_id = int(sys.argv[2])-1
  sign = float(sys.argv[3])

  mol   = readmol(xyzfile, qm_config.basis)
  coeff = np.load(xyzfile+'.mo.npy')
  X     = np.load(xyzfile+'.X.npy')
  occ  = mol.nelectron//2
  x_mo = X[state_id]
  x_ao = coeff[:,:occ] @ x_mo @ coeff[:,occ:].T

  with mol.with_common_orig((0,0,0)):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
  el_dip = np.einsum('xij,ji->x', ao_dip, x_ao)
  if sign < 0:
    el_dip *= -1.0

  with np.printoptions(formatter={'float': '{: .5f}'.format}):
    print(xyzfile, el_dip)


if __name__ == "__main__":
  main()

