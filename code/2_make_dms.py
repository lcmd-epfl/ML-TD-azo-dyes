#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import scf
from utils import readmol,compute_j
import qm_config


def mo2ao(mat, coeff):
  return np.dot(coeff, np.dot(mat, coeff.T))


def main():
  xyzfile = sys.argv[1]
  state_id = []
  for i in range(2, len(sys.argv)):
    state_id.append(int(sys.argv[i])-1)

  mol   = readmol(xyzfile, qm_config.basis)
  coeff = np.load(xyzfile+'.mo.npy')
  X     = np.load(xyzfile+'.X.npy')
  occ = mol.nelectron//2

  for i in range(len(state_id)):
    x = X[state_id[i]]
    hole_mo = np.dot(x, x.T)
    part_mo = np.dot(x.T, x)
    hole_ao = mo2ao(hole_mo, coeff[:,:occ])
    part_ao = mo2ao(part_mo, coeff[:,occ:])
    hole_coulomb = np.array([ compute_j(mol, hole_ao) ])
    part_coulomb = np.array([ compute_j(mol, part_ao) ])
    np.savetxt(xyzfile+".st"+str(i+1)+"_dm_hole.dat", hole_ao)
    np.savetxt(xyzfile+".st"+str(i+1)+"_dm_part.dat", part_ao)
    np.savetxt(xyzfile+".st"+str(i+1)+"_coulomb_hole.dat", hole_coulomb)
    np.savetxt(xyzfile+".st"+str(i+1)+"_coulomb_part.dat", part_coulomb)


if __name__ == "__main__":
  main()

