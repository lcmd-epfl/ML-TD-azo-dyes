#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import df
from utils import readmol,number_of_electrons,eri_pqi
import qm_config


def main():
  xyzfile = sys.argv[1]
  mol     = readmol(xyzfile, qm_config.basis)
  auxmol  = df.make_auxmol(mol, qm_config.basis2)

  eri3c = eri_pqi(mol, auxmol)
  eri2c = auxmol.intor('int2c2e_sph')
  np.save(xyzfile+'.jmat.npy', eri2c)

  N = mol.nelectron
  files=(
    ('ground', xyzfile+'.coulomb.dat',          xyzfile+'.dm.dat',          N, xyzfile+'.dm_fit.dat'         ),
    ('hole_1', xyzfile+'.st1_coulomb_hole.dat', xyzfile+'.st1_dm_hole.dat', 1, xyzfile+'.st1_dm_hole_fit.dat'),
    ('part_1', xyzfile+'.st1_coulomb_part.dat', xyzfile+'.st1_dm_part.dat', 1, xyzfile+'.st1_dm_part_fit.dat'),
    ('hole_2', xyzfile+'.st2_coulomb_hole.dat', xyzfile+'.st2_dm_hole.dat', 1, xyzfile+'.st2_dm_hole_fit.dat'),
    ('part_2', xyzfile+'.st2_coulomb_part.dat', xyzfile+'.st2_dm_part.dat', 1, xyzfile+'.st2_dm_part_fit.dat'))

  for i in files:
    (title, file_coul, file_dm, n_correct, file_out) = i
    dm = np.loadtxt(file_dm)
    ej = np.loadtxt(file_coul)
    b = np.einsum('ijp,ij->p', eri3c, dm)
    c = np.linalg.solve(eri2c, b)
    error = ej - np.dot(c,b)
    n = number_of_electrons(c, auxmol)
    print(xyzfile+'\t'+title, "Error: % .3e Eh ( %.1e %%)  Electrons: %.3e ( % .1e )"%( error, error/ej*100.0, n, n-n_correct ))
    np.savetxt(file_out, c)


if __name__ == "__main__":
  main()

