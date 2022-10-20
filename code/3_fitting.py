#!/usr/bin/env python3

import sys
import numpy as np
import qstack
from qstack.fields import decomposition
import qm_config


def main():
    xyzfile = sys.argv[1]
    mol     = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis)
    auxmol  = qstack.compound.make_auxmol(mol, qm_config.basis2)

    eri2c, eri3c = decomposition.get_integrals(mol, auxmol)[1:]
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
        c = decomposition.get_coeff(dm, eri2c, eri3c)
        error = decomposition.decomposition_error(ej, c, eri2c)
        n = decomposition.number_of_electrons_deco(auxmol, c)
        print(xyzfile+'\t'+title, "Error: % .3e Eh ( %.1e %%)  Electrons: %.3e ( % .1e )"%( error, error/ej*100.0, n, n-n_correct ))
        np.savetxt(file_out, c)


if __name__ == "__main__":
  main()
