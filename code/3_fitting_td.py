#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import df
from utils import readmol,compute_j,number_of_electrons_vec,eri_pqi
import qm_config


def main():
  xyzfile = sys.argv[1]
  state_id = [int(sys.argv[2])-1, int(sys.argv[3])-1]
  azo_id   = [int(sys.argv[4])-1, int(sys.argv[5])-1]

  mol   = readmol(xyzfile, qm_config.basis)
  coeff = np.load(xyzfile+'.mo.npy')
  X     = np.load(xyzfile+'.X.npy')

  auxmol = df.make_auxmol(mol, qm_config.basis2)
  eri3c  = eri_pqi(mol, auxmol)
  eri2c  = auxmol.intor('int2c2e_sph')

  occ    = mol.nelectron//2
  azo_ao = auxmol.offset_ao_by_atom()[azo_id[0]:azo_id[1]+1,2:]
  q      = number_of_electrons_vec(auxmol)

  for i,state in enumerate(state_id):

    x_mo = X[state]
    x_ao = coeff[:,:occ] @ x_mo @ coeff[:,occ:].T

    coulomb = compute_j(mol, x_ao)
    b = np.einsum('ijp,ij->p', eri3c, x_ao)
    c = np.linalg.solve(eri2c, b)
    error = coulomb - np.dot(c,b)

    cq = c*q
    n  = sum(cq)
    n1 = sum(cq[azo_ao[0,0]:azo_ao[0,1]])
    n2 = sum(cq[azo_ao[1,0]:azo_ao[1,1]])
    if(n1<0.0):
      c = -c

    print(xyzfile, 'state'+str(i+1), "Error: % .3e Eh ( %.1e %%)  Electrons: % .1e total, % .2e left N, % .2e right N"%( error, error/coulomb*100.0, n, n1, n2 ))
    np.savetxt(xyzfile+'.st'+str(i+1)+'_transition_fit.dat', c)


if __name__ == "__main__":
  main()

