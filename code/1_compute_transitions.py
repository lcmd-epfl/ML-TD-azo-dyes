#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import scf,tdscf
from utils import readmol
import qm_config

def do_scf(mol, func, dm0=None):
  mf = scf.RKS(mol)
  mf.xc = func
  mf.kernel(dm0=dm0)
  dm = mf.make_rdm1()
  return (mf, dm)

def do_ci(mf, nstates):
  td = mf.TDA()
  td.nstates = nstates
  td.verbose = 5
  e, coeffs = td.kernel()
  td.analyze()
  return td

def compute_j(mf, dm):
  j,k = mf.get_jk()
  return np.einsum('ij,ij',j,dm)


def main():
  xyzfile = sys.argv[1]
  mol = readmol(xyzfile, qm_config.basis)
  try:
    dm0 = np.loadtxt(xyzfile+'.dm.dat')
    (mf, dm) = do_scf(mol, qm_config.func, dm0)
  except:
    (mf, dm) = do_scf(mol, qm_config.func)
    coulomb = np.array([compute_j(mf, dm)])
    np.savetxt(xyzfile+".coulomb.dat", coulomb)
    np.savetxt(xyzfile+'.dm.dat', dm)
    np.save(xyzfile+'.mo.npy', mf.mo_coeff)
  td = do_ci(mf, qm_config.nstates)
  X = np.sqrt(2.0) * np.array([ xy[0] for xy in td.xy ])
  np.save(xyzfile+'.X.npy', X)


if __name__ == "__main__":
  main()

