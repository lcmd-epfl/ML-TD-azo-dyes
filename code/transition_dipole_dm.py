#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config

def main():
    xyzfile = sys.argv[1]
    state_id = int(sys.argv[2])-1
    sign = float(sys.argv[3])

    mol   = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis)
    coeff = np.load(xyzfile+'.mo.npy')
    X     = np.load(xyzfile+'.X.npy')

    x_ao   = qstack.fields.excited.get_transition_dm(mol, X[state_id], coeff)
    el_dip = qstack.fields.moments.first(mol, x_ao)
    if sign < 0:
        el_dip *= -1.0

    with np.printoptions(formatter={'float': '{: .5f}'.format}):
        print(xyzfile, el_dip)

if __name__ == "__main__":
  main()
