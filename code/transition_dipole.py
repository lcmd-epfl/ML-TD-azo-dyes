#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config

def main():
    xyzfile = sys.argv[1]
    c       = np.loadtxt(sys.argv[2])
    mol     = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis2)
    el_dip  = qstack.fields.moments.first(mol, c)
    with np.printoptions(formatter={'float': '{: .5f}'.format}):
        print(xyzfile, el_dip)

if __name__ == "__main__":
  main()
