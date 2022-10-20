#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config

def main():
    xyzfile = sys.argv[1]
    hole    = np.loadtxt(sys.argv[2])
    part    = np.loadtxt(sys.argv[3])
    mol     = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis2)
    dist, hole_extent, part_extent = qstack.fields.excited.exciton_properties(mol, hole, part)
    print(xyzfile, "\tdist = %e  hole_size = %e  part_size = %e" %(dist, hole_extent, part_extent))

if __name__ == "__main__":
    main()
