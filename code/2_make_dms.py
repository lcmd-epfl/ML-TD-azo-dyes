#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config

def main():
    xyzfile = sys.argv[1]
    state_id = [int(arg)-1 for arg in sys.argv[2:]]

    mol   = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis)
    coeff = np.load(xyzfile+'.mo.npy')
    X     = np.load(xyzfile+'.X.npy')

    for i, state in enumerate(state_id):
        hole_ao, part_ao = qstack.fields.excited.get_holepart(mol, X[state], coeff)
        hole_coulomb = np.array([qstack.fields.decomposition.get_self_repulsion(mol, hole_ao)])
        part_coulomb = np.array([qstack.fields.decomposition.get_self_repulsion(mol, part_ao)])
        np.savetxt(xyzfile+".st"+str(i+1)+"_dm_hole.dat", hole_ao)
        np.savetxt(xyzfile+".st"+str(i+1)+"_dm_part.dat", part_ao)
        np.savetxt(xyzfile+".st"+str(i+1)+"_coulomb_hole.dat", hole_coulomb)
        np.savetxt(xyzfile+".st"+str(i+1)+"_coulomb_part.dat", part_coulomb)


if __name__ == "__main__":
    main()
