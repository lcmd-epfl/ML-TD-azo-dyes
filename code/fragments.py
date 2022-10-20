#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config


def main():
    xyz_file  = sys.argv[1]
    frag_file = sys.argv[2]
    hole_file = sys.argv[3]
    part_file = sys.argv[4]

    mol = qstack.compound.xyz_to_mol(xyz_file, qm_config.basis2)
    fragments = qstack.compound.fragments_read(frag_file)
    hole = np.loadtxt(hole_file)
    part = np.loadtxt(part_file)
    dm_atoms = {q: np.load(qm_config.path_sph+q+'.npy') for q in qm_config.elements}

    omega_hole_atom, omega_part_atom = qstack.fields.hirshfeld.hirshfeld_charges(mol, [hole, part], dm_atoms, qm_config.basis_at, dominant=True, occupations=True, grid_level=qm_config.grid_level)
    omega_hole_frag, omega_part_frag = qstack.compound.fragment_partitioning(fragments, [omega_hole_atom, omega_part_atom], normalize=True)

    with np.printoptions(formatter={'float': '{:6.3f}'.format}):
        print(xyz_file, 'H', omega_hole_frag, 'P', omega_part_frag)


if __name__ == "__main__":
    main()
