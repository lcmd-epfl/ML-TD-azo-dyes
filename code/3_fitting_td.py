#!/usr/bin/env python3

import sys
import numpy as np
import qstack
from qstack.fields import decomposition
import qm_config

def main():
    xyzfile  = sys.argv[1]
    state_id = [int(sys.argv[2])-1, int(sys.argv[3])-1]
    azo_id   = [int(sys.argv[4])-1, int(sys.argv[5])-1]

    mol   = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis)
    coeff = np.load(xyzfile+'.mo.npy')
    X     = np.load(xyzfile+'.X.npy')

    auxmol = qstack.compound.make_auxmol(mol, qm_config.basis2)
    eri2c, eri3c = decomposition.get_integrals(mol, auxmol)[1:]

    occ    = mol.nelectron//2
    azo_ao = auxmol.offset_ao_by_atom()[azo_id[0]:azo_id[1]+1,2:]
    q      = decomposition.number_of_electrons_deco_vec(auxmol)

    for i, state in enumerate(state_id):
        x_ao = qstack.fields.excited.get_transition_dm(mol, X[state], coeff)
        coulomb = decomposition.get_self_repulsion(mol, x_ao)
        c = decomposition.get_coeff(x_ao, eri2c, eri3c)
        error = decomposition.decomposition_error(coulomb, c, eri2c)

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
