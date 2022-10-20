#!/usr/bin/env python3

import sys
import numpy as np
import qstack
import qm_config

def main():
    xyzfile = sys.argv[1]
    mol = qstack.compound.xyz_to_mol(xyzfile, qm_config.basis)
    try:
        dm0 = np.loadtxt(xyzfile+'.dm.dat')
        (mf, dm) = qstack.fields.dm.get_converged_mf(mol, qm_config.func, dm0)
    except:
        (mf, dm) = qstack.fields.dm.get_converged_mf(mol, qm_config.func)
        coulomb = np.array([qstack.fields.decomposition.get_self_repulsion(mf, dm)])
        np.savetxt(xyzfile+".coulomb.dat", coulomb)
        np.savetxt(xyzfile+'.dm.dat', dm)
        np.save(xyzfile+'.mo.npy', mf.mo_coeff)
    td = qstack.fields.excited.get_cis(mf, qm_config.nstates)
    X  = qstack.fields.excited.get_cis_tdm(td)
    np.save(xyzfile+'.X.npy', X)

if __name__ == "__main__":
    main()
