#!/usr/bin/env python3

import numpy as np
from pyscf import gto,scf,data
import qm_config

for q in qm_config.elements:
  mol = gto.M(atom=[[q, [0,0,0]]], spin=data.elements.ELEMENTS_PROTON[q]%2, basis=qm_config.basis_at)
  dm = scf.hf.init_guess_by_atom(mol)
  np.save(qm_config.path_sph+q+'.npy', dm)
