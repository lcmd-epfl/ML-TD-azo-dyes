#!/usr/bin/env python3

import numpy as np
import qstack
import qm_config

dms = qstack.fields.hirshfeld.spherical_atoms(qm_config.elements, qm_config.basis_at)
for q in dms:
    np.save(qm_config.path_sph + q + '.npy', dms[q])
