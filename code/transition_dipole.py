#!/usr/bin/env python3

import sys
import numpy as np
from utils import readmol,r_c
import qm_config

xyzfile = sys.argv[1]
c       = np.loadtxt(sys.argv[2])
mol     = readmol(xyzfile, qm_config.basis2)
el_dip  = r_c(c, mol)
with np.printoptions(formatter={'float': '{: .5f}'.format}):
  print(xyzfile, el_dip)
