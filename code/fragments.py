#!/usr/bin/env python3

import sys
import numpy as np
from pyscf import gto,dft,data
from utils import readmol
import qm_config


def readfrag(frag_file):
  fragments = []
  with open(frag_file, 'r') as f:
    for line in f.readlines():
      fragments.append(np.fromstring(line, dtype=int, sep=' ')-1)
  return fragments

def atom_contributions(c, ao, tot_weights):
  tmp = np.einsum('i,pi->p', c, ao)
  return np.einsum('p,ap->a', tmp, tot_weights)

def hirshfeld_weights(mol_full, grid_coord, atm_dm):
    # atomic densities
    grid_n = len(grid_coord)
    rho_atm = np.zeros((mol_full.natm, grid_n), dtype=float)
    for i in range(mol_full.natm):
      q = mol_full._atom[i][0]
      mol_atm    = gto.M(atom=mol_full._atom[i:i+1], basis=qm_config.basis_at, spin=data.elements.ELEMENTS_PROTON[q]%2, unit='Bohr')
      ao_atm     = dft.numint.eval_ao(mol_atm, grid_coord)
      rho_atm[i] = dft.numint.eval_rho(mol_atm, ao_atm, atm_dm[q])

    # get hirshfeld weights
    rho = rho_atm.sum(axis=0)
    idx = np.where(rho > 0)[0]
    h_weights = np.zeros_like(rho_atm)
    for i in range(mol_full.natm):
      h_weights[i,idx] = rho_atm[i,idx]/rho[idx]

    # get dominant hirshfeld weights
    for point in range(grid_n):
      i = np.argmax(h_weights[:,point])
      h_weights[:,point] = np.zeros(mol_full.natm)
      h_weights[i,point] = 1.0
    return h_weights


def main():
  xyz_file  = sys.argv[1]
  frag_file = sys.argv[2]
  hole_file = sys.argv[3]
  part_file = sys.argv[4]

  # load molecule
  mol = readmol(xyz_file, qm_config.basis2)
  # load fragment definition
  fragments = readfrag(frag_file)
  # load fields
  hole = np.loadtxt(hole_file)
  part = np.loadtxt(part_file)

  # construct integration grid
  g = dft.gen_grid.Grids(mol)
  g.level = qm_config.grid_level
  g.build()

  # load atomic DMs
  dm_atoms = {}
  for q in qm_config.elements:
    dm_atoms[q] = np.load(qm_config.path_sph+q+'.npy')

  # compute atomic weights
  h_weights   = hirshfeld_weights(mol, g.coords, dm_atoms)
  tot_weights = np.einsum('p,ap->ap', g.weights, h_weights)

  # atom partitioning
  ao = dft.numint.eval_ao(mol, g.coords)
  omega_hole_atom = atom_contributions(hole, ao, tot_weights)
  omega_part_atom = atom_contributions(part, ao, tot_weights)

  # fragment partitioning
  omega_hole_frag = np.zeros(len(fragments))
  omega_part_frag = np.zeros_like(omega_hole_frag)
  for i, k in enumerate(fragments):
    omega_hole_frag[i] = omega_hole_atom[k].sum()
    omega_part_frag[i] = omega_part_atom[k].sum()

  # normalized fragment partitioning
  tot_hole = omega_hole_frag.sum()     # should be close to 1
  tot_part = omega_part_frag.sum()     # should be close to 1
  omega_hole_frag *= 100.0 / tot_hole
  omega_part_frag *= 100.0 / tot_part

  with np.printoptions(formatter={'float': '{:6.3f}'.format}):
    print(xyz_file, 'H', omega_hole_frag, 'P', omega_part_frag)


if __name__ == "__main__":
  main()

