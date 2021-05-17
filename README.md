# ML-TD-azo-dyes

This code supports the paper
> Sergi Vela, Alberto Fabrizio, Ksenia R. Briling, and Clémence Corminboeuf,<br>
> “Machine-Learning the Transition Density of the Productive Excited States of Azo-dyes”<br>
> [`ChemRxiv:XXXXXX`](https://chemrxiv.org/xxxxxx)<br>

It consists of two parts: the scripts to run *ab initio* computations
and the scripts to compute the excited states descriptors.

## Contents

* [Requirements](#requirements)
* [1. *Ab initio* computations](#1-ab-initio-computations)
  * [1.1. Compute transitions](#11-compute-transitions)
  * [1.2. Extract hole and particle density matrices](#12-extract-hole-and-particle-density-matrices)
  * [1.3. Density fitting](#13-density-fitting)
* [2. Decriptors](#2-decriptors)
  * [Transition dipole moments](#transition-dipole-moments)
  * [Exciton properties](#exciton-properties)
  * [Fragment decomposition](#fragment-decomposition)

## Requirements

* `python >= 3.6`
* `numpy >= 1.16`
* [`pyscf >= 1.6`](https://github.com/pyscf/pyscf)

## 1. *Ab initio* computations

### 1.1. Compute transitions

The script [`1_compute_transitions.py`](code/1_compute_transitions.py)
runs a TDDFT computation withing the TDA approximation
of 8 lowest excited states at the ωB97X-D/def2-SVP level (defined in [`qm_config.py`](code/qm_config.py)).

Usage:
```
$ code/1_compute_transitions.py <molecule>.xyz
```
Example:
```
$ code/1_compute_transitions.py examples/C1-13-2-3.xyz > examples/C1-13-2-3.xyz.transition.out
```
Output files:
|File                                      | Description
|---                                       | ---
|`examples/C1-13-2-3.xyz.mo.npy`           | ground-state molecular orbital vectors
|`examples/C1-13-2-3.xyz.dm.dat`           | ground-state density matrix
|`examples/C1-13-2-3.xyz.coulomb.dat`      | electrostatic self-repulsion of the ground-state density (a number)
|`examples/C1-13-2-3.xyz.X.npy`            | response vectors (nstates×occ×virt) normalized to 1
|`examples/C1-13-2-3.xyz.transition.out`   | `pyscf` output

### 1.2. Extract hole and particle density matrices

The script [`2_make_dms.py`](code/2_make_dms.py)
saves hole and particle density matrices (atomic orbital basis) of selected states to separate files.

Usage:
```
$ code/2_make_dms.py <molecule>.xyz <state1> <state2>
```
Example (for the S₁ and S₂ states):
```
$ code/2_make_dms.py examples/C1-13-2-3.xyz 1 2
```
Output files:
|File                                                 | Description
|---                                                  | ---
|`examples/C1-13-2-3.xyz.st{1,2}_dm_part.dat`         | particle density matrices
|`examples/C1-13-2-3.xyz.st{1,2}_dm_hole.dat`         | hole density matrices
|`examples/C1-13-2-3.xyz.st{1,2}_coulomb_part.dat`    | electrostatic self-repulsion of particle densities
|`examples/C1-13-2-3.xyz.st{1,2}_coulomb_hole.dat`    | electrostatic self-repulsion of hole densities

### 1.3. Density fitting

1. The script [`3_fitting.py`](code/3_fitting.py)
decomposes the ground-state and hole and particle densities saved on the previous steps onto an atom-centered basis
(cc-pVQZ/JKFIT, defined in [`qm_config.py`](code/qm_config.py)).

Usage:
```
$ code/3_fitting.py <molecule>.xyz
```
Example:
```
$ code/3_fitting.py examples/C1-13-2-3.xyz
```
Output files:
|File                                                 | Description
|---                                                  | ---
|`examples/C1-13-2-3.xyz.dm_fit.dat`                  | fitting coefficients for the ground-state density
|`examples/C1-13-2-3.xyz.st{1,2}_dm_part_fit.dat`     | fitting coefficients for the particle densities
|`examples/C1-13-2-3.xyz.st{1,2}_dm_hole_fit.dat`     | fitting coefficients for the hole densities

Output: absolute and relative fitting errors and the number of particles in the fitted field,
```
examples/C1-13-2-3.xyz  ground  Error: 4.790e-04 Eh ( 1.0e-05 %)  Electrons: 1.660e+02 (  1.3e-03 )
examples/C1-13-2-3.xyz  hole_1  Error: 8.813e-07 Eh ( 2.0e-04 %)  Electrons: 1.000e+00 (  1.3e-05 )
examples/C1-13-2-3.xyz  part_1  Error: 8.911e-07 Eh ( 2.9e-04 %)  Electrons: 1.000e+00 (  2.5e-05 )
examples/C1-13-2-3.xyz  hole_2  Error: 3.224e-07 Eh ( 1.8e-04 %)  Electrons: 1.000e+00 (  4.4e-05 )
examples/C1-13-2-3.xyz  part_2  Error: 5.648e-07 Eh ( 2.4e-04 %)  Electrons: 1.000e+00 (  3.3e-05 )
```

2. The script [`3_fitting_td.py`](code/3_fitting_td.py)
decomposes the transition density. The sign is determined so that the density on the left Nitrogen is positive.

Usage:
```
$ code/3_fitting_td.py <molecule>.xyz <state1> <state2> <number_of_left_N> <number_of_right_N>
```
Example:
```
$ code/3_fitting_td.py examples/C1-13-2-3.xyz  1 2  9 10
```
Output files:
|File                                                 | Description
|---                                                  | ---
|`examples/C1-13-2-3.xyz.st{1,2}_transition_fit.dat`  | fitting coefficients for the transition densities

Output:
```
examples/C1-13-2-3.xyz state1 Error:  5.041e-07 Eh ( 4.0e-03 %)  Electrons:  7.6e-07 total, -4.90e-03 left N, -2.11e-03 right N
examples/C1-13-2-3.xyz state2 Error:  2.021e-07 Eh ( 1.6e-03 %)  Electrons:  1.2e-05 total, -1.62e-01 left N,  1.27e-01 right N
```
For the S₂ state, the sign of the density on the left N is negative thus the fitting coefficients were multiplied by -1 before saving.


## 2. Decriptors

### Transition dipole moments

The scripts [`transition_dipole_dm.py`](code/transition_dipole_dm.py) and [`transition_dipole.py`](code/transition_dipole.py)
compute the transition dipole moments. ***NB:*** the output values should be multiplied by √2.
* *Ab initio* transition dipole moments:
```
$ code/transition_dipole_dm.py <molecule>.xyz <state> <sign_factor>
```

Example:
```
$ code/transition_dipole_dm.py examples/C1-13-2-3.xyz 2 -1

examples/C1-13-2-3.xyz [-0.68928  2.10718  1.53424]
```

* Decomposed / predicted transition dipole moments:
```
$ code/transition_dipole.py <molecule>.xyz <transition-density-coefficients>
```
Example:
```
$ code/transition_dipole.py examples/C1-13-2-3.xyz examples/C1-13-2-3.xyz.st2_transition_fit.dat

examples/C1-13-2-3.xyz [-0.68919  2.10692  1.53400]
```

### Exciton properties

The scripts [`exciton_properties_dm.py`](code/exciton_properties_dm.py) and [`exciton_properties.py`](code/exciton_properties.py)
compute the hole–particle distances and hole and particle sizes.
* *Ab initio* properties:
```
$ code/exciton_properties_dm.py <molecule>.xyz <hole_density_matrix> <particle_density_matrix>
```
Example:
```
$ code/exciton_properties_dm.py examples/C1-13-2-3.xyz examples/C1-13-2-3.xyz.st2_dm_{hole,part}.dat

examples/C1-13-2-3.xyz  dist = 2.598634e+00  hole_size = 7.848500e+00  part_size = 5.676174e+00
```
* Decomposed / predicted properties:
```
$ code/exciton_properties.py <molecule>.xyz <hole_coefficients> <particle_coefficients>
```
Example:
```
$ code/exciton_properties.py examples/C1-13-2-3.xyz examples/C1-13-2-3.xyz.st2_dm_{hole,part}_fit.dat

examples/C1-13-2-3.xyz  dist = 2.599404e+00  hole_size = 7.847751e+00  part_size = 5.675416e+00
```

### Fragment decomposition

The script [`fragments.py`](code/fragments.py) computes the hole and particle contribution of each fragment:
```
$ code/fragments.py <molecule>.xyz <fragment_definition> <hole_coefficients> <particle_coefficients>
```

Example:
```
$ code/fragments.py C1-13-2-3.{xyz,frag} C1-13-2-3.xyz.st2_dm_{hole,part}_fit.dat

C1-13-2-3.xyz H [ 4.243 25.179  7.802 32.888 29.888] P [ 1.867 20.024 37.237 36.807  4.066]
```

(The atomic densities were computed with [`spherical_atoms.py`](code/spherical_atoms.py) and are stored in [`spherical_atoms/`](spherical_atoms/).)

