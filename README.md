# ML-TD-azo-dyes

This code supports the paper
> Sergi Vela, Alberto Fabrizio, Ksenia R. Briling, and Clémence Corminboeuf,<br>
> “Machine-Learning the Transition Density of the Productive Excited States of Azo-dyes”<br>
> [`ChemRxiv:XXXXXX`](https://chemrxiv.org/xxxxxx)<br>

It consists of ... ... and ... ... .

## Requirements
* `python >= 3.6`
* `numpy >= 1.16`
* [`pyscf >= 1.6`](https://github.com/pyscf/pyscf)

## Usage

### Ab initio computations

#### 1. Compute transitions within TDDFT

The script
`code/1_compute_transitions.py`
runs an *ab initio* computation of 8 lowest excited states at the ωB97X-D/def2-SVP level.
Example:
```
code/1_compute_transitions.py examples/C1-13-2-3.xyz > examples/C1-13-2-3.xyz.transition.out
```
Output:
|file                                      | description
|---                                       | ---
|`examples/C1-13-2-3.xyz.mo.npy`           | ground-state molecular orbital vectors
|`examples/C1-13-2-3.xyz.dm.dat`           | ground-state density matrix
|`examples/C1-13-2-3.xyz.coulomb.dat`      | electrostatic self-repulsion of the ground-state density (a number)
|`examples/C1-13-2-3.xyz.X.npy`            | response vectors (nstates\*occ\*virt)
|`examples/C1-13-2-3.xyz.transition.out`   | `pyscf` output

#### 2. Get hole and particle density matrices

#### 3. Perform density fitting

### Decriptors

#### Calculate the hole–particle distance

#### Calculate the hole and particle sizes

#### Calculate the transition dipole moment
