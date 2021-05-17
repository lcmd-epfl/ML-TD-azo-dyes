import numpy
from pyscf import gto,scf

def readmol(fin, basis, charge=0):
  f = open(fin, "r")
  molxyz = '\n'.join(f.read().split('\n')[2:])
  f.close()
  mol = gto.Mole()
  mol.atom = molxyz
  mol.basis = basis
  mol.charge = charge
  mol.build()
  return mol


def compute_j(mol, dm):
  j,k = scf.hf.get_jk(mol, dm)
  return numpy.einsum('ij,ij',j,dm)


def eri_pqi(mol, auxmol):
  pmol  = mol + auxmol
  eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
  return eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)


def number_of_electrons_vec(mol):
  nel = []
  i = 0
  for iat in range(mol.natm):
    j = 0
    q = mol._atom[iat][0]
    max_l = mol._basis[q][-1][0]
    numbs = [x[0] for x in mol._basis[q]]
    for n in range(numbs.count(0)):
      a, w = mol._basis[q][j][1]
      nel.append( w * pow (2.0*numpy.pi/a, 0.75) )
      i += 1
      j += 1
    for l in range(1,max_l+1):
      n_l = numbs.count(l)
      i += n_l * (2*l+1)
      nel.extend([0]*n_l * (2*l+1))
      j += n_l
  return numpy.array(nel)


def number_of_electrons(c, mol):
  q = number_of_electrons_vec(mol)
  return q@c


def r_c(rho, mol):
  r  = numpy.zeros(3)
  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    coord = mol.atom_coords()[iat]
    for gto in mol._basis[q]:
      l, [a, c] = gto
      #print(l, a ,c)
      if(l==0):
        I0 = c * (2.0*numpy.pi/a)**0.75
        r   += I0 * rho[i] * coord
        i+=1
      elif(l==1):
        I1 = c * (2.0*numpy.pi)**0.75 / (a**1.25)
        r   += I1 * rho[i:i+3]
        i+=3
      else:
        i+=2*l+1
  return r


def r2_c(rho, mol):
  N  = 0.0           # <1>
  r  = np.zeros(3)   # <r>
  r2 = 0.0           # <r^2>
  i=0
  for iat in range(mol.natm):
    q = mol._atom[iat][0]
    coord = mol.atom_coords()[iat]
    for gto in mol._basis[q]:
      l, [a, c] = gto
      if(l==0):
        I0 = c * (2.0*np.pi/a)**0.75
        I2 = c * 3.0 * (np.pi**0.75) / (a**1.75 * 2.0**0.25)
        N   += I0 * rho[i]
        r   += I0 * rho[i] * coord
        r2  += I0 * rho[i] * (coord@coord)
        r2  += I2 * rho[i]
        i+=1
      elif(l==1):
        I1 = c * (2.0*np.pi)**0.75 / (a**1.25)
        temp = I1 * rho[i:i+3]
        r   += temp
        r2  += 2.0*(temp@coord)
        i+=3
      else:
        i+=2*l+1
  return N, r, r2

