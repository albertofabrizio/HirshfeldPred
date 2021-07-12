import numpy as np
import ase.io as aio
from pyscf import gto, dft
from pyscf.dft import numint
import argparse
import json

parser = argparse.ArgumentParser(description='Process the input.')
parser.add_argument('--mol', type=str, nargs='+', dest='filename',
                    help='Molecular structure in xyz format')
parser.add_argument('--basis', type=str, nargs='+', dest='basis',
                    help='Basis set for DFT computation')
parser.add_argument('--auxbasis', type=str, nargs='+', dest='auxbasis',
                    help='Basis set for decomposition')
parser.add_argument('--func', type=str, nargs='+', dest='xc_func',
                    help='XC functional for computation')


args = parser.parse_args()


def read_xyz(filename):
    return aio.read(filename, format='xyz')


def convert_ASE_PySCF(molecules, basis, auxbasis):
    # Take an ASE atom and return  PySCF mol and auxmol objects.

    sym = molecules.get_chemical_symbols()
    pos = molecules.positions
    tmp_mol = []
    for s, (x, y, z) in zip(sym, pos):
        tmp_mol.append([s, (x, y, z)])

    pyscf_mol = gto.M(atom=tmp_mol, basis=basis, charge=1)

    with open(auxbasis, 'r') as f:
        auxbasis = eval(f.read())

    pyscf_auxmol = gto.M(atom=tmp_mol, basis=auxbasis, charge=1)

    return pyscf_mol, pyscf_auxmol


def get_integrals(mol, auxmol):

    S = auxmol.intor('int1e_ovlp_sph')
    pmol = mol + auxmol

    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(
        0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

    return S, eri2c, eri3c


def get_coeff(dm, eri2c, eri3c):
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    c = np.linalg.solve(eri2c, rho)
    return c


def main():
    print('Reading molecule')
    compounds = aio.read(args.filename[0]+".xyz", format='xyz')
    mol, auxmol = convert_ASE_PySCF(compounds, args.basis[0], args.auxbasis[0])
    print('Performing DFT computation')
    dm = np.load(args.filename[0]+"_dm.npy")
    print('Computing integrals')
    S, eri2c, eri3c = get_integrals(mol, auxmol)
    print('Computing coefficients')
    c = get_coeff(dm, eri2c, eri3c)
    print(c.shape)
    
    g = dft.gen_grid.Grids(mol)
    g.level = 3
    g.build()
    coords = g.coords
    grid_weights = g.weights

    ao_sph = numint.eval_ao(auxmol, coords)

    print(np.dot(np.dot(ao_sph, c),grid_weights))

    print('Saving coefficients')
    np.save('coeff_'+args.filename[0], c)


if __name__ == "__main__":
    main()
