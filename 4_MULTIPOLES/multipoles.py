import numpy as np
from pyscf import gto
import argparse

##### CONSTANTS #####

AU2DEBYE = 2.541746

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description='This program computes the dominant hirshfeld charges of a molecular system.')

parser.add_argument('--mol', type=str, dest='filename', help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--frag', type=str, dest='frag', help='Define which atom belongs to which molecule', required=True)
parser.add_argument('--auxbasis', type=str, dest='auxbasis', help='Basis set used for decomposition', required=True)
parser.add_argument('--hirchrg', type=str, dest='hirchrg', help='Path to the file containing Hirshfeld charges', required=True)
parser.add_argument('--charge', type=int, dest='charge', default=0, help='(optional) Total charge of the system (default = 0)')

args = parser.parse_args()

########################## Helper Functions ##########################

def readmol(fin, basis, charge=0):
    """ Read xyz and return pyscf-mol object """

    f = open(fin, "r")
    molxyz = '\n'.join(f.read().split('\n')[2:])
    f.close()
    mol = gto.Mole()
    mol.atom = molxyz
    mol.basis = basis
    mol.charge = charge
    mol.build()

    return mol

def dipole_moment(q,r,origin):
    """ Compute the (discrete) dipole moment given the partial charges. """
    return np.dot(q, r-origin)


########################## Main ##########################

def main():
    """ Main """

    # Create pyscf-mol object and load the fragments
    xyz_filename = args.filename
    mol = readmol(xyz_filename, args.auxbasis, charge = args.charge)
    frag = np.loadtxt(args.frag, usecols=1, dtype=np.int)
    chgs = np.loadtxt(args.hirchrg)

    # For each molecule compute the center of charge and the dipole moment.

    nmol = max(frag)
    dipoles = np.zeros((nmol,4))
    for i in range(nmol):
        idx = np.where(frag==i)[0]
        qi = mol.atom_charges()[idx]
        ri = mol.atom_coords()[idx]
        origin = np.einsum('k,kx->x', qi,ri) / sum(qi) # Center of nuclear charges
        dipoles[i,:3] = dipole_moment(chgs[idx],ri,origin)
        dipoles[i,3] = np.linalg.norm(dipoles[i,:3])

    np.savetxt("mol_dipoles.dat", dipoles, fmt="%8.6f")
    np.savetxt("mol_dipoles_debye.dat", dipoles*AU2DEBYE, fmt="%4.2f")


if __name__ == "__main__":
    main()