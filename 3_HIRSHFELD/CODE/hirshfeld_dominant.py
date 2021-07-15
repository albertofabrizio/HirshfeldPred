import argparse
from pyscf import gto, dft
from pyscf.dft import numint
import numpy as np
import os

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program computes the dominant hirshfeld charges of a molecular system.')

parser.add_argument('--mol', type=str, dest='filename',
                    help='Path to molecular structure in xyz format', required=True)
parser.add_argument('--auxbasis', type=str, dest='auxbasis',
                    help='Basis set used for decomposition', required=True)
parser.add_argument('--coeff', type=str, dest='coeff_mol',
                    help='Coefficient for the system', required=True)

parser.add_argument('--sphbasis', type=str, dest='sphbasis',
                    help='Basis set for spherical atoms', required=True)
parser.add_argument('--sphcoeff', type=str, dest='coeff_sph', default="./atoms/",
                    help='Path to the directory containig the spherical atom coefficients')

parser.add_argument('--isS', dest='isS', default = False, action='store_true',
                    help="Whether or not the overlap metric was used for projection.")
parser.add_argument('--isfile', dest='isfile', action='store_true',
                    help="Whether or not the auxbasis is the name of an external file to read [default: False].")

parser.add_argument('--isdft', dest='isdft', action='store_true',
                    help="Whether or not the spherical DM originated from a DFT computation.")
parser.add_argument('--func', dest='func', type=str,
                    help="DFT functional used in the spherical DM computation.")

parser.add_argument('--charge', type=int, nargs='?', dest='charge', default=0,
                    help='(optional) Total charge of the system (default = 0)')

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

def make_grid(mol, levl = 3):
    """ Construct an atom centered Becke grid """
    g = dft.gen_grid.Grids(mol)
    g.level = levl
    g.build()
    coords = g.coords
    grid_weights = g.weights

    return coords, grid_weights

def ao_on_grid(mol,sph_mol, coords):
    """Evalute orbital values at each point of the grid"""

    ao = numint.eval_ao(mol, coords)
    ao_sph = numint.eval_ao(sph_mol, coords)

    return ao, ao_sph

def hirsfheld_chrges(mol, sph_mol, coeff_mol, sph_coeff, sph_shell_start):
    """ Compute Dominant Hirshfeld Charges"""

    # Make the grid
    coords, weights = make_grid(mol)

    # Compute the value of orbitals at each point of the grid
    ao, ao_sph = ao_on_grid(mol,sph_mol,coords)

    # Compute the total density
    dens = np.dot(ao, coeff_mol)

    # Compute the integrand
    intgrd = np.zeros((sph_mol.natm, ao_sph.shape[0]))

    for i in range(sph_mol.natm):

        # Lookup element in dictionary of spherical coefficients
        key = sph_mol.elements[i]
        coeff_i = sph_coeff[key]

        # Compute shell start and end for each atom
        start = sph_shell_start[i]
        if i < sph_mol.natm-1:
            end = sph_shell_start[i+1]
        else:
            end = None

        # Compute rho pro-atomic
        intgrd[i,:] = np.dot(ao_sph[:,start:end], coeff_i)

    # Find the dominant atom at each point
    idx = np.argmax(intgrd, axis=0)  
    intgrd.fill(0)

    # Naive implementation, rethink it later
    for point in range(intgrd.shape[1]):
        intgrd[idx[point], point] = 1.0

    # Multiply and integrate.
    q = np.dot( np.multiply(intgrd, dens), weights)
    
    return q

########################## Main ##########################

def main():
    """ Main """

    # Create pyscf-mol object
    xyz_filename = args.filename
    mol = readmol(xyz_filename, args.auxbasis, charge = args.charge)

    # Read or load the auxiliary basis set
    if args.isfile:
        with open( args.sphbasis,'r') as f:
            sphbasis = eval(f.read())
    else:
        sphbasis = args.sphbasis

    sph_mol = readmol(xyz_filename, sphbasis, charge = args.charge)
    

    # Store the starting index for each atom
    sph_shell_start = []
    for atoms in range(sph_mol.natm):
        sph_shell_start.append(sph_mol.search_shell_id(atoms,0))

    # Load coefficients for molecule
    coeff_mol = np.loadtxt(args.coeff_mol)

    # Load coefficients for each spherical atom
    uniq_el = set(mol.elements)

    if args.isS:
        name = 'S_coeff_'
    else:
        name = 'J_coeff_'

    if args.isdft:
        base = os.path.basename(args.sphbasis)+'_'+args.func
    else:
        base = os.path.basename(args.sphbasis)

    sph_coeff = {}
    for i in uniq_el:
        if not args.coeff_sph.endswith("/"):
            sph_coeff[i] = np.load(args.coeff_sph+"/"+name+i+'_'+base+".npy")

        else:
            sph_coeff[i] = np.load(args.coeff_sph+name+i+'_'+base+".npy")

    # Get dominant hirshfeld charges
    
    q = hirsfheld_chrges(mol, sph_mol, coeff_mol, sph_coeff, sph_shell_start)

    # Get partial charges and save

    partial_charges = mol.atom_charges() - q
    np.savetxt('partial_charges.dat', partial_charges, fmt="%4.2f")


if __name__ == "__main__":
    main()
