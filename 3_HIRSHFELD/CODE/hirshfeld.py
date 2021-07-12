import argparse
from pyscf import gto, dft
from pyscf.dft import numint
import numpy as np

########################## Parsing user defined input ##########################
parser = argparse.ArgumentParser(description='This program computes the classical hirshfeld charges of a molecular system.')

parser.add_argument('--mol', type=str, dest='filename',
                    help='Path to molecular structure in xyz format')
parser.add_argument('--auxbasis', type=str, dest='auxbasis',
                    help='Basis set used for decomposition')
parser.add_argument('--coeff', type=str, dest='coeff_mol',
                    help='Coefficient for the system')
parser.add_argument('--sphauxbasis', type=str, dest='sphauxbasis',
                    help='Basis set for spherical atoms')
parser.add_argument('--sphcoeff', type=str, dest='coeff_sph', default="./atoms/",
                    help='Path to the directory containig the spherical atom coefficients')
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
    """ Compute Classical Hirshfeld Charges"""

    # Make the grid
    coords, weights = make_grid(mol)

    # Compute the value of orbitals at each point of the grid
    ao, ao_sph = ao_on_grid(mol,sph_mol,coords)

    # Compute the total density
    dens = np.dot(ao, coeff_mol)

    # Compute the integrand
    intgrd = np.zeros((sph_mol.natm, ao.shape[0]))
    
    all_sph_coeff = []
    for i in range(sph_mol.natm):

        # Lookup element in dictionary of spherical coefficients
        key = sph_mol.elements[i]
        coeff_i = sph_coeff[key]

        # Append coefficients for atom in the total list
        all_sph_coeff.append(coeff_i)

        # Compute numerator

        start = sph_shell_start[i]
        if i < sph_mol.natm-1:
            end = sph_shell_start[i+1]
        else:
            end = None

        intgrd[i,:] = np.multiply(np.dot(ao_sph[:,start:end], coeff_i), dens)

    # Compute denominator

    flat_sph_coeff = [item for sublist in all_sph_coeff for item in sublist]
    all_sph_coeff = np.array(flat_sph_coeff)
    
    den = np.dot(ao_sph, all_sph_coeff) + 1E-15 # Avoid division by zero

    # Divide and integrate.

    q = np.dot( np.divide(intgrd, den), weights)
    
    return q

########################## Main ##########################

def main():
    """ Main """

    # Create pyscf-mol object
    xyz_filename = args.filename
    mol = readmol(xyz_filename, args.auxbasis, charge = args.charge)

    # Load spherical atom basis and create a spherical molecule object
    with open(args.sphauxbasis, 'r') as f:
        s_basis = eval(f.read())

    sph_mol = readmol(xyz_filename, s_basis, charge = args.charge)

    # Store the starting index for each atom
    sph_shell_start = []
    for atoms in range(sph_mol.natm):
        sph_shell_start.append(sph_mol.search_shell_id(atoms,0))

    # Load coefficients for molecule
    coeff_mol = np.loadtxt(args.coeff_mol)

    # Load coefficients for each spherical atom
    uniq_el = set(mol.elements)

    sph_coeff = {}
    for i in uniq_el:
        if not args.coeff_sph.endswith("/"):
            sph_coeff[i] = np.load(args.coeff_sph+"/"+"coeff_"+i+".npy")
        else:
            sph_coeff[i] = np.load(args.coeff_sph+"coeff_"+i+".npy")

    # Get classical hirshfeld charges
    
    q = hirsfheld_chrges(mol, sph_mol, coeff_mol, sph_coeff, sph_shell_start)

    # Get partial charges and save

    partial_charges = mol.atom_charges() - q
    print(partial_charges)
    np.savetxt('partial_charges.dat', partial_charges, fmt="%4.2f")


if __name__ == "__main__":
    main()