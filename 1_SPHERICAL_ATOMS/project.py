import numpy as np
from pyscf import gto, dft
import argparse

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description="This programs projects the spherical averaged density matrix @ HF level onto a given basis set.")

parser.add_argument('--atom', type=str, required=True, dest='atm', help="The periodic table symbol for the atom.")
parser.add_argument('--dm', type=str, required=True, dest='dmfile', help="The path for the density matrix.")
parser.add_argument('--basis', type=str, required=True, dest='base', help="The name of the basis set used for the DM computation.")
parser.add_argument('--auxbasis', type=str, required=True, dest='auxbase', help="The name of the basis set for the projection.")
parser.add_argument('--isS', dest='isS', help="Whether or not using the overlap metric for projection.", default = False, action='store_true')
parser.add_argument('--isfile', type=bool, dest='isfile', help="Whether or not the auxbasis is the name of an external file to read [default: False].", default = False)

args = parser.parse_args()

########################## Helper Functions ##########################

def get_integrals(mol, auxmol):
    """ Compute the overlap, the two-center and the three-center Coulomb integrals of the auxiliary basis functions.
    
    Returns:
        The S, J2, and J3 integrals, 2D/3D ndarray

    """

    # Compute overlap
    S = auxmol.intor('int1e_ovlp_sph')

    # Create a fake mol object concatenating mol and auxmol
    pmol = mol + auxmol

    # Compute Coulomb integrals
    eri2c = auxmol.intor('int2c2e_sph')
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)

    return S, eri2c, eri3c


def get_coeff_J(dm, eri2c, eri3c):
    """ Compute the expansion coefficients using the J-metric.
    
    Returns:
        The coefficients, 1D ndarray

    """

    rho = np.einsum('ijp,ij->p', eri3c, dm)
    c = np.linalg.solve(eri2c, rho)
    return c

def get_coeff_S(mol, auxmol, dm):
    """ Compute the expansion coefficients using the S-metric.
    
    Returns:
        The coefficients, 1D ndarray

    """

    # Grid Setup
    g = dft.gen_grid.Grids(mol)
    g.level = 3
    g.build()

    coords = g.coords
    grid_weights = g.weights

    # AO on grid
    ao = dft.numint.eval_ao(mol, coords)
    ao_aux = dft.numint.eval_ao(auxmol, coords)

    # Projection integrals
    rho = np.einsum('pi, pj, ij -> p', ao, ao, dm)
    proj = np.einsum('p, p, pk -> k', rho, grid_weights, ao_aux)

    # Overlap integral
    S = auxmol.intor('int1e_ovlp_sph')

    # Coefficients
    c = np.linalg.solve(S, proj)

    return c

def number_of_electrons(rho, mol):
    """ Compute the number of electrons given coefficients for one atom.
    
    Returns:
        The nuymber of electron, float

    """

    # Get information about the basis
    bas_info = np.array(mol._bas)
    basis = mol._basis[args.atm]

    # Identify spherical GTOs
    spherical_info = bas_info[bas_info[:,1] == 0]

    # Loop over all the contracted spherical GTOs    
    nel = 0
    start = 0
    for n in range(spherical_info.shape[0]):
        primitives = np.array(basis[n][1:])

        # Get exponents and coefficients for this cGTO
        es = primitives[:,0]
        cs = primitives[:,1:]

        cs = np.einsum('pi, p -> pi', cs, pow(2*es/np.pi, 0.75))

        # Compute cGTO self-overlap and normalize
        ss = 0
        for ia in range(len(cs)):
            for ib in range(len(cs)):

                esa = es[ia]
                esb = es[ib]

                ss += cs[ia] * cs[ib] * pow(np.pi/(esa + esb),1.5)                

        fact = 1/np.sqrt(ss)

        for i in range(cs.shape[1]):
            cs[:,i] *= fact[i]

        # Compute integral
        integral = pow(np.pi/es, 1.5)
        integral = np.dot(integral,cs)

        # Compute number of electrons
        stop = start + integral.shape[0]
        nel += np.dot(rho[start:stop],integral)
        start = stop
    
    return nel

########################## Main ##########################

def main():

    # Read in inputs
    atm = args.atm.upper()
    print("Projecting DM of atom.", atm)    

    # Define spin in function of element

    spin_dict = {"H":1, "HE":0,
    "LI":1, "BE":0, "B":1, "C":2, "N":3, "O":2, "F":1, "NE":0,
    "NA":1, "MG":0, "AL":1, "SI":2, "P":3,"S":2,"CL":1, "AR":0,
    "BR":1}
    
    # Instance pyscf mol object
    mol = gto.M(atom=atm+''' 0 0 0''', basis=args.base, spin=spin_dict[atm])
    auxmol = gto.M(atom=atm+''' 0 0 0''', basis=args.auxbase, spin=spin_dict[atm])

    # Loading the density matrix
    dm = np.load(args.dmfile)

    # Compute basis set integrals
    print('Computing integrals.')
    S, eri2c, eri3c = get_integrals(mol, auxmol)

    # Compute coefficients
    print('Computing coefficients.')

    if args.isS:
        print("Warning: coefficients using the overlap metric are less accurate and are computed by numerical integration.")
        c = get_coeff_S(mol, auxmol, dm)
    else:
        c = get_coeff_J(dm, eri2c, eri3c)
        
    # Check the number of electrons after density-fitting.
    print('Checking number of electrons.')
    n = number_of_electrons(c, auxmol)
    print("Integrated number of electrons:", n)

    print('Saving coefficients')
    if args.isS:
        name = 'S_coeff_'
    else:
        name = 'J_coeff_'        
    
    np.save(name+args.atm+'_'+args.auxbase, c)


if __name__ == "__main__":
    main()
