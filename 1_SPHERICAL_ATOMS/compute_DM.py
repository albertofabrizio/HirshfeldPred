import copy
import scipy
import numpy as np
from pyscf import gto
from pyscf.scf import atom_hf, addons
import argparse

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description="This programs computes the spherical averaged density matrix @ HF level for a given atom and basis set.")

parser.add_argument('--atom', type=str, required=True, dest='atm', help="The periodic table symbol for the atom.")
parser.add_argument('--basis', type=str, required=True, dest='base', help="The name of the basis set for the computation.")

args = parser.parse_args()

########################## Helper Functions ##########################
def init_guess_by_atom(mol):
    '''Generate initial guess density matrix from superposition of atomic HF
    density matrix.  The atomic HF is occupancy averaged RHF

    Returns:
        Density matrix, 2D ndarray
    '''
    atm_scf = atom_hf.get_atm_nrhf(mol)
    mo = []
    mo_occ = []
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        if symb in atm_scf:
            e_hf, e, c, occ = atm_scf[symb]
        else:
            symb = mol.atom_pure_symbol(ia)
            e_hf, e, c, occ = atm_scf[symb]
        mo.append(c)
        mo_occ.append(occ)

    mo = scipy.linalg.block_diag(*mo)
    mo_occ = np.hstack(mo_occ)

    pmol = copy.copy(mol)
    pmol.cart = False
    c = addons.project_mo_nr2nr(pmol, mo, mol)
    dm = np.dot(c*mo_occ, c.T)

    return dm

########################## Main ##########################

def main():

    # Read in inputs
    atm = args.atm.upper()
    print("Computing density matrix for atom ", atm)    

    # Define spin in function of element

    spin_dict = {"H":1, "HE":0,
    "LI":1, "BE":0, "B":1, "C":2, "N":3, "O":2, "F":1, "NE":0,
    "NA":1, "MG":0, "AL":1, "SI":2, "P":3,"S":2,"CL":1, "AR":0,
    "BR":1}
    
    # Instance pyscf mol object
    mol = gto.M(atom=atm+''' 0 0 0''', basis=args.base, spin=spin_dict[atm])

    # Compute and save DM
    dm = init_guess_by_atom(mol)
    np.save(atm+'_'+args.base+'_dm.npy',dm)

if __name__== "__main__":
    main()
