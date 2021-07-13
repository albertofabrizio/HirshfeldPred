import numpy as np
from pyscf import gto, df
import argparse

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description="This programs optimize an even tempered basis to minimize the number of electrons integration error.")

parser.add_argument('--atom', type=str, required=True, dest='atm', help="The periodic table symbol for the atom.")
parser.add_argument('--dm', type=str, required=True, dest='dmfile', help="The path for the density matrix.")
parser.add_argument('--basis', type=str, required=True, dest='base', help="The name of the basis set used for the DM computation.")
parser.add_argument('--isS', dest='isS', help="Whether or not using the overlap metric for projection.", default = False, action='store_true')


args = parser.parse_args()

########################## Helper Functions ##########################


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

    auxbasis = df.addons.aug_etb(mol, beta=2.0)
    print(auxbasis)


if __name__ == "__main__":
    main()
