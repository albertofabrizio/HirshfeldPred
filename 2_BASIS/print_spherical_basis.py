import numpy as np
from pyscf import gto
import json
import argparse

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description="This programs computes the spherical averaged density matrix @ HF level for a given atom and basis set.")

parser.add_argument('--atom', type=str, nargs='+', required=True, dest='atm', help="The periodic table symbol for the atom.")
parser.add_argument('--basis', type=str, required=True, dest='base', help="The name of the basis set for the computation.")

args = parser.parse_args()

########################## Main ##########################

def main():

    # Read in inputs
    atm = []
    for i in args.atm:
        atm.append(i.upper())
    
    print("Printing spherical basis for atoms: ", atm)

    # Define spin in function of element

    spin_dict = {"H":1, "HE":0,
    "LI":1, "BE":0, "B":1, "C":2, "N":3, "O":2, "F":1, "NE":0,
    "NA":1, "MG":0, "AL":1, "SI":2, "P":3,"S":2,"CL":1, "AR":0,
    "BR":1}
    
    full_dict = {}
    for i in atm:
        # Instance pyscf mol object
        mol = gto.M(atom=i+''' 0 0 0''', basis=args.base, spin=spin_dict[i])
        base = mol._basis[i]
        bas_info = np.array(mol._bas)
        spherical_info = bas_info[bas_info[:,1] == 0]
        n_sph = spherical_info.shape[0]
        full_dict[i] = base[:n_sph]

    a_file = open(args.base+".json", "w")
    json.dump(full_dict, a_file)
    a_file.close()

if __name__== "__main__":
    main()

