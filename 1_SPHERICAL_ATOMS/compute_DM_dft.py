import numpy as np
from pyscf import gto, dft
from pyscf.lib import param
from pyscf.data import elements
from pyscf.dft.rks import RKS
import argparse

########################## Parsing user defined input ##########################

parser = argparse.ArgumentParser(description="This programs computes the spherical averaged density matrix @ HF level for a given atom and basis set.")

parser.add_argument('--atom', type=str, required=True, dest='atm', help="The periodic table symbol for the atom.")
parser.add_argument('--basis', type=str, required=True, dest='base', help="The name of the basis set for the computation.")
parser.add_argument('--func', type=str, required=True, dest='func', help="The chosen xc density functional.")


args = parser.parse_args()

########################## Helper Functions ##########################
def _angular_momentum_for_each_ao(mol):
    ao_ang = np.zeros(mol.nao, dtype=np.int)
    ao_loc = mol.ao_loc_nr()
    for i in range(mol.nbas):
        p0, p1 = ao_loc[i], ao_loc[i+1]
        ao_ang[p0:p1] = mol.bas_angular(i)
    return ao_ang

def frac_occ(symb, l, atomic_configuration=elements.NRSRHF_CONFIGURATION):
    nuc = gto.charge(symb)
    if l < 4 and atomic_configuration[nuc][l] > 0:
        ne = atomic_configuration[nuc][l]
        nd = (l * 2 + 1) * 2
        ndocc = ne.__floordiv__(nd)
        frac = (float(ne) / nd - ndocc) * 2
    else:
        ndocc = frac = 0
    return ndocc, frac

class AtomSphericAverageDFT(RKS):
    def __init__(self, mol):
        self._eri = None
        self.atomic_configuration = elements.NRSRHF_CONFIGURATION
        RKS.__init__(self, mol, xc=args.func)

        # The default initial guess minao does not have super-heavy elements
        if mol.atom_charge(0) > 96:
            self.init_guess = '1e'

    def eig(self, f, s):
        mol = self.mol
        ao_ang = _angular_momentum_for_each_ao(mol)

        nao = mol.nao
        mo_coeff = []
        mo_energy = []

        for l in range(param.L_MAX):
            degen = 2 * l + 1
            idx = np.where(ao_ang == l)[0]
            nao_l = len(idx)

            if nao_l > 0:
                nsh = nao_l // degen
                f_l = f[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                s_l = s[idx[:,None],idx].reshape(nsh, degen, nsh, degen)
                # Average over angular parts
                f_l = np.einsum('piqi->pq', f_l) / degen
                s_l = np.einsum('piqi->pq', s_l) / degen

                e, c = self._eigh(f_l, s_l)

                mo_energy.append(np.repeat(e, degen))

                mo = np.zeros((nao, nsh, degen))
                for i in range(degen):
                    mo[idx[i::degen],:,i] = c
                mo_coeff.append(mo.reshape(nao, nao_l))

        return np.hstack(mo_energy), np.hstack(mo_coeff)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''spherically averaged fractional occupancy'''
        mol = self.mol
        symb = mol.atom_symbol(0)

        nelec_ecp = mol.atom_nelec_core(0)
        coreshl = gto.ecp.core_configuration(nelec_ecp)

        occ = []
        for l in range(param.L_MAX):
            n2occ, frac = frac_occ(symb, l, self.atomic_configuration)
            degen = 2 * l + 1
            idx = mol._bas[:,gto.ANG_OF] == l
            nbas_l = mol._bas[idx,gto.NCTR_OF].sum()
            if l < 4:
                n2occ -= coreshl[l]
                assert n2occ <= nbas_l

                occ_l = np.zeros(nbas_l)
                occ_l[:n2occ] = 2
                if frac > 0:
                    occ_l[n2occ] = frac
                occ.append(np.repeat(occ_l, degen))
            else:
                occ.append(np.zeros(nbas_l * degen))

        return np.hstack(occ)

    def get_grad(self, mo_coeff, mo_occ, fock=None):
        return 0

    def scf(self, *args, **kwargs):
        kwargs['dump_chk'] = False
        return RKS.scf(self, *args, **kwargs)


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
    mol_avg = AtomSphericAverageDFT(mol)
    mol_avg.atomic_configuration = elements.NRSRHF_CONFIGURATION
    mol_avg.verbose = 0
    mol_avg.run()

    c, occ = mol_avg.mo_coeff, mol_avg.mo_occ
    dm = np.dot(c*occ, c.T)

    np.save(atm+'_'+args.base+'_'+args.func+'_dm.npy',dm)

if __name__== "__main__":
    main()
