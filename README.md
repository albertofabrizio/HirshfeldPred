# HirshfeldPred
Compute Classical and Dominant Hirshfeld Charges Given Density Predictions

## SPHERICAL ATOMS

**Compute the spherical averaged density matrix (DM) of all the elements up to Ar @ HF level.**

Usage:
`compute_DM.py [-h] [--atom ATM] [--basis BASE]`

optional arguments: \
  -h, --help    show this help message and exit \
  --atom ATM    The periodic table symbol for the atom. \
  --basis BASE  The name of the basis set for the computation. \

**Compute the spherical averaged density matrix (DM) of all the elements up to Ar @ DFT level (user defined XC).**

Usage:
`compute_DM_dft.py [-h] --atom ATM --basis BASE --func FUNC`

optional arguments:\
  -h, --help    show this help message and exit \
  --atom ATM    The periodic table symbol for the atom. \
  --basis BASE  The name of the basis set for the computation. \
  --func FUNC   The chosen xc density functional. \

**Project the DM onto a specified basis set. **

Usage:
`project.py [-h] --atom ATM --dm DMFILE --basis BASE --auxbasis AUXBASE [--isS] [--isfile]`

optional arguments: \
  -h, --help          show this help message and exit \
  --atom ATM          The periodic table symbol for the atom. \
  --dm DMFILE         The path for the density matrix. \
  --basis BASE        The name of the basis set used for the DM computation. \
  --auxbasis AUXBASE  The name of the basis set (or path) for the projection. \
  --isS               Whether or not using the overlap metric for projection. \
  --isfile            Whether or not the auxbasis is the name of an external file to read [default: False]. 

## BASIS

**Take a user defined basis set and save its spherical part in a json format.**

Usage:
`print_spherical_basis.py [-h] --atom ATM [ATM ...] --basis BASE`

optional arguments: \
  -h, --help            show this help message and exit. \
  --atom ATM [ATM ...]  The periodic table symbol for the atom. \
  --basis BASE          The name of the basis set for the computation. \

## HIRSHFELD

**Compute the dominant Hirshfeld partial charges given spherical atoms and predicted densities.**

Usage:
`hirshfeld_dominant.py [-h] --mol FILENAME --auxbasis AUXBASIS --coeff COEFF_MOL --sphbasis SPHBASIS [--sphcoeff COEFF_SPH] [--isS] [--isfile] [--isdft]
                             [--func FUNC] [--charge [CHARGE]]`

optional arguments: \
  -h, --help            show this help message and exit \
  --mol FILENAME        Path to molecular structure in xyz format \
  --auxbasis AUXBASIS   Basis set used for decomposition \
  --coeff COEFFMOL     Coefficient for the system \
  --sphbasis SPHBASIS   Basis set for spherical atoms \
  --sphcoeff COEFFSPH  Path to the directory containig the spherical atom coefficients \
  --isS                 Whether or not the overlap metric was used for projection. \
  --isfile              Whether or not the auxbasis is the name of an external file to read (default: False). \
  --isdft               Whether or not the spherical DM originated from a DFT computation. \
  --func FUNC           DFT functional used in the spherical DM computation. \
  --charge [CHARGE]     (optional) Total charge of the system (default = 0) 

 **Compute the classical Hirshfeld partial charges given spherical atoms and predicted densities.**

Usage:
`hirshfeld_classical.py [-h] --mol FILENAME --auxbasis AUXBASIS --coeff COEFF_MOL --sphbasis SPHBASIS [--sphcoeff COEFF_SPH] [--isS] [--isfile] [--isdft]
                             [--func FUNC] [--charge [CHARGE]]`

