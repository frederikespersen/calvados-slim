"""
src.calvados.constants
----------------------
Amino-acid lookup tables and physical constants shared across modules.
"""

############################################################
# Amino-acid dictionaries
############################################################

# All residue names recognised in PDB files, including protonation variants
AA_NAMES: list[str] = [
    "GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
    "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
    "HID", "ASH", "HIE", "HIP", "HSD", "HSE", "HSP",
]

ONE_TO_THREE: dict[str, str] = {
    "G": "GLY", "A": "ALA", "V": "VAL", "L": "LEU", "I": "ILE",
    "F": "PHE", "W": "TRP", "Y": "TYR", "D": "ASP", "N": "ASN",
    "E": "GLU", "K": "LYS", "Q": "GLN", "M": "MET", "S": "SER",
    "T": "THR", "C": "CYS", "P": "PRO", "H": "HIS", "R": "ARG",
}

THREE_TO_ONE: dict[str, str] = {v: k for k, v in ONE_TO_THREE.items()}

# Electron counts per element symbol used for electron-density COE calculation
ELECTRON_COUNTS: dict[str, int] = {"C": 6, "N": 7, "O": 8, "H": 1, "S": 16}


############################################################
# Physical constants
############################################################

# Gas constant in kcal⋅K⁻¹⋅mol⁻¹
R_KCAL: float = 8.31446261815324 * 0.238846 * 1e-3

# Avogadro's number
NA: float = 6.022e23


############################################################
# Helpers
############################################################

def process_protonation(resname: str) -> str:
    """Normalise non-standard protonation-state residue names to canonical form.

    Parameters
    ----------
    resname : str
        Three-letter residue name, possibly a protonation variant.

    Returns
    -------
    str
        Canonical three-letter residue name.
    """
    if resname == "ASH":
        return "ASP"
    if resname in {"HIE", "HID", "HIP", "HSD", "HSE", "HSP"}:
        return "HIS"
    return resname
