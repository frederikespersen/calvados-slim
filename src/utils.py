"""
src.utils
------------------------------------------------------------------------
Miscellaneous utilitary functions.
"""


#······································································#
# Imports
#······································································#

import warnings
from Bio.PDB import MMCIFParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
    


#······································································#
# File conversion
#······································································#

def cif_to_pdb(cif_in, pdb_out) -> None:
    with warnings.catch_warnings():
        # Suppress common non-fatal parsing warnings (e.g., missing disordered atoms)
        warnings.simplefilter('ignore', PDBConstructionWarning)
        parser = MMCIFParser()
        structure = parser.get_structure("pdb", cif_in)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_out)
