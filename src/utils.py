"""
src.utils
------------------------------------------------------------------------
Miscellaneous utilitary functions.
"""


import warnings
import pandas as pd
import numpy as np
import os
from Bio import PDB, SeqIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller
import gemmi


#······································································#
# File manipulation
#······································································#

def replace_copy_file(ori_path, new_path, ori_str, new_str):
    with open(ori_path, "r", encoding="utf-8") as file:
        content = file.read()
    updated_content = content.replace(ori_str, new_str)
    with open(new_path, "w", encoding="utf-8") as file:
        file.write(updated_content)

#······································································#

def cif_to_pdb(cif_in, pdb_out) -> None:
    with warnings.catch_warnings():
        # Suppress common non-fatal parsing warnings (e.g., missing disordered atoms)
        warnings.simplefilter('ignore', PDBConstructionWarning)
        parser = PDB.MMCIFParser()
        structure = parser.get_structure("pdb", cif_in)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_out)



#······································································#
# File parsing
#······································································#

def parse_specific_interactions(fspecific_interactions):
    spec = []
    with open(fspecific_interactions,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            left, right, params = line.split('|')
            name_i, bead_i = left.split()
            name_j, bead_j = right.split()
            s, k = params.split()
            spec.append([name_i, int(bead_i), name_j, int(bead_j), float(s), k]) # 1-based
    return pd.DataFrame(spec, columns=['Acceptor chain', 'Acceptor residue', 'Donor chain', 'Donor residue', 'Equilibrium distance', 'Well depth'])



#······································································#
# Structure files
#······································································#

def _label_to_auth_map(cif_path: str) -> dict[str, str]:
    """label_asym_id -> auth_asym_id, read from _atom_site. 1:1 by construction."""
    block = gemmi.cif.read(cif_path).sole_block()
    table = block.find('_atom_site.', ['label_asym_id', 'auth_asym_id'])
    m = {}
    for row in table:
        m.setdefault(row[0], row[1])   # first occurrence suffices; mapping is a function
    return m

#······································································#

def pdbfix_select_chains(input_pdb, output_pdb, chainids: list[str], add_missing: bool = True) -> None:

    fixer = PDBFixer(filename=input_pdb)   # Must be SEQRES-bearing file

    # OpenMM sets chain.id = label_asym_id for CIFs, which can differ from the
    # author chain id (what ChimeraX / the PDB header use). Relabel to author ids
    # so that `chainids` are interpreted as author chains, matching the viewer.
    if str(input_pdb).lower().endswith('.cif'):
        label2auth = _label_to_auth_map(input_pdb)
        for c in fixer.topology.chains():
            c.id = label2auth.get(c.id, c.id)

    # Sanity check: requested author chains actually exist
    present = {c.id for c in fixer.topology.chains()}
    missing = [c for c in chainids if c not in present]
    if missing:
        raise ValueError(f'Requested author chains {missing} not found in {input_pdb} '
                         f'(present: {sorted(present)})')

    # Isolate chains (by index, so duplicate ids from relabeling are handled safely)
    chains_to_remove = [i for i, c in enumerate(fixer.topology.chains()) if c.id not in chainids]
    fixer.removeChains(chainIndices=chains_to_remove)

    # Remove water and heterogens
    fixer.removeHeterogens(keepWater=False)

    # Reconstruct missing heavy atoms
    if add_missing:
        fixer.findMissingResidues()
    else:
        fixer.missingResidues = {}

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Delete hydrogen atoms to satisfy SMOG topology requirements
    modeller = Modeller(fixer.topology, fixer.positions)
    hydrogens = [atom for atom in modeller.topology.atoms() if atom.element.symbol == 'H']
    modeller.delete(hydrogens)

    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)



#······································································#
# BioPython
#······································································#

class PeptideChainsSelect(PDB.Select):
    def __init__(self, chain_ids: list[str]):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        """Retain only chains matching the specified IDs."""
        return 1 if chain.get_id() in self.chain_ids else 0

    def accept_residue(self, residue):
        """
        Evaluate residue identity to exclude waters and ligands.
        """
        # Method A: Strict Canonical Amino Acid Filter
        # Evaluates if the residue is one of the 20 standard amino acids.
        return 1 if PDB.Polypeptide.is_aa(residue, standard=True) else 0

#······································································#

def seq_from_pdb(pdb: str, chainid: str) -> str:
    extension = os.path.splitext(pdb)[-1]
    if extension == '.cif':
        parser = 'cif-seqres'
    elif extension == '.pdb':
        parser = 'pdb-seqres'
    else:
        raise ValueError(f"Structure type '{os.path.splitext(pdb)[-1].upper()}' not supported!")
    for record in SeqIO.parse(pdb, parser):
        if record.annotations.get("chain") == chainid:
            return str(record.seq)
    else:
        raise KeyError(f"Chain ID '{chainid}' was not found in '{pdb}'!")



#······································································#
# Slurm
#······································································#

def estimate_cpus(num_particles: int) -> int:
    if num_particles < 150:
        return 4
    elif num_particles < 1000:
        return 8
    else:
        return 16

#······································································#

def estimate_mem(num_particles: int) -> str:
    gb = np.ceil((4.0 + (num_particles * 1000) / (1024**3)) * 1.5)
    return f"{gb + (gb % 2):.0f}G"
