"""
src.analysis.coarsegrain
------------------------------------------------------------------------
Utils for analysis of structures in a coarse grained manner.
"""

import os
from typing import Literal
import numpy as np
from Bio import PDB
import MDAnalysis as mda
import warnings
from src.calvados.calvados.sequence import seq_from_pdb

from src import constants


#······································································#
# Input validation
#······································································#

def validate_chain_pdb(pdb_path, expected_seq, use_com=False):
    seq, n_termini, c_termini = seq_from_pdb(pdb_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(pdb_path)

    # one chain only
    assert len(u.segments) == 1, \
        f"{pdb_path}: expected 1 chain, found {len(u.segments)} ({[s.segid for s in u.segments]})"

    # nothing non-standard leaked through (would also have raised in seq_from_pdb)
    assert 'X' not in seq, f"{pdb_path}: unmapped residue(s) -> X in parsed sequence"

    # the bead atom CALVADOS will actually read must exist for every residue
    if not use_com:
        n_ca = len(u.select_atoms('name CA'))
        assert n_ca == len(seq), \
            f"{pdb_path}: {len(seq)} residues but {n_ca} CA atoms (gap residue missing CA?)"

    # sequence identity, with a positional diff on failure
    if seq != expected_seq:
        if len(seq) != len(expected_seq):
            raise AssertionError(
                f"{pdb_path}: length {len(seq)} != expected {len(expected_seq)}")
        i = next(k for k in range(len(seq)) if seq[k] != expected_seq[k])
        lo, hi = max(0, i - 10), i + 10
        raise AssertionError(
            f"{pdb_path}: sequence mismatch at residue {i+1}\n"
            f"  got     : ...{seq[lo:hi]}...\n"
            f"  expected: ...{expected_seq[lo:hi]}...")
    return seq



#······································································#
# Bead centers
#······································································#

def get_ca(entity) -> np.ndarray:
    """Returns the C-alpha coordinates...

    * For a single residue, if entity is of type residue
    * For all residues in chain, if entity is of type chain

    Parameters
    ----------
    entity : Residue, Chain
        Either a Residue or Chain BioPython object

    Returns
    -------
    np.ndarray
        A float array of size ``(N,3)``, where N is the number of residues in the entity.
    """

    # If entity is residue, get CA atom coordinates
    if type(entity) == PDB.Residue.Residue:
        return entity['CA'].get_coord()

    # If entity is chain, perform iterative invocation on all residues
    elif type(entity) == PDB.Chain.Chain:
        return np.array([get_ca(res) for res in entity])

    else:
        raise TypeError(f"Function not supported for entity {type(entity)}")


def get_com(entity) -> np.ndarray:
    """Returns the center-of-mass (COM) coordinates...

    * For a single residue, if entity is of type residue
    * For all residues in chain, if entity is of type chain

    Parameters
    ----------
    entity : Residue, Chain
        Either a Residue or Chain BioPython object

    Returns
    -------
    np.ndarray
        A float array of size ``(N,3)``, where N is the number of residues in the entity.
    """

    # If entity is residue, get mass-weighted average of atom corodinates
    if type(entity) == PDB.Residue.Residue:
        atoms = [a for a in entity]
        coords = np.array([a.get_coord() for a in atoms])
        masses = np.array([constants.element.mass[a.element] for a in atoms])
        return np.average(coords, weights=masses, axis=0)

    # If entity is chain, perform iterative invocation on all residues
    elif type(entity) == PDB.Chain.Chain:
        return np.array([get_com(res) for res in entity])

    else:
        raise TypeError(f"Function not supported for entity {type(entity)}")
    

#······································································#
# SLiM contacts curation
#······································································#

def dists(
        structure_file: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        pair_residue_idxs,
        chain1_cg: Literal['CA', 'COM']='COM',
        chain2_cg: Literal['CA', 'COM']='CA',
    ) -> np.ndarray:

    # Loading structure
    extension = os.path.splitext(structure_file)[-1]
    if extension == '.pdb':
        parser = PDB.PDBParser(QUIET=True)
    elif extension == '.cif':
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        raise ValueError(f"Only PDB or CIF file parsing supported, not {extension.upper()}!")
    structure = parser.get_structure('struc', structure_file)

    # Subsetting chains
    chain1 = structure[0][chain1_id]
    chain2 = structure[0][chain2_id]

    # Determining auth resnum for first residue in each chain
    chain1_authresids = [res.get_id()[1] for res in chain1]
    chain2_authresids = [res.get_id()[1] for res in chain2]

    # Subsetting residues
    pair_residues = [(chain1[chain1_authresids[res1_idx]], chain2[chain2_authresids[res2_idx]]) for (res1_idx, res2_idx) in pair_residue_idxs]

    # Determining equilibrium distance from structure
    pair_eqdists = []
    for (res1, res2) in pair_residues:
        if chain1_cg == 'CA':
            res1_coords = get_ca(res1)
        elif chain1_cg == 'COM':
            res1_coords = get_com(res1)
        if chain2_cg == 'CA':
            res2_coords = get_ca(res2)
        elif chain2_cg == 'COM':
            res2_coords = get_com(res2)
        eqdist = np.linalg.norm(res1_coords - res2_coords)
        pair_eqdists.append(eqdist)

    return np.array(pair_eqdists)
