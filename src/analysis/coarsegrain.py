"""
src.analysis.coarsegrain
------------------------------------------------------------------------
Utils for analysis of structures in a coarse grained manner.
"""

import os
import warnings
from typing import Literal
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.groups import Residue, ResidueGroup, AtomGroup
from src.calvados.calvados.sequence import seq_from_pdb

from src import utils


#······································································#
# Input validation
#······································································#

def validate_chain_pdb(pdb_path: str, expected_seq: str, use_com: bool = False) -> str:
    """Check that a single-chain PDB is consistent with an expected sequence.

    Asserts that the file contains exactly one chain, that no residue mapped
    to an unknown (``X``) code, that every residue carries the bead atom
    CALVADOS will read (a ``CA`` atom, unless ``use_com``), and that the
    parsed sequence matches ``expected_seq``. On a sequence mismatch the error
    points to the first differing residue with surrounding context.

    Parameters
    ----------
    pdb_path : str
        Path to the single-chain structure file.
    expected_seq : str
        The one-letter sequence the structure is expected to contain.
    use_com : bool, optional
        If ``True``, skip the per-residue ``CA`` check (centre-of-mass beads
        do not require a ``CA`` atom), by default ``False``.

    Returns
    -------
    str
        The validated sequence parsed from the structure.

    Raises
    ------
    ValueError
        If any consistency check fails.
    """
    seq, n_termini, c_termini = seq_from_pdb(pdb_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(pdb_path)

    # one chain only
    if len(u.segments) != 1:
        raise ValueError(
            f"{pdb_path}: expected 1 chain, found {len(u.segments)} ({[s.segid for s in u.segments]})")

    # nothing non-standard leaked through (would also have raised in seq_from_pdb)
    if 'X' in seq:
        raise ValueError(f"{pdb_path}: unmapped residue(s) -> X in parsed sequence")

    # the bead atom CALVADOS will actually read must exist for every residue
    if not use_com:
        n_ca = len(u.select_atoms('name CA'))
        if n_ca != len(seq):
            raise ValueError(
                f"{pdb_path}: {len(seq)} residues but {n_ca} CA atoms (gap residue missing CA?)")

    # sequence identity, with a positional diff on failure
    if seq != expected_seq:
        if len(seq) != len(expected_seq):
            raise ValueError(
                f"{pdb_path}: length {len(seq)} != expected {len(expected_seq)}")
        i = next(k for k in range(len(seq)) if seq[k] != expected_seq[k])
        lo, hi = max(0, i - 10), i + 10
        raise ValueError(
            f"{pdb_path}: sequence mismatch at residue {i+1}\n"
            f"  got     : ...{seq[lo:hi]}...\n"
            f"  expected: ...{expected_seq[lo:hi]}...")
    return seq



#······································································#
# Bead centers
#······································································#

def _as_residues(entity: ResidueGroup | AtomGroup) -> ResidueGroup:
    """Return the residues of an ``AtomGroup`` or ``ResidueGroup`` as a group."""
    return entity.residues if isinstance(entity, AtomGroup) else entity

#······································································#

def get_ca_coords(entity: Residue | ResidueGroup | AtomGroup) -> np.ndarray:
    """Return alpha-carbon coordinates for a residue or a whole chain.

    * For a ``Residue``, returns that residue's ``CA`` coordinate.
    * For a chain (``ResidueGroup``/``AtomGroup``), returns the ``CA``
      coordinate of every residue.

    Parameters
    ----------
    entity : MDAnalysis.core.groups.Residue, ResidueGroup, or AtomGroup
        A single residue or a group of residues / atoms.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` for a residue, or ``(N, 3)`` for a chain of ``N``
        residues (Angstrom).

    Raises
    ------
    TypeError
        If ``entity`` is not a supported MDAnalysis group.
    """
    if isinstance(entity, Residue):
        return entity.atoms.select_atoms('name CA').positions[0]
    elif isinstance(entity, (ResidueGroup, AtomGroup)):
        return np.array([res.atoms.select_atoms('name CA').positions[0] for res in _as_residues(entity)])
    else:
        raise TypeError(f"Function not supported for entity {type(entity)}")

#······································································#

def get_com_coords(entity: Residue | ResidueGroup | AtomGroup) -> np.ndarray:
    """Return centre-of-mass coordinates for a residue or a whole chain.

    * For a ``Residue``, returns that residue's mass-weighted atom centroid.
    * For a chain (``ResidueGroup``/``AtomGroup``), returns the centre of
      mass of every residue.

    Parameters
    ----------
    entity : MDAnalysis.core.groups.Residue, ResidueGroup, or AtomGroup
        A single residue or a group of residues / atoms.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` for a residue, or ``(N, 3)`` for a chain of ``N``
        residues (Angstrom).

    Raises
    ------
    TypeError
        If ``entity`` is not a supported MDAnalysis group.
    """
    if isinstance(entity, Residue):
        return entity.atoms.center_of_mass()
    elif isinstance(entity, (ResidueGroup, AtomGroup)):
        return np.array([res.atoms.center_of_mass() for res in _as_residues(entity)])
    else:
        raise TypeError(f"Function not supported for entity {type(entity)}")



#······································································#
# SLiM contacts curation
#······································································#

def _bead_coords(residue: Residue, mode: Literal['CA', 'COM']) -> np.ndarray:
    """Return a residue's bead coordinate for the given representation.

    Parameters
    ----------
    residue : MDAnalysis.core.groups.Residue
        The residue to reduce to a single bead.
    mode : Literal['CA', 'COM']
        ``'CA'`` for the alpha-carbon position, ``'COM'`` for the
        centre of mass.

    Returns
    -------
    np.ndarray
        Shape ``(3,)`` bead coordinate (Angstrom).
    """
    return get_ca_coords(residue) if mode == 'CA' else get_com_coords(residue)

#······································································#

def bead_distances(
        structure_file: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        pair_residue_idxs: np.ndarray,
        chain1_bead: Literal['CA', 'COM']='COM',
        chain2_bead: Literal['CA', 'COM']='CA',
    ) -> np.ndarray:
    """Compute coarse-grained distances for selected inter-chain residue pairs.

    For each requested pair of residues (one from each chain), reduces the
    two residues to their chosen bead representation (alpha-carbon or
    centre-of-mass) and returns the Euclidean distance between the beads. Used
    to assign equilibrium distances to SLiM contacts.

    Parameters
    ----------
    structure_file : os.PathLike
        Path to the structure file (``.pdb`` or ``.cif``).
    chain1_id : str
        Author chain id of the first chain.
    chain2_id : str
        Author chain id of the second chain.
    pair_residue_idxs : array_like
        Sequence of ``(i, j)`` 0-based residue index pairs, where ``i`` indexes
        into ``chain1_id`` and ``j`` into ``chain2_id``.
    chain1_bead : Literal['CA', 'COM'], optional
        Bead representation for the first chain's residues, by default ``'COM'``.
    chain2_bead : Literal['CA', 'COM'], optional
        Bead representation for the second chain's residues, by default ``'CA'``.

    Returns
    -------
    np.ndarray
        1-D array of inter-bead distances (Angstrom), one per input pair.
    """
    # Load the structure and isolate each chain's residues (in file order)
    universe = mda.Universe(structure_file)
    chain1 = utils.select_chain(universe, chain1_id).residues
    chain2 = utils.select_chain(universe, chain2_id).residues

    # Distance between the chosen beads for each requested residue pair
    pair_eqdists = []
    for (res1_idx, res2_idx) in pair_residue_idxs:
        res1 = chain1[res1_idx]
        res2 = chain2[res2_idx]
        eqdist = np.linalg.norm(_bead_coords(res1, chain1_bead) - _bead_coords(res2, chain2_bead))
        pair_eqdists.append(eqdist)

    return np.array(pair_eqdists)
