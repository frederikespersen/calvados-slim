"""
src.utils
------------------------------------------------------------------------
Miscellaneous utility helpers used across the curation and analysis pipeline.
"""


import pandas as pd
import numpy as np
import os
import MDAnalysis as mda
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller
import gemmi


#······································································#
# File manipulation
#······································································#

def copy_with_substitution(src_path: str, dst_path: str, old: str, new: str) -> None:
    """Copy a text file, substituting a placeholder string along the way.

    Parameters
    ----------
    src_path : str
        Path to the source file to read.
    dst_path : str
        Path to write the substituted copy to.
    old : str
        Substring (placeholder) to search for.
    new : str
        Replacement string for every occurrence of ``old``.
    """
    with open(src_path, "r", encoding="utf-8") as file:
        content = file.read()
    updated_content = content.replace(old, new)
    with open(dst_path, "w", encoding="utf-8") as file:
        file.write(updated_content)



#······································································#
# File parsing
#······································································#

def parse_specific_interactions(fspecific_interactions: str) -> pd.DataFrame:
    """Parse a CALVADOS ``specific_interactions.txt`` file into a table.

    Each non-comment line encodes one specific inter-molecular contact in the
    format ``name_i bead_i | name_j bead_j | eqdist k``, where bead indices are
    1-based. Blank lines and lines starting with ``#`` are ignored.

    Parameters
    ----------
    fspecific_interactions : str
        Path to the ``specific_interactions.txt`` file.

    Returns
    -------
    pandas.DataFrame
        One row per contact with columns ``['Acceptor chain', 'Acceptor
        residue', 'Donor chain', 'Donor residue', 'Equilibrium distance',
        'Well depth']``.
    """
    values = []
    with open(fspecific_interactions,'r') as f:
        for line in f.readlines():
            line = line.strip()
            if (not line) or line.startswith('#'):
                continue
            left, right, params = line.split('|')
            chain_i, bead_i = left.split()
            chain_j, bead_j = right.split()
            s, k = params.split()
            values.append([chain_i, int(bead_i), chain_j, int(bead_j), float(s), k]) # 1-based
    return pd.DataFrame(values, columns=['Acceptor chain', 'Acceptor residue', 'Donor chain', 'Donor residue', 'Equilibrium distance', 'Well depth'])



#······································································#
# Structure files
#······································································#

def _label_to_auth_map(cif_path: str) -> dict[str, str]:
    """Map mmCIF ``label_asym_id`` to ``auth_asym_id`` from ``_atom_site``.

    The mapping is a function (1:1 by construction), so the first occurrence
    of each label id suffices.

    Most often, ``label_asym_id`` and ``auth_asym_id`` are the same in a
    mmCIF file. When it isn't, expectations on chain numbering may lead to
    errors. This establishes author numbering ``auth_asym_id`` as the
    only numbering practise.

    Parameters
    ----------
    cif_path : str
        Path to the mmCIF file.

    Returns
    -------
    dict[str, str]
        Mapping from internal label chain ids to author chain ids.
    """
    block = gemmi.cif.read(cif_path).sole_block()
    table = block.find('_atom_site.', ['label_asym_id', 'auth_asym_id'])
    m = {}
    for row in table:
        m.setdefault(row[0], row[1])   # first occurrence suffices; mapping is a function
    return m

#······································································#

def select_and_clean_chains(input_path: str, output_path: str, chain_ids: list[str], add_missing: bool = True) -> None:
    """Isolate selected chains and clean a structure with PDBFixer.

    Keeps only the requested author chains, strips water and heterogens,
    optionally rebuilds missing residues/atoms, replaces non-standard
    residues, removes hydrogens (for SMOG compatibility), and writes the
    result to ``output_path``. For mmCIF input, OpenMM's label chain ids are
    relabelled to author chain ids so that ``chain_ids`` matches what viewers
    such as ChimeraX and the PDB header display.

    Parameters
    ----------
    input_path : str
        Path to the input structure (``.pdb`` or ``.cif``). Must carry SEQRES
        records for missing-residue reconstruction to work.
    output_path : str
        Path to write the cleaned PDB file.
    chain_ids : list[str]
        Author chain ids to retain.
    add_missing : bool, optional
        If ``True`` (default), reconstruct missing residues and heavy atoms;
        if ``False``, leave gaps in place (e.g. for contact analysis from
        experimental structures where unresolved regions must not be modelled).

    Raises
    ------
    ValueError
        If any requested chain id is not present in the input structure.
    """
    fixer = PDBFixer(filename=input_path)   # Must be SEQRES-bearing file

    # Ensure that `chain_ids` are interpreted as author chains, to match expectations from structure viewers
    if str(input_path).lower().endswith('.cif'):
        label2auth = _label_to_auth_map(input_path)
        for c in fixer.topology.chains():
            c.id = label2auth.get(c.id, c.id)

    # Sanity check: requested author chains actually exist
    present = {c.id for c in fixer.topology.chains()}
    missing = [c for c in chain_ids if c not in present]
    if missing:
        raise ValueError(f'Requested author chains {missing} not found in {input_path} '
                         f'(present: {sorted(present)})')

    # Isolate chains (by index, so duplicate ids from relabeling are handled safely)
    chains_to_remove = [i for i, c in enumerate(fixer.topology.chains()) if c.id not in chain_ids]
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

    with open(output_path, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)



#······································································#
# MDAnalysis
#······································································#

def select_chain(universe: mda.Universe, chain_id: str) -> mda.AtomGroup:
    """Select all atoms of a single (author) chain from a ``Universe``.

    Chain identifiers map to different topology attributes depending on the
    reader (``segid`` for mmCIF, ``chainID`` for PDB), so both are tried.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The structure to select from.
    chain_id : str
        Author chain id to select.

    Returns
    -------
    MDAnalysis.AtomGroup
        Atoms belonging to ``chain_id``.

    Raises
    ------
    KeyError
        If no atoms with the requested chain id are found.
    """
    for keyword in ('segid', 'chainID'):
        try:
            atoms = universe.select_atoms(f'{keyword} {chain_id}')
        except Exception:
            continue
        if atoms.n_atoms > 0:
            return atoms
    raise KeyError(f"Chain id '{chain_id}' was not found in the structure!")



#······································································#
# Slurm
#······································································#

def estimate_cpus(num_particles: int) -> int:
    """Heuristic number of CPUs to request for a system of a given size.

    Parameters
    ----------
    num_particles : int
        Total number of coarse-grained particles (beads) in the system.

    Returns
    -------
    int
        Suggested CPU count (4, 8, or 16).
    """
    if num_particles < 150:
        return 4
    elif num_particles < 1000:
        return 8
    else:
        return 16

#······································································#

def estimate_memory(num_particles: int) -> str:
    """Heuristic memory request (rounded up to an even GB) for a system.

    Parameters
    ----------
    num_particles : int
        Total number of coarse-grained particles (beads) in the system.

    Returns
    -------
    str
        Memory request as a Slurm-style string, e.g. ``"8G"``.
    """
    gb = np.ceil((4.0 + (num_particles * 1000) / (1024**3)) * 1.5)
    return f"{gb + (gb % 2):.0f}G"
