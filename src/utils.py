"""
src.utils
------------------------------------------------------------------------
Miscellaneous utility helpers used across the curation and analysis pipeline.
"""


import pandas as pd
import numpy as np
import os
import urllib.request
from io import StringIO
import MDAnalysis as mda
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller
import gemmi
import struct
import re
import tempfile
import warnings
from typing import Union


#······································································#
# File download
#······································································#

def download_cif(pdb_id: str, pdb_path: str) -> None:
    """
    Downloads an mmCIF from the RCSB Protein Data Bank.

    Parameters
    ----------
    pdb_id : str
        The 4-character PDB identifier (case-insensitive).
    pdb_dir : str
        The directory path where the downloaded file should be stored.
    """    
    # Ensure target directory exists
    pdb_dir = os.path.dirname(pdb_path)
    assert os.path.exists(pdb_dir), f"Directory '{pdb_dir}' to save {pdb_id.upper()}.cif to does not exist!"
    
    # Construct standard URL and local destination path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    
    # Execute retrieval
    print(f"Retrieving {pdb_id.upper()}.cif from RCSB PDB")
    filename, headers = urllib.request.urlretrieve(url, pdb_path)



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
    format ``name_i bead_i | name_j bead_j | eqdist eps_go``, where bead
    indices are 1-based. Blank lines and lines starting with ``#`` are
    ignored.

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
            s, eps_go = params.split()
            values.append([chain_i, int(bead_i), chain_j, int(bead_j), float(s), eps_go]) # 1-based
    return pd.DataFrame(values, columns=['Acceptor chain', 'Acceptor residue', 'Donor chain', 'Donor residue', 'Equilibrium distance', 'Well depth'])

#······································································#

def get_dcd_frame_count(dcd_path: str) -> Union[int, None]:
    """Gets the number of frames in a DCD-file.

    Parameters
    ----------
    dcd_path : str
        The path to a DCD file.

    Returns
    -------
    n_frames : int
        The number of frames the DCD file, as specified by its header.
    """
    try:
        with open(dcd_path, 'rb') as f:
            # Skip the first record length (4 bytes) and the 'CORD' ID (4 bytes)
            f.seek(8)
            # Read the first 4-byte integer of the header array
            n_frames = struct.unpack('i', f.read(4))[0]
            return n_frames
    except FileNotFoundError:
        return None


#······································································#

def get_log_speed(log_path: str) -> Union[int, None]:
    """Gets the mean speed reported in a OpenMM log file.

    Parameters
    ----------
    log_path : str
        The path to a LOG file.

    Returns
    -------
    ns_day : float
        The mean simulation speed reported in the OpenMM log file, in ns/day.
    """
    try:
        # Load file
        log = pd.read_csv(log_path, delimiter='\t')

        # First row always reports speed of 0; Do not include this.
        log = log.iloc[1:]

        # Return mean
        return log['Speed (ns/day)'].mean()
        
    except:
        return None



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

def _cif_input_for_pdbfixer(cif_path: str) -> str:
    """Return a CIF path safe for PDBFixer, working around two mmCIF quirks.

    PDBFixer reads mmCIF through OpenMM's ``PDBxFile`` and reconstructs missing
    residues by aligning the resolved residues to the ``entity_poly_seq``
    (SEQRES). Two situations break this:

    - **Unknown unit cell.** Some structures (typically NMR) record a
      placeholder cell (``_cell.length_a = '?'``). ``PDBxFile`` parses
      ``_cell`` values as floats and raises ``ValueError`` on ``'?'``. The cell
      is meaningless for the single-chain, coarse-grained use here.
    - **Insertion codes.** When author numbering uses insertion codes (e.g.
      ``113A``), PDBFixer's SEQRES alignment fails and it reconstructs *no*
      missing residues at all -- silently dropping unresolved tags/loops.

    When either is present, a sanitised copy is written to a temporary file:
    the cell is reset to a trivial box and polymer residues are renumbered to
    their SEQRES position (``label_seq``), which drops insertion codes while
    preserving the resolved-vs-SEQRES gap structure. The cleaned per-chain PDB's
    residue numbering is not used downstream (domain annotations live in the
    construct frame; SLiM contacts read the original structure), so this
    renumbering is safe. Files needing neither fix are returned unchanged.

    Parameters
    ----------
    cif_path : str
        Path to the input mmCIF file.

    Returns
    -------
    str
        ``cif_path`` unchanged, or the path to a sanitised temporary copy.
    """
    bad_cell = gemmi.cif.read(cif_path).sole_block().find_value('_cell.length_a') in ('?', '.')
    structure = gemmi.read_structure(cif_path)
    structure.setup_entities()
    has_icode = any(
        res.seqid.icode not in (' ', '\0', '')
        for model in structure for chain in model for res in chain.get_polymer()
    )
    if not (bad_cell or has_icode):
        return cif_path

    if has_icode:
        for model in structure:
            for chain in model:
                for res in chain.get_polymer():
                    if res.label_seq is not None:
                        res.seqid.num = res.label_seq
                        res.seqid.icode = ' '
    if bad_cell:
        structure.cell = gemmi.UnitCell(1, 1, 1, 90, 90, 90)
        structure.spacegroup_hm = 'P 1'

    tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(cif_path))
    structure.make_mmcif_document().write_file(tmp)
    return tmp

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
    is_cif = str(input_path).lower().endswith('.cif')
    # Work around mmCIF quirks (unknown unit cell, insertion codes) that break
    # OpenMM's reader or PDBFixer's missing-residue reconstruction
    fixer = PDBFixer(filename=_cif_input_for_pdbfixer(input_path) if is_cif else input_path)   # Must be SEQRES-bearing file

    # Ensure that `chain_ids` are interpreted as author chains, to match expectations from structure viewers
    if is_cif:
        label2auth = _label_to_auth_map(input_path)
        for c in fixer.topology.chains():
            c.id = label2auth.get(c.id, c.id)
        # Relabel the SEQRES sequences too, else findMissingResidues (which
        # matches chains to sequences by id) sees label ids on the sequences
        # but author ids on the chains and reconstructs no missing residues.
        for s in fixer.sequences:
            s.chainId = label2auth.get(s.chainId, s.chainId)

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

def is_pdb(pdb_id: str) -> bool:
    """Evaluates whether a string conforms to the standard 4-character PDB ID format.
    
    Parameters
    ----------
    pdb_id : str
        The string to evaluate.
    
    Returns
    -------
    bool
        True if the input is a valid legacy PDB ID (one digit from 1-9 
        followed by three alphanumeric characters), False otherwise.
    """
    return bool(re.fullmatch(r'[1-9][A-Za-z0-9]{3}', pdb_id))



#······································································#
# MDAnalysis
#······································································#

def load_structure(structure_file: os.PathLike) -> mda.Universe:
    """Load a PDB or mmCIF structure into an MDAnalysis ``Universe``.

    MDAnalysis has no mmCIF topology parser (confirmed through 2.9.0), so it
    cannot load ``.cif`` files directly. ``.cif`` inputs are therefore converted
    to PDB in-memory with gemmi (which preserves author chain ids and residue
    numbering) before being handed to MDAnalysis; ``.pdb`` inputs are read
    directly.

    Waters and non-polymer heterogens (ions, buffer molecules, ``UNX``
    unknown-density placeholders) are dropped from ``.cif`` inputs before
    conversion. They are never part of the (protein-only) analysis, and their
    high author sequence numbers otherwise trip MDAnalysis' PDB "resid looping"
    heuristic (which adds 10000 whenever a residue number drops by >5000 between
    consecutive atoms), silently offsetting every chain gemmi writes after a
    per-chain water block.

    Parameters
    ----------
    structure_file : str
        Path to the structure file (``.pdb`` or ``.cif``).

    Returns
    -------
    MDAnalysis.Universe
        The loaded structure.
    """
    if os.path.splitext(structure_file)[-1].lower() in ('.cif', '.mmcif'):
        structure = gemmi.read_structure(str(structure_file))
        structure.remove_ligands_and_waters()
        structure.setup_entities()
        with warnings.catch_warnings():
            # A placeholder (1 A^3) unit cell is irrelevant to these
            # non-periodic analyses; silence MDAnalysis' warning about it.
            warnings.filterwarnings('ignore', message='1 A.3 CRYST1 record')
            return mda.Universe(StringIO(structure.make_pdb_string()), format='PDB')
    return mda.Universe(structure_file)

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
