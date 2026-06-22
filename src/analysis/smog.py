"""
src.analysis.smog
------------------------------------------------------------------------
Shadow Contact Map (SCM) calculation via the SMOG2 structure-based model.
"""


import os
import tempfile
import numpy as np
import pandas as pd
import subprocess

from src import utils


#······································································#
# SMOG invocation
#······································································#

def _smog_bin() -> str:
    """Return SMOG2's ``bin`` directory from the ``SMOGBIN`` environment variable.

    Returns
    -------
    str
        Path to SMOG2's ``bin`` directory.

    Raises
    ------
    FileNotFoundError
        If the ``SMOGBIN`` environment variable is not set.
    """
    try:
        return os.environ['SMOGBIN']
    except KeyError:
        raise FileNotFoundError(
            "src.analysis.smog requires the environment variable 'SMOGBIN', "
            "specifying SMOG2's bin directory, to be set so that SMOG analyses "
            "can be run!"
        )


def _run_smog(structure_file: os.PathLike, chain_ids: list[str], smog_bin: str, work_dir: str) -> tuple[str, str]:
    """Select the given chains and run SMOG2 to obtain a contact list.

    Isolates the requested chains (without rebuilding missing loops, whose
    contacts cannot be inferred), then runs ``smog_adjustPDB`` and ``smog2``
    inside ``work_dir``.

    Parameters
    ----------
    structure_file : os.PathLike
        Path to the input structure (``.pdb`` or ``.cif``).
    chain_ids : list[str]
        Author chain ids to retain.
    smog_bin : str
        Path to SMOG2's ``bin`` directory.
    work_dir : str
        Directory in which intermediate and output files are written.

    Returns
    -------
    tuple[str, str]
        ``(adjusted_pdb, contacts_file)`` paths produced by SMOG2.
    """
    raw_pdb = os.path.join(work_dir, 'raw.pdb')
    adjusted_pdb = os.path.join(work_dir, 'adjusted.pdb')
    smog_output = os.path.join(work_dir, 'smog')

    utils.select_and_clean_chains(structure_file, raw_pdb, chain_ids=chain_ids, add_missing=False)

    subprocess.run(
        ['sh', f'{smog_bin}/smog_adjustPDB', '-i', raw_pdb, '-o', adjusted_pdb,
         '-removewater', '-renumber', '-d'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
    )
    subprocess.run(
        ['sh', f'{smog_bin}/smog2', '-i', adjusted_pdb, '-o', smog_output,
         '-dname', smog_output, '-AA'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True,
    )

    return adjusted_pdb, f'{smog_output}.contacts'


#······································································#
# Contact-list reduction
#······································································#

def _parse_atom_table(adjusted_pdb: str) -> pd.DataFrame:
    """Build an atom-serial lookup table from a SMOG-adjusted PDB.

    Parameters
    ----------
    adjusted_pdb : str
        Path to the ``smog_adjustPDB`` output.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``['chain_id', 'res_no', 'chain_no']`` indexed by
        ``(chain_no, atom_id)``; ``chain_no`` is a 1-based chain ordinal
        matching the indexing used in SMOG's ``.contacts`` file.
    """
    records = []
    with open(adjusted_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    atom_id = int(line[6:11].strip())
                    chain_id = line[21].strip()
                    res_no = int(line[22:26].strip())
                    records.append((chain_id, res_no, atom_id))
                except ValueError:
                    continue
    atom_table = pd.DataFrame(records, columns=['chain_id', 'res_no', 'atom_id'])
    atom_table['chain_no'] = pd.factorize(atom_table['chain_id'])[0] + 1
    return atom_table.set_index(['chain_no', 'atom_id'])


def _atomic_to_residue_contacts(
        smog_contacts: pd.DataFrame,
        atom_table: pd.DataFrame,
        chain1_id: str,
        chain2_id: str,
    ) -> np.ndarray:
    """Reduce SMOG's atomic contact list to an inter-chain residue contact map.

    Parameters
    ----------
    smog_contacts : pandas.DataFrame
        SMOG ``.contacts`` rows with columns
        ``['chain_no_1', 'atom_id_1', 'chain_no_2', 'atom_id_2']``.
    atom_table : pandas.DataFrame
        Atom lookup table from :func:`_parse_atom_table`.
    chain1_id, chain2_id : str
        Author chain ids of the two chains of interest.

    Returns
    -------
    np.ndarray
        Boolean ``(N1, N2)`` map of inter-chain residue contacts.
    """
    chain_1_len = atom_table[atom_table['chain_id'] == chain1_id]['res_no'].nunique()
    chain_2_len = atom_table[atom_table['chain_id'] == chain2_id]['res_no'].nunique()
    contacts = np.zeros((chain_1_len, chain_2_len), dtype=bool)

    for _, row in smog_contacts.iterrows():
        chain_id_1, res_no_1 = atom_table.loc[(row['chain_no_1'], row['atom_id_1'])]
        chain_id_2, res_no_2 = atom_table.loc[(row['chain_no_2'], row['atom_id_2'])]
        res_idx_1, res_idx_2 = res_no_1 - 1, res_no_2 - 1

        if (chain_id_1 == chain1_id) and (chain_id_2 == chain2_id):
            contacts[res_idx_1, res_idx_2] = True
        elif (chain_id_1 == chain2_id) and (chain_id_2 == chain1_id):
            contacts[res_idx_2, res_idx_1] = True

    return contacts


#······································································#
# Public interface
#······································································#

def shadow_contact_map(
        structure_file: os.PathLike,
        chain1_id: str,
        chain2_id: str,
    ) -> np.ndarray:
    """Compute the inter-chain Shadow Contact Map between two chains.

    Runs SMOG2's all-atom shadow algorithm on the two selected chains and
    reduces the atomic contact list to a Boolean residue-by-residue map of
    inter-chain contacts. Missing loops are intentionally not reconstructed,
    since contacts cannot be inferred for unresolved regions. Requires the
    ``SMOGBIN`` environment variable to point at SMOG2's ``bin`` directory.

    Parameters
    ----------
    structure_file : os.PathLike
        Path to the input structure (``.pdb`` or ``.cif``).
    chain1_id : str
        Author chain id of the first chain.
    chain2_id : str
        Author chain id of the second chain.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(N1, N2)`` where entry ``[i, j]`` is ``True``
        if residue ``i`` of ``chain1_id`` contacts residue ``j`` of
        ``chain2_id``.

    Raises
    ------
    FileNotFoundError
        If the ``SMOGBIN`` environment variable is not set.
    """
    smog_bin = _smog_bin()

    with tempfile.TemporaryDirectory() as work_dir:
        adjusted_pdb, contacts_file = _run_smog(
            structure_file, [chain1_id, chain2_id], smog_bin, work_dir,
        )
        atom_table = _parse_atom_table(adjusted_pdb)
        smog_contacts = pd.read_csv(
            contacts_file,
            delimiter=' ',
            names=['chain_no_1', 'atom_id_1', 'chain_no_2', 'atom_id_2'],
        )
        return _atomic_to_residue_contacts(smog_contacts, atom_table, chain1_id, chain2_id)
