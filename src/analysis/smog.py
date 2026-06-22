"""
src.analysis.smog
------------------------------------------------------------------------
For analysis of SMOG analyses.
"""


from genericpath import exists
import os
import tempfile
import numpy as np
import pandas as pd
import subprocess
from Bio import PDB

from src import utils


#······································································#
# Contacts
#······································································#

def scm(
        input_structure: os.PathLike,
        chain1_id: str,
        chain2_id: str,
    ) -> np.ndarray:

    try:
        smog_bin = os.environ['SMOGBIN']
    except KeyError:
        raise FileNotFoundError(f"{__name__}.scm() requires "\
        "the environment variable 'SMOGBIN' specifying SMOG's bin"\
        "to be specifiedm such that SMOG analyses can be run!")

    # Creating temporary directory for SMOG run
    with tempfile.TemporaryDirectory() as temp_dir:

        raw_pdb = os.path.join(temp_dir, 'raw.pdb')
        adjusted_pdb = os.path.join(temp_dir, 'adjusted.pdb')

        # Selecting chains of interest (without adding missing loops, for which contacts are not deducible) and saving to PDB
        utils.pdbfix_select_chains(input_structure, raw_pdb, chainids=[chain1_id, chain2_id], add_missing=False)

        # Running SMOG PDB adjustment and SMOG analysis
        smog_output = os.path.join(temp_dir, 'smog')
        subprocess.run([
            'sh', f'{smog_bin}/smog_adjustPDB',
            '-i', raw_pdb,
            '-o', adjusted_pdb,
            '-removewater',
            '-renumber',
            '-d'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        subprocess.run([
            'sh', f'{smog_bin}/smog2',
            '-i', adjusted_pdb,
            '-o', smog_output,
            '-dname', smog_output,
            '-AA',
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )

        # Parsing a residue mapping in SMOG input PDB
        residue_map = []
        with open(adjusted_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        atom_serial = int(line[6:11].strip())
                        res_name = line[17:20].strip()
                        chain_id = line[21].strip()
                        res_seq = int(line[22:26].strip())
                        residue_map += [(chain_id, res_seq, atom_serial)]
                    except ValueError:
                        continue
        residue_map = pd.DataFrame(residue_map, columns=['chainid', 'resno', 'atomid'])
        residue_map['chainno'] = pd.factorize(residue_map['chainid'])[0] + 1
        residue_map.set_index(['chainno','atomid'], inplace=True)

        # Loading SMOG output
        smog_contacts = pd.read_csv(
            f'{smog_output}.contacts',
            delimiter=' ',
            names=['chainno_1', 'atomid_1', 'chainno_2', 'atomid_2']
        )

        # Initialising empty contact map
        chain_1_len = residue_map[residue_map['chainid'] == chain1_id]['resno'].nunique()
        chain_2_len = residue_map[residue_map['chainid'] == chain2_id]['resno'].nunique()
        contacts = np.zeros((chain_1_len, chain_2_len), dtype=bool)

        # Looking for inter-chain AA contacts
        for _, row in smog_contacts.iterrows():
            cid_1, rno_1 = residue_map.loc[(row['chainno_1'], row['atomid_1'])]
            cid_2, rno_2 = residue_map.loc[(row['chainno_2'], row['atomid_2'])]
            ridx_1 = rno_1 - 1
            ridx_2 = rno_2 - 1

            if (cid_1 == chain1_id) & (cid_2 == chain2_id):
                contacts[ridx_1, ridx_2] = True
            elif (cid_1 == chain2_id) & (cid_2 == chain1_id):
                contacts[ridx_2, ridx_1] = True

        return contacts
