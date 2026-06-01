"""
src.analysis.smog
------------------------------------------------------------------------
For analysis of SMOG analyses.
"""


import numpy as np
import pandas as pd



#······································································#
# Contacts
#······································································#

def scm_res(
        input_pdb,
        output_contacts,
        chainid_1,
        chainid_2
    ) -> np.ndarray:

    # Parsing a residue mapping in SMOG input PDB
    residue_map = []
    with open(input_pdb, 'r') as f:
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
    residue_map = pd.DataFrame(residue_map, columns=['chainid', 'resid', 'atomid'])
    residue_map['chainno'] = pd.factorize(residue_map['chainid'])[0] + 1
    residue_map.set_index(['chainno','atomid'], inplace=True)

    # Loading SMOG output
    smog_output = pd.read_csv(
        output_contacts,
        delim_whitespace=True,
        names=['chainno_1', 'atomid_1', 'chainno_2', 'atomid_2']
    )

    # Initialising empty contact map
    chain_1_len = residue_map[residue_map['chainid'] == chainid_1]['resid'].nunique()
    chain_2_len = residue_map[residue_map['chainid'] == chainid_2]['resid'].nunique()
    smog_contacts = np.zeros((chain_1_len, chain_2_len), dtype=bool)

    # Looking for inter-chain AA contacts
    for _, row in smog_output.iterrows():
        cid_1, rid_1 = residue_map.loc[(row['chainno_1'], row['atomid_1'])]
        cid_2, rid_2 = residue_map.loc[(row['chainno_2'], row['atomid_2'])]

        if (cid_1 == chainid_1) & (cid_2 == chainid_2):
            smog_contacts[rid_1-1, rid_2-1] = True
        elif (cid_1 == chainid_2) & (cid_2 == chainid_1):
            smog_contacts[rid_2-1, rid_1-1] = True

    return smog_contacts
