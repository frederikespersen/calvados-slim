"""
src.pipeline.curate_contacts
------------------------------------------------------------------------
Generates a CSV of two-chain contacts defined from SMOG and PAE
analysis of an AlphaFold3 structure.
"""


import argparse
import pandas as pd
import numpy as np
import os
import subprocess
from src.utils import cif_to_pdb
from src.analysis.alphafold import inter_chain_analysis as af_analysis
from src.analysis.smog import scm_res as shadow_contact_map



#······································································#
# Main
#······································································#

if __name__ == '__main__':

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--af_model_cif', type=str, help="Path to AlphaFold3 CIF model file")
    parser.add_argument('-f', '--af_fulldata_json', type=str, help="Path to AlphaFold3 JSON full data file")
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory")
    parser.add_argument('-1', '--chain1', type=str, help="Chain identifier of first chain for analysis")
    parser.add_argument('-2', '--chain2', type=str, help="Chain identifier of second chain for analysis")
    parser.add_argument('-p', '--pae_max', type=float, help="PAE cutoff [Å] for contact definition (default 5Å)", default=5)
    parser.add_argument('-d', '--dist_max', type=float, help="Minimum inter-residue distance cutoff [Å] for contact definition (default 10Å)", default=10)
    parser.add_argument('-s', '--smog_bin', type=str, help="Path to SMOG bin/ directory", default='~/software/smog-2.5/bin')

    args = parser.parse_args()

    # Curating directory
    output_dir = args.output_dir
    misc_dir = os.path.join(output_dir, "misc")
    os.makedirs(misc_dir, exist_ok=True)

    # Converting input structure to PDB for SMOG compatability
    input_pdb = os.path.join(output_dir, "af_model.pdb")
    adjusted_pdb = os.path.join(misc_dir, "adjusted_af_model.pdb")
    cif_to_pdb(args.af_model_cif, input_pdb)
    smog_bin = os.path.expanduser(args.smog_bin)

    # Preprocessing PDB with SMOG
    subprocess.run([
        'sh', f'{smog_bin}/smog_adjustPDB',
        '-i', input_pdb,
        '-o', adjusted_pdb,
        '-removewater',
        '-insertTER',
        '-d'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Computing SMOG Shadow Contact Map
    subprocess.run([
        'sh', f'{smog_bin}/smog2',
        '-i', adjusted_pdb,
        '-o', misc_dir+'/smog',
        '-dname', misc_dir+'/smog',
        '-AA',
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    smog_output = os.path.join(misc_dir, 'smog.contacts')

    # Running analyses
    pae, pinc, min_dist, avg_plddt = af_analysis(
        args.af_model_cif,
        args.af_fulldata_json,
        args.chain1, args.chain2
    )
    scm = shadow_contact_map(
        adjusted_pdb,
        smog_output,
        args.chain1, args.chain2
    )

    # Defining contacts
    contacts = scm & (pae <= args.pae_max) & (min_dist <= args.dist_max) & (pinc >= 0.5)

    # Saving residue pairs
    contacts_resids = (np.argwhere(contacts) + 1).astype(str)
    contacts_resids_df = pd.DataFrame(contacts_resids, columns=[f'chain_{args.chain1}', f'chain_{args.chain2}'])
    contacts_resids_df.to_csv(os.path.join(output_dir, 'contact_pairs.csv'), index=False)

    # Writing ChimeraX selection statement
    chimeraX_str = f'sel /{args.chain1}:'+','.join(np.unique(contacts_resids[:,0])) \
                 + f' |  /{args.chain2}:'+','.join(np.unique(contacts_resids[:,1]))
    with open(os.path.join(output_dir, 'chimeraX.txt'), 'w') as file:
        file.write(chimeraX_str)

    # Saving map results
    np.save(os.path.join(output_dir, 'contacts.npy'), contacts)
    np.save(os.path.join(output_dir, 'pae.npy'), pae)
    np.save(os.path.join(output_dir, 'pinc.npy'), pinc)
    np.save(os.path.join(output_dir, 'min_dist.npy'), min_dist)
    np.save(os.path.join(output_dir, 'scm.npy'), scm)
    np.save(os.path.join(output_dir, 'avg_plddt.npy'), avg_plddt)
