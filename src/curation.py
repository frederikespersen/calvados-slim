"""
src.curation
------------------------------------------------------------------------
Functions for curating CALVADOS systems.
"""


import json
import os
import logging
import warnings
import numpy as np
import pandas as pd
from Bio import PDB

from src.analysis import smog, alphafold, coarsegrain
from src import utils
from src.calvados.calvados import cfg, sequence


#······································································#
# Simulation-wise curation
#······································································#

class Simulation:

    @staticmethod
    def from_jsons(
        system: dict,
        system_dir: dict,
        conditions: dict,
        sampling: dict,
        slim_go_k: float,
        sim_dir: str,
        slurm_scripts_dir: str='data/configurations/slurm',
    ):
        
        # Copying and editing specific_interactions.txt for system
        utils.replace_copy_file(
            os.path.join(system_dir, 'specific_interactions.txt'),
            os.path.join(sim_dir, 'specific_interactions.txt'),
            ori_str='<k>', new_str=f'{slim_go_k}'
        )

        # Setting up configuration file
        config = cfg.Config(
            # Conditions
            sysname = system['ID'],
            box = system['Box sides'],
            temp = conditions['Temperature'], # K
            ionic = conditions['Ionic strength'], # molar
            pH = conditions['pH'],
            topol = 'grid', # two molecules -> grid placement

            # Runtime
            wfreq = sampling['Stride'], # dcd writing interval, 1 = 10 fs
            steps = sampling['Stride']*sampling['Frames'], # number of simulation steps
            runtime = 0,
            platform = 'CUDA',
            restart = 'checkpoint',
            frestart = 'restart.chk',
            verbose = True,

            # Specific intermolecular interactions
            specific_interactions = True,
            specific_interaction_type = 'go',
            fspecific_interactions = os.path.join('.', 'specific_interactions.txt'),
            specific_cutoff = 2.0, # nm
        )
        config.write(sim_dir)

        # Setting up components files
        particles = 0
        components = cfg.Components(
            fresidues = os.path.abspath(conditions["CALVADOS residue parameters"])
        )
        for chain in system['Chains']:
            particles += chain['Copies'] * len(chain['Sequence'])
            # IDPs
            if chain['PDB'] == '':
                components.add(
                    name = chain['ID'],
                    molecule_type = 'protein',
                    nmol = chain['Copies'],
                    ffasta = os.path.join(system_dir, f"{chain['ID']}.fasta"),
                    charge_termini = 'both',
                    restraint = False,
                    use_com = False,
                )
            # MDPs
            else:
                # PDB
                if chain['PDB AlphaFold data JSON'] == '':
                    mdp_kwarg = dict(
                        restraint_type = 'harmonic',
                        fdomains = os.path.join(system_dir, f"{chain['ID']}.yaml"),
                        k_harmonic = 700.,
                    )
                # AlphaFold
                else:
                    mdp_kwarg = dict(
                        restraint_type = 'go',
                        colabfold = 2,
                        k_go = 15.,
                    )
                components.add(
                    name = chain['ID'],
                    molecule_type = 'protein',
                    nmol = chain['Copies'],
                    charge_termini = 'both',
                    pdb_folder = system_dir,
                    use_com = True,
                    restraint = True,
                    **mdp_kwarg
                )
        components.write(sim_dir)

        # Generating submission script
        for script in os.listdir(slurm_scripts_dir):
            utils.replace_copy_file(
                os.path.join(slurm_scripts_dir, script),
                os.path.join(sim_dir, script),
                ori_str='<JOB-NAME>', new_str=system['ID']
            )
            if 'cpu' in script:
                utils.replace_copy_file(
                    os.path.join(sim_dir, script),
                    os.path.join(sim_dir, script),
                    ori_str='<CPUS-PER-TASK>', new_str=str(utils.estimate_cpus(particles))
                )
                utils.replace_copy_file(
                    os.path.join(sim_dir, script),
                    os.path.join(sim_dir, script),
                    ori_str='<MEM>', new_str=utils.estimate_mem(particles)
                )



#······································································#
# System-wise curation
#······································································#

class System:

    @staticmethod
    def from_json(
        system: dict,
        dir='exp/scans/',
        overwrite: bool=True,
    ):
        
        # Creating directories for system
        system_dir = os.path.join(dir, system['ID'])
        os.makedirs(system_dir, exist_ok=overwrite)

        # Setting up logging
        log_file = os.path.join(system_dir, 'setup.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file, filemode='a',
            force=True
        )
        log = logging.getLogger(__name__)

        # Parsing chains
        log.info("·"*80)
        log.info(f"Parsing chains of '{system['ID']}' system to CALVADOS input")
        log.info("·"*80)

        for chain in system['Chains']:
            log.info(f"Chain '{chain['ID']}' ('{chain['Name']}')")

            # Validating specifications
            construct_len = chain['Construct']['end'] - chain['Construct']['start'] + 1
            sequence_len = len(chain['Sequence'])
            assert construct_len == sequence_len, f"Construct specifications do not agree with sequence length for chain '{chain['ID']}'"
            
            # Treating chains without PDBs as IDPs
            if chain['PDB'] == '':
                log.info(f"\t- Parsing chain as an IDP, as no chain PDB is specified")

                # Appending chain sequence to IDP FASTA
                chain_info_fasta = os.path.join(system_dir, f"{chain['ID']}.fasta")
                log.info(f"\t- Appending sequence to '{chain_info_fasta}'")
                with open(chain_info_fasta, 'w') as fasta:
                    fasta.write(f">{chain['ID']}\n")
                    fasta.write(f"{chain['Sequence']}\n")

            # Treating chains with PDBs as MDPs
            else:
                log.info(f"\t- Parsing chain as a MDP, as chain PDB is specified")

                # Loading chain structure
                chain_input_pdb = os.path.join(system_dir, f"{chain['ID']}.pdb")
                log.info(f"\t- Selecting chain '{chain['PDB chain ID']}' and pasting to '{chain_input_pdb}'")
                utils.pdbfix_select_chains(chain['PDB'], chain_input_pdb, chainids=[chain['PDB chain ID']])

                # Validating the PDB sequence matches expected
                pdb_sequence, _nter, _cter = sequence.seq_from_pdb(chain_input_pdb)
                assert pdb_sequence == chain['Sequence'], f"The sequence extracted by CALVADOS from '{chain_input_pdb}' does not match the construct specified for chain '{chain['ID']}'"

                # Treating MDPs without PAEs as experimental structures
                if chain['PDB AlphaFold data JSON'] == '':
                    log.info(f"\t- Parsing MDP PDB as an experimental structure, as chain PDB AlphaFold data JSON is not specified")

                    # Writing domain annotation
                    assert len(chain['PDB Domains']) >= 1, "Domain annotation must accompany an experimental structure!"

                    # Writing domains to YAML in input directory
                    chain_domain_yaml = os.path.join(system_dir, f"{chain['ID']}.yaml")
                    log.info(f"\t- Writing chain '{chain['PDB chain ID']}' domain annotations to '{chain_domain_yaml}'")
                    domain_shift = chain['Construct']['start'] - 1
                    with open(chain_domain_yaml, 'w') as domain_yaml:
                        domain_yaml.write(f"{chain['ID']}:")
                        for domain in chain['PDB Domains']:
                            domain_yaml.write(f"\n- [{domain['start'] - domain_shift},{domain['end'] - domain_shift}]")

                # Treating MDPs with PAEs as AlphaFold structures
                else:
                    log.info(f"\t- Parsing MDP PDB as an AlphaFold structure, as chain PDB AlphaFold data JSON is specified")
                    
                    # Loading AlphaFold data file
                    chain_input_pae = os.path.join(system_dir, f"{chain['ID']}.json")
                    log.info(f"\t- Writing chain '{chain['PDB chain ID']}' intra-chain PAEs to '{chain_input_pae}'")
                    with open(chain['PDB AlphaFold data JSON'], 'r') as data_json:
                        data = json.load(data_json)

                    # Writing intra chain PAEs as JSON in input directory
                    intra_mask = np.array(data['token_chain_ids']) == chain['PDB chain ID']
                    pae = np.array(data['pae'])
                    pae = pae[intra_mask][:,intra_mask]
                    with open(chain_input_pae, 'w') as pae_json:
                        json.dump({'pae': pae.tolist()}, pae_json)

            log.info("")

        # Preparing TXT file for specific interactions in input directory
        input_specific_interactions = os.path.join(system_dir, 'specific_interactions.txt')
        if os.path.exists(input_specific_interactions):
            os.remove(input_specific_interactions)

        # Parsing SLiMs
        log.info("·"*80)
        log.info(f"Parsing SLiMs of '{system['ID']}' system to CALVADOS input")
        log.info("·"*80)
        
        for slim in system['SLiMs']:
            log.info(f"SLiM '{slim['ID']}' ('{slim['Name']}')")
            
            # Verifying JSON references
            log.info(f"\t- Verifying that acceptor '{slim['Acceptor ID']}' and donor '{slim['Donor ID']}' chains are found in system")
            for chain in system['Chains']:
                if chain['ID'] == slim['Acceptor ID']:
                    acceptor_chain = chain
                    break
            else:
                raise ValueError(f"SLiM Acceptor chain '{slim['Acceptor ID']}' not found in System '{system['Name']}' chains!")
            for chain in system['Chains']:
                if chain['ID'] == slim['Donor ID']:
                    donor_chain = chain
                    break
            else:
                raise ValueError(f"SLiM Donor chain '{slim['Donor ID']}' not found in System '{system['Name']}' chains!")
            
            assert os.path.exists(slim['PDB']), f"Specified structure for SLiM '{slim['ID']}' does not exist! ({slim['PDB']})"

            # Running contact analyses
            log.info("\t- Performing contact analysis")

            # SMOG Shadow Contact Map (SCM) for PDB
            log.info("\t\t- Calculating SMOG Shadow Contact Map (SCM) from SLiM PDB")
            scm = smog.scm(
                slim['PDB'],
                slim['PDB Acceptor chain ID'],
                slim['PDB Donor chain ID']
            )

            # For experimental SLiM PDBs, SCM is sufficient
            if slim['PDB AlphaFold data JSON'] == '':
                log.info(f"\t\t- Parsing SLiM PDB as an experimental structure, as SLiM PDB AlphaFold data JSON is not specified")

                log.info(f"\t\t- Defining contacts")
                contacts = scm

            # For AlphaFold SLiM PDBs, PAE and Pinc is considered along SCM
            else:
                log.info(f"\t\t- Parsing SLiM PDB as an AlphaFold structure, as SLiM PDB AlphaFold data JSON is specified")

                log.info("\t\t- Calculating AlphaFold contact metrics (PAE, Pinc)")
                pae = 5 > alphafold.inter_pae(
                    slim['PDB AlphaFold data JSON'],
                    slim['PDB Acceptor chain ID'],
                    slim['PDB Donor chain ID']
                )
                pinc = 0.5 < alphafold.inter_pinc(
                    slim['PDB'],
                    slim['PDB AlphaFold data JSON'],
                    slim['PDB Acceptor chain ID'],
                    slim['PDB Donor chain ID']
                )

                log.info(f"\t\t- Defining contacts")
                contacts = scm & pae & pinc

            # Inferring residue IDs in SLiM frame of reference
            slim_residxs_pdb = np.argwhere(contacts)
            log.info(f"\t\t- Found {contacts.sum()} pairwise contacts")

            # Calculating equilibrium distances from structure, by COM-CA distances
            log.info(f"\t\t- Calculating contact distances")
            slim_eqdists = coarsegrain.dists(
                slim['PDB'],
                chain1_id=slim['PDB Acceptor chain ID'], chain1_cg='COM',
                chain2_id=slim['PDB Donor chain ID'], chain2_cg='CA',
                pair_residue_idxs=slim_residxs_pdb
            )
            slim_eqdists /= 10 # Å > nm

            # Mapping contacts from PDB indexing (i.e. only resolved residues) to full construct sequence
            # Determining AUTHRES numbering for SLiM PDB
            extension = os.path.splitext(slim['PDB'])[-1]
            if extension == '.pdb':
                parser = PDB.PDBParser(QUIET=True)
            elif extension == '.cif':
                parser = PDB.MMCIFParser(QUIET=True)
            else:
                raise ValueError(f"Only PDB or CIF file parsing supported, not {extension.upper()}!")
            structure = parser.get_structure("protein", slim['PDB'])
            authres_acc_pdb = np.array([res.get_id()[1] for res in structure[0][slim['PDB Acceptor chain ID']].get_residues() if PDB.Polypeptide.is_aa(res, standard=True)])
            authres_don_pdb = np.array([res.get_id()[1] for res in structure[0][slim['PDB Donor chain ID']].get_residues() if PDB.Polypeptide.is_aa(res, standard=True)])
            # Determining AUTHRES numbering from sequence constructs
            authres_acc_seq_first = acceptor_chain['Construct']['start']
            authres_acc_seq_len = acceptor_chain['Construct']['end'] - acceptor_chain['Construct']['start'] + 1
            authres_acc_seq = np.arange(authres_acc_seq_first, authres_acc_seq_first+authres_acc_seq_len)
            authres_don_seq_first = donor_chain['Construct']['start']
            authres_don_seq_len = donor_chain['Construct']['end'] - donor_chain['Construct']['start'] + 1
            authres_don_seq = np.arange(authres_don_seq_first, authres_don_seq_first+authres_don_seq_len)
            # Index mapping between AUTHRES numbering
            residx_map_acc = np.argwhere(authres_acc_pdb[:,np.newaxis] == authres_acc_seq[np.newaxis,:])
            residx_map_acc = {residx_pdb: residx_seq for residx_pdb, residx_seq in residx_map_acc}
            residx_map_don = np.argwhere(authres_don_pdb[:,np.newaxis] == authres_don_seq[np.newaxis,:])
            residx_map_don = {residx_pdb: residx_seq for residx_pdb, residx_seq in residx_map_don}
            # Mapping contacts
            slim_residxs = np.zeros_like(slim_residxs_pdb)
            slim_authresids = np.zeros_like(slim_residxs_pdb)
            for i, (acc_residxs_pdb, don_residxs_pdb) in enumerate(slim_residxs_pdb):
                slim_residxs[i,0] = residx_map_acc[acc_residxs_pdb]
                slim_residxs[i,1] = residx_map_don[don_residxs_pdb]
                slim_authresids[i,0] = authres_acc_pdb[acc_residxs_pdb]
                slim_authresids[i,1] = authres_don_pdb[don_residxs_pdb]
            
            # Removing contacts between excluded residues
            #if slim['PDB Acceptor exclude'] or slim['PDB Donor exclude']:
            slim_acceptor_exclude = np.isin(slim_authresids[:,0], slim['PDB Acceptor exclude'])
            slim_donor_exclude = np.isin(slim_authresids[:,1], slim['PDB Donor exclude'])
            slim_exclude = slim_acceptor_exclude | slim_donor_exclude
            log.info(f"\t- Excluding {slim_exclude.sum()} contacts")
            slim_residxs = slim_residxs[~slim_exclude]
            slim_eqdists = slim_eqdists[~slim_exclude]

            # Raising error if contacts are far away after coarsegraining
            if (slim_eqdists > 2.0).any():
                log.warning(f"\t- There are {(slim_eqdists > 2.0).sum():.0f} contacts with an equilibrium distance >2 nm after coarsegraining!")

            # Mapping numbering from SLiM frame of reference to System frame of reference
            log.info("\t- Mapping residue indices between System and SLiM constructs")
            slim_resids = slim_residxs + 1

            # Appending SLiM contacts to specific interactions TXT
            log.info(f"\t- Appending contacts to '{input_specific_interactions}'")
            with open(input_specific_interactions, 'a') as file:
                for (acceptor_resid, donor_resid), eqdist in zip(slim_resids, slim_eqdists):
                    file.write(f"{acceptor_chain['ID']} {acceptor_resid} | {donor_chain['ID']} {donor_resid} | {eqdist:.4f} <k>\n")

            # Files for visualisation of contacts
            with open(os.path.join(system_dir, f"{slim['ID']}_chimerax.text"), 'w') as file:
                file.write("set bgColor white\nstyle stick\ncolor bychain\nshow cartoons\nhide\n")
                file.write(f"hide ~/{slim['PDB Acceptor chain ID']},{slim['PDB Donor chain ID']} cartoons\n")
                file.write("sel ")
                file.write(f"/{slim['PDB Acceptor chain ID']}")
                file.write(":" + ",".join(np.unique(slim_authresids[:,0][~slim_exclude]).astype(str)))
                file.write(" | ")
                file.write(f"/{slim['PDB Donor chain ID']}")
                file.write(":" + ",".join(np.unique(slim_authresids[:,1][~slim_exclude]).astype(str)))
                file.write("\nview sel")

            log.info("")
