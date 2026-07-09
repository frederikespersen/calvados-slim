"""
src.curation
------------------------------------------------------------------------
Functions for curating CALVADOS systems.
"""


import json
import os
import re
import logging
import numpy as np
import MDAnalysis as mda
from typing import Literal

from src.analysis import bsa, alphafold, coarsegrain
from src import utils, constants
from src.calvados.calvados import cfg, sequence


#······································································#
# Simulation-wise curation
#······································································#

class Simulation:
    """Namespace for generating a single CALVADOS simulation directory."""

    @staticmethod
    def _write_config(
        system: dict,
        stride: int,
        frames: int,
        temperature: float,
        ionic_strength: float,
        pH: float,
        sim_dir: str,
        platform: Literal['GPU', 'CPU', 'LOCAL'],
    ) -> None:
        """Write the CALVADOS ``config.yaml`` for one simulation.

        Parameters
        ----------
        system : dict
            A single system entry (see ``data/curation/scheme.json``).
        stride : int
            Write stride (simulation steps between written frames).
        frames : int
            Total number of frames to sample to DCD.
        temperature : float
            Temperature, in Kelvin.
        ionic_strength : float
            Ionic strength, in molar.
        pH : float
            The pH.
        sim_dir : str
            Output directory for the simulation.
        platform : Literal['GPU', 'CPU', 'LOCAL']
            Which platform to specify for the config YAML and submssion scripts.
            ``'GPU'`` -> CUDA (cluster), ``'LOCAL'`` -> OpenCL (local GPU),
            ``'CPU'`` -> CPU.

        """

        config = cfg.Config(
            # Conditions
            sysname = system['ID'],
            box = system['Box sides'],
            temp = temperature, # K
            ionic = ionic_strength, # molar
            pH = pH,
            topol = 'grid', # two molecules -> grid placement

            # Runtime
            wfreq = stride, # dcd writing interval, 1 = 10 fs
            steps = stride*frames, # number of simulation steps
            runtime = 0,
            platform = {'GPU': 'CUDA', 'LOCAL': 'OpenCL', 'CPU': 'CPU'}[platform],
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

    #··································································#

    @staticmethod
    def _write_components(system: dict, fresidues: str, system_dir: str, sim_dir: str) -> int:
        """Write the CALVADOS components file and return the particle count.

        Each chain is added according to its specification: chains without a
        PDB as IDPs (sequence only), chains with a PDB and no AlphaFold data
        with harmonic domain restraints, and chains with AlphaFold data with
        Go restraints derived from the PAE.

        Parameters
        ----------
        system : dict
            A single system entry.
        fresidues : str
            Path to the CALVADOS residue-parameter table (``residues.csv``).
        system_dir : str
            Directory holding the curated input files for ``system``.
        sim_dir : str
            Output directory for the simulation.

        Returns
        -------
        int
            Total number of coarse-grained particles in the system.
        """
        particles = 0
        components = cfg.Components(
            fresidues = os.path.abspath(fresidues)
        )
        for chain in system['Chains']:
            particles += chain['Copies'] * len(chain['Sequence'])

            # IDPs (no structure) carry only their sequence
            if chain['PDB'] == '':
                components.add(
                    name = chain['ID'],
                    molecule_type = 'protein',
                    nmol = chain['Copies'],
                    ffasta = os.path.join(system_dir, f"{chain['ID']}.fasta"),
                    charge_termini = False, # No charged termini; We assume the SLiM is
                    restraint = False,
                    use_com = False,
                )
                continue

            # MDPs use harmonic (experimental) or Go (AlphaFold) restraints
            if chain['PDB AlphaFold data JSON'] == '':
                restraint_kwargs = dict(
                    restraint_type = 'harmonic',
                    fdomains = os.path.join(system_dir, f"{chain['ID']}.yaml"),
                    k_harmonic = 700.,
                )
            else:
                restraint_kwargs = dict(
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
                **restraint_kwargs
            )
        components.write(sim_dir)
        return particles
    
    #··································································#

    @staticmethod
    def _write_submission_scripts(system: dict, particles: int, slurm_scripts_dir: str, sim_dir: str, platform: Literal['GPU', 'CPU', 'LOCAL']) -> None:
        """Instantiate the templated Slurm submission scripts for one simulation.

        Substitutes the job name into every script and, for CPU scripts, fills
        in the estimated CPU and memory requests.

        Parameters
        ----------
        system : dict
            A single system entry; provides the job name (``ID``).
        particles : int
            Total particle count, used to size CPU/memory requests.
        slurm_scripts_dir : str
            Directory of templated Slurm scripts.
        sim_dir : str
            Output directory for the simulation.
        platform : Literal['GPU', 'CPU', 'LOCAL']
            Which platform to specify for the config YAML and submssion scripts.
        """
        for script in os.listdir(slurm_scripts_dir):
            if platform.lower() not in script:
                continue
            # 'local_gpu.sh' also matches the 'gpu' keyword; reserve it for LOCAL
            if platform == 'GPU' and 'local' in script:
                continue
            utils.copy_with_substitution(
                os.path.join(slurm_scripts_dir, script),
                os.path.join(sim_dir, script),
                old='<JOB-NAME>', new=system['ID']
            )
            utils.copy_with_substitution(
                os.path.join(sim_dir, script),
                os.path.join(sim_dir, script),
                old='<MEM>', new=utils.estimate_memory(particles)
            )
                
    
    #··································································#
    
    @staticmethod
    def build(
        system: dict,
        system_dir: str,
        eps_go: float,
        sim_dir: str,
        platform: Literal['GPU', 'CPU', 'LOCAL'],
        stride: int,
        frames: int,
        temperature: float=298.15,
        ionic_strength: float=0.150,
        pH: float=7.0,
        fresidues: str='data/configurations/residues.csv',
        slurm_scripts_dir: str='data/configurations/slurm',
    ) -> None:
        """Write all input files for one runnable CALVADOS simulation.

        Consumes the curated inputs produced by :meth:`System.curate`
        (FASTA/PDB/domain/PAE files and ``specific_interactions.txt`` in
        ``system_dir``) together with run conditions and writes a CALVADOS
        config, components file, the per-run ``specific_interactions.txt``
        (with the SLiM Go well depth filled in), and Slurm submission scripts
        into ``sim_dir``.

        Sampling (``stride`` and ``frames``) is mandatory; thermodynamic
        conditions default to standard physiological values.

        Parameters
        ----------
        system : dict
            A single system entry (see ``data/curation/scheme.json``).
        system_dir : str
            Directory holding the curated input files for ``system``.
        eps_go : float
            Well depth (``ε_Go``) substituted into the SLiM specific
            interactions, in place of the ``<eps_go>`` template placeholder.
        sim_dir : str
            Output directory for the simulation; created files are written here.
        platform : Literal['GPU', 'CPU', 'LOCAL']
            Which platform to specify for the config YAML and submssion scripts.
            ``'GPU'`` -> CUDA (cluster), ``'LOCAL'`` -> OpenCL (local GPU),
            ``'CPU'`` -> CPU.
        stride : int
            Write stride (simulation steps between written frames).
        frames : int
            Total number of frames to sample to DCD.
        temperature : float, optional
            Temperature, in Kelvin, by default 298.15.
        ionic_strength : float, optional
            Ionic strength, in molar, by default 0.150.
        pH : float, optional
            The pH, by default 7.0.
        fresidues : str, optional
            Path to the CALVADOS residue-parameter table, by default
            ``'data/configurations/residues.csv'``.
        slurm_scripts_dir : str, optional
            Directory of templated Slurm scripts to instantiate, by default
            ``'data/configurations/slurm'``.
        """

        # Creating directories, non-overwriting
        os.makedirs(sim_dir, exist_ok=False)
        os.makedirs(os.path.join(sim_dir, 'logs'))

        # Copy the system's specific interactions, filling in the Go well depth
        utils.copy_with_substitution(
            os.path.join(system_dir, 'specific_interactions.txt'),
            os.path.join(sim_dir, 'specific_interactions.txt'),
            old='<eps_go>', new=f'{eps_go}'
        )

        Simulation._write_config(
            system, stride, frames, temperature, ionic_strength, pH,
            sim_dir, platform,
        )
        particles = Simulation._write_components(system, fresidues, system_dir, sim_dir)
        Simulation._write_submission_scripts(system, particles, slurm_scripts_dir, sim_dir, platform)



#······································································#
# System-wise curation
#······································································#

class System:
    """Namespace for curating CALVADOS inputs from a system specification."""

    #··········································································#
    # Logging
    #··········································································#

    @staticmethod
    def _init_logger(system_dir: str) -> logging.Logger:
        """Configure and return the per-system ``setup.log`` logger."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=os.path.join(system_dir, 'setup.log'), filemode='a',
            force=True
        )
        return logging.getLogger(__name__)

    #··································································#
    
    @staticmethod
    def _log_header(log: logging.Logger, message: str) -> None:
        """Write a banner-delimited header line to the log."""
        log.info("·"*80)
        log.info(message)
        log.info("·"*80)

    #··········································································#
    # Per-chain curation
    #··········································································#

    @staticmethod
    def _write_idp_fasta(chain: dict, system_dir: str, log: logging.Logger) -> None:
        """Write an IDP chain's sequence to ``<ID>.fasta``."""
        fasta_path = os.path.join(system_dir, f"{chain['ID']}.fasta")
        log.info(f"\t- Appending sequence to '{fasta_path}'")
        with open(fasta_path, 'w') as fasta:
            fasta.write(f">{chain['ID']}\n")
            fasta.write(f"{chain['Sequence']}\n")

    #··································································#
    
    @staticmethod
    def _write_domain_yaml(chain: dict, system_dir: str, log: logging.Logger) -> None:
        """Write an experimental MDP chain's domain annotation to ``<ID>.yaml``.

        Domain boundaries are shifted from author numbering into the chain's
        own construct frame of reference.
        """
        if len(chain['PDB Domains']) < 1:
            raise ValueError(f"Domain annotation must accompany an experimental structure (chain '{chain['ID']}')!")
        yaml_path = os.path.join(system_dir, f"{chain['ID']}.yaml")
        log.info(f"\t- Writing chain '{chain['PDB chain ID']}' domain annotations to '{yaml_path}'")
        domain_shift = chain['Construct']['start'] - 1
        with open(yaml_path, 'w') as domain_yaml:
            domain_yaml.write(f"{chain['ID']}:")
            for domain in chain['PDB Domains']:
                domain_yaml.write(f"\n- [{domain['start'] - domain_shift},{domain['end'] - domain_shift}]")

    #··································································#
    
    @staticmethod
    def _write_intra_pae(chain: dict, system_dir: str, log: logging.Logger) -> None:
        """Write an AlphaFold MDP chain's intra-chain PAE submatrix to ``<ID>.json``."""
        pae_path = os.path.join(system_dir, f"{chain['ID']}.json")
        log.info(f"\t- Writing chain '{chain['PDB chain ID']}' intra-chain PAEs to '{pae_path}'")
        with open(chain['PDB AlphaFold data JSON'], 'r') as data_json:
            data = json.load(data_json)
        intra_mask = np.array(data['token_chain_ids']) == chain['PDB chain ID']
        pae = np.array(data['pae'])[intra_mask][:, intra_mask]
        with open(pae_path, 'w') as pae_json:
            json.dump({'pae': pae.tolist()}, pae_json)

    #··································································#
    
    @staticmethod
    def _curate_chain(chain: dict, system_dir: str, log: logging.Logger) -> None:
        """Curate one chain's CALVADOS inputs (IDP, experimental MDP, or AlphaFold MDP)."""
        log.info(f"Chain '{chain['ID']}' ('{chain['Name']}')")

        # Validate that the construct range matches the sequence length
        construct_len = chain['Construct']['end'] - chain['Construct']['start'] + 1
        if construct_len != len(chain['Sequence']):
            raise ValueError(
                f"Construct specifications do not agree with sequence length for chain '{chain['ID']}'")

        # Chains without a PDB are treated as IDPs (sequence only)
        if chain['PDB'] == '':
            log.info("\t- Parsing chain as an IDP, as no chain PDB is specified")
            System._write_idp_fasta(chain, system_dir, log)
            log.info("")
            return

        # Chains with a PDB are treated as MDPs: select the chain and clean it.
        # AlphaFold MDPs use Go restraints whose scale is read back from the PDB
        # B-factor column, so the source pLDDT must be preserved on that path.
        is_alphafold = chain['PDB AlphaFold data JSON'] != ''
        log.info(f"\t- Parsing chain as a MDP, as chain PDB '{chain['PDB']}' is specified")
        chain_input_pdb = os.path.join(system_dir, f"{chain['ID']}.pdb")
        log.info(f"\t- Selecting chain '{chain['PDB chain ID']}' and pasting to '{chain_input_pdb}'")
        utils.select_and_clean_chains(chain['PDB'], chain_input_pdb,
                                      chain_ids=[chain['PDB chain ID']], preserve_bfactors=is_alphafold)

        # The cleaned PDB sequence must match the specified construct
        pdb_sequence, _nter, _cter = sequence.seq_from_pdb(chain_input_pdb)
        if pdb_sequence != chain['Sequence']:
            raise ValueError(
                f"The sequence extracted by CALVADOS from '{chain_input_pdb}' does not match the construct specified for chain '{chain['ID']}'")

        # MDPs without AlphaFold data are experimental (domain restraints);
        # MDPs with AlphaFold data use Go restraints derived from the PAE
        if not is_alphafold:
            log.info("\t- Parsing MDP PDB as an experimental structure, as chain PDB AlphaFold data JSON is not specified")
            System._write_domain_yaml(chain, system_dir, log)
        else:
            log.info("\t- Parsing MDP PDB as an AlphaFold structure, as chain PDB AlphaFold data JSON is specified")
            System._write_intra_pae(chain, system_dir, log)

        log.info("")

    #··········································································#
    # Per-SLiM curation
    #··········································································#

    @staticmethod
    def _find_chain(system: dict, chain_id: str, role: str) -> dict:
        """Return the system chain with ``chain_id``, or raise if absent.

        Parameters
        ----------
        system : dict
            A single system entry.
        chain_id : str
            The chain ``ID`` to look up.
        role : str
            Human-readable role (e.g. ``'Acceptor'``) used in the error message.

        Returns
        -------
        dict
            The matching chain entry.

        Raises
        ------
        ValueError
            If no chain in the system has the requested id.
        """
        for chain in system['Chains']:
            if chain['ID'] == chain_id:
                return chain
        raise ValueError(f"SLiM {role} chain '{chain_id}' not found in System '{system['Name']}' chains!")

    #··································································#
    
    @staticmethod
    def _compute_slim_contacts(slim: dict, log: logging.Logger) -> np.ndarray:
        """Compute the Boolean inter-chain contact map for one SLiM.

        Always uses a buried-surface-area contact map (see
        :func:`src.analysis.bsa.buried_contact_map`); for AlphaFold structures
        the map is further intersected with PAE (< 5 Å) and Pinc (> 0.5) filters.

        Parameters
        ----------
        slim : dict
            A single SLiM entry.
        log : logging.Logger
            Logger for progress messages.

        Returns
        -------
        np.ndarray
            Boolean ``(N_acc, N_don)`` contact map in the SLiM PDB frame.
        """
        log.info("\t\t- Calculating buried-surface-area (BSA) contact map from SLiM PDB")
        contacts = bsa.buried_contact_map(slim['PDB'], slim['PDB Acceptor chain ID'], slim['PDB Donor chain ID'])

        if slim['PDB AlphaFold data JSON'] == '':
            log.info("\t\t- Parsing SLiM PDB as an experimental structure, as SLiM PDB AlphaFold data JSON is not specified")
            log.info("\t\t- Defining contacts")
            return contacts

        log.info("\t\t- Parsing SLiM PDB as an AlphaFold structure, as SLiM PDB AlphaFold data JSON is specified")
        log.info("\t\t- Calculating AlphaFold contact metrics (PAE, Pinc)")
        pae = 5 > alphafold.inter_chain_pae(
            slim['PDB AlphaFold data JSON'], slim['PDB Acceptor chain ID'], slim['PDB Donor chain ID']
        )
        pinc = 0.5 < alphafold.inter_chain_pinc(
            slim['PDB'], slim['PDB AlphaFold data JSON'], slim['PDB Acceptor chain ID'], slim['PDB Donor chain ID']
        )
        log.info("\t\t- Defining contacts")
        return contacts & pae & pinc

    #··································································#
    
    @staticmethod
    def _pdb_author_resids(universe: mda.Universe, chain_id: str) -> np.ndarray:
        """Author residue ids of the standard amino acids in a structure chain."""
        standard_resnames = set(constants.residue.name3to1)
        residues = utils.select_chain(universe, chain_id).residues
        return np.array([res.resid for res in residues if res.resname in standard_resnames])

    #··································································#
    
    @staticmethod
    def _authres_index_map(pdb_authres: np.ndarray, construct: dict) -> dict:
        """Map PDB residue indices to construct residue indices via author numbering.

        Parameters
        ----------
        pdb_authres : np.ndarray
            Author residue ids of the resolved residues, in PDB order.
        construct : dict
            The chain's ``Construct`` (``start``/``end`` author residue ids).

        Returns
        -------
        dict
            Mapping from 0-based PDB residue index to 0-based construct index.
        """
        construct_authres = np.arange(construct['start'], construct['end'] + 1)
        pairs = np.argwhere(pdb_authres[:, np.newaxis] == construct_authres[np.newaxis, :])
        return {pdb_idx: construct_idx for pdb_idx, construct_idx in pairs}

    #··································································#
    
    @staticmethod
    def _map_contacts_to_construct(
        slim: dict,
        acceptor_chain: dict,
        donor_chain: dict,
        contact_pdb_idxs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map SLiM-PDB contact indices onto full-construct residue numbering.

        Resolved-residue indices in the SLiM PDB are translated, via shared
        author residue numbering, into the acceptor/donor construct frames.

        Parameters
        ----------
        slim : dict
            A single SLiM entry.
        acceptor_chain, donor_chain : dict
            The system chain entries referenced by the SLiM.
        contact_pdb_idxs : np.ndarray
            ``(N, 2)`` array of ``(acceptor_idx, donor_idx)`` PDB residue index
            pairs (from ``np.argwhere`` on the contact map).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(construct_res_idxs, authres_ids, in_construct)``. The first two
            are ``(N, 2)``: the 0-based construct indices and the author residue
            ids of each contact pair. ``in_construct`` is a length-``N`` boolean
            marking the pairs whose acceptor and donor both fall within their
            construct windows; the construct indices of the rest are meaningless
            (left at zero) and should be dropped by the caller.
        """
        universe = utils.load_structure(slim['PDB'])
        acc_authres_pdb = System._pdb_author_resids(universe, slim['PDB Acceptor chain ID'])
        don_authres_pdb = System._pdb_author_resids(universe, slim['PDB Donor chain ID'])
        acc_map = System._authres_index_map(acc_authres_pdb, acceptor_chain['Construct'])
        don_map = System._authres_index_map(don_authres_pdb, donor_chain['Construct'])

        construct_res_idxs = np.zeros_like(contact_pdb_idxs)
        authres_ids = np.zeros_like(contact_pdb_idxs)
        in_construct = np.zeros(len(contact_pdb_idxs), dtype=bool)
        for i, (acc_idx, don_idx) in enumerate(contact_pdb_idxs):
            authres_ids[i, 0] = acc_authres_pdb[acc_idx]
            authres_ids[i, 1] = don_authres_pdb[don_idx]
            # A contact residue only maps if its author id lies within the
            # construct window; a longer crystallised chain (e.g. tandem SLiM
            # motifs on one peptide) can contact residues beyond it.
            if acc_idx in acc_map and don_idx in don_map:
                construct_res_idxs[i, 0] = acc_map[acc_idx]
                construct_res_idxs[i, 1] = don_map[don_idx]
                in_construct[i] = True
        return construct_res_idxs, authres_ids, in_construct

    #··································································#
    
    @staticmethod
    def _write_chimerax_script(slim: dict, authres_ids: np.ndarray, system_dir: str, scale: float=20) -> None:
        """Write a ChimeraX script selecting a SLiM's contacting residues."""
        acc_resids = ",".join(np.unique(authres_ids[:, 0]).astype(str))
        don_resids = ",".join(np.unique(authres_ids[:, 1]).astype(str))
        pdb_id = os.path.splitext(os.path.basename(slim['PDB']))[0]
        with open(os.path.join(system_dir, f"{slim['ID']}_chimerax.cxc"), 'w') as file:
            if utils.is_pdb(pdb_id):
                file.write(f"open {pdb_id.upper()}\n")
            file.write("set bgColor white\nstyle stick\ncolor bychain\nshow cartoons\nhide\n")
            file.write(f"hide ~/{slim['PDB Acceptor chain ID']},{slim['PDB Donor chain ID']} cartoons\n")
            file.write("sel ")
            file.write(f"/{slim['PDB Acceptor chain ID']}:{acc_resids}")
            file.write(" | ")
            file.write(f"/{slim['PDB Donor chain ID']}:{don_resids}")
            file.write("\nshow sel surfaces")
            file.write("\nview sel")
            file.write(f"\nscale {scale*10:.0f}")


    #··································································#
    
    @staticmethod
    def _curate_slim(slim: dict, system: dict, system_dir: str, fspecific_interactions: str, log: logging.Logger) -> None:
        """Derive one SLiM's specific contacts and append them to the system inputs."""
        log.info(f"SLiM '{slim['ID']}' ('{slim['Name']}')")

        # Resolve the acceptor/donor chains the SLiM refers to
        log.info(f"\t- Verifying that acceptor '{slim['Acceptor ID']}' and donor '{slim['Donor ID']}' chains are found in system")
        acceptor_chain = System._find_chain(system, slim['Acceptor ID'], 'Acceptor')
        donor_chain = System._find_chain(system, slim['Donor ID'], 'Donor')
        if not os.path.exists(slim['PDB']):
            raise FileNotFoundError(f"Specified structure for SLiM '{slim['ID']}' does not exist! ({slim['PDB']})")

        # Define inter-chain contacts and their equilibrium distances
        log.info("\t- Performing contact analysis")
        contacts = System._compute_slim_contacts(slim, log)
        contact_pdb_idxs = np.argwhere(contacts)
        log.info(f"\t\t- Found {contacts.sum()} pairwise contacts")

        log.info("\t\t- Calculating contact distances")
        eqdists = coarsegrain.bead_distances(
            slim['PDB'],
            chain1_id=slim['PDB Acceptor chain ID'], chain1_bead='COM',
            chain2_id=slim['PDB Donor chain ID'], chain2_bead='CA',
            pair_residue_idxs=contact_pdb_idxs
        )
        eqdists /= 10 # Å > nm

        # Map contacts from PDB (resolved-residue) indexing onto the full constructs
        log.info("\t- Mapping residue indices between System and SLiM constructs")
        construct_res_idxs, authres_ids, in_construct = System._map_contacts_to_construct(
            slim, acceptor_chain, donor_chain, contact_pdb_idxs
        )

        # Drop contacts to residues outside either construct window: the
        # simulated chains span only their constructs, so such contacts (e.g.
        # to a flanking residue of an adjacent motif on a longer peptide chain,
        # or a distant crystal-packing partner) cannot be represented.
        log.info(f"\t- Dropping {(~in_construct).sum()} contacts outside the construct windows")
        construct_res_idxs = construct_res_idxs[in_construct]
        eqdists = eqdists[in_construct]
        authres_ids = authres_ids[in_construct]

        # Drop contacts involving explicitly excluded residues
        exclude_mask = (
            np.isin(authres_ids[:, 0], slim['PDB Acceptor exclude'])
            | np.isin(authres_ids[:, 1], slim['PDB Donor exclude'])
        )
        log.info(f"\t- Excluding {exclude_mask.sum()} contacts")
        construct_res_idxs = construct_res_idxs[~exclude_mask]
        eqdists = eqdists[~exclude_mask]
        authres_ids = authres_ids[~exclude_mask]

        # Warn about contacts that are far apart once coarse-grained
        if (eqdists > 2.0).any():
            log.warning(f"\t- There are {(eqdists > 2.0).sum():.0f} contacts with an equilibrium distance >2 nm after coarsegraining!")

        # Append the contacts (1-based residue ids) to the specific interactions file
        log.info(f"\t- Appending contacts to '{fspecific_interactions}'")
        resids = construct_res_idxs + 1
        if len(resids) == 0:
            raise ValueError("No contacts identified!")
        with open(fspecific_interactions, 'a') as file:
            for (acceptor_resid, donor_resid), eqdist in zip(resids, eqdists):
                file.write(f"{acceptor_chain['ID']} {acceptor_resid} | {donor_chain['ID']} {donor_resid} | {eqdist:.4f} <eps_go>\n")

        # ChimeraX script for visual inspection of the contacts
        System._write_chimerax_script(slim, authres_ids, system_dir, system['Box sides'][0])

        log.info("")

    #··········································································#
    # Public interface
    #··········································································#

    @staticmethod
    def curate(
        system: dict,
        output_dir: str='exp/scans/',
        overwrite: bool=False,
    ) -> None:
        """Curate all CALVADOS input files for one system from its spec.

        For each chain, writes a FASTA (IDPs), a cleaned PDB and domain YAML
        (experimental MDPs), or a cleaned PDB and intra-chain PAE JSON
        (AlphaFold MDPs). For each SLiM, derives inter-chain contacts by
        combining a buried-surface-area contact map (with AlphaFold PAE/Pinc
        filters, when AlphaFold data is available), maps the contacts onto the
        full construct numbering, computes COM-CA equilibrium distances, and
        appends them to ``specific_interactions.txt``.

        A ChimeraX selection script is written per SLiM for visual inspection,
        and a ``setup.log`` records the curation steps. All outputs are placed
        in ``<output_dir>/<system ID>/``.

        Parameters
        ----------
        system : dict
            A single system entry (see ``data/curation/scheme.json``).
        output_dir : str, optional
            Parent directory in which the per-system folder is created, by
            default ``'exp/scans/'``.
        overwrite : bool, optional
            Passed as ``exist_ok`` when creating the system directory.

        Raises
        ------
        ValueError
            If construct/sequence lengths disagree, a parsed PDB sequence does
            not match the specified construct, domain annotation is missing for
            an experimental MDP, or a SLiM references an acceptor/donor chain id
            absent from the system.
        FileNotFoundError
            If a SLiM's specified structure does not exist.
        """
        # Set up the system directory and logging
        system_dir = os.path.join(output_dir, system['ID'])
        os.makedirs(system_dir, exist_ok=overwrite)
        log = System._init_logger(system_dir)

        # Curate each chain into CALVADOS input files
        System._log_header(log, f"Parsing chains of '{system['ID']}' system to CALVADOS input")
        for chain in system['Chains']:
            System._curate_chain(chain, system_dir, log)

        # Start the specific interactions file fresh
        fspecific_interactions = os.path.join(system_dir, 'specific_interactions.txt')
        if os.path.exists(fspecific_interactions):
            os.remove(fspecific_interactions)

        # Curate each SLiM's specific contacts
        System._log_header(log, f"Parsing SLiMs of '{system['ID']}' system to CALVADOS input")
        for slim in system['SLiMs']:
            System._curate_slim(slim, system, system_dir, fspecific_interactions, log)
