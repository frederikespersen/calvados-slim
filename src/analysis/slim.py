"""
src.analysis.slim
------------------------------------------------------------------------
Utils for analysis of SLiM interactions in MD trajectories.
"""


import os
import numpy as np
import mdtraj as md
from typing import Union
import pandas as pd 

from src import constants
from src import utils
from src.analysis.blocking import BlockAnalysis


#······································································#
# Thermodynamics
#······································································#

def _calc_Kd(fraction_bound: float, Pt: float, Lt: Union[float,None]=None) -> float:
    Lt = Pt if Lt is None else Lt
    fb = fraction_bound
    if fb <= 0.0: return np.inf
    if fb >= 1.0: return 0
    Kd = Pt * fb + Lt / fb - Pt - Lt
    return Kd



#······································································#
# Two-chain SLiM simulations
#······································································#

class TwoChain:

    @staticmethod
    def get_Kd(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            contact_eqdist_fold: float=2,
            contact_fraction_bound: float=0.1,
        ) -> float:
        """
        Assumes a SLiM acceptor (first chain in topology, one copy) and a SLiM doner (second chain in topology, one copy)
        """

        # Load contacts
        specific_interactions = utils.parse_specific_interactions(specific_interactions_txt)

        # Load trajectory
        traj = md.load(traj_dcd, top=top_pdb)

        # Calculate concentration from topology
        chains = 1
        volume = traj.unitcell_volumes[0] * 1e-24 # L
        concentration = (chains) / (constants.N_A * volume) # M

        # Calculate contact fraction in traj
        contact_indices = specific_interactions.iloc[:,[1,3]].astype(int).values - 1
        contact_eqdists = specific_interactions.iloc[:,4].astype(float).values
        acceptor_resids = traj.topology.select('chainid 0')
        donor_resids = traj.topology.select('chainid 1')
        contact_resiuds = np.vstack((acceptor_resids[contact_indices[:,0]], donor_resids[contact_indices[:,1]])).T
        contact_dists = md.compute_distances(traj, contact_resiuds, periodic=True)

        # Kd estimate
        bound_state = (contact_dists <= contact_eqdists*contact_eqdist_fold).mean(axis=1) > contact_fraction_bound
        fraction_bound = bound_state.mean()
        Kd = _calc_Kd(fraction_bound, concentration)

        return Kd
    
    @staticmethod
    def get_Kd_sem(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            contact_eqdist_fold: float=2,
            contact_fraction_bound: float=0.1,
            n_SE: float=1,
            to_csv: str=None,
        ) -> tuple[float, float, float]:
        """
        Assumes a SLiM acceptor (first chain in topology, one copy) and a SLiM doner (second chain in topology, one copy)
        """

        # Load contacts
        specific_interactions = utils.parse_specific_interactions(specific_interactions_txt)

        # Load trajectory
        traj = md.load(traj_dcd, top=top_pdb)

        # Calculate concentration from topology
        chains = 1
        volume = traj.unitcell_volumes[0] * 1e-24 # L
        concentration = (chains) / (constants.N_A * volume) # M

        # Calculate contact fraction in traj
        contact_indices = specific_interactions.iloc[:,[1,3]].astype(int).values - 1
        contact_eqdists = specific_interactions.iloc[:,4].astype(float).values
        acceptor_resids = traj.topology.select('chainid 0')
        donor_resids = traj.topology.select('chainid 1')
        contact_resiuds = np.vstack((acceptor_resids[contact_indices[:,0]], donor_resids[contact_indices[:,1]])).T
        contact_dists = md.compute_distances(traj, contact_resiuds, periodic=True)

        # Kd blocking estimates
        bound_state = (contact_dists <= contact_eqdists*contact_eqdist_fold).mean(axis=1) > contact_fraction_bound
        ba = BlockAnalysis(bound_state)
        ba.SEM()
        fraction_bound = ba.av
        fraction_bound_sem = ba.sem
        Kd = _calc_Kd(fraction_bound, concentration)
        Kd_upper = _calc_Kd(max(0, fraction_bound - n_SE*fraction_bound_sem), concentration)
        Kd_lower = _calc_Kd(min(1, fraction_bound + n_SE*fraction_bound_sem), concentration)

        # Optionally saving to CSV
        if to_csv:
            pd.DataFrame({
                'Mean Kd': [Kd],
                'Mean Kd -1 SEM': [Kd_lower],
                'Mean Kd +1 SEM': [Kd_upper]
            }).to_csv(to_csv, index=False)

        return Kd, Kd_lower, Kd_upper
    