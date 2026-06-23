"""
src.analysis.slim
------------------------------------------------------------------------
Utils for analysis of SLiM interactions in MD trajectories.
"""


import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds
from typing import Union
import pandas as pd

from src import constants
from src import utils
from src.calvados.calvados.analysis import BlockAnalysis


#······································································#
# Thermodynamics
#······································································#

def _kd_from_fraction_bound(fraction_bound: float, protein_total: float, ligand_total: Union[float,None]=None) -> float:
    """Dissociation constant from a bound fraction and total concentrations.

    Inverts the 1:1 binding equilibrium ``P + L <-> PL`` for the
    dissociation constant ``Kd`` given the fraction of the limiting species
    that is bound and the total protein/ligand concentrations.

    Parameters
    ----------
    fraction_bound : float
        Fraction of complex formed, in ``[0, 1]``.
    protein_total : float
        Total protein (acceptor) concentration, in molar.
    ligand_total : float or None, optional
        Total ligand (donor) concentration in molar; defaults to
        ``protein_total`` when ``None`` (equimolar two-chain box).

    Returns
    -------
    float
        Dissociation constant ``Kd`` in molar (``inf`` if nothing is bound,
        ``0`` if fully bound).
    """
    ligand_total = protein_total if ligand_total is None else ligand_total
    fb = fraction_bound
    if fb <= 0.0: return np.inf
    if fb >= 1.0: return 0
    Kd = protein_total * fb + ligand_total / fb - protein_total - ligand_total
    return Kd

#······································································#

def dG(Kd: float, T: float = 298.15) -> float:
    """Binding free energy from a dissociation constant.

    Computes ``ΔG = R·T·ln(Kd / C°)`` relative to a standard state of
    ``C° = 1 M``. Since ``C° = 1 M`` is unity, ``Kd`` is simply passed in
    molar (``ΔG = R·T·ln(Kd)`` with ``Kd`` numerically in M).

    Parameters
    ----------
    Kd : float
        Dissociation constant in molar.
    T : float, optional
        Temperature in Kelvin, by default ``298.15`` (25 °C).

    Returns
    -------
    float
        Binding free energy in kJ/mol (``-inf`` when ``Kd`` is ``0``).
    """
    if Kd == 0:
        return -np.inf
    return constants.R * T * np.log(Kd)



#······································································#
# Two-chain SLiM simulations
#······································································#

class TwoChain:
    """Affinity estimation for two-chain (1 acceptor + 1 donor) SLiM runs."""

    @staticmethod
    def _bound_state(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            contact_eqdist_fold: float=2,
            contact_fraction_bound: float=0.1,
        ) -> tuple[np.ndarray, float]:
        """Per-frame bound state and box concentration for a two-chain run.

        Loads the trajectory and its specific contacts, then flags each frame
        as bound when more than ``contact_fraction_bound`` of the contacts are
        within ``contact_eqdist_fold`` times their equilibrium distance.
        Assumes one SLiM acceptor (chain 0) and one SLiM donor (chain 1).

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str
            Path to the trajectory DCD.
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt`` defining the
            contacts and their equilibrium distances.
        contact_eqdist_fold : float, optional
            Multiple of each contact's equilibrium distance below which the
            contact counts as formed, by default ``2``.
        contact_fraction_bound : float, optional
            Minimum fraction of formed contacts for a frame to count as bound,
            by default ``0.1``.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(bound_state, concentration)`` where ``bound_state`` is a Boolean
            array (one entry per frame) and ``concentration`` is the single-copy
            box concentration in molar.
        """
        # Load contacts and trajectory. The DCDReader emits a 3.0-deprecation
        # notice about making independent timesteps; it doesn't affect this
        # estimator (positions are read fresh per frame), so silence just it.
        specific_interactions = utils.parse_specific_interactions(specific_interactions_txt)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="DCDReader currently makes independent timesteps")
            u = mda.Universe(top_pdb, traj_dcd)

        # This estimator assumes a two-chain box (acceptor = chain 0, donor =
        # chain 1) with CALVADOS' one bead per residue; validate before indexing.
        if u.segments.n_segments != 2:
            raise ValueError(
                f"Expected a two-chain topology (acceptor + donor), but found "
                f"{u.segments.n_segments} chain(s) in '{top_pdb}'."
            )
        for chain_idx, segment in enumerate(u.segments):
            if segment.atoms.n_atoms != segment.residues.n_residues:
                raise ValueError(
                    f"Chain {chain_idx} of '{top_pdb}' is not one bead per residue "
                    f"({segment.atoms.n_atoms} beads, {segment.residues.n_residues} residues)."
                )

        # Bead atom indices per chain (one bead per residue, so the 0-based
        # residue index within a chain indexes its bead atoms directly).
        acceptor_atoms = u.segments[0].atoms.indices
        donor_atoms = u.segments[1].atoms.indices

        # Contacting bead-atom index pairs and their equilibrium distances (nm)
        contact_indices = specific_interactions.iloc[:, [1, 3]].astype(int).values - 1
        contact_eqdists = specific_interactions.iloc[:, 4].astype(float).values
        pair_a = acceptor_atoms[contact_indices[:, 0]]
        pair_b = donor_atoms[contact_indices[:, 1]]

        # Per-frame minimum-image distances (Å), converted to nm
        contact_dists = np.empty((u.trajectory.n_frames, len(pair_a)))
        box_volume_L = None
        for f, ts in enumerate(u.trajectory):
            if f == 0:
                # NVT ensemble: box volume is constant, so frame 0 suffices
                box_volume_L = ts.volume * 1e-27  # Å³ -> L
            positions = u.atoms.positions
            contact_dists[f] = calc_bonds(positions[pair_a], positions[pair_b], box=ts.dimensions)
        contact_dists /= 10.0  # Å -> nm

        # Concentration from the box volume (one copy of each chain)
        concentration = 1 / (constants.N_A * box_volume_L) # M

        # Flag frames as bound
        bound_state = (contact_dists <= contact_eqdists*contact_eqdist_fold).mean(axis=1) > contact_fraction_bound

        return bound_state, concentration

    @staticmethod
    def get_kd(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            contact_eqdist_fold: float=2,
            contact_fraction_bound: float=0.1,
        ) -> float:
        """Estimate the dissociation constant from a two-chain trajectory.

        Assumes the topology holds exactly one SLiM acceptor (chain 0) and one
        SLiM donor (chain 1). A frame is counted as bound when more than
        ``contact_fraction_bound`` of the specific contacts are within
        ``contact_eqdist_fold`` times their equilibrium distance; the bound
        fraction is converted to ``Kd`` using the box concentration.

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str
            Path to the trajectory DCD.
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt`` defining the
            contacts and their equilibrium distances.
        contact_eqdist_fold : float, optional
            Multiple of each contact's equilibrium distance below which the
            contact counts as formed, by default ``2``.
        contact_fraction_bound : float, optional
            Minimum fraction of formed contacts for a frame to count as bound,
            by default ``0.1``.

        Returns
        -------
        float
            Estimated dissociation constant ``Kd`` in molar.
        """
        bound_state, concentration = TwoChain._bound_state(
            top_pdb, traj_dcd, specific_interactions_txt,
            contact_eqdist_fold, contact_fraction_bound,
        )

        # Kd estimate
        fraction_bound = bound_state.mean()
        Kd = _kd_from_fraction_bound(fraction_bound, concentration)

        return Kd
    
    @staticmethod
    def get_kd_sem(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            contact_eqdist_fold: float=2,
            contact_fraction_bound: float=0.1,
            n_sem: float=1,
            csv_path: Union[str, None]=None,
        ) -> tuple[float, float, float]:
        """Estimate ``Kd`` with a blocking-based confidence interval.

        Like :meth:`get_kd`, but propagates the standard error of the bound
        fraction (obtained by reblocking the per-frame bound state with
        CALVADOS's ``BlockAnalysis``) into lower/upper ``Kd`` bounds at
        ``n_sem`` standard errors. Assumes one SLiM acceptor (chain 0) and one
        SLiM donor (chain 1).

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str
            Path to the trajectory DCD.
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt``.
        contact_eqdist_fold : float, optional
            Multiple of each contact's equilibrium distance below which the
            contact counts as formed, by default ``2``.
        contact_fraction_bound : float, optional
            Minimum fraction of formed contacts for a frame to count as bound,
            by default ``0.1``.
        n_sem : float, optional
            Number of standard errors used for the bounds, by default ``1``.
        csv_path : str, optional
            If given, write the mean and bounds to this CSV path.

        Returns
        -------
        tuple[float, float, float]
            ``(Kd, Kd_lower, Kd_upper)`` in molar, where the bounds correspond
            to ``±n_sem`` standard errors on the bound fraction.
        """
        bound_state, concentration = TwoChain._bound_state(
            top_pdb, traj_dcd, specific_interactions_txt,
            contact_eqdist_fold, contact_fraction_bound,
        )

        # Kd blocking estimates
        ba = BlockAnalysis(bound_state)
        ba.SEM()
        fraction_bound = ba.av
        fraction_bound_sem = ba.sem
        Kd = _kd_from_fraction_bound(fraction_bound, concentration)
        Kd_upper = _kd_from_fraction_bound(max(0, fraction_bound - n_sem*fraction_bound_sem), concentration)
        Kd_lower = _kd_from_fraction_bound(min(1, fraction_bound + n_sem*fraction_bound_sem), concentration)

        # Optionally saving to CSV
        if csv_path:
            pd.DataFrame({
                'Mean Kd': [Kd],
                'Mean Kd -1 SEM': [Kd_lower],
                'Mean Kd +1 SEM': [Kd_upper]
            }).to_csv(csv_path, index=False)

        return Kd, Kd_lower, Kd_upper
    