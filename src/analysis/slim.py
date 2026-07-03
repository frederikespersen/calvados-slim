"""
src.analysis.slim
------------------------------------------------------------------------
Utils for analysis of SLiM interactions in MD trajectories.
"""


import warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds
from scipy.stats import gaussian_kde
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
# Native-contact order parameter
#······································································#

# Default switching-function parameters, rescaled from Best, Hummer & Eaton's
# atomistic 50 nm⁻¹/1.8 for CALVADOS bead-scale contacts (see
# :func:`_native_contact_fraction`). Used whenever calibration is disabled or
# falls back for lack of near-equilibrium frames (:func:`_calibrate_switching_params`).
_DEFAULT_BETA_Q = 8.0 # nm⁻¹
_DEFAULT_LAMBDA_Q = 1.6
_DEFAULT_BOUND_Q = 0.22

#······································································#

def _native_contact_fraction(
        contact_dists: np.ndarray,
        contact_eqdists: np.ndarray,
        beta_q: float = _DEFAULT_BETA_Q,
        lambda_q: float = _DEFAULT_LAMBDA_Q,
    ) -> np.ndarray:
    """Compute the native contact fraction (``Q``) of specific contacts.

    Scores each frame with a smooth sigmoid switching function per contact
    and averages over all specific contacts, rather than hard-thresholding a
    subset of them. It implements the native-contact order parameter of Best,
    Hummer and Eaton [1]_.

    Parameters
    ----------
    contact_dists : numpy.ndarray
        Per-frame, per-contact distances, shape ``(n_frames, n_contacts)``,
        in nm.
    contact_eqdists : numpy.ndarray
        Equilibrium distance ``r0`` per contact, shape ``(n_contacts,)``, nm.
    beta_q : float, optional
        Sharpness of the switching function, in nm⁻¹; larger values approach
        a hard cutoff, by default ``8.0``. Best, Hummer and Eaton [1]_ use
        ``50`` nm⁻¹ (5 Å⁻¹) for atomistic native contacts; that value is not
        reused here, since one-bead-per-residue CALVADOS contacts fluctuate
        on a much larger (nm, not Å) length scale, and a switching function
        that sharp would reintroduce the hard-cutoff recrossing problem this
        order parameter is meant to avoid. Prefer calibrating this
        empirically per trajectory with :func:`_calibrate_switching_params`
        over hand-picking a value.
    lambda_q : float, optional
        Tolerance on the equilibrium distance -- a contact is "native-like"
        up to ``lambda_q * r0`` -- by default ``1.6`` (vs. ``1.8`` in [1]_; see
        ``beta_q`` for why the atomistic value doesn't transfer directly).

    Returns
    -------
    numpy.ndarray
        Per-frame ``Q`` in ``[0, 1]``, shape ``(n_frames,)``.

    Model
    -----
    For frame ``X`` and native contact ``(i, j)`` with equilibrium distance
    ``r_ij0``, a sigmoid switching function assigns a soft, continuous degree
    of contact formation:

    .. math:: q_{ij}(X) = \\frac{1}{1 + \\exp\\!\\big[\\beta_Q\\,(r_{ij}(X) - \\lambda\\, r_{ij}^0)\\big]}

    ``Q`` is then the mean over all ``N`` native contacts:

    .. math:: Q(X) = \\frac{1}{N} \\sum_{(i,j)} q_{ij}(X)

    so that a frame only scores high if most native contacts are
    simultaneously near-native, rather than any arbitrary fraction of them.

    References
    ----------
    .. [1] Best R.B., Hummer G., Eaton W.A. (2013) Native contacts determine
       protein folding mechanisms in atomistic simulations. Proceedings of
       the National Academy of Sciences, 110(44), 17874-17879.
       https://doi.org/10.1073/pnas.1311599110
    """
    switch = 1.0 / (1.0 + np.exp(beta_q * (contact_dists - lambda_q*contact_eqdists)))
    return switch.mean(axis=1)

#······································································#

def _calibrate_switching_params(
        contact_dists: np.ndarray,
        contact_eqdists: np.ndarray,
        near_native_max_fold: float = 2.0,
        min_near_native_frames: int = 20,
        tolerance_sigmas: float = 3.0,
        beta_q_fallback: float = _DEFAULT_BETA_Q,
        lambda_q_fallback: float = _DEFAULT_LAMBDA_Q,
    ) -> tuple[float, float]:
    """Empirically calibrate ``beta_q``/``lambda_q`` from a trajectory's own contacts.

    Estimates each contact's thermal spread directly from its own
    near-equilibrium frames - those within ``near_native_max_fold`` of that
    contact's ``r0``. ``beta_q`` is set from the typical spread so the
    switching function's transition width matches the system's own observed
    noise, and ``lambda_q`` is set so a contact stays "native-like" up to
    ``tolerance_sigmas`` spreads above its equilibrium distance.

    An *absolute* proximity criterion is used rather than a fixed quantile
    of ranked distances (e.g. "closest 20% of frames"): a quantile always
    returns *something*, even for a contact that never actually forms, in
    which case it measures the spread of a freely diffusing pair rather than
    a formed contact's thermal noise -- yielding a near-zero ``beta_q`` and a
    ``lambda_q`` covering most of the box. Contacts (or whole trajectories) with
    too few genuinely near-equilibrium frames fall back to
    ``beta_q_fallback``/``lambda_q_fallback`` instead.

    Parameters
    ----------
    contact_dists : numpy.ndarray
        Per-frame, per-contact distances, shape ``(n_frames, n_contacts)``,
        in nm.
    contact_eqdists : numpy.ndarray
        Equilibrium distance ``r0`` per contact, shape ``(n_contacts,)``, nm.
    near_native_max_fold : float, optional
        Multiple of ``r0`` within which a frame counts as near-equilibrium
        for that contact, by default ``2.0`` (matching the fold used by the
        legacy hard-cutoff estimator).
    min_near_native_frames : int, optional
        Minimum number of near-equilibrium frames a contact needs before its
        spread is trusted, by default ``20``.
    tolerance_sigmas : float, optional
        Number of estimated spreads above equilibrium at which a contact is
        still considered native-like, by default ``3.0``.
    beta_q_fallback : float, optional
        Value returned if no contact has enough near-equilibrium frames to
        calibrate from, by default ``15.0``.
    lambda_q_fallback : float, optional
        Value returned if no contact has enough near-equilibrium frames to
        calibrate from, by default ``1.4``.

    Returns
    -------
    tuple[float, float]
        ``(beta_q, lambda_q)`` calibrated from the trajectory, in nm⁻¹ and
        dimensionless respectively.

    References
    ----------
    .. [1] Best R.B., Hummer G., Eaton W.A. (2013) Native contacts determine
       protein folding mechanisms in atomistic simulations. Proceedings of
       the National Academy of Sciences, 110(44), 17874-17879.
       https://doi.org/10.1073/pnas.1311599110
    """
    near_native_mask = contact_dists <= near_native_max_fold*contact_eqdists
    valid = near_native_mask.sum(axis=0) >= min_near_native_frames
    if not valid.any():
        warnings.warn(
            "No specific contact came within "
            f"{near_native_max_fold}x its equilibrium distance for at least "
            f"{min_near_native_frames} frames; falling back to "
            f"beta_q={beta_q_fallback}, lambda_q={lambda_q_fallback}. This usually "
            "means the trajectory never approaches a bound state."
        )
        return beta_q_fallback, lambda_q_fallback

    sigma = np.array([
        contact_dists[near_native_mask[:, j], j].std()
        for j in np.flatnonzero(valid)
    ])
    sigma = np.maximum(sigma, 1e-3)  # Floor against near-zero spread on short/rigid contacts

    beta_q = 1.0 / np.median(sigma)
    lambda_q = 1.0 + tolerance_sigmas * np.median(sigma / contact_eqdists[valid])

    return beta_q, lambda_q

#······································································#

def _bound_q(
        Q: np.ndarray,
        n_grid: int = 500,
        min_mode_separation: float = 0.05,
        bound_q_fallback: float = _DEFAULT_BOUND_Q,
    ) -> float:
    """Data-driven bound/unbound dividing point on a ``Q`` trace.

    Evaluates a Gaussian kernel density estimate of the ``Q`` trace on a fine
    grid and returns the density minimum between its two tallest,
    sufficiently separated modes, i.e. the dividing surface at the
    trajectory's own apparent free-energy barrier.
    
    A KDE is used instead of a
    fixed-bin-count histogram because ``Q`` is often heavily imbalanced
    across a scan (some runs never bind, some are essentially always bound,
    some sit in between), and a fixed bin count either over-resolves a
    sparse tail into spurious noise or under-resolves a dominant, narrow
    mode; Silverman's rule instead adapts smoothing to the data itself.
    
    Falls back to ``bound_q_fallback``, ``Q = 0.22``, which has been
    observed to often segregate the ``Q``distribution well, if the trace has
    negligible spread or is effectively unimodal (e.g. an always-bound or
    always-unbound run, where a data-driven barrier isn't a meaningful
    concept).

    Parameters
    ----------
    Q : numpy.ndarray
        Per-frame native contact fraction, as from
        :func:`_native_contact_fraction`.
    n_grid : int, optional
        Number of points on the evaluation grid over ``[0, 1]``, by default
        ``500``.
    min_mode_separation : float, optional
        Minimum ``Q``-distance between two density peaks for them to count
        as distinct modes, by default ``0.05``. Guards against a dominant
        mode's shoulder being mistaken for a second mode on a fine grid.
    bound_q_fallback : float, optional
        Value returned if ``bound_q`` estimation fails.

    Returns
    -------
    float
        The ``Q`` value at the barrier separating the two modes.
    """
    if Q.std() < 1e-6:
        return bound_q_fallback

    grid = np.linspace(0.0, 1.0, n_grid)
    density = gaussian_kde(Q, bw_method="silverman")(grid)

    # Local density maxima, including plateaus running up to either edge
    is_peak = np.zeros(n_grid, dtype=bool)
    is_peak[1:-1] = (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])
    is_peak[0] = density[0] > density[1]
    is_peak[-1] = density[-1] > density[-2]
    peaks = np.flatnonzero(is_peak)
    if peaks.size < 2:
        return bound_q_fallback

    # Tallest peak, then the tallest peak far enough away to be a distinct mode
    peaks = peaks[np.argsort(density[peaks])[::-1]]
    mode_a = peaks[0]
    mode_b = next(
        (p for p in peaks[1:] if abs(grid[p] - grid[mode_a]) >= min_mode_separation),
        None,
    )
    if mode_b is None:
        return bound_q_fallback

    # Deepest point in the density between the two modes
    mode_lo, mode_hi = sorted((mode_a, mode_b))
    valley_offset = np.argmin(density[mode_lo:mode_hi + 1])
    return grid[mode_lo + valley_offset]


#······································································#
# Two-chain SLiM simulations
#······································································#

class TwoChain:
    """Affinity estimation for two-chain (1 acceptor + 1 donor) SLiM runs."""

    @staticmethod
    def _load_contacts(
            top_pdb: str,
            traj_dcd: Union[str, list[str]],
            specific_interactions_txt: str,
        ) -> tuple[np.ndarray, np.ndarray, float]:
        """Load a two-chain run's per-frame specific-contact distances.

        Assumes one SLiM acceptor (chain 0) and one SLiM donor (chain 1)
        with CALVADOS' one bead per residue.

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str or list[str]
            Path to the reference run's trajectory DCD(s).
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt`` defining the
            contacts and their equilibrium distances.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, float]
            ``(contact_dists, contact_eqdists, concentration)``: per-frame,
            per-contact distances, shape ``(n_frames, n_contacts)``, nm;
            each contact's equilibrium distance, shape ``(n_contacts,)``,
            nm; and the single-copy box concentration, in molar.
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

        return contact_dists, contact_eqdists, concentration

    #··································································#

    @staticmethod
    def fit_switching_params(
            top_pdb: str,
            traj_dcd: Union[str, list[str]],
            specific_interactions_txt: str,
            csv_path: Union[str, None]=None,
        ) -> tuple[float, float, float]:
        """Fit ``(beta_q, lambda_q, bound_q)`` once, from a single reference run.

        A scan over many runs of the same system (e.g. varying Gō potential
        well depth) should share one switching-function definition and one
        bound/unbound dividing surface, rather than letting every run
        calibrate its own: a poorly-sampled run (e.g. one that only rarely
        and incidentally touches a higher-``Q`` region) can have
        :func:`_bound_q` lock onto a handful of frames as a spurious
        "bound" mode, producing an affinity outlier uncorrelated with any
        real difference in binding. Call this once on a reference run whose
        ``Q`` distribution is clearly and robustly bimodal, then pass its
        return values into :meth:`get_kd`/:meth:`get_kd_sem` (with
        ``calibrate=False``) for every other run of the same system.

        Parameters
        ----------
        top_pdb : str
            Path to the reference run's topology PDB.
        traj_dcd : str or list[str]
            Path to the reference run's trajectory DCD(s).
            A list of paths can be passed to concatenate trajectories.
        specific_interactions_txt : str
            Path to the reference run's ``specific_interactions.txt``.
        csv_path : str, optional
            If given, write the fitted parameters to this CSV path.

        Returns
        -------
        tuple[float, float, float]
            ``(beta_q, lambda_q, bound_q)`` fitted from the reference run.
        """
        contact_dists, contact_eqdists, _ = TwoChain._load_contacts(
            top_pdb, traj_dcd, specific_interactions_txt,
        )
        beta_q, lambda_q = _calibrate_switching_params(contact_dists, contact_eqdists)
        Q = _native_contact_fraction(contact_dists, contact_eqdists, beta_q, lambda_q)
        bound_q = _bound_q(Q)

        # Optionally saving to CSV
        if csv_path:
            pd.DataFrame({
                'beta_q': [beta_q],
                'lambda_q': [lambda_q],
                'bound_q': [bound_q],
            }).to_csv(csv_path, index=False)

        return beta_q, lambda_q, bound_q

    #··································································#

    @staticmethod
    def _bound_state(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            beta_q: float=_DEFAULT_BETA_Q,
            lambda_q: float=_DEFAULT_LAMBDA_Q,
            bound_q: Union[float, None]=None,
        ) -> tuple[np.ndarray, float]:
        """Per-frame bound state and box concentration for a two-chain run.

        Loads the trajectory and its specific contacts, scores each frame
        with the native contact fraction ``Q`` (:func:`_native_contact_fraction`),
        and flags a frame as bound when ``Q`` exceeds the bound/unbound
        barrier. Assumes one SLiM acceptor (chain 0) and one SLiM donor
        (chain 1).

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str
            Path to the trajectory DCD.
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt`` defining the
            contacts and their equilibrium distances.
        beta_q : float, optional
            Sharpness of the per-contact switching function, in nm⁻¹, by
            default ``15.0``.
        lambda_q : float, optional
            Tolerance factor on each contact's equilibrium distance, by
            default ``1.4``.
        bound_q : float or None, optional
            Bound/unbound dividing point on ``Q``.
            If not specified, it will be calculated from the estimated
            ``Q`` distribution using :func:`_bound_q`.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(bound_state, concentration)`` where ``bound_state`` is a Boolean
            array (one entry per frame) and ``concentration`` is the single-copy
            box concentration in molar.
        """
        contact_dists, contact_eqdists, concentration = TwoChain._load_contacts(
            top_pdb, traj_dcd, specific_interactions_txt,
        )

        # Flag frames as bound: Q above the bound/unbound barrier
        Q = _native_contact_fraction(contact_dists, contact_eqdists, beta_q, lambda_q)
        bound_state = Q > (bound_q if bound_q is not None else _bound_q(Q))

        return bound_state, concentration

    #··································································#

    @staticmethod
    def get_kd(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            beta_q: float=_DEFAULT_BETA_Q,
            lambda_q: float=_DEFAULT_LAMBDA_Q,
            bound_q: Union[float, None]=None,
        ) -> float:
        """Estimate the dissociation constant from a two-chain trajectory.

        Assumes the topology holds exactly one SLiM acceptor (chain 0) and one
        SLiM donor (chain 1). A frame is counted as bound when its native
        contact fraction ``Q`` exceeds the bound/unbound barrier (see
        :meth:`_bound_state`); the bound fraction is converted to ``Kd``
        using the box concentration.

        Parameters
        ----------
        top_pdb : str
            Path to the topology PDB.
        traj_dcd : str
            Path to the trajectory DCD.
        specific_interactions_txt : str
            Path to the run's ``specific_interactions.txt`` defining the
            contacts and their equilibrium distances.
        beta_q : float, optional
            Sharpness of the per-contact switching function, in nm⁻¹, by
            default ``15.0``. Ignored if ``calibrate`` is ``True``. See
            :func:`_native_contact_fraction`.
        lambda_q : float, optional
            Tolerance factor on each contact's equilibrium distance, by
            default ``1.4``. Ignored if ``calibrate`` is ``True``. See
            :func:`_native_contact_fraction`.
        bound_q : float or None, optional
            Bound/unbound dividing point on ``Q``.
            If not specified, it will be calculated from the estimated
            ``Q`` distribution using :func:`_bound_q`.

        Returns
        -------
        float
            Estimated dissociation constant ``Kd`` in molar.
        """
        bound_state, concentration = TwoChain._bound_state(
            top_pdb, traj_dcd, specific_interactions_txt,
            beta_q, lambda_q, bound_q,
        )

        # Kd estimate
        fraction_bound = bound_state.mean()
        Kd = _kd_from_fraction_bound(fraction_bound, concentration)

        return Kd

    #··································································#

    @staticmethod
    def get_kd_sem(
            top_pdb: str,
            traj_dcd: str,
            specific_interactions_txt: str,
            beta_q: float=_DEFAULT_BETA_Q,
            lambda_q: float=_DEFAULT_LAMBDA_Q,
            bound_q: Union[float, None]=None,
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
        beta_q : float, optional
            Sharpness of the per-contact switching function, in nm⁻¹, by
            default ``15.0``. Ignored if ``calibrate`` is ``True``. See
            :func:`_native_contact_fraction`.
        lambda_q : float, optional
            Tolerance factor on each contact's equilibrium distance, by
            default ``1.4``. Ignored if ``calibrate`` is ``True``. See
            :func:`_native_contact_fraction`.
        bound_q : float or None, optional
            Bound/unbound dividing point on ``Q``.
            If not specified, it will be calculated from the estimated
            ``Q`` distribution using :func:`_bound_q`.
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
            beta_q, lambda_q, bound_q,
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
