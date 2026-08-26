"""
src.analysis.mbar
------------------------------------------------------------------------
Preliminary MBAR analysis along the Gō well depth ``eps_go``.

The Gō potential is linear in ``eps_go``: a configuration ``X`` has Gō energy
``eps_go * G(X)`` with the collective variable ``G`` defined in
:func:`go_collective_variable`. Writing the total reduced potential (``NVT``) of
sample ``X`` at well depth ``eps_go`` as

.. math:: u_{\\varepsilon}(X) = \\beta\\,\\big[U_0(X) + \\varepsilon\\, G(X)\\big],
   \\qquad \\beta = 1/(R T),

the ``eps_go``-independent CALVADOS energy ``U₀`` is a per-sample constant
common to every state, and MBAR is invariant to such shifts: it cancels in every
reduced-potential difference ``u_{\\varepsilon'} - u_{\\varepsilon} =
\\beta(\\varepsilon' - \\varepsilon)\\,G``. Reweighting between ``eps_go`` states
therefore depends only on the single, cheaply computable coordinate ``G(X)``,
making MBAR here a 1-D reweighting along ``eps_go``.

The same linearity extends the reweighting to a second axis. ``G`` is a sum over
contacts, ``G = Σ_{i∈E} g_i``, so deleting a subset ``S`` of the Gō edges is just
another reduced potential of the same samples, ``u(X) = βε\\,(G(X) - G_S(X))``,
and the sampled trajectories can be re-expressed on an edited contact graph
without re-simulating. That two-dimensional (``eps_go`` × edge-subset)
reweighting keeps the parent's bound indicator, so it answers how the free energy
of forming the original native pose changes as Gō terms are removed. It cannot
create sampling: the log-weight variance grows as ``(βε)² Var(G_S)``, so
``N_eff/N ≈ exp[-(βε)² Var(G_S)]`` and aggressive edits collapse the estimate,
which the reported ``N_eff_minor`` makes visible.

Trajectory files are passed as explicit paths (see :func:`state_samples`), so no
simulation-directory layout is assumed here.

MBAR is implemented here (``pymbar`` is not a project dependency).

References
----------
.. [Shirts2008] Shirts, M.R. & Chodera, J.D. (2008) Statistically optimal
   analysis of samples from multiple equilibrium states. J. Chem. Phys. 129,
   124105. https://doi.org/10.1063/1.2978177
.. [Klimovich2015] Klimovich, P.V., Shirts, M.R. & Mobley, D.L. (2015)
   Guidelines for the analysis of free energy calculations. J. Comput.-Aided
   Mol. Des. 29, 397-411. https://doi.org/10.1007/s10822-015-9840-9
.. [Kish1965] Kish, L. (1965) Survey Sampling. Wiley. (effective sample size /
   design effect)
.. [Clementi2000] Clementi, C., Nymeyer, H. & Onuchic, J.N. (2000) Topological
   and energetic factors ... a structure-based (Gō) model. J. Mol. Biol. 298,
   937-953. https://doi.org/10.1006/jmbi.2000.3693
.. [Jovanovic1997] Jovanovic, B.D. & Levy, P.S. (1997) A look at the rule of
   three. Am. Stat. 51, 137-139. https://doi.org/10.1080/00031305.1997.10473947
.. [Kunsch1989] Künsch, H.R. (1989) The jackknife and the bootstrap for general
   stationary observations. Ann. Stat. 17, 1217-1241.
   https://doi.org/10.1214/aos/1176347265
.. [Politis1992] Politis, D.N. & Romano, J.P. (1992) A circular block-resampling
   procedure for stationary data. In Exploring the Limits of Bootstrap, Wiley.
"""


import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import logsumexp

from src import constants
from src.analysis import slim


#······································································#
# Constants
#······································································#

# Gō / specific-interaction cutoff (``src/curation.py``: specific_cutoff), nm.
SPECIFIC_CUTOFF = 2.0

# Effective-sample-size floor below which a reweighted estimate is untrustworthy.
DEFAULT_NEFF_MIN = 50.0


#······································································#
# Gō collective variable
#······································································#

def go_contact_wells(
        contact_dists: np.ndarray,
        contact_eqdists: np.ndarray,
        cutoff: float = SPECIFIC_CUTOFF,
    ) -> np.ndarray:
    """Per-frame, per-contact Gō 12-10 well values ``g_i(X)``.

    The resolved form of :func:`go_collective_variable`, which is their sum over
    contacts. Keeping the contacts separate allows the Gō graph to be
    edited after the fact: any subset ``E'`` of the edges has collective variable
    ``G_{E'} = Σ_{i∈E'} g_i`` computable from the same frames, which is the basis
    of :func:`reweight_subgraph_curve`.

    Parameters
    ----------
    contact_dists : numpy.ndarray
        Per-frame, per-contact distances, shape ``(n_frames, n_contacts)``, nm.
    contact_eqdists : numpy.ndarray
        Equilibrium distance per contact, shape ``(n_contacts,)``, nm.
    cutoff : float, optional
        Distance beyond which a contact contributes nothing, by default
        :data:`SPECIFIC_CUTOFF`.

    Returns
    -------
    numpy.ndarray
        Well value per frame and contact, shape ``(n_frames, n_contacts)``, each
        in ``[-1, 0]`` up to the repulsive core (``-1`` at ``r_ij = s_ij``).
    """
    sr = contact_eqdists[None, :] / contact_dists
    well = 5.0 * sr**12 - 6.0 * sr**10
    return np.where(contact_dists <= cutoff, well, 0.0)


def go_collective_variable(
        contact_dists: np.ndarray,
        contact_eqdists: np.ndarray,
        cutoff: float = SPECIFIC_CUTOFF,
    ) -> np.ndarray:
    """Per-frame Gō collective variable ``G(X)``.

    Sums each specific contact's structure-based (Gō) 12-10 well shape,
    truncated at ``cutoff`` [Clementi2000]_:

    .. math:: G(X) = \\sum_{(i,j)}
        \\left[\\, 5\\left(\\tfrac{s_{ij}}{r_{ij}}\\right)^{12}
        - 6\\left(\\tfrac{s_{ij}}{r_{ij}}\\right)^{10} \\right]\\,
        \\mathbb{1}\\!\\left[r_{ij} \\le r_c\\right],

    where ``s_ij`` is the contact's equilibrium distance and ``r_ij`` its
    instantaneous distance. Each term equals ``-1`` at its minimum
    (``r_ij = s_ij``), so the Gō energy at well depth ``eps_go`` is exactly
    ``eps_go * G``. ``G ≈ 0`` when unbound (contacts beyond ``cutoff``) and
    ``G ≈ -N_contacts`` when fully bound. Because MBAR needs only energy
    differences between ``eps_go`` states, ``G`` is the sole quantity required
    (see the module docstring).

    Parameters
    ----------
    contact_dists : numpy.ndarray
        Per-frame, per-contact distances, shape ``(n_frames, n_contacts)``, nm.
    contact_eqdists : numpy.ndarray
        Equilibrium distance per contact, shape ``(n_contacts,)``, nm.
    cutoff : float, optional
        Distance beyond which a contact contributes nothing, by default
        :data:`SPECIFIC_CUTOFF`.

    Returns
    -------
    numpy.ndarray
        ``G`` per frame, shape ``(n_frames,)``.
    """
    return go_contact_wells(contact_dists, contact_eqdists, cutoff).sum(axis=1)


#······································································#
# Sample loading (explicit paths; no directory-layout assumption)
#······································································#

def state_samples(
        top_pdb: str,
        traj_dcd: str | list[str],
        specific_interactions_txt: str,
        bound_q: float,
        beta_q: float = slim._DEFAULT_BETA_Q,
        lambda_q: float = slim._DEFAULT_LAMBDA_Q,
        cutoff: float = SPECIFIC_CUTOFF,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Gō collective variable and bound indicator for one ``eps_go`` state.

    Loads a single state's specific-contact distances from the given files
    (concatenating multiple DCDs if a list is passed), then returns its Gō
    collective variable :func:`go_collective_variable` and per-frame bound
    indicator ``[Q > bound_q]`` (``Q`` the native contact fraction; see
    :func:`src.analysis.slim._native_contact_fraction`). The caller supplies the
    file paths, so no simulation-directory layout is assumed here.

    Parameters
    ----------
    top_pdb : str
        Path to the topology PDB.
    traj_dcd : str or list[str]
        Path(s) to the trajectory DCD(s) of this state.
    specific_interactions_txt : str
        Path to the state's ``specific_interactions.txt``.
    bound_q : float
        Bound/unbound dividing point on ``Q`` (per-system calibrated value).
    beta_q, lambda_q : float, optional
        Switching-function parameters for ``Q``; defaults match
        :mod:`src.analysis.slim`.
    cutoff : float, optional
        Gō contact cutoff (nm), by default :data:`SPECIFIC_CUTOFF`.
    stride : int, optional
        Keep every ``stride``-th frame (reduces autocorrelation and cost).
        Applied at the trajectory-reader level, so skipped frames are never
        read or scored, by default ``1``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        ``(G, bound, c_box)``: per-frame Gō CV, per-frame Boolean bound state
        (as floats), and the single-copy box concentration in molar.
    """
    wells, bound, c_box = state_contact_samples(
        top_pdb, traj_dcd, specific_interactions_txt, bound_q,
        beta_q=beta_q, lambda_q=lambda_q, cutoff=cutoff, stride=stride)
    return wells.sum(axis=1), bound, c_box


def state_contact_samples(
        top_pdb: str,
        traj_dcd: str | list[str],
        specific_interactions_txt: str,
        bound_q: float,
        beta_q: float = slim._DEFAULT_BETA_Q,
        lambda_q: float = slim._DEFAULT_LAMBDA_Q,
        cutoff: float = SPECIFIC_CUTOFF,
        stride: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-contact Gō wells and bound indicator for one ``eps_go`` state.

    The contact-resolved counterpart of :func:`state_samples`, which sums over
    contacts. Use this when the Gō graph is to be edited after sampling, since
    the per-contact wells are what let any edge subset be re-summed later
    (:func:`reweight_subgraph_curve`).

    The bound indicator is unchanged: it is always the parent system's
    ``Q > bound_q`` over its full native contact set, because the question a
    contact-subgraph reweighting answers is how the free energy of forming the
    original pose responds to deleting Gō terms, not how a differently
    defined complex would bind.

    Parameters
    ----------
    top_pdb : str
        Path to the topology PDB.
    traj_dcd : str or list[str]
        Path(s) to the trajectory DCD(s) of this state.
    specific_interactions_txt : str
        Path to the state's ``specific_interactions.txt``.
    bound_q : float
        Bound/unbound dividing point on ``Q`` (per-system calibrated value).
    beta_q, lambda_q, cutoff, stride : optional
        As in :func:`state_samples`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        ``(wells, bound, c_box)``: per-frame per-contact well values with shape
        ``(n_frames, n_contacts)``, per-frame Boolean bound state (as floats),
        and the single-copy box concentration in molar.
    """
    contact_dists, contact_eqdists, c_box = slim.TwoChain._load_contacts(
        top_pdb, traj_dcd, specific_interactions_txt, stride=stride)
    wells = go_contact_wells(contact_dists, contact_eqdists, cutoff)
    Q = slim._native_contact_fraction(contact_dists, contact_eqdists, beta_q, lambda_q)
    return wells, (Q > bound_q).astype(float), c_box


#······································································#
# MBAR estimator
#······································································#

class MBAR:
    """Minimal MBAR estimator for reweighting along ``eps_go``.

    Solves the MBAR free energies of the sampled states, then reweights the
    pooled samples to any target state. Reduced potentials are dimensionless
    energies ``u = U / (R T)``; only the ``eps_go``-dependent part is needed
    (see the module docstring).

    MBAR is the maximum-likelihood, minimum-variance estimator of the state free
    energies [Shirts2008]_. The ``f_k`` (dimensionless free energies) solve the
    self-consistent equations

    .. math:: f_i = -\\ln \\sum_{n=1}^{N}
        \\frac{\\exp(-u_i(x_n))}{\\sum_{k=1}^{K} N_k \\exp\\!\\big(f_k - u_k(x_n)\\big)},

    which are the stationarity conditions of a strictly convex log-likelihood,
    so the solution is unique up to an additive gauge (fixed here by
    ``f_k[0] = 0``). ``N = Σ_k N_k`` is the total pooled sample count.

    Parameters
    ----------
    u_kn : numpy.ndarray
        Reduced potential of every pooled sample at every sampled state,
        shape ``(n_states, n_samples)``.
    N_k : numpy.ndarray
        Number of samples drawn from each sampled state, shape ``(n_states,)``.
    tol : float, optional
        Convergence tolerance on the free energies, by default ``1e-10``.
    max_iter : int, optional
        Maximum self-consistent iterations, by default ``5000``.

    Attributes
    ----------
    f_k : numpy.ndarray
        Converged dimensionless free energies (gauge-fixed to ``f_k[0] = 0``).
    """

    def __init__(self, u_kn: np.ndarray, N_k: np.ndarray, tol: float = 1e-10, max_iter: int = 5000):
        self.u_kn = np.asarray(u_kn, dtype=float)
        self.N_k = np.asarray(N_k, dtype=float)
        with np.errstate(divide="ignore"):
            self._logN = np.log(self.N_k)
        f = np.zeros(len(self.N_k))
        for _ in range(max_iter):
            log_denom = logsumexp(self._logN[:, None] + f[:, None] - self.u_kn, axis=0)
            f_new = -logsumexp(-self.u_kn - log_denom[None, :], axis=1)
            f_new -= f_new[0]
            if np.max(np.abs(f_new - f)) < tol:
                f = f_new
                break
            f = f_new
        else:
            warnings.warn("MBAR did not fully converge; overlap may be poor.")
        self.f_k = f
        # Cache the per-sample log mixture denominator at the converged f_k.
        self._log_denom = logsumexp(self._logN[:, None] + f[:, None] - self.u_kn, axis=0)

    #··································································#

    def reweight(self, u_n: np.ndarray) -> tuple[np.ndarray, float]:
        """Normalized sample weights and effective sample size at a state.

        The (normalized) MBAR weight of sample ``n`` at a target state with
        reduced potential ``u_n`` is

        .. math:: W_n \\propto
            \\frac{\\exp(-u_n(x_n))}{\\sum_k N_k \\exp\\!\\big(f_k - u_k(x_n)\\big)},
            \\qquad \\sum_n W_n = 1,

        and the reliability of the reweighted estimate is summarised by Kish's
        effective sample size [Kish1965]_,

        .. math:: N_{\\mathrm{eff}} = 1 \\big/ \\sum_n W_n^2 \\in [1, N].

        A single dominant weight gives ``N_eff ≈ 1`` (untrustworthy); uniform
        weights give ``N_eff = N``.

        Parameters
        ----------
        u_n : numpy.ndarray
            Reduced potential of every pooled sample at the target state,
            shape ``(n_samples,)``.

        Returns
        -------
        tuple[numpy.ndarray, float]
            ``(weights, N_eff)``.
        """
        log_w = -u_n - self._log_denom
        weights = np.exp(log_w - logsumexp(log_w))
        N_eff = 1.0 / np.sum(weights**2)
        return weights, N_eff

    #··································································#

    def expectation(self, values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        """Reweighted mean and its variance for a per-sample observable.

        Returns ``⟨A⟩ = Σ_n W_n A_n`` and the fixed-weight reweighting variance

        .. math:: \\operatorname{Var}(\\langle A\\rangle) \\approx
            \\sum_n W_n^2\\,(A_n - \\langle A\\rangle)^2 .

        This correctly diverges where the effective sample size collapses, but
        (being first-order and treating the weights as fixed) it ignores the
        uncertainty in the ``f_k`` and so mildly under-states the full MBAR
        uncertainty [Shirts2008]_; use a block bootstrap for publication errors.

        Parameters
        ----------
        values : numpy.ndarray
            Per-sample observable ``A_n``, shape ``(n_samples,)``.
        weights : numpy.ndarray
            Normalized weights at the target state (from :meth:`reweight`).

        Returns
        -------
        tuple[float, float]
            ``(mean, variance)``.
        """
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights**2 * (values - mean)**2))
        return mean, variance

    #··································································#

    def overlap_matrix(self, state_index: np.ndarray) -> np.ndarray:
        """State overlap matrix from sample responsibilities.

        Using the posterior probability that sample ``n`` came from state ``k``,

        .. math:: p_{k\\mid n} =
            \\frac{N_k \\exp(f_k - u_k(x_n))}{\\sum_{k'} N_{k'} \\exp(f_{k'} - u_{k'}(x_n))},
            \\qquad \\sum_k p_{k\\mid n} = 1,

        entry ``O_ij`` is the mean responsibility to state ``j`` of the samples
        actually drawn from state ``i`` (rows sum to 1). A diagonal near 1 flags
        an isolated state (poor overlap); a sizeable adjacent off-diagonal
        ``O_{i,i+1}`` means neighbouring ``eps_go`` states share configuration
        space, so MBAR can bridge them. As a rule of thumb, adjacent overlap
        below ~0.03 indicates an effectively broken link [Klimovich2015]_.

        Parameters
        ----------
        state_index : numpy.ndarray
            Source-state index of each pooled sample, shape ``(n_samples,)``.

        Returns
        -------
        numpy.ndarray
            Overlap matrix, shape ``(n_states, n_states)``.
        """
        log_p = (self._logN[:, None] + self.f_k[:, None] - self.u_kn - self._log_denom[None, :])
        p = np.exp(log_p)                       # (K, N), columns sum to 1 over states
        K = len(self.N_k)
        overlap = np.zeros((K, K))
        for i in range(K):
            mask = state_index == i
            if mask.any():
                overlap[i] = p[:, mask].mean(axis=1)
        return overlap


#······································································#
# Preliminary MBAR analysis
#······································································#

def _dg_with_sigma(f_bound: float, f_var: float, c_box: float, temperature: float) -> tuple[float, float]:
    """``dG`` and its 1σ from a bound fraction and its variance.

    With ``Kd = C_box (1-f) / f`` (single-copy box, see
    :func:`src.analysis.slim._kd_from_fraction_bound`) and
    ``dG = R T \\ln Kd``, propagating ``Var(f)`` by the delta method gives

    .. math:: \\Delta G = R T\\,[\\ln C_{\\mathrm{box}} + \\ln(1-f) - \\ln f],
        \\qquad
        \\sigma_{\\Delta G} = R T\\left(\\frac{1}{1-f} + \\frac{1}{f}\\right)\\sigma_f .

    The prefactor diverges as ``f -> 0`` or ``1``, correctly flagging the
    plateaus/censored regime where ``dG`` is unresolved. Being first-order and
    fixed-weight it under-states the true uncertainty (it ignores the ``f_k``
    error and time-correlation); toggle on :func:`bootstrap_dg` for an honest,
    correlation-aware band.
    """
    if f_bound <= 0.0 or f_bound >= 1.0:
        return (np.inf if f_bound <= 0.0 else -np.inf), np.inf
    Kd = slim._kd_from_fraction_bound(f_bound, c_box)
    dG = slim.dG(Kd)
    rt = constants.R * temperature
    dG_sigma = rt * (1.0 / (1.0 - f_bound) + 1.0 / f_bound) * np.sqrt(max(f_var, 0.0))
    return dG, dG_sigma


#······································································#
# Block-bootstrap uncertainty (correlation-aware, toggleable)
#······································································#

def _circular_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample indices for a circular moving-block bootstrap of length ``n``.

    Draws ``⌈n / block_size⌉`` blocks of ``block_size`` consecutive frames from
    random (wrap-around) start points and concatenates them, truncated to ``n``
    [Kunsch1989]_ [Politis1992]_. Resampling blocks rather than single frames
    preserves the trajectory's autocorrelation, so the bootstrap spread reflects
    the true (correlated) sampling error rather than an ``iid`` under-estimate.
    ``block_size = 1`` reduces to the ordinary ``iid`` bootstrap.

    Parameters
    ----------
    n : int
        Number of frames in the state.
    block_size : int
        Block length in frames (choose ``≈ 2τ``, twice the autocorrelation time).
    rng : numpy.random.Generator
        Random generator (seed it for reproducibility).

    Returns
    -------
    numpy.ndarray
        Integer indices into ``[0, n)``, shape ``(n,)``.
    """
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
    return idx.reshape(-1)[:n]


def bootstrap_dg(
        states: list[dict],
        c_box: float,
        eps_go_grid: np.ndarray,
        n_boot: int = 200,
        block_size: int = 1,
        temperature: float = 298.15,
        seed: int | None = None,
        min_finite_frac: float = 0.5,
    ) -> pd.DataFrame:
    """Block-bootstrap ``dG(eps_go)`` uncertainty band along the grid.

    Repeatedly resamples each sampled state's frames with a circular moving-block
    bootstrap (:func:`_circular_block_indices`), re-solves MBAR on the resample,
    reweights to every grid ``eps_go`` and recomputes ``dG``. The spread of the
    replicate ``dG`` values is a correlation-aware uncertainty that, unlike the
    fixed-weight delta method in :func:`_dg_with_sigma`, also captures the
    error in the MBAR free energies ``f_k`` and the frame-to-frame correlation,
    at the cost of re-solving MBAR ``n_boot`` times. Censored grid points (a
    replicate with ``f_bound`` at 0 or 1, i.e. ``dG`` unresolved) contribute no
    value; a point kept below ``min_finite_frac`` finite replicates is reported
    as ``inf`` sigma rather than a falsely tight band.

    Parameters
    ----------
    states : list[dict]
        Sampled states, each ``{'eps_go', 'G', 'bound'}`` (see
        :func:`state_samples`).
    c_box : float
        Single-copy box concentration in molar.
    eps_go_grid : numpy.ndarray
        Target ``eps_go`` grid to evaluate (pass the same grid as the point
        estimate to align the band with it).
    n_boot : int, optional
        Number of bootstrap replicates, by default ``200``.
    block_size : int, optional
        Moving-block length in frames, by default ``1`` (``iid``); set to
        ``≈ 2τ`` (the decorrelation time from the statistical inefficiency
        ``g = 1 + 2τ``) to respect autocorrelation.
    temperature : float, optional
        Temperature in Kelvin, by default ``298.15``.
    seed : int or None, optional
        Seed for the bootstrap generator (set for reproducibility), by default
        ``None``.
    min_finite_frac : float, optional
        Minimum fraction of replicates that must yield a finite ``dG`` at a grid
        point for its sigma to be reported (else ``inf``), by default ``0.5``.

    Returns
    -------
    pandas.DataFrame
        Columns ``['eps_go', 'dG_boot', 'dG_boot_sigma', 'boot_finite']``: the
        bootstrap mean ``dG`` and standard deviation per grid point (``inf``
        sigma / ``nan`` mean where too few replicates resolve ``dG``), and the
        count of finite replicates there.
    """
    rng = np.random.default_rng(seed)
    eps_states = np.array([s["eps_go"] for s in states], dtype=float)
    G_states = [np.asarray(s["G"], dtype=float) for s in states]
    bound_states = [np.asarray(s["bound"], dtype=float) for s in states]
    N_k = np.array([len(g) for g in G_states], dtype=float)
    beta = 1.0 / (constants.R * temperature)
    eps_go_grid = np.asarray(eps_go_grid, dtype=float)

    dg_boot = np.full((n_boot, len(eps_go_grid)), np.nan)
    for b in range(n_boot):
        G_res, bound_res = [], []
        for g, bnd in zip(G_states, bound_states):
            idx = _circular_block_indices(len(g), block_size, rng)
            G_res.append(g[idx])
            bound_res.append(bnd[idx])
        G_all = np.concatenate(G_res)
        bound_all = np.concatenate(bound_res)
        u_kn = beta * eps_states[:, None] * G_all[None, :]
        with warnings.catch_warnings():           # replicate-level non-convergence is expected noise
            warnings.simplefilter("ignore")
            mbar = MBAR(u_kn, N_k)
        for j, eps in enumerate(eps_go_grid):
            weights, _ = mbar.reweight(beta * eps * G_all)
            f_bound = float(np.sum(weights * bound_all))
            if 0.0 < f_bound < 1.0:
                dg_boot[b, j] = slim.dG(slim._kd_from_fraction_bound(f_bound, c_box))

    finite = np.isfinite(dg_boot).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # empty-slice means for fully-censored points
        mean = np.nanmean(dg_boot, axis=0)
        sigma = np.nanstd(dg_boot, axis=0, ddof=1)
    enough = finite >= max(2, int(np.ceil(min_finite_frac * n_boot)))
    sigma = np.where(enough, sigma, np.inf)
    mean = np.where(finite > 0, mean, np.nan)
    return pd.DataFrame({"eps_go": eps_go_grid, "dG_boot": mean,
                         "dG_boot_sigma": sigma, "boot_finite": finite})


#······································································#
# Reweighted dG(eps_go) curve
#······································································#

def reweight_curve(
        states: list[dict],
        c_box: float,
        eps_go_grid: np.ndarray,
        temperature: float = 298.15,
        errors: str = "bootstrap",
        n_boot: int = 200,
        block_size: int = 1,
        seed: int | None = None,
    ) -> pd.DataFrame:
    """Reweight pooled scan states onto an ``eps_go`` grid -> ``dG(eps_go)``.

    The single reweighting entry point for the ``eps_go`` scans: builds MBAR from
    the pooled (converged) states and evaluates the bound fraction, ``dG`` and its
    uncertainty at every grid ``eps_go``. The returned schema is *identical*
    whichever error method is selected, so ``errors='delta'`` and
    ``errors='bootstrap'`` are interchangeable for downstream code; only
    the provenance of the ``dG_sigma`` column changes, never its name or shape.

    Parameters
    ----------
    states : list[dict]
        Sampled states, each ``{'eps_go', 'G', 'bound'}`` (see
        :func:`state_samples`).
    c_box : float
        Single-copy box concentration in molar.
    eps_go_grid : numpy.ndarray
        Target ``eps_go`` grid to evaluate. May extend past the sampled range --
        reweighting to unsampled ``eps_go`` is a deliberate use here (it flags
        which new states to run). Extrapolated points simply carry a collapsing
        ``N_eff``/``N_eff_minor`` and a wide (bootstrap) ``dG_sigma``, which is the
        honest signal that they are predictions, not resolved estimates.
    temperature : float, optional
        Temperature in Kelvin, by default ``298.15``.
    errors : {'bootstrap', 'delta'}, optional
        Source of the ``dG_sigma`` column. ``'bootstrap'`` (default) is the
        correlation-aware circular block bootstrap (:func:`bootstrap_dg`), which
        also captures the ``f_k`` uncertainty; ``'delta'`` is the cheap
        fixed-weight delta method (:func:`_dg_with_sigma`) that needs a single
        MBAR solve but under-states the true error. Both populate the *same*
        ``dG_sigma`` column.
    n_boot, block_size, seed : optional
        Bootstrap controls, ignored when ``errors='delta'`` (see
        :func:`bootstrap_dg`).

    Returns
    -------
    pandas.DataFrame
        Columns ``['eps_go', 'f_bound', 'dG', 'dG_sigma', 'N_eff', 'N_eff_minor']``.
        ``N_eff_minor = N_eff · min(f, 1−f)`` is the effective sample size of the
        rarer (minor) state: it collapses in overlap gaps and at censored edges, and is
        the basis for both the reliability flag and the sampling-value score.
    """
    if errors not in ("bootstrap", "delta"):
        raise ValueError(f"errors must be 'bootstrap' or 'delta', got {errors!r}")

    eps_states = np.array([s["eps_go"] for s in states], dtype=float)
    G_all = np.concatenate([s["G"] for s in states])
    bound_all = np.concatenate([s["bound"] for s in states])
    N_k = np.array([len(s["G"]) for s in states], dtype=float)
    beta = 1.0 / (constants.R * temperature)
    grid = np.asarray(eps_go_grid, dtype=float)

    est = MBAR(beta * eps_states[:, None] * G_all[None, :], N_k)
    rows = []
    for eps in grid:
        weights, N_eff = est.reweight(beta * eps * G_all)
        f_bound, f_var = est.expectation(bound_all, weights)
        dG, dG_sigma = _dg_with_sigma(f_bound, f_var, c_box, temperature)
        rows.append((eps, f_bound, dG, dG_sigma, N_eff, N_eff * min(f_bound, 1.0 - f_bound)))
    curve = pd.DataFrame(rows, columns=["eps_go", "f_bound", "dG", "dG_sigma", "N_eff", "N_eff_minor"])

    if errors == "bootstrap":
        boot = bootstrap_dg(states, c_box, grid, n_boot=n_boot, block_size=block_size,
                            temperature=temperature, seed=seed)
        curve["dG_sigma"] = boot["dG_boot_sigma"].to_numpy()   # swap in correlation-aware sigma
    return curve


#······································································#
# Per-system reweighting (multiprocessing-friendly entry point)
#······································································#

def reweight_system(
        system_id: str,
        dGs_subset: pd.DataFrame,
        sims: pd.DataFrame,
        bound_q: float,
        eps_go_grid: np.ndarray,
        stride: int = 1,
        temperature: float = 298.15,
        errors: str = "bootstrap",
        n_boot: int = 200,
        block_size: int = 1,
        seed: int | None = None,
    ) -> pd.DataFrame:
    """Load one system's sampled ``eps_go`` states and reweight onto the grid.

    Loads and pools every converged ``eps_go`` state for ``system_id``, then
    calls :func:`reweight_curve`. Systems are independent of one another (each
    loads its own trajectories and solves its own MBAR/bootstrap), so this is
    the unit of work for dispatching the reweighting loop across processes,
    e.g. with :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    system_id : str
        System identifier, matching the ``System`` column/index of
        ``dGs_subset`` and ``sims``.
    dGs_subset : pandas.DataFrame
        This system's rows of the scan summary (already filtered to the
        converged trajectories the caller wants pooled), with an ``eps_go``
        column.
    sims : pandas.DataFrame
        Simulation directory table with ``System``, ``eps_go`` and ``Path``
        columns (see ``src.utils.get_sim_dirs``).
    bound_q : float
        Bound/unbound dividing point on ``Q`` for this system (see
        :func:`state_samples`).
    eps_go_grid : numpy.ndarray
        Target ``eps_go`` grid, passed through to :func:`reweight_curve`.
    stride, temperature, errors, n_boot, block_size, seed : optional
        Passed through to :func:`state_samples` (``stride``) and
        :func:`reweight_curve` (the rest).

    Returns
    -------
    pandas.DataFrame
        This system's reweighted curve (see :func:`reweight_curve`).
    """
    states = []
    c_box = None
    for eps_go, _ in dGs_subset.groupby("eps_go"):
        sims_subset = sims[(sims["System"] == system_id) & (sims["eps_go"] == eps_go)]

        traj_dcds = sims_subset["Path"] + f"/{system_id}.dcd"
        top_pdb = sims_subset["Path"].iloc[0] + "/top.pdb"
        spec_txt = sims_subset["Path"].iloc[0] + "/specific_interactions.txt"

        G, bound, c_box = state_samples(top_pdb, traj_dcds, spec_txt, bound_q=bound_q, stride=stride)
        states.append({"eps_go": eps_go, "G": G, "bound": bound})

    states.sort(key=lambda s: s["eps_go"])

    return reweight_curve(states, c_box, eps_go_grid, temperature=temperature, errors=errors,
                          n_boot=n_boot, block_size=block_size, seed=seed)


#······································································#
# Contact-subgraph reweighting  (eps_go × edge subset, no new simulation)
#······································································#

def _keep_mask(keep, n_contacts: int) -> np.ndarray:
    """Normalise a contact selection to a boolean mask over ``n_contacts``."""
    if keep is None:
        return np.ones(n_contacts, dtype=bool)
    mask = np.asarray(keep)
    if mask.dtype == bool:
        if mask.shape != (n_contacts,):
            raise ValueError(
                f"Boolean keep mask has shape {mask.shape}, expected ({n_contacts},) "
                f"to match the sampled contact set."
            )
    else:
        if mask.size and (mask.min() < 0 or mask.max() >= n_contacts):
            raise ValueError(
                f"Contact indices must lie in [0, {n_contacts}); got "
                f"[{mask.min()}, {mask.max()}]."
            )
        index, mask = mask, np.zeros(n_contacts, dtype=bool)
        mask[index] = True
    if not mask.any():
        raise ValueError("keep selects no contacts; the edited graph would have no Gō term.")
    return mask


def _stack_states(states: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool contact-resolved states into ``(eps, wells, bound, N_k)``."""
    if not states:
        raise ValueError("No states given.")
    missing = [k for k in ("eps_go", "wells", "bound") if k not in states[0]]
    if missing:
        raise KeyError(
            f"States must carry {missing}; build them with state_contact_samples(), "
            f"not state_samples() (which has already summed over contacts)."
        )
    widths = {np.shape(s["wells"])[1] for s in states}
    if len(widths) != 1:
        raise ValueError(
            f"States disagree on the contact count ({sorted(widths)}); every state must "
            f"be sampled on the same native contact set for a subgraph to be defined."
        )
    return (np.array([s["eps_go"] for s in states], dtype=float),
            np.concatenate([np.asarray(s["wells"], dtype=float) for s in states], axis=0),
            np.concatenate([np.asarray(s["bound"], dtype=float) for s in states]),
            np.array([len(s["wells"]) for s in states], dtype=float))


def _subgraph_rows(estimator, beta, wells, bound, mask, c_box, grid, temperature) -> list[tuple]:
    """Evaluate one edited graph over the ``eps_go`` grid (shared by both entry points)."""
    G_kept = wells[:, mask].sum(axis=1)
    rows = []
    for eps in grid:
        weights, n_eff = estimator.reweight(beta * eps * G_kept)
        f_bound, f_var = estimator.expectation(bound, weights)
        dG, dG_sigma = _dg_with_sigma(f_bound, f_var, c_box, temperature)
        rows.append((eps, f_bound, dG, dG_sigma, n_eff, n_eff * min(f_bound, 1.0 - f_bound)))
    return rows


def reweight_subgraph_curve(
        states: list[dict],
        c_box: float,
        eps_go_grid: np.ndarray,
        keep=None,
        temperature: float = 298.15,
    ) -> pd.DataFrame:
    """Reweight sampled states onto an edited contact graph -> ``dG(eps_go)``.

    The two-dimensional counterpart of :func:`reweight_curve`: it reweights in
    ``eps_go`` and in the Gō edge set at once, so an existing scan yields the
    curve the system would have shown had it been simulated with only the
    contacts in ``keep``. Because the Gō energy is a sum of per-contact wells,
    both moves are the same operation, a change of reduced potential on frames
    already in hand, and the result is exact in the MBAR sense, with no new
    simulation:

    .. math:: u(X;\\varepsilon, E') = \\beta\\varepsilon \\sum_{i \\in E'} g_i(X).

    The MBAR mixture is built from the full contact set, the Hamiltonian the
    frames were generated under; only the target state
    uses the subset. Note the corollary: passing these states to
    :func:`bootstrap_dg` with the subgraph ``G`` substituted would be wrong, as
    that would claim the samples were drawn from the edited graph.

    The bound indicator is the parent's throughout (see
    :func:`state_contact_samples`), so this measures how the free energy of
    forming the original native pose responds to deleting Gō terms.

    Reweighting cannot manufacture sampling. Deleting edges whose wells fluctuate
    strongly inflates the log-weight variance as ``(βε)² Var(G_S)`` over the
    removed set ``S``, and ``N_eff/N ≈ exp[-(βε)² Var(G_S)]`` collapses. Trust a
    row only where ``N_eff_minor`` stays comfortably above ~5; deleting the
    loosest contacts is cheapest, since they are the least often formed and so
    carry the least variance.

    Parameters
    ----------
    states : list[dict]
        Sampled states, each ``{'eps_go', 'wells', 'bound'}`` from
        :func:`state_contact_samples`. Every state must share one contact set.
    c_box : float
        Single-copy box concentration in molar.
    eps_go_grid : numpy.ndarray
        Target ``eps_go`` grid to evaluate.
    keep : array-like or None, optional
        Contacts retained in the edited graph, as a boolean mask of length
        ``n_contacts`` or an array of contact indices, ordered as in the state's
        ``specific_interactions.txt``. ``None`` (default) keeps every contact and
        so reproduces :func:`reweight_curve`, the natural self-check, which
        agrees to ~1e-10 kJ/mol wherever ``f_bound`` is more than ~1e-6 from
        saturation. Closer than that the two can differ by ~0.1 kJ/mol on
        identical weights, because ``Kd = C(1-f)²/f`` cancels catastrophically as
        ``f -> 1``; those points carry ``N_eff_minor = 0`` and are not estimates
        at all. Filter on ``N_eff_minor`` rather than trusting a finite ``dG``.
    temperature : float, optional
        Temperature in Kelvin, by default ``298.15``.

    Returns
    -------
    pandas.DataFrame
        Columns ``['eps_go', 'f_bound', 'dG', 'dG_sigma', 'N_eff',
        'N_eff_minor']``, the same schema as :func:`reweight_curve`, so the two
        are interchangeable downstream. ``dG_sigma`` is the fixed-weight delta
        method only: the block bootstrap is not offered here because it would
        re-solve MBAR per replicate per variant, which is prohibitive when
        scanning many edited graphs.

    Raises
    ------
    KeyError
        If the states carry ``'G'`` rather than ``'wells'`` (i.e. came from
        :func:`state_samples`, which has already summed over contacts).
    ValueError
        If the states disagree on the contact count, or ``keep`` is the wrong
        shape, out of range, or empty.
    """
    eps_states, wells, bound, N_k = _stack_states(states)
    mask = _keep_mask(keep, wells.shape[1])
    beta = 1.0 / (constants.R * temperature)
    grid = np.asarray(eps_go_grid, dtype=float)

    estimator = MBAR(beta * eps_states[:, None] * wells.sum(axis=1)[None, :], N_k)
    rows = _subgraph_rows(estimator, beta, wells, bound, mask, c_box, grid, temperature)
    return pd.DataFrame(rows, columns=["eps_go", "f_bound", "dG", "dG_sigma",
                                       "N_eff", "N_eff_minor"])


def reweight_subgraph_system(
        system_id: str,
        dGs_subset: pd.DataFrame,
        sims: pd.DataFrame,
        bound_q: float,
        eps_go_grid: np.ndarray,
        keeps: dict,
        stride: int = 1,
        temperature: float = 298.15,
    ) -> pd.DataFrame:
    """Reweight one system onto many edited contact graphs, loading it once.

    The per-system entry point for contact-graph editing, mirroring
    :func:`reweight_system`. Trajectories are read and MBAR is solved once,
    then every variant in ``keeps`` is evaluated against that single estimator --
    which is the whole economy of the approach, since the MBAR solve dominates
    and a parent typically spawns tens of edited graphs. Systems are independent,
    so this is again the unit of work to dispatch across processes.

    Parameters
    ----------
    system_id : str
        System identifier, matching the ``System`` column of ``dGs_subset`` and
        ``sims``.
    dGs_subset : pandas.DataFrame
        This system's rows of the scan summary, already filtered to the converged
        trajectories to pool, with an ``eps_go`` column.
    sims : pandas.DataFrame
        Simulation directory table with ``System``, ``eps_go`` and ``Path``
        columns (see ``src.utils.get_sim_dirs``).
    bound_q : float
        Bound/unbound dividing point on ``Q`` for this system.
    eps_go_grid : numpy.ndarray
        Target ``eps_go`` grid, passed through to :func:`reweight_subgraph_curve`.
    keeps : dict
        Variant name -> contact selection (boolean mask or index array; see
        ``keep`` in :func:`reweight_subgraph_curve`). Include an unedited entry
        to carry the parent curve through the same code path as its children.
    stride, temperature : optional
        Passed through to :func:`state_contact_samples` and the reweighting.

    Returns
    -------
    pandas.DataFrame
        Long-form curves, the :func:`reweight_subgraph_curve` schema prefixed by
        ``['System', 'variant']``.
    """
    states, c_box = [], None
    for eps_go, _ in dGs_subset.groupby("eps_go"):
        sims_subset = sims[(sims["System"] == system_id) & (sims["eps_go"] == eps_go)]
        if not len(sims_subset):
            continue
        wells, bound, c_box = state_contact_samples(
            sims_subset["Path"].iloc[0] + "/top.pdb",
            list(sims_subset["Path"] + f"/{system_id}.dcd"),
            sims_subset["Path"].iloc[0] + "/specific_interactions.txt",
            bound_q=bound_q, stride=stride)
        states.append({"eps_go": eps_go, "wells": wells, "bound": bound})
    states.sort(key=lambda s: s["eps_go"])

    eps_states, wells, bound, N_k = _stack_states(states)
    beta = 1.0 / (constants.R * temperature)
    grid = np.asarray(eps_go_grid, dtype=float)
    estimator = MBAR(beta * eps_states[:, None] * wells.sum(axis=1)[None, :], N_k)

    frames = []
    for variant, keep in keeps.items():
        mask = _keep_mask(keep, wells.shape[1])
        curve = pd.DataFrame(
            _subgraph_rows(estimator, beta, wells, bound, mask, c_box, grid, temperature),
            columns=["eps_go", "f_bound", "dG", "dG_sigma", "N_eff", "N_eff_minor"])
        curve.insert(0, "variant", variant)
        curve.insert(0, "System", system_id)
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


#······································································#
# Per-system overlap diagnostics (multiprocessing-friendly entry point)
#······································································#

def overlap_system(
        system_id: str,
        dGs_subset: pd.DataFrame,
        sims: pd.DataFrame,
        bound_q: float,
        stride: int = 1,
        temperature: float = 298.15,
    ) -> dict:
    """Load one system's sampled ``eps_go`` states and diagnose the MBAR overlap chain.

    Loads and pools every converged ``eps_go`` state for ``system_id`` (same
    loading as :func:`reweight_system`), then builds MBAR from the *sampled*
    states only: no fine grid, no bootstrap, since only the discrete link
    structure between consecutive scanned states is wanted here, not the
    reweighted curve. A weak adjacent overlap means those two states barely
    share configuration space, so any target ``eps_go`` between them rests on
    an unbridged MBAR extrapolation however many frames either state has (see
    :meth:`MBAR.overlap_matrix`); its midpoint is where a new simulation would
    repair the gap. This is the cheap connectivity half of what
    :func:`preliminary_mbar` computes, split out so it can be dispatched
    per-system (e.g. with :class:`concurrent.futures.ProcessPoolExecutor`)
    without also paying for a fine-grid bootstrap reweight.

    Parameters
    ----------
    system_id : str
        System identifier, matching the ``System`` column/index of
        ``dGs_subset`` and ``sims``.
    dGs_subset : pandas.DataFrame
        This system's rows of the scan summary (already filtered to the
        converged trajectories the caller wants pooled), with an ``eps_go``
        column.
    sims : pandas.DataFrame
        Simulation directory table with ``System``, ``eps_go`` and ``Path``
        columns (see ``src.utils.get_sim_dirs``).
    bound_q : float
        Bound/unbound dividing point on ``Q`` for this system (see
        :func:`state_samples`).
    stride, temperature : optional
        Passed through to :func:`state_samples` (``stride``) and the reduced
        potential (``temperature``).

    Returns
    -------
    dict
        ``eps_go_sampled`` (sorted array of sampled ``eps_go``),
        ``adjacent_overlap`` (list of ``(eps_i, eps_j, overlap)`` for each
        consecutive sampled pair), ``weakest_link`` (the entry with lowest
        overlap, or ``None`` if fewer than two states), ``self_N_eff`` (each
        sampled state's own MBAR effective sample size; low means poorly
        connected to the rest of the chain) and ``state_transitions`` (each
        state's own ``(n_bind, n_unbind)`` count; few means that state's run
        itself is under-sampled and should be extended rather than bridged).
    """
    states = []
    for eps_go, _ in dGs_subset.groupby("eps_go"):
        sims_subset = sims[(sims["System"] == system_id) & (sims["eps_go"] == eps_go)]

        traj_dcds = sims_subset["Path"] + f"/{system_id}.dcd"
        top_pdb = sims_subset["Path"].iloc[0] + "/top.pdb"
        spec_txt = sims_subset["Path"].iloc[0] + "/specific_interactions.txt"

        G, bound, _ = state_samples(top_pdb, traj_dcds, spec_txt, bound_q=bound_q, stride=stride)
        states.append({"eps_go": eps_go, "G": G, "bound": bound})

    states.sort(key=lambda s: s["eps_go"])
    eps_states = np.array([s["eps_go"] for s in states], dtype=float)

    if len(states) < 2:
        return {"eps_go_sampled": eps_states, "adjacent_overlap": [], "weakest_link": None,
                "self_N_eff": {}, "state_transitions": {}}

    G_all = np.concatenate([s["G"] for s in states])
    N_k = np.array([len(s["G"]) for s in states], dtype=float)
    state_index = np.concatenate([np.full(len(s["G"]), i) for i, s in enumerate(states)])
    beta = 1.0 / (constants.R * temperature)

    u_kn = beta * eps_states[:, None] * G_all[None, :]
    est = MBAR(u_kn, N_k)

    overlap = est.overlap_matrix(state_index)
    adj = [(eps_states[i], eps_states[i + 1], float(overlap[i, i + 1]))
           for i in range(len(eps_states) - 1)]
    weakest = min(adj, key=lambda t: t[2]) if adj else None

    self_neff, state_transitions = {}, {}
    for i, s in enumerate(states):
        _, ne = est.reweight(u_kn[i])
        self_neff[float(s["eps_go"])] = float(ne)
        n_bind, n_unbind = slim._state_transitions(s["bound"].astype(bool))
        state_transitions[float(s["eps_go"])] = (n_bind, n_unbind)

    return {"eps_go_sampled": eps_states, "adjacent_overlap": adj, "weakest_link": weakest,
            "self_N_eff": self_neff, "state_transitions": state_transitions}


def preliminary_mbar(
        states: list[dict],
        c_box: float,
        eps_go_grid: np.ndarray | None = None,
        dg_window: tuple[float, float] = (-47.0, -23.0),
        temperature: float = 298.15,
        neff_min: float = DEFAULT_NEFF_MIN,
        min_unbound_eff: float = 5.0,
        errors: str = "bootstrap",
        n_boot: int = 200,
        block_size: int = 1,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, dict]:
    """Reweight a system's scans along ``eps_go`` and propose the next sim.

    Parameters
    ----------
    states : list[dict]
        Sampled states, each ``{'eps_go', 'G', 'bound'}`` (``G`` and ``bound``
        from :func:`state_samples`).
    c_box : float
        Single-copy box concentration in molar.
    eps_go_grid : numpy.ndarray or None, optional
        Fine target ``eps_go`` grid; defaults to 201 points spanning the sampled
        range (slightly extended).
    dg_window : tuple[float, float], optional
        Target ``dG`` window (kJ/mol) within which to size the next simulation,
        by default the core SLiM window ``(-47, -23)``.
    temperature : float, optional
        Temperature in Kelvin, by default ``298.15``.
    neff_min : float, optional
        Effective-sample-size floor for "reachable" estimates, by default
        :data:`DEFAULT_NEFF_MIN`.
    min_unbound_eff : float, optional
        Minimum effective number of unbound samples (``N_eff * (1 - f_bound)``)
        for a target ``eps_go`` to count as "reachable". In the ``f -> 1`` tail
        the reweighted ensemble retains no unbound frames, so ``(1 - f)``, and
        hence ``dG``, is unresolved even though the fixed-weight variance looks
        small. This is the rule-of-three censoring bound [Jovanovic1997]_: with
        zero observed unbound events in ``N_eff`` independent samples, the 95%
        bound on the unbound fraction is ``≈ 3/N_eff``, so ``dG`` is only
        bounded, not resolved, below it. By default ``5.0``.
    errors : {'bootstrap', 'delta'}, optional
        Source of the curve's ``dG_sigma`` column, passed through to
        :func:`reweight_curve`; ``'bootstrap'`` (default) is correlation-aware,
        ``'delta'`` is the cheap fixed-weight method. Interchangeable I/O: the
        column name and shape are identical either way.
    n_boot : int, optional
        Bootstrap replicates when ``errors='bootstrap'``, by default ``200``.
    block_size : int, optional
        Moving-block length in frames for the bootstrap (``≈ 2τ`` to respect
        autocorrelation); only used when ``errors='bootstrap'``, by default ``1``.
    seed : int or None, optional
        Seed for the bootstrap generator (reproducibility); only used when
        ``errors='bootstrap'``, by default ``None``.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        ``(curve, diagnostics)``. ``curve`` has columns
        ``['eps_go', 'f_bound', 'dG', 'dG_sigma', 'N_eff', 'N_eff_minor']`` over the
        grid (see :func:`reweight_curve`). ``diagnostics``
        holds the sampled ``eps_go``/``N_k``, the overlap matrix, the weakest
        adjacent overlap link, the reachable ``dG`` edge, per-state self
        effective sample size and transition counts (extend-vs-add), and the
        suggested next ``eps_go``.
    """
    eps_states = np.array([s["eps_go"] for s in states], dtype=float)
    G_all = np.concatenate([s["G"] for s in states])
    N_k = np.array([len(s["G"]) for s in states], dtype=float)
    state_index = np.concatenate([np.full(len(s["G"]), i) for i, s in enumerate(states)])
    beta = 1.0 / (constants.R * temperature)

    if eps_go_grid is None:
        lo, hi = eps_states.min(), eps_states.max()
        eps_go_grid = np.linspace(lo, hi + 0.5 * (hi - lo) / max(len(eps_states) - 1, 1), 201)
    eps_go_grid = np.asarray(eps_go_grid, dtype=float)

    # reweighted curve (delta or bootstrap dG_sigma; identical schema either way)
    curve = reweight_curve(states, c_box, eps_go_grid, temperature=temperature,
                           errors=errors, n_boot=n_boot, block_size=block_size, seed=seed)

    # MBAR object retained for the connectivity diagnostics below.
    # Reduced potentials: u_kn = beta * eps_go_k * G_n (U0 cancels in MBAR).
    u_kn = beta * eps_states[:, None] * G_all[None, :]
    mbar = MBAR(u_kn, N_k)

    # Diagnostics.
    overlap = mbar.overlap_matrix(state_index)
    adj = [(eps_states[i], eps_states[i + 1], float(overlap[i, i + 1]))
           for i in range(len(eps_states) - 1)]
    weakest = min(adj, key=lambda t: t[2]) if adj else None

    # extend-vs-add signals: MBAR ESS at each sampled state (connectivity;
    # low => isolated state) and the state's own binding/unbinding transition
    # count (internal sampling; few transitions => extend that run).
    self_neff, state_transitions = {}, {}
    for i, s in enumerate(states):
        _, ne = mbar.reweight(u_kn[i])
        self_neff[float(s["eps_go"])] = float(ne)
        n_bind, n_unbind = slim._state_transitions(s["bound"].astype(bool))
        state_transitions[float(s["eps_go"])] = (n_bind, n_unbind)

    # reachable (most negative) dG that still has enough effective unbound
    # samples to resolve (1 - f); this stops the edge before the f->1 tail,
    # where dG looks confident but is actually censored.
    unbound_eff = curve.N_eff * (1.0 - curve.f_bound)
    ok = curve[(curve.N_eff >= neff_min) & np.isfinite(curve.dG) & (unbound_eff >= min_unbound_eff)]
    reachable = ok.loc[ok.dG.idxmin(), ["eps_go", "dG", "dG_sigma"]].to_dict() if len(ok) else None

    # suggested next eps_go: max dG variance inside the target dG window
    in_win = curve[(curve.dG <= dg_window[1]) & (curve.dG >= dg_window[0]) & np.isfinite(curve.dG_sigma)]
    if len(in_win):
        suggest = float(in_win.loc[in_win.dG_sigma.idxmax(), "eps_go"])
        suggest_reason = "max dG variance within target window"
    elif weakest is not None:
        suggest = round(0.5 * (weakest[0] + weakest[1]), 2)   # bridge the weakest link
        suggest_reason = "midpoint of weakest overlap link (window not yet reached)"
    else:
        suggest, suggest_reason = None, "n/a"

    diagnostics = {
        "eps_go_sampled": eps_states,
        "N_k": N_k,
        "overlap_matrix": overlap,
        "adjacent_overlap": adj,
        "weakest_link": weakest,
        "self_N_eff": self_neff,
        "state_transitions": state_transitions,
        "reachable_dG_edge": reachable,
        "suggested_next_eps_go": suggest,
        "suggested_reason": suggest_reason,
    }
    return curve, diagnostics


#······································································#
# Held-out validation: reweighting vs interpolation between simulations
#······································································#

# Interpolation schemes benchmarked against reweighting, cheapest last.
INTERPOLATORS = ("linear", "quadratic", "pchip", "nearest")


def interpolate_curve(
        train_eps_go: np.ndarray,
        train_dG: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, np.ndarray]:
    """Interpolated ``dG`` at ``targets`` from simulated ``dG(eps_go)`` points.

    The no-reweighting baseline: draw a curve through the per-simulation
    trajectory ``dG`` estimates and read it off at the target ``eps_go``. Unlike
    reweighting, every scheme here passes exactly through its anchors, so it
    inherits their sampling error, and none of them extrapolate; targets
    outside ``[min, max]`` of ``train_eps_go`` come back ``nan`` rather than
    silently continuing the last segment.

    Parameters
    ----------
    train_eps_go : numpy.ndarray
        Sampled ``eps_go`` of the training simulations, strictly increasing.
    train_dG : numpy.ndarray
        Their trajectory ``dG`` estimates in kJ/mol (finite values only).
    targets : numpy.ndarray
        ``eps_go`` values to predict.

    Returns
    -------
    dict[str, numpy.ndarray]
        One prediction array per scheme in :data:`INTERPOLATORS`; ``nan`` where
        a scheme is undefined (outside the hull, or too few anchors: 2 for
        ``linear``/``pchip``, 3 for ``quadratic``).
    """
    targets = np.asarray(targets, dtype=float)
    predictions = {name: np.full(len(targets), np.nan) for name in INTERPOLATORS}
    if len(train_eps_go) == 0:
        return predictions

    predictions["nearest"] = train_dG[np.abs(targets[:, None] - train_eps_go[None, :]).argmin(axis=1)]
    if len(train_eps_go) >= 2:
        predictions["linear"] = interp1d(train_eps_go, train_dG, kind="linear", bounds_error=False)(targets)
        predictions["pchip"] = PchipInterpolator(train_eps_go, train_dG, extrapolate=False)(targets)
    if len(train_eps_go) >= 3:
        predictions["quadratic"] = interp1d(train_eps_go, train_dG, kind="quadratic", bounds_error=False)(targets)
    return predictions


def holdout_system(
        system_id: str,
        dGs_subset: pd.DataFrame,
        sims: pd.DataFrame,
        bound_q: float,
        coarse: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        stride: int = 1,
        temperature: float = 298.15,
    ) -> pd.DataFrame:
    """Score reweighting against interpolation on held-out simulations.

    Two held-out experiments, both judged against the trajectory ``dG`` of a
    simulation withheld from every predictor:

    ``'refinement'``
        train on the coarse scan (``eps_go`` in ``coarse``), predict every
        refinement simulation, the operational question of whether reweighting
        was a sound basis for choosing where to simulate next.
    ``'LOO'``
        train on all-but-one sampled state, predict the held-out one; every
        simulation serves as a test point once.

    Each system's states are loaded once and MBAR is re-solved per fold, so the
    cost is dominated by the folds, not the trajectory I/O. Systems are
    independent, making this the unit of work for
    :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    system_id : str
        System identifier, matching the ``System`` column of ``dGs_subset`` and
        ``sims``.
    dGs_subset : pandas.DataFrame
        This system's rows of the scan summary; unconverged states (never bound
        or never unbound) and non-finite ``dG`` are dropped here, since they can
        serve neither as training anchors nor as truth.
    sims : pandas.DataFrame
        Simulation directory table with ``System``, ``eps_go`` and ``Path``
        columns (see ``src.utils.get_scan_dirs``).
    bound_q : float
        Bound/unbound dividing point on ``Q`` for this system.
    coarse : tuple[float, ...], optional
        The ``eps_go`` values constituting the original coarse scan, by default
        ``(0, 1, 2, 3, 4, 5)``.
    stride, temperature : optional
        Passed through to :func:`state_samples` (``stride``) and
        :func:`reweight_curve` (``temperature``).

    Returns
    -------
    pandas.DataFrame
        One row per (experiment, target ``eps_go``) with the truth
        (``dG_true``, ``dG_true_sem``, ``traj_transitions``), the fold geometry
        (``n_train``, ``gap`` to the nearest training state, ``bracketed``), the
        reweighted prediction and its support (``pred_mbar``, ``mbar_sigma``,
        ``N_eff``, ``N_eff_minor``), and one ``pred_<scheme>`` column per entry
        of :data:`INTERPOLATORS`. Empty if fewer than three states converge.
    """
    converged = dGs_subset[~dGs_subset.traj_never_binds & ~dGs_subset.traj_never_unbinds]
    converged = converged[np.isfinite(converged.dG)]

    states, c_box = {}, None
    for eps_go in sorted(converged.eps_go.unique()):
        paths = sims[sims.eps_go == eps_go]["Path"]
        if len(paths) == 0:                              # summarised state whose run is gone/too short
            continue
        G, bound, c_box = state_samples(
            paths.iloc[0] + "/top.pdb", paths + f"/{system_id}.dcd",
            paths.iloc[0] + "/specific_interactions.txt", bound_q=bound_q, stride=stride)
        states[eps_go] = {"eps_go": eps_go, "G": G, "bound": bound}
    if len(states) < 3:
        return pd.DataFrame()

    truth = converged.set_index("eps_go")
    sampled = np.array(sorted(states))

    # (experiment, training eps_go, target eps_go) per fold
    folds = []
    coarse_eps = np.array([e for e in sampled if e in coarse])
    refine_eps = np.array([e for e in sampled if e not in coarse])
    if len(coarse_eps) >= 2 and len(refine_eps) > 0:
        folds.append(("refinement", coarse_eps, refine_eps))
    folds += [("LOO", sampled[sampled != eps_go], np.array([eps_go])) for eps_go in sampled]

    rows = []
    for experiment, train_eps_go, targets in folds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # fold-level non-convergence is a reported outcome
            curve = reweight_curve([states[e] for e in train_eps_go], c_box, targets,
                                   temperature=temperature, errors="delta")
        interpolated = interpolate_curve(train_eps_go, truth.loc[train_eps_go, "dG"].to_numpy(), targets)

        for i, target in enumerate(targets):
            state = truth.loc[target]
            rows.append({
                "System": system_id,
                "experiment": experiment,
                "eps_go": target,
                "dG_true": state.dG,
                "dG_true_sem": 0.5 * (state["dG+1SEM"] - state["dG-1SEM"]),
                "traj_transitions": state.traj_transitions,
                "n_train": len(train_eps_go),
                "gap": np.abs(target - train_eps_go).min(),
                "bracketed": bool(train_eps_go.min() <= target <= train_eps_go.max()),
                "pred_mbar": curve.dG.iloc[i],
                "mbar_sigma": curve.dG_sigma.iloc[i],
                "N_eff": curve.N_eff.iloc[i],
                "N_eff_minor": curve.N_eff_minor.iloc[i],
                **{f"pred_{name}": values[i] for name, values in interpolated.items()},
            })
    return pd.DataFrame(rows)
