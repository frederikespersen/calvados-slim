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
common to every state, and MBAR is invariant to such shifts -- it cancels in
every reduced-potential *difference* ``u_{\\varepsilon'} - u_{\\varepsilon} =
\\beta(\\varepsilon' - \\varepsilon)\\,G``. Reweighting between ``eps_go`` states
therefore depends only on the single, cheaply computable coordinate ``G(X)``,
making MBAR here an (essentially exact) 1-D reweighting along ``eps_go``.

From existing scans this module:

1. reweights to a fine ``eps_go`` grid -> ``dG(eps_go)`` with an uncertainty band;
2. builds the state overlap matrix to locate broken links in the ``eps_go`` chain;
3. reports the most negative (tightest-affinity) ``dG`` still reachable at
   acceptable variance;
4. distinguishes "extend this run" (low internal effective sample size) from
   "add a new state" (an overlap gap); and
5. proposes the next ``eps_go`` to simulate -- the point of maximum ``dG``
   variance inside the target window.

This module takes *explicit file paths* (see :func:`state_samples`); the mapping
from systems to trajectory files is the caller's responsibility, so no
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
    *differences* between ``eps_go`` states, ``G`` is the sole quantity required
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
    sr = contact_eqdists[None, :] / contact_dists
    well = 5.0 * sr**12 - 6.0 * sr**10
    well = np.where(contact_dists <= cutoff, well, 0.0)
    return well.sum(axis=1)


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
    contact_dists, contact_eqdists, c_box = slim.TwoChain._load_contacts(
        top_pdb, traj_dcd, specific_interactions_txt, stride=stride)
    G = go_collective_variable(contact_dists, contact_eqdists, cutoff)
    Q = slim._native_contact_fraction(contact_dists, contact_eqdists, beta_q, lambda_q)
    return G, (Q > bound_q).astype(float), c_box


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
        Reduced potential of every pooled sample at every *sampled* state,
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

    With ``Kd = C_box (1-f)^2 / f`` (equimolar single-copy box) and
    ``dG = R T \\ln Kd``, propagating ``Var(f)`` by the delta method gives

    .. math:: \\Delta G = R T\\,[\\ln C_{\\mathrm{box}} + 2\\ln(1-f) - \\ln f],
        \\qquad
        \\sigma_{\\Delta G} = R T\\left(\\frac{2}{1-f} + \\frac{1}{f}\\right)\\sigma_f .

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
    dG_sigma = rt * (2.0 / (1.0 - f_bound) + 1.0 / f_bound) * np.sqrt(max(f_var, 0.0))
    return dG, dG_sigma


#······································································#
# Block-bootstrap uncertainty (correlation-aware, toggleable)
#······································································#

def _circular_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Resample indices for a circular moving-block bootstrap of length ``n``.

    Draws ``⌈n / block_size⌉`` blocks of ``block_size`` consecutive frames from
    random (wrap-around) start points and concatenates them, truncated to ``n``
    [Kunsch1989]_ [Politis1992]_. Resampling *blocks* rather than single frames
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
    replicate ``dG`` values is a correlation-aware uncertainty that -- unlike the
    fixed-weight delta method in :func:`_dg_with_sigma` -- also captures the
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
# Reweighted dG(eps_go) curve -- single entry point
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
    ``errors='bootstrap'`` are drop-in interchangeable for downstream code -- only
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
        *rarer* (minor) state: it collapses in overlap gaps and at censored edges, and is
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
        Minimum effective number of *unbound* samples (``N_eff * (1 - f_bound)``)
        for a target ``eps_go`` to count as "reachable". In the ``f -> 1`` tail
        the reweighted ensemble retains no unbound frames, so ``(1 - f)`` -- and
        hence ``dG`` -- is unresolved even though the fixed-weight variance looks
        small. This is the rule-of-three censoring bound [Jovanovic1997]_: with
        zero observed unbound events in ``N_eff`` independent samples, the 95%
        bound on the unbound fraction is ``≈ 3/N_eff``, so ``dG`` is only
        bounded, not resolved, below it. By default ``5.0``.
    errors : {'bootstrap', 'delta'}, optional
        Source of the curve's ``dG_sigma`` column, passed through to
        :func:`reweight_curve`; ``'bootstrap'`` (default) is correlation-aware,
        ``'delta'`` is the cheap fixed-weight method. Interchangeable I/O -- the
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

    # --- diagnostics ---
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
