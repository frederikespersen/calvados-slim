"""
src.analysis.preprocessing
------------------------------------------------------------------------
Cleaning of ``eps_go``-scan free-energy data before modelling.

At high Go strength the complex is always bound within sampling, so the unbound
population falls below what a finite trajectory resolves and ``dG`` saturates at
a sampling-limited floor. Those censored points carry no signal and bias a fit,
so the floor is computed and the points that have reached it dropped.

The floor follows from the estimator in :mod:`src.analysis.slim`. For an
equimolar two-chain box the dissociation constant is ``Kd = C·(1-f)²/f`` with
bound fraction ``f`` and single-copy concentration ``C = C_box``; near
saturation (``f -> 1``) this is ``Kd ≈ C·ε²`` in the unbound fraction
``ε = 1-f``, so ``dG = RT·ln(Kd) ≈ RT·ln(C·ε²)``. The smallest resolvable ``ε``
is one frame out of ``N``, ``ε_min ≈ 1/N``, giving

    dG_floor ≈ RT·ln(C_box / N_frames²).
"""


import numpy as np
import pandas as pd

from src import constants


#······································································#
# Constants
#······································································#

# Frames per scan trajectory (``data/configurations/sampling.json`` -> "Scan").
DEFAULT_N_FRAMES = 10000

# Default temperature (``data/configurations/conditions.json`` -> "Default").
DEFAULT_TEMPERATURE = 298.15  # K


#······································································#
# Sampling-limited floor
#······································································#

def sampling_floor(
        c_box: float | np.ndarray,
        n_frames: int = DEFAULT_N_FRAMES,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> float | np.ndarray:
    """Sampling-limited ``dG`` floor of a bound-fraction estimator.

    The most negative ``dG`` a scan can resolve, set by the finite trajectory
    length (see the module docstring for the derivation):
    ``dG_floor = RT·ln(C_box / n_frames²)``.

    Parameters
    ----------
    c_box : float or numpy.ndarray
        Single-copy box concentration(s) in molar (see
        :func:`src.analysis.features.box_features`).
    n_frames : int, optional
        Number of frames per scan trajectory, by default
        :data:`DEFAULT_N_FRAMES`.
    temperature : float, optional
        Temperature in Kelvin, by default :data:`DEFAULT_TEMPERATURE`.

    Returns
    -------
    float or numpy.ndarray
        The ``dG`` floor in kJ/mol, matching the shape of ``c_box``.
    """
    return constants.R * temperature * np.log(np.asarray(c_box) / n_frames**2)


def _as_aligned(c_box: float | np.ndarray | pd.Series, index: pd.Index) -> np.ndarray:
    """Broadcast ``c_box`` to a plain array aligned with a frame's index."""
    if isinstance(c_box, pd.Series):
        return c_box.reindex(index).to_numpy()
    return np.broadcast_to(np.asarray(c_box, dtype=float), (len(index),))


#······································································#
# Floor filtering
#······································································#

def floor_hit_mask(
        scan: pd.DataFrame,
        c_box: float | np.ndarray | pd.Series,
        n_frames: int = DEFAULT_N_FRAMES,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> pd.Series:
    """Flag scan points that have reached the sampling-limited floor.

    A point is floor-hit when its more negative confidence bound ``dG-1SEM`` is
    at or below :func:`sampling_floor`, i.e. indistinguishable from full
    saturation. A ``-inf`` bound (bound in every frame) is included; a ``+inf``
    bound (nothing bound) is the opposite, unbound regime, handled by
    ``drop_nonfinite`` in :func:`filter_floor_hit`.

    Parameters
    ----------
    scan : pandas.DataFrame
        Scan table with one row per ``(system, eps_go)``, carrying ``dG-1SEM``.
    c_box : float, numpy.ndarray or pandas.Series
        Single-copy box concentration(s) in molar. A scalar is broadcast; an
        array is used positionally; a ``Series`` is aligned on ``scan.index``.
    n_frames : int, optional
        Frames per scan trajectory, by default :data:`DEFAULT_N_FRAMES`.
    temperature : float, optional
        Temperature in Kelvin, by default :data:`DEFAULT_TEMPERATURE`.

    Returns
    -------
    pandas.Series
        Boolean mask (indexed like ``scan``), ``True`` where floor-hit.
    """
    floor = sampling_floor(_as_aligned(c_box, scan.index), n_frames, temperature)
    return pd.Series(scan["dG-1SEM"].to_numpy() <= floor, index=scan.index)


def filter_floor_hit(
        scan: pd.DataFrame,
        c_box: float | np.ndarray | pd.Series,
        n_frames: int = DEFAULT_N_FRAMES,
        temperature: float = DEFAULT_TEMPERATURE,
        drop_nonfinite: bool = True,
        return_mask: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Drop scan points that are floor-hit (and, by default, non-finite).

    Removes the saturated, sampling-limited points flagged by
    :func:`floor_hit_mask`; with ``drop_nonfinite`` it also removes rows whose
    ``dG`` is not finite (the fully-unbound ``eps_go`` points at the opposite
    end), leaving a table suitable for regression.

    Parameters
    ----------
    scan : pandas.DataFrame
        Scan table with one row per ``(system, eps_go)``.
    c_box : float, numpy.ndarray or pandas.Series
        Single-copy box concentration(s) in molar (see :func:`floor_hit_mask`).
    n_frames : int, optional
        Frames per scan trajectory, by default :data:`DEFAULT_N_FRAMES`.
    temperature : float, optional
        Temperature in Kelvin, by default :data:`DEFAULT_TEMPERATURE`.
    drop_nonfinite : bool, optional
        Also drop rows with a non-finite ``dG``, by default ``True``.
    return_mask : bool, optional
        If ``True``, also return the boolean mask of removed rows, by default
        ``False``.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.Series]
        The filtered table, or ``(filtered, removed_mask)`` when
        ``return_mask`` is ``True``.
    """
    removed = floor_hit_mask(scan, c_box, n_frames, temperature)
    if drop_nonfinite:
        removed = removed | ~np.isfinite(scan["dG"].to_numpy())
    filtered = scan[~removed].copy()
    return (filtered, removed) if return_mask else filtered


#······································································#
# Non-finite dG filling (local linear extrapolation of the censored tails)
#······································································#

def _fill_curve_nonfinite(x: np.ndarray, y: np.ndarray, n_fit: int) -> tuple[np.ndarray, np.ndarray]:
    """Fill non-finite ``y`` of one sorted curve; return ``(y_filled, filled_mask)``.

    ``x`` must be sorted ascending. Interior non-finite points (between the first
    and last finite point) are linearly interpolated from the finite neighbours;
    the leading/trailing non-finite runs are each replaced by the line through
    the nearest ``n_fit`` finite points at that end (a local tangent). Only the
    finite (never the just-filled) values feed the end fits.
    """
    y = y.astype(float, copy=True)
    finite = np.isfinite(y)
    filled = ~finite
    if finite.sum() < 2:
        raise ValueError(
            f"Need at least 2 finite points to extrapolate, got {int(finite.sum())}.")
    pos = np.arange(len(y))
    idx = np.flatnonzero(finite)
    lo_i, hi_i = idx[0], idx[-1]

    interior = filled & (pos > lo_i) & (pos < hi_i)
    if interior.any():
        y[interior] = np.interp(x[interior], x[finite], y[finite])

    k = min(n_fit, len(idx))
    below = pos < lo_i
    if below.any():
        m, b = np.polyfit(x[idx[:k]], y[idx[:k]], 1)
        y[below] = m * x[below] + b
    above = pos > hi_i
    if above.any():
        m, b = np.polyfit(x[idx[-k:]], y[idx[-k:]], 1)
        y[above] = m * x[above] + b
    return y, filled


def fill_nonfinite_dG(scan: pd.DataFrame, n_fit: int = 3) -> pd.DataFrame:
    """Fill censored (``±inf``) ``dG`` values by local linear extrapolation.

    Functional PCA needs a complete curve on the common ``eps_go`` grid, but at
    the extreme Gō strengths the bound fraction saturates (``f -> 1``, tight
    binding) or vanishes (``f -> 0``, no binding) and ``dG`` is reported as
    ``-inf`` / ``+inf`` (see :func:`src.analysis.mbar._dg_with_sigma`). This
    replaces those non-finite ends, within each ``System`` curve, by the line
    through the nearest ``n_fit`` finite points at that end
    (:func:`_fill_curve_nonfinite`). The filled points are placeholders keeping
    the fPCA basis well-defined, not resolved free energies, and are marked in
    ``dG_extrapolated`` so they can be down-weighted downstream.

    Parameters
    ----------
    scan : pandas.DataFrame
        Curve table with one row per ``(System, eps_go)`` (e.g.
        ``mbar_curves.csv``). Rows are sorted by ``eps_go`` within each system
        before extrapolating.
    n_fit : int, optional
        Number of finite points at each end used for the local linear fit, by
        default ``3`` (``2`` gives the endpoint tangent; larger averages
        curvature over a wider window).

    Returns
    -------
    pandas.DataFrame
        A copy of ``scan`` (original row order preserved) with a finite ``dG``
        everywhere and a Boolean ``dG_extrapolated``.

    Raises
    ------
    ValueError
        If any system has fewer than two finite ``dG`` values to anchor the
        extrapolation.
    """
    out = scan.copy()
    dG = out["dG"].to_numpy(dtype=float, copy=True)
    eps_go = out["eps_go"].to_numpy()
    filled_all = np.zeros(len(out), dtype=bool)
    for _, pos in out.groupby("System", sort=False).indices.items():
        order = pos[np.argsort(eps_go[pos], kind="stable")]
        dG[order], filled_all[order] = _fill_curve_nonfinite(eps_go[order], dG[order], n_fit)
    out["dG"] = dG
    out["dG_extrapolated"] = filled_all
    return out


#······································································#
# Convergence filtering (binding/unbinding reversibility)
#······································································#

def filter_reversible(
        scan: pd.DataFrame,
        return_mask: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Keep only runs that sampled both binding and unbinding.

    Drops rows flagged as unconverged by the ``Q``-derived state-transition
    warnings (see :func:`src.analysis.slim._warn_never_binds` /
    :func:`src.analysis.slim._warn_never_unbinds`): ``no binding`` marks runs
    whose SLiM never associates (``dG = +inf``, unusable) and ``no unbinding``
    runs whose SLiM never dissociates (``dG`` only a lower bound, biased). A row
    survives only when both are ``False``, so the trajectory observed the full
    bind/unbind cycle.

    An alternative to :func:`filter_floor_hit`: instead of testing whether
    ``dG`` reached the sampling-limited floor, it tests whether the reaction was
    reversible in the trajectory.

    Parameters
    ----------
    scan : pandas.DataFrame
        Scan table carrying the ``no binding`` and ``no unbinding`` flags.
    return_mask : bool, optional
        If ``True``, also return the Boolean mask of removed rows, by default
        ``False``.

    Returns
    -------
    pandas.DataFrame or tuple[pandas.DataFrame, pandas.Series]
        The reversibly-sampled rows, or ``(filtered, removed_mask)`` when
        ``return_mask`` is ``True``.

    Raises
    ------
    KeyError
        If either flag column is absent from ``scan``.
    """
    missing = {"no binding", "no unbinding"} - set(scan.columns)
    if missing:
        raise ValueError(
            f"Scan table is missing convergence-flag columns: {sorted(missing)}. "
            "Regenerate dGs.csv with the state-transition flags active."
        )
    removed = scan["no binding"].astype(bool) | scan["no unbinding"].astype(bool)
    filtered = scan[~removed].copy()
    return (filtered, removed) if return_mask else filtered
