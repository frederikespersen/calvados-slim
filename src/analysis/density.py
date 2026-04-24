"""
src.analysis.density
--------------------
Slab-geometry density-profile analysis and phase-coexistence concentration
estimation for CALVADOS simulations.
"""

############################################################
# Imports
############################################################

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from scipy.optimize import least_squares

from .blocking import BlockAnalysis
from ..calvados.constants import R_KCAL


############################################################
# Formatting utilities
############################################################

def significant_digit_str(value: float, error: float) -> str:
    """Format a measurement as ``value ± error`` at the correct precision.

    The number of decimal places is inferred from the magnitude of ``error``.

    Parameters
    ----------
    value : float
        Central estimate.
    error : float
        Uncertainty (determines displayed precision).

    Returns
    -------
    str
        LaTeX-compatible string, e.g. ``"1.23 ± 0.04"`` or
        ``"(12 ± 3)×10²"``.
    """
    if str(value) == "nan" or str(error) == "nan":
        return f"{value}\u00B1{error}"

    parts = str(error).split(".")
    int_part = int(parts[0])

    if int_part != 0:
        exponent = len(str(int_part)) - 1
        if exponent == 0:
            return f"{int(np.round(value + 1e-8, 0))}\u00B1{int(np.round(error + 1e-8, 0))}"
        v = value / 10 ** exponent
        e = error / 10 ** exponent
        return (
            f"({int(np.round(v + 1e-8))}\u00B1{int(np.round(e + 1e-8))})"
            f"$\\times{{10}}^{exponent}$"
        )

    # error < 1: find first non-zero decimal digit
    digit = 1
    for ch in parts[1]:
        if ch == "0":
            digit += 1
        else:
            break

    v_str = str(np.round(value, digit))
    n_dec = len(v_str.split(".")[1]) if "." in v_str else 0
    padding = "0" * max(0, digit - n_dec)
    return f"{v_str}{padding}\u00B1{np.round(error, digit)}"


############################################################
# Hyperbolic interface profile
############################################################

def _tanh_profile(
    z: np.ndarray,
    c_dil: float,
    c_den: float,
    z_mid: float,
    width: float,
) -> np.ndarray:
    """Hyperbolic-tangent density profile for a liquid slab.

    Parameters
    ----------
    z : np.ndarray
        Position along z-axis (nm).
    c_dil, c_den : float
        Dilute- and dense-phase concentrations.
    z_mid : float
        Position of the interface (nm).
    width : float
        Interface width parameter (nm).

    Returns
    -------
    np.ndarray
        Concentration at each z.
    """
    return 0.5 * (c_dil + c_den) + 0.5 * (c_den - c_dil) * np.tanh((np.abs(z) - z_mid) / width)


def _fit_interface(
    z: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Fit the tanh profile to one half of the density histogram.

    Parameters
    ----------
    z : np.ndarray
        Bin centres (nm) — one half of the symmetric slab.
    h : np.ndarray
        Mean concentration per bin (mM).

    Returns
    -------
    np.ndarray
        Fitted parameters ``[c_dil, c_den, z_mid, width]``.
    """
    residuals = lambda p, *args: args[1] - _tanh_profile(args[0], *p)
    result = least_squares(residuals, x0=[1.0, 1.0, 1.0, 1.0], args=[z, h],
                           bounds=([0.0] * 4, [100.0] * 4))
    return result.x


############################################################
# Phase-coexistence analysis
############################################################

def calc_profile(
    cwd: str,
    dataset: str,
    record: str,
    t_min: int = 1200,
    t_max: Optional[int] = None,
    plot: bool = False,
    pden: float = 0.5,
    pdil: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute dilute- and dense-phase concentrations from slab density profiles.

    Loads the per-component density profiles saved by
    :func:`calvados.trajectory.calc_slab_profiles`, fits a hyperbolic-tangent
    interface model, and uses :class:`analysis.blocking.BlockAnalysis` to
    estimate the SEM.

    Parameters
    ----------
    cwd : str
        Root working directory.
    dataset : str
        Dataset subdirectory.
    record : str
        Record identifier.
    t_min : int, optional
        First frame to include (equilibration skip).
    t_max : int, optional
        Last frame to include; ``None`` means all.
    plot : bool, optional
        Save a diagnostic figure for each component.
    pden : float, optional
        Dense-phase cutoff as a multiple of the interface width inside the slab.
    pdil : float, optional
        Dilute-phase cutoff as a multiple of the interface width outside the slab.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(value, error)`` DataFrames indexed by component name.
    """
    config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/0/config_sim.yaml"))
    components = config["components"]
    temp = config["temp"]

    value = pd.DataFrame(dtype=object)
    error = pd.DataFrame(dtype=object)

    for name, comp in components.items():
        h = np.load(f"{cwd}/{dataset}/{record}/{record}_sel_profile_{name}.npy")
        N = len(comp["seq"])
        # Convert bead density (beads/nm³) to mM
        conv = 1e4 / 6.022 / N
        h = h[t_min:t_max] * conv

        lz = h.shape[1] + 1
        edges = np.arange(-lz / 2.0, lz / 2.0, 1) / 10.0
        dz = (edges[1] - edges[0]) / 2.0
        z = edges[:-1] + dz
        hm = h.mean(axis=0)

        # Fit both halves of the symmetric slab independently for robustness
        z_pos, z_neg = z[z > 0], z[z < 0]
        h_pos, h_neg = hm[z > 0], hm[z < 0]
        params_pos = _fit_interface(z_pos, h_pos)
        params_neg = _fit_interface(z_neg, h_neg)

        # Interface positions delimiting dense and dilute regions
        cuts_den = [params_pos[2] - pden * params_pos[3],
                    -params_neg[2] + pden * params_neg[3]]
        cuts_dil = [params_pos[2] + pdil * params_pos[3],
                    -params_neg[2] - pdil * params_neg[3]]

        if not (0.5 < abs(cuts_dil[1] / cuts_dil[0]) < 2.0):
            print(f"WARNING: profile not converged for {name}")

        mask_den = np.logical_and(z < cuts_den[0], z > cuts_den[1])
        mask_dil = np.logical_or(z > cuts_dil[0], z < cuts_dil[1])

        dil_series = np.apply_along_axis(lambda a: a[mask_dil].mean(), 1, h)
        den_series = np.apply_along_axis(lambda a: a[mask_den].mean(), 1, h)

        ba_dil = BlockAnalysis(dil_series)
        ba_den = BlockAnalysis(den_series)
        ba_dil.SEM()
        ba_den.SEM()

        value.loc[name, f"{name}_dil"] = ba_dil.av
        value.loc[name, f"{name}_den"] = ba_den.av
        error.loc[name, f"{name}_dil"] = ba_dil.sem
        error.loc[name, f"{name}_den"] = ba_den.sem

        dG = R_KCAL * temp * np.log(dil_series.mean() / den_series.mean())
        print(f"{name}: c_dil = {dil_series.mean():.3f} mM, "
              f"c_den = {den_series.mean():.3f} mM, "
              f"ΔG = {dG:.4f} kcal/mol")

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].plot(z, hm)
            for c1, c2 in zip(cuts_den, cuts_dil):
                axes[0].axvline(c1, color="gray", linestyle="--")
                axes[0].axvline(c2, color="black", linestyle=":")
            axes[0].set(xlabel="z [nm]", ylabel="Concentration [mM]", yscale="log", title=name)
            axes[1].plot(dil_series)
            axes[1].set(xlabel="Frame", ylabel="c_dil [mM]")
            fig.tight_layout()
            plt.savefig(f"{cwd}/{dataset}/{record}/{name}_profile.pdf", dpi=300)
            plt.close(fig)

        value.to_pickle(f"{cwd}/{dataset}/{record}/value_{name}.pkl")
        error.to_pickle(f"{cwd}/{dataset}/{record}/error_{name}.pkl")

    return value, error


############################################################
# Plotting
############################################################

def plot_density_profile(
    cwd: str,
    dataset: str,
    record: str,
    ax: Axes,
    color: str = "blue",
    cycle: int = 0,
    label: Optional[str] = None,
    n_skip: int = 1200,
) -> None:
    """Overlay a density profile and interface markers on an existing axis.

    Parameters
    ----------
    cwd : str
        Root working directory.
    dataset : str
        Dataset subdirectory.
    record : str
        Record identifier.
    ax : matplotlib.axes.Axes
        Target axes.
    color : str, optional
        Line colour.
    cycle : int, optional
        Cycle from which to read the config.
    label : str, optional
        Legend label.
    n_skip : int, optional
        Equilibration frames to discard before computing the mean profile.
    """
    config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml"))
    L = config["L"]
    Lz = config["Lz"]
    fasta = config["seq"]
    N = len(fasta)

    profile_path = f"{cwd}/{dataset}/{record}/{record}.npy"
    if not os.path.isfile(profile_path):
        print(f"Profile not found: {profile_path}")
        return

    h = np.load(profile_path)
    conv = 1.0 / 6.022 / N / L / L / 0.1 * 1e4
    h = h[n_skip:] * conv  # mM

    lz = h.shape[1] + 1
    edges = np.arange(-lz / 2.0, lz / 2.0, 1) / 10.0
    z = edges[:-1] + (edges[1] - edges[0]) / 2.0
    hm = h.mean(axis=0)

    z_pos, z_neg = z[z > 0], z[z < 0]
    params_pos = _fit_interface(z_pos, hm[z > 0])
    params_neg = _fit_interface(z_neg, hm[z < 0])

    cuts_den = [params_pos[2] - 0.5 * params_pos[3],
                -params_neg[2] + 0.5 * params_neg[3]]
    cuts_dil = [params_pos[2] + 6.0 * params_pos[3],
                -params_neg[2] - 6.0 * params_neg[3]]

    ax.plot(z, hm * 1000, color=color, label=label, zorder=99)
    for c1, c2 in zip(cuts_den, cuts_dil):
        ax.axvline(c1, color=color, linestyle="--", linewidth=0.8)
        ax.axvline(c2, color=color, linestyle=":",  linewidth=0.8)
    ax.set_yscale("log")
