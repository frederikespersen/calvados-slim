"""
src.calvados.parameters
-----------------------
Loading and generating residue-level force-field parameters for CALVADOS.
"""

############################################################
# Imports
############################################################

import copy

import numpy as np
import pandas as pd


############################################################
# Parameter loading
############################################################

def load_parameters(flib: str, dataset: str, initial_type: str) -> pd.DataFrame:
    """Load residue force-field parameters from a CSV file.

    Selects the appropriate lambda column based on the CALVADOS variant and
    returns a DataFrame indexed by the one-letter amino-acid code.

    Parameters
    ----------
    flib : str
        Root directory containing dataset subdirectories.
    dataset : str
        Name of the dataset subdirectory.
    initial_type : str
        CALVADOS model variant: ``"C1"``, ``"C2"``, ``"C3"``, ``"0.5"``, or
        ``"Ran"``.

    Returns
    -------
    pd.DataFrame
        Residue parameters with index ``one`` (single-letter codes).
    """
    if initial_type in {"C1", "C2", "C3"}:
        residues = pd.read_csv(f"{flib}/{dataset}/residues_pub.csv").set_index("one", drop=False)
    else:
        residues = pd.read_csv(f"{flib}/{dataset}/residues_-1.csv").set_index("one", drop=False)
    return residues


def create_parameters(flib: str, dataset: str, initial_type: str, tune_W: float = 0.0) -> pd.DataFrame:
    """Write the active lambda column and persist the updated CSV.

    Copies the appropriate CALVADOS variant column into the ``lambdas`` column
    so downstream code always reads from a single, consistent field.

    Parameters
    ----------
    flib : str
        Root directory containing dataset subdirectories.
    dataset : str
        Name of the dataset subdirectory.
    initial_type : str
        CALVADOS variant: ``"C1"``, ``"C2"``, or ``"C3"``.
    tune_W : float, optional
        If non-zero and ``initial_type`` is ``"C2"``, overrides the tryptophan
        lambda with this value.

    Returns
    -------
    pd.DataFrame
        Updated residue parameters.
    """
    residues = pd.read_csv(f"{flib}/{dataset}/residues_pub.csv").set_index("one")
    col = {"C1": "CALVADOS1", "C2": "CALVADOS2", "C3": "CALVADOS3"}[initial_type]
    residues["lambdas"] = residues[col]
    if initial_type == "C2" and tune_W != 0.0:
        residues.loc["W", "lambdas"] = tune_W
    residues.to_csv(f"{flib}/{dataset}/residues_pub.csv")
    return residues


############################################################
# Parameter generation
############################################################

def gen_params_dh(
    residues: pd.DataFrame,
    fasta: list[str],
    temp: float,
    pH: float,
    ionic: float,
) -> tuple[list[float], float]:
    """Compute Debye-Hückel (Yukawa) interaction parameters.

    Assigns terminal charges, adjusts histidine protonation, and derives
    per-residue Yukawa prefactors and the inverse Debye length.

    Parameters
    ----------
    residues : pd.DataFrame
        Residue parameters indexed by one-letter code.
    fasta : list[str]
        Sequence as a list of one-letter residue codes.
    temp : float
        Temperature in Kelvin.
    pH : float
        Solution pH.
    ionic : float
        Ionic strength in molar.

    Returns
    -------
    tuple[list[float], float]
        ``(yukawa_eps, yukawa_kappa)`` — per-residue prefactors in
        nm·kJ/mol and the inverse Debye length in nm⁻¹.
    """
    kT = 8.3145 * temp * 1e-3  # kJ/mol
    seq = copy.deepcopy(fasta)
    r = residues.copy()

    # pH-dependent histidine charge
    r.loc["H", "q"] = 1.0 / (1 + 10 ** (pH - 6))

    # Terminal residues carry +1 (N-term) and -1 (C-term) extra charges
    r.loc["X"] = r.loc[seq[0]]
    r.loc["Z"] = r.loc[seq[-1]]
    r.loc["X", "q"] = r.loc[fasta[0], "q"] + 1.0
    r.loc["Z", "q"] = r.loc[fasta[-1], "q"] - 1.0
    seq[0] = "X"
    seq[-1] = "Z"

    # Bjerrum length lB in nm
    fepsw = lambda T: 5321 / T + 233.76 - 0.9297 * T + 0.1417e-2 * T**2 - 0.8292e-6 * T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / kT

    yukawa_eps = [r.loc[a, "q"] * np.sqrt(lB * kT) for a in seq]
    yukawa_kappa = np.sqrt(8 * np.pi * lB * ionic * 6.022 / 10)
    return yukawa_eps, yukawa_kappa


def gen_params_lj(
    residues: pd.DataFrame,
    fasta: list[str],
) -> tuple[list[str], list[str]]:
    """Prepare per-residue Lennard-Jones type lists with terminal modifications.

    Terminal residues receive modified molecular weights to account for
    NH₂ (+2 Da) and OH (+16 Da) capping groups.

    Parameters
    ----------
    residues : pd.DataFrame
        Residue parameters indexed by one-letter code.
    fasta : list[str]
        Sequence as a list of one-letter residue codes.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(fasta_with_termini, unique_types)`` — sequence with terminal
        pseudo-residue labels, and the sorted unique type list.
    """
    seq = fasta.copy()
    r = residues.copy()
    r.loc["X"] = r.loc[seq[0]]
    r.loc["Z"] = r.loc[seq[-1]]
    r.loc["X", "MW"] += 2
    r.loc["Z", "MW"] += 16
    seq[0] = "X"
    seq[-1] = "Z"
    types = list(np.unique(seq))
    return seq, types
