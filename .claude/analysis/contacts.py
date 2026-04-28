"""
src.analysis.contacts
---------------------
Contact-fraction and binding-affinity analysis for multi-chain Go-model simulations.
"""

############################################################
# Imports
############################################################

import numpy as np
import mdtraj as md
import yaml


############################################################
# Concentration conversion
############################################################

def protein_concentration(
    chains: int,
    lx: float,
    ly: float,
    lz: float,
) -> float:
    """Convert chain count and box volume to molar concentration.

    Parameters
    ----------
    chains : int
        Number of protein chains in the simulation box.
    lx, ly, lz : float
        Box dimensions in nm.

    Returns
    -------
    float
        Concentration in nM.
    """
    NA = 6.02e23
    volume_L = lx * ly * lz * 1e-24  # nm³ → L
    return (chains * 1e9) / (NA * volume_L)  # nM


############################################################
# Contact-fraction computation
############################################################

def compute_contact_fraction(
    traj: md.Trajectory,
    go_pairs: np.ndarray,
    n_pairs_per_repeat: int,
    n_repeats: int,
    n_chains: int,
    go_cutoff: float,
) -> np.ndarray:
    """Compute the fractional contact formation across all repeats and chains.

    For each unique contact type (defined by its position within one repeat),
    a contact is counted if *any* copy of that pair is within ``go_cutoff``.
    The fraction is then normalised by the total number of native contacts.

    Parameters
    ----------
    traj : md.Trajectory
        Full trajectory loaded with mdtraj.
    go_pairs : np.ndarray
        Shape ``(M, 2)`` — all Go-pair atom indices.
    n_pairs_per_repeat : int
        Number of Go pairs in one PRM repeat.
    n_repeats : int
        Number of repeats per chain.
    n_chains : int
        Number of chains of each component.
    go_cutoff : float
        Distance threshold for a contact to be counted (nm).

    Returns
    -------
    np.ndarray
        Shape ``(n_frames,)`` — contact fraction per frame in [0, 1].
    """
    n_total = n_pairs_per_repeat * n_repeats * n_chains
    distances = md.compute_distances(traj, go_pairs)  # (n_frames, M)
    contact_count = np.zeros(len(traj))

    for i in range(n_pairs_per_repeat):
        # Indices of all copies of the i-th pair type across repeats and chains
        copy_indices = np.arange(i, len(go_pairs), n_pairs_per_repeat)
        pair_set = go_pairs[copy_indices]
        d_set = distances[:, copy_indices]  # (n_frames, n_copies)

        # Each unique source atom contributes at most one contact
        for src_atom in set(pair_set[:, 0]):
            mask = pair_set[:, 0] == src_atom
            contact_count += np.any(d_set[:, mask] < go_cutoff, axis=1).astype(int)

    return contact_count / n_total


############################################################
# Binding affinity estimation
############################################################

def estimate_kd(
    contact_fraction: np.ndarray,
    chains: int,
    lx: float,
    ly: float,
    lz: float,
    bound_cutoff: float = 0.05,
) -> float:
    """Estimate the dissociation constant K_d from contact-fraction statistics.

    Uses the two-state binding approximation:
    ``K_d = P_t * f_c + L_t / f_c - P_t - L_t``
    where ``P_t = L_t`` is the total protein concentration and ``f_c`` is the
    fraction of frames classified as bound.

    Parameters
    ----------
    contact_fraction : np.ndarray
        Per-frame contact fractions.
    chains : int
        Number of chains (one type, both equal).
    lx, ly, lz : float
        Box dimensions in nm.
    bound_cutoff : float, optional
        Minimum contact fraction to classify a frame as bound.

    Returns
    -------
    float
        K_d in μM.
    """
    Pt = protein_concentration(chains, lx, ly, lz) * 1e-3  # nM → μM
    Lt = Pt
    fc = np.mean(contact_fraction > bound_cutoff)
    if fc == 0.0 or fc == 1.0:
        return float("nan")
    kd = Pt * fc + Lt / fc - Pt - Lt
    return float(kd)


############################################################
# High-level analysis
############################################################

def multichain_contact_analysis(
    cwd: str,
    dataset: str,
    record: str,
    cycle: int = 0,
    bound_cutoff: float = 0.05,
) -> dict:
    """Load a centred trajectory and compute contact fraction and K_d.

    Parameters
    ----------
    cwd : str
        Root working directory.
    dataset : str
        Dataset subdirectory.
    record : str
        Record identifier (e.g. ``"PRM1_SH31_1_1"``).
    cycle : int, optional
        Simulation cycle from which to read config.
    bound_cutoff : float, optional
        Contact-fraction threshold for bound classification.

    Returns
    -------
    dict
        Keys: ``contact_fraction`` (array), ``kd_uM`` (float),
        ``log_kd`` (float), ``fc`` (float).
    """
    run_dir = f"{cwd}/{dataset}/{record}/{cycle}"
    config = yaml.safe_load(open(f"{run_dir}/config_sim.yaml"))

    go_pairs       = np.load(f"{run_dir}/Go_pairs_list.npy")
    n_per_repeat   = len(np.load(f"{run_dir}/oneRepeat_pair_list.npy"))
    go_cutoff      = config["Go_pairsCutoff"]
    comp_order     = config["components_order"]
    Lx = Ly        = config["L"]
    Lz             = config["Lz"]

    repeats1 = int(record.split("_")[0].replace("PRM", ""))
    chains1  = int(record.split("_")[2])

    traj = md.load_dcd(
        f"{cwd}/{dataset}/{record}/traj_centered.dcd",
        f"{cwd}/{dataset}/{record}/traj_centered.pdb",
    )

    cf = compute_contact_fraction(
        traj, go_pairs, n_per_repeat, repeats1, chains1, go_cutoff
    )
    kd  = estimate_kd(cf, chains1, Lx, Ly, Lz, bound_cutoff)
    fc  = float(np.mean(cf > bound_cutoff))

    return {
        "contact_fraction": cf,
        "kd_uM": kd,
        "log_kd": float(np.log(kd)) if not np.isnan(kd) else float("nan"),
        "fc": fc,
    }
