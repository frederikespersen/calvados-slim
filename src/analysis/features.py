"""
src.analysis.features
------------------------------------------------------------------------
Features for predicting the Go strength ``eps_go`` needed to reach a target
binding free energy, computed from a curated system entry (``systems.json``),
its ``specific_interactions.txt`` contacts, and the CALVADOS residue table.
"""


import numpy as np
import pandas as pd

from src import constants
from src import utils
from src.analysis import slim
from src.calvados.calvados.analysis import ah_potential, yukawa_potential
from src.calvados.calvados.interactions import genParamsDH


#······································································#
# CALVADOS constants and default conditions
#······································································#

# Ashbaugh-Hatch / Debye-Hückel constants, mirroring CALVADOS' defaults
# (``src/calvados/calvados/data/default_config.yaml``); ``eps_lj`` is given in
# kcal/mol there and converted to kJ/mol in ``sim.py`` (``*= 4.184``).
EPS_LJ = 0.2 * 4.184  # kJ/mol
CUTOFF_LJ = 2.0       # nm
CUTOFF_YU = 4.0       # nm

# Default thermodynamic conditions (``data/configurations/conditions.json``).
DEFAULT_CONDITIONS = {"Temperature": 298.15, "Ionic strength": 0.150, "pH": 7.0}

# Residues counted as aromatic for composition features.
AROMATIC = frozenset("FWY")


#······································································#
# Loaders and chain resolution
#······································································#

def load_residue_params(path: str) -> pd.DataFrame:
    """Load a CALVADOS residue-parameter table indexed by one-letter code.

    Parameters
    ----------
    path : str
        Path to a ``residues.csv`` (e.g. ``data/configurations/residues.csv``).

    Returns
    -------
    pandas.DataFrame
        The table indexed by the ``one`` column, carrying at least the
        ``lambdas``, ``sigmas`` and ``q`` columns used here.

    Raises
    ------
    ValueError
        If any of the required columns is missing.
    """
    residues = pd.read_csv(path).set_index("one")
    missing = {"lambdas", "sigmas", "q"} - set(residues.columns)
    if missing:
        raise ValueError(f"Residue table '{path}' is missing columns: {sorted(missing)}.")
    return residues


def resolve_slim(system: dict, slim_index: int = 0) -> tuple[dict, dict, dict]:
    """Resolve the acceptor/donor chains backing a system's SLiM.

    Parameters
    ----------
    system : dict
        A single system entry from ``systems.json``.
    slim_index : int, optional
        Index into the system's ``SLiMs`` list, by default ``0``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(slim, acceptor_chain, donor_chain)``.

    Raises
    ------
    KeyError
        If the SLiM's ``Acceptor ID`` or ``Donor ID`` matches no chain.
    """
    slim_entry = system["SLiMs"][slim_index]
    chains = {chain["ID"]: chain for chain in system["Chains"]}
    try:
        acceptor_chain = chains[slim_entry["Acceptor ID"]]
        donor_chain = chains[slim_entry["Donor ID"]]
    except KeyError as exc:
        raise KeyError(
            f"SLiM '{slim_entry['ID']}' references chain {exc} not present in "
            f"system '{system['ID']}'."
        ) from exc
    return slim_entry, acceptor_chain, donor_chain


def _contact_residue_params(
        specific_interactions: pd.DataFrame,
        acceptor_seq: str,
        donor_seq: str,
        residues: pd.DataFrame,
    ) -> pd.DataFrame:
    """Attach per-contact residue parameters (λ, σ, q) to a contact table.

    Maps each contact's 1-based acceptor/donor residue index onto the
    corresponding residue in the (full, authored) chain sequence and looks up
    its CALVADOS parameters, returning the combining-rule values used by the
    Ashbaugh-Hatch and Debye-Hückel terms.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts (see :func:`src.utils.parse_specific_interactions`).
    acceptor_seq, donor_seq : str
        Full one-letter chain sequences (the CALVADOS beads).
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code.

    Returns
    -------
    pandas.DataFrame
        ``specific_interactions`` with added per-contact columns
        ``['lam_A', 'sig_A', 'q_A', 'lam_D', 'sig_D', 'q_D', 'lam_ij',
        'sig_ij', 'q_ij']``.
    """
    out = specific_interactions.copy()
    acc_res = [acceptor_seq[i - 1] for i in out["Acceptor residue"]]
    don_res = [donor_seq[i - 1] for i in out["Donor residue"]]
    for tag, res in (("A", acc_res), ("D", don_res)):
        out[f"lam_{tag}"] = residues.loc[res, "lambdas"].to_numpy()
        out[f"sig_{tag}"] = residues.loc[res, "sigmas"].to_numpy()
        out[f"q_{tag}"] = residues.loc[res, "q"].to_numpy()
    # Lorentz-Berthelot-style combining rules used by CALVADOS' AH term, and
    # the charge product entering the Debye-Hückel term.
    out["lam_ij"] = 0.5 * (out["lam_A"] + out["lam_D"])
    out["sig_ij"] = 0.5 * (out["sig_A"] + out["sig_D"])
    out["q_ij"] = out["q_A"] * out["q_D"]
    return out


def _count_clusters(residue_indices: np.ndarray, gap: int = 1) -> tuple[int, int]:
    """Count contiguous clusters among 1-based residue indices.

    Parameters
    ----------
    residue_indices : numpy.ndarray
        Residue indices (any order, possibly repeated).
    gap : int, optional
        Maximum sequence separation still joined into one cluster, by default
        ``1`` (consecutive residues).

    Returns
    -------
    tuple[int, int]
        ``(n_clusters, largest_cluster_size)`` in residues; ``(0, 0)`` for an
        empty input.
    """
    unique = np.unique(residue_indices)
    if unique.size == 0:
        return 0, 0
    # A new cluster starts wherever the jump to the next index exceeds ``gap``.
    breaks = np.where(np.diff(unique) > gap)[0]
    clusters = np.split(unique, breaks + 1)
    return len(clusters), max(len(c) for c in clusters)


#······································································#
# A. Box and thermodynamic conditioning
#······································································#

def box_features(system: dict) -> dict:
    """Box-size features setting the standard-state offset of ``Kd``.

    Parameters
    ----------
    system : dict
        A single system entry (uses ``Box sides`` in nm).

    Returns
    -------
    dict
        ``{'V_box_nm3', 'V_box_L', 'C_box'}`` with the box volume in nm³ and L
        and the single-copy concentration ``C_box`` in molar.
    """
    v_nm3 = float(np.prod(system["Box sides"]))
    v_L = v_nm3 * 1e-24  # nm³ -> L
    return {
        "V_box_nm3": v_nm3,
        "V_box_L": v_L,
        "C_box": 1.0 / (constants.N_A * v_L),
    }


#······································································#
# B. Chain length and composition
#······································································#

def _composition(seq: str, residues: pd.DataFrame) -> dict:
    """Composition fractions for a single sequence (helper for both chains)."""
    n = len(seq)
    lam = residues.loc[list(seq), "lambdas"].to_numpy()
    q = residues.loc[list(seq), "q"].to_numpy()
    return {
        "L": n,
        "f_P": seq.count("P") / n,
        "f_G": seq.count("G") / n,
        "f_arom": sum(seq.count(a) for a in AROMATIC) / n,
        "f_charged": np.count_nonzero(q) / n,
        "Q": float(q.sum()),
        "lam_mean": float(lam.mean()),
    }


def chain_features(acceptor_seq: str, donor_seq: str, residues: pd.DataFrame) -> dict:
    """Length, composition and net-charge features of both chains.

    Parameters
    ----------
    acceptor_seq, donor_seq : str
        Full one-letter chain sequences.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code.

    Returns
    -------
    dict
        Per-chain features suffixed ``_A`` (acceptor) and ``_D`` (donor):
        length ``L``, proline/glycine/aromatic/charged fractions, net charge
        ``Q`` and mean hydrophobicity ``lam_mean``.
    """
    out = {}
    for tag, seq in (("A", acceptor_seq), ("D", donor_seq)):
        for key, value in _composition(seq, residues).items():
            out[f"{key}_{tag}"] = value
    return out


#······································································#
# C. Contact count and topology (the Go scalers)
#······································································#

def contact_topology_features(
        specific_interactions: pd.DataFrame,
        len_acceptor: int,
        len_donor: int,
    ) -> dict:
    """Count and topology features of the native-contact set.

    The Go enthalpy scales as ``-eps_go*N_C``, so these counting features are
    the primary scalers of the ``dG``-vs-``eps_go`` slope. Footprint spans,
    valencies, contact clusters (multivalent subsites) and contact-graph
    degree statistics describe how those contacts are distributed.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts (see :func:`src.utils.parse_specific_interactions`).
    len_acceptor, len_donor : int
        Chain lengths, used for interface-fraction features.

    Returns
    -------
    dict
        Contact counts (``N_C``, ``N_RCA``, ``N_RCD``), valencies, donor
        footprint span/density, interface fractions, cluster counts and
        donor-side degree statistics.
    """
    acc = specific_interactions["Acceptor residue"].to_numpy()
    don = specific_interactions["Donor residue"].to_numpy()

    n_c = len(specific_interactions)
    n_rca = np.unique(acc).size
    n_rcd = np.unique(don).size
    span_d = int(don.max() - don.min() + 1) if n_c else 0

    n_sites_d, s_site_d = _count_clusters(don)
    # Donor-side coordination: how many acceptor residues each donor residue
    # engages (the binding "anchors" form first and dominate the transition).
    deg_d = pd.Series(acc).groupby(don).nunique().to_numpy() if n_c else np.array([])

    return {
        "N_C": n_c,
        "N_RCA": n_rca,
        "N_RCD": n_rcd,
        "val_A": n_c / n_rca if n_rca else 0.0,
        "val_D": n_c / n_rcd if n_rcd else 0.0,
        "L_SD": span_d,
        "rho_C": n_c / span_d if span_d else 0.0,
        "f_int_A": n_rca / len_acceptor,
        "f_int_D": n_rcd / len_donor,
        "N_sites_D": n_sites_d,
        "S_site_D": s_site_d,
        "deg_max_D": int(deg_d.max()) if deg_d.size else 0,
        "deg_std_D": float(deg_d.std()) if deg_d.size else 0.0,
    }


#······································································#
# D. Contact geometry (equilibrium distances)
#······································································#

def contact_geometry_features(specific_interactions: pd.DataFrame) -> dict:
    """Equilibrium-distance features of the native contacts.

    Tighter contacts (shorter equilibrium distance) form more readily and
    contribute a deeper, narrower Go well, so a distance-weighted "effective"
    contact count often tracks the ``dG``-vs-``eps_go`` slope better than the
    raw count.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses the ``Equilibrium distance`` column (nm).

    Returns
    -------
    dict
        Mean/std/min/max equilibrium distance, the distance-weighted effective
        contact number ``N_eff`` (``Σ exp(-r0/σ̄)``) and the short-contact
        fraction ``f_short`` (within one mean σ of contact).
    """
    r0 = specific_interactions["Equilibrium distance"].to_numpy(dtype=float)
    if r0.size == 0:
        return {"r0_mean": 0.0, "r0_std": 0.0, "r0_min": 0.0, "r0_max": 0.0,
                "N_eff": 0.0, "f_short": 0.0}
    scale = r0.mean()
    return {
        "r0_mean": float(r0.mean()),
        "r0_std": float(r0.std()),
        "r0_min": float(r0.min()),
        "r0_max": float(r0.max()),
        "N_eff": float(np.exp(-r0 / scale).sum()),
        "f_short": float((r0 < scale).mean()),
    }


#······································································#
# E. Confinement / chain entropy
#······································································#

def confinement_features(
        specific_interactions: pd.DataFrame,
        len_donor: int,
    ) -> dict:
    """Donor tethering features that set the entropic cost of binding.

    Disordered donor residues flanking the contact footprint must be confined
    on binding; longer flanks raise the entropic penalty and the ``eps_go``
    needed to reach a given ``dG``.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses the donor residue indices.
    len_donor : int
        Donor chain length.

    Returns
    -------
    dict
        N-/C-terminal flank lengths (residues outside the donor footprint) and
        the flank fraction of the donor.
    """
    don = specific_interactions["Donor residue"].to_numpy()
    if don.size == 0:
        return {"L_flankN_D": 0, "L_flankC_D": 0, "f_flank_D": 0.0}
    flank_n = int(don.min() - 1)
    flank_c = int(len_donor - don.max())
    return {
        "L_flankN_D": flank_n,
        "L_flankC_D": flank_c,
        "f_flank_D": (flank_n + flank_c) / len_donor,
    }


#······································································#
# F/G. Interface charge and hydrophobicity
#······································································#

def interface_features(contacts_with_params: pd.DataFrame) -> dict:
    """Charge and hydrophobicity features of the contacting interface.

    Built from the per-contact residue parameters: electrostatic
    complementarity (the sign of each contact's charge product) and the
    summed/mean Ashbaugh-Hatch hydrophobicity that drives the
    ``eps_go``-independent background attraction.

    Parameters
    ----------
    contacts_with_params : pandas.DataFrame
        Output of :func:`_contact_residue_params`.

    Returns
    -------
    dict
        Interface net charges, charge-product sum, attractive/repulsive contact
        counts, and summed/mean combined hydrophobicity over contacts.
    """
    q_ij = contacts_with_params["q_ij"].to_numpy()
    lam_ij = contacts_with_params["lam_ij"].to_numpy()
    # Interface net charge per side (unique contacting residues only).
    q_int_a = contacts_with_params.groupby("Acceptor residue")["q_A"].first().sum()
    q_int_d = contacts_with_params.groupby("Donor residue")["q_D"].first().sum()
    return {
        "Qint_A": float(q_int_a),
        "Qint_D": float(q_int_d),
        "Qint_prod": float(q_int_a * q_int_d),
        "q_pair_sum": float(q_ij.sum()),
        "N_attr": int((q_ij < 0).sum()),
        "N_rep": int((q_ij > 0).sum()),
        "sum_lam": float(lam_ij.sum()),
        "mean_lam": float(lam_ij.mean()) if lam_ij.size else 0.0,
    }


#······································································#
# H. CALVADOS background energetics (the dG intercept at eps_go = 0)
#······································································#

def background_energy_features(
        contacts_with_params: pd.DataFrame,
        conditions: dict = DEFAULT_CONDITIONS,
        eps_lj: float = EPS_LJ,
        cutoff_lj: float = CUTOFF_LJ,
        cutoff_yu: float = CUTOFF_YU,
    ) -> dict:
    """``eps_go``-independent CALVADOS background energy over the native
    contacts.

    Evaluates the Ashbaugh-Hatch and Debye-Hückel (Yukawa) potentials at each
    contact's equilibrium distance using the same forms and constants as the
    simulations, and sums them. This is the generic attraction already present
    at ``eps_go = 0`` and thus an estimate of the intercept of the
    ``dG``-vs-``eps_go`` relationship (it is a contact-geometry enthalpy
    proxy, not a calibrated binding free energy).

    Parameters
    ----------
    contacts_with_params : pandas.DataFrame
        Output of :func:`_contact_residue_params` (provides ``lam_ij``,
        ``sig_ij``, ``q_ij`` and ``Equilibrium distance``).
    conditions : dict, optional
        Thermodynamic conditions (``Temperature``, ``Ionic strength``); by
        default :data:`DEFAULT_CONDITIONS`.
    eps_lj, cutoff_lj, cutoff_yu : float, optional
        Ashbaugh-Hatch prefactor (kJ/mol) and LJ/Yukawa cutoffs (nm); by
        default the CALVADOS values :data:`EPS_LJ`, :data:`CUTOFF_LJ`,
        :data:`CUTOFF_YU`.

    Returns
    -------
    dict
        ``{'E_AH', 'E_DH', 'E_bg'}`` in kJ/mol (Ashbaugh-Hatch, Debye-Hückel,
        and their sum over all contacts).
    """
    r0 = contacts_with_params["Equilibrium distance"].to_numpy(dtype=float)
    lam_ij = contacts_with_params["lam_ij"].to_numpy()
    sig_ij = contacts_with_params["sig_ij"].to_numpy()
    q_ij = contacts_with_params["q_ij"].to_numpy()

    eps_yu, k_yu = genParamsDH(conditions["Temperature"], conditions["Ionic strength"])

    e_ah = sum(
        ah_potential(r, s, eps_lj, l, cutoff_lj)
        for r, s, l in zip(r0, sig_ij, lam_ij)
    )
    e_dh = sum(
        yukawa_potential(r, eps_yu * q, k_yu, rc_yu=cutoff_yu)
        for r, q in zip(r0, q_ij)
    )
    return {"E_AH": float(e_ah), "E_DH": float(e_dh), "E_bg": float(e_ah + e_dh)}


#······································································#
# I. Physics-informed composites (need the target dG)
#······································································#

def target_features(
        dG_target: float,
        n_contacts: int,
        background_energy: float,
        conditions: dict = DEFAULT_CONDITIONS,
    ) -> dict:
    """Composite features conditioning on the target free energy.

    Combines the target with the contact count and the ``eps_go = 0``
    background to form physically-motivated predictors. ``eps_go_estimate``
    is a crude first-principles guess from ``dG_target ≈ E_bg - eps_go*N_C``
    (each bound contact adds ``-eps_go``); it is an uncalibrated heuristic,
    not a substitute for the fitted model.

    Parameters
    ----------
    dG_target : float
        Target binding free energy in kJ/mol.
    n_contacts : int
        Number of native contacts ``N_C``.
    background_energy : float
        ``eps_go``-independent background energy ``E_bg`` (kJ/mol).
    conditions : dict, optional
        Thermodynamic conditions (uses ``Temperature``); by default
        :data:`DEFAULT_CONDITIONS`.

    Returns
    -------
    dict
        ``Kd_target`` (M), the per-contact target energy, the background-to-
        target energy gap and the heuristic ``eps_go_estimate``.
    """
    rt = constants.R * conditions["Temperature"]
    gap = background_energy - dG_target
    return {
        "dG_target": dG_target,
        "Kd_target": float(np.exp(dG_target / rt)),
        "dG_per_contact": dG_target / n_contacts if n_contacts else np.nan,
        "dG_gap": gap,
        "eps_go_estimate": max(0.0, gap / n_contacts) if n_contacts else np.nan,
    }


#······································································#
# Orchestration
#······································································#

def featurize(
        system: dict,
        specific_interactions_txt: str,
        residues: pd.DataFrame,
        dG_target: float | None = None,
        slim_index: int = 0,
        conditions: dict = DEFAULT_CONDITIONS,
    ) -> dict:
    """Assemble all features for a single SLiM of a system.

    Resolves the acceptor/donor chains, parses the contacts and attaches their
    residue parameters once, then concatenates every ``*_features`` group into
    a single flat row. The target-conditioned composites are only included when
    ``dG_target`` is given.

    Parameters
    ----------
    system : dict
        A single system entry from ``systems.json``.
    specific_interactions_txt : str
        Path to the system's ``specific_interactions.txt``.
    residues : pandas.DataFrame
        Residue-parameter table (see :func:`load_residue_params`).
    dG_target : float or None, optional
        Target binding free energy in kJ/mol; when given, adds the
        target-conditioned composites, by default ``None``.
    slim_index : int, optional
        Index into the system's ``SLiMs`` list, by default ``0``.
    conditions : dict, optional
        Thermodynamic conditions; by default :data:`DEFAULT_CONDITIONS`.

    Returns
    -------
    dict
        One flat row of features, prefixed by ``System`` and the resolved
        ``SLiM`` id.
    """
    slim_entry, acceptor_chain, donor_chain = resolve_slim(system, slim_index)
    acceptor_seq = acceptor_chain["Sequence"]
    donor_seq = donor_chain["Sequence"]

    contacts = utils.parse_specific_interactions(specific_interactions_txt)
    contacts = _contact_residue_params(contacts, acceptor_seq, donor_seq, residues)

    row = {"System": system["ID"], "SLiM": slim_entry["ID"]}
    row.update(box_features(system))
    row.update(chain_features(acceptor_seq, donor_seq, residues))
    row.update(contact_topology_features(contacts, len(acceptor_seq), len(donor_seq)))
    row.update(contact_geometry_features(contacts))
    row.update(confinement_features(contacts, len(donor_seq)))
    row.update(interface_features(contacts))
    row.update(background_energy_features(contacts, conditions))

    if dG_target is not None:
        row.update(target_features(dG_target, row["N_C"], row["E_bg"], conditions))

    return row
