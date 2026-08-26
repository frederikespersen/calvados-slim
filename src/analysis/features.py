"""
src.analysis.features
------------------------------------------------------------------------
Zero-shot features for predicting the Go well depth ``eps_go`` a CALVADOS
two-chain SLiM-domain system needs to reach a target binding free energy.

Everything is computed from static inputs: a curated ``systems.json`` entry, its
``specific_interactions.txt`` native contacts, the CALVADOS residue table and,
for the interface-area group, the complex structure. No trajectory is required.
The acceptor is the folded domain, the donor the (disordered) SLiM peptide.

Features are expressed in the coarse-grained frame the simulation integrates:
one bead per residue at its Calpha, carrying its CALVADOS ``lambda``, ``sigma``
and ``q``. A feature describing physics the force field has no term for cannot
drive the binding free energy it is asked to predict, so all-atom surfaces,
backbone rigidity (no angle or dihedral term) and pi-stacking (no aromatic term)
are absent.

:func:`featurize` assembles the default row from the groups the sections below
are organised into, ordered by the role the quantity plays in the simulation
rather than by how it is computed: the box and chain setup, the Go term, the
interface surface, the background energy at ``eps_go`` = 0, the acceptor as an
object, the entropic cost, and composites.

The Go-term and interface groups follow from thermodynamic perturbation theory on
the Go potential. For the single-copy box the exact statement is
``d(dG)/d(eps_go) = (1+f)(<G>_b - <G>_u)`` with ``<G> ~ -N_formed``: the location
of the transition is set by how many native contacts the bound ensemble can
satisfy, and its steepness by ``Var(N_formed)``.

Two groups are implemented but not wired into :func:`featurize`, as they require
optional dependencies and did not improve prediction: the configurational-entropy
group :func:`entropy_features` (``sparrow`` and ``localCIDER``) and the ESM-2
embedding :func:`esm_embedding` (``fair-esm`` and ``torch``).

Every feature function returns a flat ``{name: value}`` dict; each feature
carries a short inline description where it is returned.
"""


import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import ConvexHull

from src import constants
from src import utils
from src.analysis import bsa
from src.analysis import slim
from src.calvados.calvados.analysis import ah_potential, yukawa_potential
from src.calvados.calvados.interactions import genParamsDH


#······································································#
# Constants
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

# Standard amino-acid one-letter codes (for the optional sequence predictors).
_STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


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


#······································································#
# Setup: the box and the two chains
#······································································#

#······································································#
# 1. Box and thermodynamic conditioning
#······································································#

def box_features(system: dict) -> dict:
    """Box-size features setting the standard-state offset of ``Kd``.

    The single-copy box concentration ``C_box`` is the reference concentration
    the equimolar two-chain ``Kd`` is measured against, so it shifts the whole
    ``dG(eps_go)`` curve vertically.

    Parameters
    ----------
    system : dict
        A single system entry (uses ``Box sides`` in nm).

    Returns
    -------
    dict
        Box volume (nm³ and L) and the single-copy concentration ``C_box`` (M).
    """
    v_nm3 = float(np.prod(system["Box sides"]))
    v_L = v_nm3 * 1e-24  # nm³ -> L
    return {
        "V_box_nm3": v_nm3,                       # box volume (nm³)
        "V_box_L": v_L,                           # box volume (litres)
        "C_box": 1.0 / (constants.N_A * v_L),     # single-copy concentration (M) = Kd standard-state offset
    }


#······································································#
# 2. Chain length and composition
#······································································#

def _composition(seq: str, residues: pd.DataFrame) -> dict:
    """Composition fractions and net charge for a single sequence."""
    n = len(seq)
    lam = residues.loc[list(seq), "lambdas"].to_numpy()
    q = residues.loc[list(seq), "q"].to_numpy()
    return {
        # f_P, f_G and f_arom were removed: CALVADOS has a uniform harmonic bond and
        # no angle, dihedral or π-stacking term, so proline rigidity, glycine
        # flexibility and aromatic stacking are chemistry the force field cannot
        # represent. Aromatics still enter through their λ.
        "L": n,                                   # chain length (residues)
        "f_charged": np.count_nonzero(q) / n,     # fraction of charged residues
        "Q": float(q.sum()),                      # net charge
        "lam_mean": float(lam.mean()),            # mean CALVADOS hydrophobicity ("stickiness")
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
        The per-sequence descriptors of :func:`_composition`, suffixed ``_A``
        (acceptor / domain) and ``_D`` (donor / SLiM).
    """
    out = {}
    for tag, seq in (("A", acceptor_seq), ("D", donor_seq)):
        for key, value in _composition(seq, residues).items():
            out[f"{key}_{tag}"] = value
    return out


#······································································#
# The Go term: what eps_go multiplies
#······································································#

#······································································#
# 3. Contact count and topology  (the Gō scalers - set the transition location)
#······································································#

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


def contact_topology_features(
        specific_interactions: pd.DataFrame,
        len_acceptor: int,
        len_donor: int,
    ) -> dict:
    """Count and topology features of the native-contact set.

    The Gō stabilisation scales as ``-eps_go*N_C``, so these counting features
    are the dominant scalers of the ``dG``-vs-``eps_go`` transition location.
    Footprint spans, valencies and contact clusters describe how those contacts
    are distributed along the two chains.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts (see :func:`src.utils.parse_specific_interactions`).
    len_acceptor, len_donor : int
        Chain lengths, used for the interface-fraction features.

    Returns
    -------
    dict
        Contact counts, valencies, donor footprint span/density, interface
        fractions, contact-cluster counts and donor-side degree statistics.
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
        "N_C": n_c,                               # native-contact count - the dominant Gō scaler
        "N_RC": n_rca + n_rcd,                    # distinct contacting residues
        "N_RCA": n_rca,                           # distinct contacting acceptor residues (acceptor footprint)
        "N_RCD": n_rcd,                           # distinct contacting donor residues (donor footprint)
        "N_RCmin": min(n_rca, n_rcd),             # distinct contacting acceptor or donor residues, whichever is smaller
        "N_RCmax": max(n_rca, n_rcd),             # distinct contacting acceptor or donor residues, whichever is smaller
        "val_A": n_c / n_rca if n_rca else 0.0,   # contacts per acceptor residue (acceptor multivalency)
        "val_D": n_c / n_rcd if n_rcd else 0.0,   # contacts per donor residue (donor multivalency)
        "L_SD": span_d,                           # donor footprint span (first→last contacting residue)
        "rho_C": n_c / span_d if span_d else 0.0, # contact density along the donor footprint
        "f_int_A": n_rca / len_acceptor,          # fraction of the acceptor in the interface
        "f_int_D": n_rcd / len_donor,             # fraction of the donor in the interface
        "N_sites_D": n_sites_d,                   # number of contiguous donor contact clusters (subsites)
        "S_site_D": s_site_d,                     # residues in the largest donor contact cluster
        "deg_max_D": int(deg_d.max()) if deg_d.size else 0,   # partners of the most-coordinated donor residue (anchor)
        "deg_std_D": float(deg_d.std()) if deg_d.size else 0.0,  # spread of donor-residue coordination
    }


#······································································#
# 4. Contact geometry  (equilibrium distances)
#······································································#

def contact_geometry_features(specific_interactions: pd.DataFrame) -> dict:
    """Equilibrium-distance features of the native contacts.

    Tighter contacts (shorter equilibrium distance) form more readily and give a
    deeper, narrower Gō well, so a distance-weighted "effective" contact count
    often tracks the transition better than the raw count.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses the ``Equilibrium distance`` column (nm).

    Returns
    -------
    dict
        Summary statistics of the contact equilibrium distances, the distance-
        weighted effective contact number ``N_eff`` and the short-contact
        fraction ``f_short``.
    """
    r0 = specific_interactions["Equilibrium distance"].to_numpy(dtype=float)
    if r0.size == 0:
        return {"r0_mean": 0.0, "r0_std": 0.0, "r0_min": 0.0, "r0_max": 0.0,
                "N_eff": 0.0, "f_short": 0.0}
    scale = r0.mean()
    return {
        "r0_mean": float(r0.mean()),              # mean contact equilibrium distance (nm) - overall looseness
        "r0_std": float(r0.std()),                # spread of contact distances
        "r0_min": float(r0.min()),                # tightest contact
        "r0_max": float(r0.max()),                # loosest contact
        "N_eff": float(np.exp(-r0 / scale).sum()),  # distance-weighted effective contact count (tight contacts weigh more)
        "f_short": float((r0 < scale).mean()),    # fraction of contacts tighter than the mean
    }


#······································································#
# 5. Contact-graph cooperativity  (how contacts are wired - sets steepness)
#······································································#

def _bipartite_components(acceptor: np.ndarray, donor: np.ndarray) -> tuple[int, int]:
    """Connected components of the bipartite acceptor-donor contact graph.

    Nodes are the contacting acceptor and donor residues; each contact is an
    edge. Independent components are separate binding sub-modules, which bind
    quasi-independently and so spread the ``dG``-vs-``eps_go`` transition out
    (shallower), whereas a single interlocked component binds cooperatively
    (steeper).

    Parameters
    ----------
    acceptor, donor : numpy.ndarray
        1-based acceptor/donor residue indices, paired per contact.

    Returns
    -------
    tuple[int, int]
        ``(n_components, largest_component_residues)``; ``(0, 0)`` if empty.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    nodes = set()
    for a, d in zip(acceptor, donor):
        na, nd = ("A", int(a)), ("D", int(d))
        nodes.update((na, nd))
        ra, rd = find(na), find(nd)
        if ra != rd:
            parent[ra] = rd
    if not nodes:
        return 0, 0
    sizes = {}
    for node in nodes:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return len(sizes), max(sizes.values())


def contact_graph_features(specific_interactions: pd.DataFrame) -> dict:
    """Cooperativity descriptors of the bipartite acceptor-donor contact graph.

    Where :func:`contact_topology_features` counts contacts (which set where
    the transition sits), these describe how the contacts are wired - a single
    interlocked interface versus several independent subsites - which sets how
    steep the transition is (the second fPCA mode). They are ratios / graph
    counts, largely invariant to the raw contact count.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts (uses the acceptor/donor residue indices).

    Returns
    -------
    dict
        Connected-component count/size/fraction, bipartite density, node-degree
        statistics, and the cyclomatic number / redundant-contact fraction.
    """
    acc = specific_interactions["Acceptor residue"].to_numpy()
    don = specific_interactions["Donor residue"].to_numpy()
    n_c = len(specific_interactions)
    if n_c == 0:
        return {"N_comp": 0, "S_comp_max": 0, "f_comp_max": 0.0, "graph_density": 0.0,
                "deg_mean": 0.0, "deg_max_A": 0, "deg_std_A": 0.0,
                "N_cycles": 0, "f_redundant": 0.0}

    n_rca = np.unique(acc).size
    n_rcd = np.unique(don).size
    n_comp, s_comp_max = _bipartite_components(acc, don)
    # Acceptor-side coordination (mirror of the donor-side degree in topology).
    deg_a = pd.Series(don).groupby(acc).nunique().to_numpy()
    # Cyclomatic number of the bipartite graph: edges beyond a spanning forest.
    # Extra edges are redundant "bridges" that interlock the interface into one
    # cooperative unit, so more cycles -> steeper transition.
    n_cycles = n_c - (n_rca + n_rcd) + n_comp
    return {
        "N_comp": n_comp,                         # number of independent binding sub-modules (components)
        "S_comp_max": s_comp_max,                 # residues in the largest binding module
        "f_comp_max": s_comp_max / (n_rca + n_rcd),  # fraction of interface residues in that module (single-unit-ness)
        "graph_density": n_c / (n_rca * n_rcd),   # bipartite contact density (how fully the footprints interconnect)
        "deg_mean": 2 * n_c / (n_rca + n_rcd),    # mean contacts per interface residue
        "deg_max_A": int(deg_a.max()),            # partners of the most-coordinated acceptor residue (anchor)
        "deg_std_A": float(deg_a.std()),          # spread of acceptor-residue coordination
        "N_cycles": int(n_cycles),                # redundant "bridge" contacts that interlock the interface
        "f_redundant": n_cycles / n_c,            # fraction of contacts that are redundant bridges (cooperativity)
    }


#······································································#
# 6. Native-distance heterogeneity  (contact-formation variance → steepness)
#······································································#

# Contact distance (nm) beyond which the short-ranged Gō well (12-10) contributes
# little at typical well depths - used for the "loose contact" fraction.
LOOSE_CONTACT_R0 = 0.8


def contact_heterogeneity_features(specific_interactions: pd.DataFrame) -> dict:
    """Spread of the native-contact distances - a contact-formation variance proxy.

    The curvature of the transition follows the fluctuation-response relation
    ``∂²dG/∂eps_go² = -β·Var(N_formed)``, so steepness (the second fPCA mode) is
    governed by how much the number of formed contacts fluctuates. The Gō well is
    short-ranged, so a contact's equilibrium distance sets how readily it forms: a
    narrow ``r0`` distribution means the contacts engage together (cooperative,
    steep transition), a broad one means they engage piecewise across a range
    of ``eps_go`` (gradual, shallow).

    Complements :func:`contact_geometry_features`, which reports only the mean,
    standard deviation and extremes; here the shape of the distribution is
    described.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses the ``Equilibrium distance`` column (nm).

    Returns
    -------
    dict
        Interquartile range, skew, relative dispersion, loose-contact fraction
        and the participation ratio / entropy of the distance-derived contact
        weights.
    """
    r0 = specific_interactions["Equilibrium distance"].to_numpy(dtype=float)
    if r0.size < 2:
        return {"r0_iqr": np.nan, "r0_skew": np.nan, "r0_cv": np.nan, "r0_p90": np.nan,
                "f_loose": np.nan, "w_participation": np.nan, "w_entropy": np.nan}

    mean, std = r0.mean(), r0.std()
    q25, q75 = np.percentile(r0, [25, 75])
    # Distance-derived contact weights (as in N_eff): how the "binding strength"
    # is shared out over contacts. A few dominant contacts -> low participation.
    weights = np.exp(-r0 / mean)
    share = weights / weights.sum()
    return {
        "r0_iqr": float(q75 - q25),                # spread of the middle half of contact distances (nm)
        "r0_skew": float(((r0 - mean) ** 3).mean() / std ** 3) if std > 0 else np.nan,  # tail asymmetry (+ = a few loose outliers)
        "r0_cv": float(std / mean) if mean > 0 else np.nan,  # relative dispersion (scale-free heterogeneity)
        "r0_p90": float(np.percentile(r0, 90)),    # upper-decile contact distance (the marginal contacts)
        "f_loose": float((r0 > LOOSE_CONTACT_R0).mean()),  # fraction of contacts too long to contribute much
        "w_participation": float(1.0 / (share ** 2).sum() / r0.size),  # effective fraction of contacts carrying the binding (1 = all equal)
        "w_entropy": float(-(share * np.log(share)).sum() / np.log(r0.size)),  # evenness of the contact-weight distribution (1 = uniform)
    }


#······································································#
# 7. Contact-graph spectra  (separability into sub-modules → steepness)
#······································································#

def contact_spectral_features(specific_interactions: pd.DataFrame) -> dict:
    """Spectral separability of the contact graph - a cooperativity descriptor.

    Where :func:`contact_graph_features` counts disconnected components, an
    interface is usually one component that is nevertheless nearly split into
    quasi-independent sub-sites. Such an interface binds in stages (two shallow
    steps rather than one sharp one), which broadens the transition - the second
    fPCA mode again.

    The algebraic connectivity (second-smallest eigenvalue ``λ₂`` of the
    normalised Laplacian) measures exactly this near-separability, and the sign
    structure of its eigenvector (the Fiedler vector) gives the best bipartition,
    whose cut fraction says how weakly the two halves are coupled.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts (uses the acceptor/donor residue indices).

    Returns
    -------
    dict
        Algebraic connectivity, Fiedler cut fraction and balance, and the
        degree-distribution entropy of the contact graph.
    """
    acc = specific_interactions["Acceptor residue"].to_numpy()
    don = specific_interactions["Donor residue"].to_numpy()
    empty = {"lambda2": np.nan, "f_cut": np.nan, "cut_balance": np.nan, "deg_entropy": np.nan}
    if len(acc) < 2:
        return empty

    # Bipartite adjacency: acceptor nodes first, then donor nodes.
    acc_nodes = {r: i for i, r in enumerate(np.unique(acc))}
    don_nodes = {r: i + len(acc_nodes) for i, r in enumerate(np.unique(don))}
    n_nodes = len(acc_nodes) + len(don_nodes)
    if n_nodes < 3:
        return empty
    adjacency = np.zeros((n_nodes, n_nodes))
    for a, d in zip(acc, don):
        i, j = acc_nodes[a], don_nodes[d]
        adjacency[i, j] += 1.0
        adjacency[j, i] += 1.0

    degree = adjacency.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    laplacian = np.eye(n_nodes) - adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    values, vectors = np.linalg.eigh(laplacian)

    fiedler = vectors[:, 1]
    side = fiedler > 0
    # Fraction of contacts crossing the best bipartition: low -> two weakly
    # coupled sub-sites that can bind separately.
    crossing = sum(1 for a, d in zip(acc, don) if side[acc_nodes[a]] != side[don_nodes[d]])
    smaller = min(int(side.sum()), int((~side).sum()))
    probability = degree / degree.sum()
    nonzero = probability[probability > 0]

    return {
        "lambda2": float(values[1]),               # algebraic connectivity (→0 = nearly separable into sub-sites)
        "f_cut": float(crossing / len(acc)),       # share of contacts bridging the two sub-sites (low = modular)
        "cut_balance": float(smaller / n_nodes),   # how evenly the interface splits (→0.5 = two equal modules)
        "deg_entropy": float(-(nonzero * np.log(nonzero)).sum() / np.log(n_nodes)),  # evenness of coordination (1 = uniform, low = anchor-dominated)
    }


#······································································#
# 8. Interface diffuseness  (cutoff sensitivity of the contact count)
#······································································#

# Cross-chain Cα-Cα cutoffs (Å) over which the contact count is profiled.
DIFFUSENESS_CUTOFFS = (6.0, 8.0, 10.0, 12.0)


def interface_diffuseness_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
    ) -> dict:
    """How sharply defined the interface is, from its contact-count profile.

    The native-contact set is a thresholded view of a continuous distance
    distribution. Counting cross-chain residue pairs over a range of cutoffs
    measures how sharply the interface is defined: a count that grows slowly with
    the cutoff means a well-separated, sharply defined interface (cooperative), a
    fast-growing one means a diffuse interface with many marginal near-contacts
    that engage progressively (gradual transition).

    Structure-only, so it is independent of the SMOG shadow-map definition used
    for ``N_C`` - a useful cross-check on that choice as well as a feature.

    Parameters
    ----------
    structure_file : str
        Path to the full-atom complex (``.pdb``/``.cif``).
    acceptor_chain_id, donor_chain_id : str
        Author chain ids of the acceptor and donor.

    Returns
    -------
    dict
        Cross-chain pair counts at the tightest/widest cutoff, their ratio, the
        logarithmic growth rate, and the closest cross-chain approach.
    """
    acc_ca = bsa.ca_coordinates(structure_file, acceptor_chain_id)
    don_ca = bsa.ca_coordinates(structure_file, donor_chain_id)
    distance = np.sqrt(((acc_ca[:, None] - don_ca[None, :]) ** 2).sum(-1))

    counts = np.array([float((distance < c).sum()) for c in DIFFUSENESS_CUTOFFS])
    tight, wide = counts[0], counts[-1]
    # Log-linear growth rate of the count with the cutoff (Å⁻¹); a compact,
    # well-separated interface grows more slowly than a diffuse one.
    positive = counts > 0
    slope = (np.polyfit(np.array(DIFFUSENESS_CUTOFFS)[positive], np.log(counts[positive]), 1)[0]
             if positive.sum() >= 2 else np.nan)

    return {
        "N_close": tight,                          # cross-chain residue pairs within the tightest cutoff (core contacts)
        "N_wide": wide,                            # pairs within the widest cutoff (core + periphery)
        "diffuseness": float(wide / tight) if tight > 0 else np.nan,  # periphery-to-core ratio (high = fuzzy interface)
        "diffuse_slope": float(slope),             # log-growth rate of the contact count with cutoff (Å⁻¹)
        "d_min_cross": float(distance.min()),      # closest cross-chain Cα approach (Å)
    }


#······································································#
# 9. Force-field relaxation of the crystal pose  (needs the complex)
#······································································#
# The Go contacts are defined from the crystal, with each ``r0`` set to the crystal
# distance, so at that geometry every contact sits at its well bottom and the Go collective
# variable is ``-N_C`` by construction. Under the full force field it is not: sterics and
# electrostatics push the donor off the crystal geometry, and mutually frustrated contacts
# cannot all stay satisfied. What survives a minimisation is an effective contact number.
#
# Thermodynamic perturbation theory on the Go term gives
# ``d(dG)/d(eps_go) = (1+f)(<G>_bound - <G>_unbound)``, so the transition is set by how many
# contacts the bound ensemble holds rather than by ``N_C``. The minimisation is the
# zero-temperature estimate of that quantity: it gives the amplitude, while the thermal
# ensemble holds fewer.
#
# Everything here is a deterministic function of structure and force field, so the group
# remains zero-shot. It is the most expensive group (~0.04 s median).

# Backbone bond parameters of the CALVADOS chain (``kb`` in components.yaml).
BOND_K = 8033.0        # kJ/mol/nm²
BOND_R0 = 0.38         # nm - Cα-Cα spacing

# Cross-chain Cα-Cα distance defining the relaxation's Go map (nm).
RELAX_GO_CUTOFF = 1.0

# Reference well depth the pose is relaxed at (kJ/mol); mid-range for the scans.
RELAX_EPS_GO = 3.0

# Minimiser settings: converge hard, since a truncated minimisation reports a
# partially-relaxed pose. Long donors have needed >5000 iterations.
RELAX_OPTIONS = {"maxiter": 20000, "maxfun": 20000, "ftol": 1e-12, "gtol": 1e-8}

# Bead separation, nm, counted as an intra-chain contact in a relaxed structure.
SELF_CONTACT_NM = 0.8
# Root of the simulation tree holding the CALVADOS starting topologies.
SIM_ROOT = "sim"
# Free-donor Langevin run: BAOAB at CALVADOS conditions.
FREE_STEPS, FREE_SEEDS, FREE_STRIDE = 12000, 2, 60
FREE_DT, FREE_FRICTION = 0.005, 20.0    # ps, 1/ps (over-damped: equilibrium
                                        # moments are friction-independent, and
                                        # strong damping keeps the integrator safe)
KB_KJ = 0.008314462618                  # kJ/mol/K
# Register scoring: tolerated mismatch in λ and in charge before a shifted register
# stops resembling the native one.
REG_LAM_SCALE, REG_Q_SCALE, REG_ALT_CUT = 0.30, 1.0, 0.6
# Per-seed jitter, nm, applied to a shared starting conformation.
START_JITTER_NM = 0.05
# Surface patches: minimum bead SASA (Å²) to count as exposed, linkage distance (Å)
# joining two beads into one patch, and the distance (Å) marking the native site.
PATCH_SASA_MIN, PATCH_LINK_A, PATCH_SITE_A = 5.0, 8.0, 8.0



def _ah_pair(r: np.ndarray, sig: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ashbaugh-Hatch energy and ``dE/dr``, cut and shifted at :data:`CUTOFF_LJ`.

    Vectorised and differentiated because the minimiser needs an analytic
    gradient over whole distance matrices; :func:`background_energy_features`
    uses the CALVADOS fork's scalar ``ah_potential`` for energy-only evaluation.

    The piecewise split at the Lennard-Jones minimum is smooth in both value and
    slope: at ``r = 2^(1/6)·sigma`` the LJ term is stationary, so both branches
    meet with zero slope.
    """
    sr6 = (sig / r) ** 6
    lj = 4 * EPS_LJ * (sr6 ** 2 - sr6)
    inside = r <= sig * 2 ** (1 / 6)
    within = r < CUTOFF_LJ
    sr6_c = (sig / CUTOFF_LJ) ** 6
    lj_c = 4 * EPS_LJ * (sr6_c ** 2 - sr6_c)
    shift = np.where(CUTOFF_LJ <= sig * 2 ** (1 / 6), lj_c + EPS_LJ * (1 - lam), lam * lj_c)
    energy = np.where(within, np.where(inside, lj + EPS_LJ * (1 - lam), lam * lj) - shift, 0.0)
    dlj = 4 * EPS_LJ * (-12 * sr6 ** 2 + 6 * sr6) / r
    return energy, np.where(within, np.where(inside, dlj, lam * dlj), 0.0)


def _dh_pair(r: np.ndarray, qq: np.ndarray, eps_yu: float, k_yu: float
             ) -> tuple[np.ndarray, np.ndarray]:
    """Debye-Hückel (Yukawa) energy and ``dE/dr``, cut and shifted at :data:`CUTOFF_YU`."""
    scaled = eps_yu * qq
    expo = np.exp(-k_yu * r)
    within = r < CUTOFF_YU
    shift = scaled * np.exp(-k_yu * CUTOFF_YU) / CUTOFF_YU
    energy = np.where(within, scaled * expo / r - shift, 0.0)
    return energy, np.where(within, -scaled * expo * (k_yu / r + 1.0 / r ** 2), 0.0)


def _go_pair(r: np.ndarray, r0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gō 12-10 well shape and ``dE/dr``; the well bottoms at ``-1`` per contact."""
    sr = r0 / r
    return 5.0 * sr ** 12 - 6.0 * sr ** 10, 60.0 * (sr ** 10 - sr ** 12) / r


def relaxation_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
        eps_go: float = RELAX_EPS_GO,
        conditions: dict = DEFAULT_CONDITIONS,
    ) -> dict:
    """Relax the coarse-grained crystal pose and report what the force field holds.

    Mirrors the production setup: the folded acceptor is held rigid (it is
    harmonically restrained to the crystal there) and the disordered donor is
    free (unrestrained, held only by the Gō contacts), so only the donor's
    coordinates are minimised. The energy is the full CALVADOS model (Ashbaugh-
    Hatch, Debye-Hückel, backbone bonds) plus the Gō term at ``eps_go``.

    Residue parameters are read off the *structure's* resolved residues (see
    :func:`src.analysis.bsa.chain_sequence`), not the authored chain sequence,
    which also contains unresolved residues and expression tags.

    Parameters
    ----------
    structure_file : str
        Path to the full-atom complex (``.pdb``/``.cif``).
    acceptor_chain_id, donor_chain_id : str
        Author chain ids of the acceptor and donor.
    residues : pandas.DataFrame
        Residue-parameter table (see :func:`load_residue_params`).
    eps_go : float, optional
        Gō well depth to relax at (kJ/mol); by default :data:`RELAX_EPS_GO`.
        The relaxed pose depends on it, since deeper wells hold more contacts, so
        it must be held fixed across systems for the features to be comparable.
    conditions : dict, optional
        Thermodynamic conditions; by default :data:`DEFAULT_CONDITIONS`.

    Returns
    -------
    dict
        Structure-derived contact count, start/end energies, the relaxed Gō
        collective variable and the fraction of the map it represents, the donor
        displacement, and the crystal's backbone-bond strain. All ``nan`` if the
        donor is too short or has no cross-chain contacts.

    Notes
    -----
    ``relax_bond_strain`` doubles as a curation check: the coarse-grained
    chain bonds consecutive resolved residues at 0.38 nm, so a large value flags
    a structure where unresolved gaps silently pull distant residues together
    (this is how ``pp1c-ppp1r2`` was caught).
    """
    empty = {"relax_n_go": np.nan, "relax_E_start": np.nan, "relax_E_min": np.nan,
             "relax_dE": np.nan, "relax_G_min": np.nan, "relax_f_kept": np.nan,
             "relax_f_formed": np.nan, "relax_rmsd": np.nan, "relax_bond_strain": np.nan}

    # Coarse-grained coordinates: Cα beads in nm (structures are in Ångström).
    acceptor = bsa.ca_coordinates(structure_file, acceptor_chain_id) / 10.0
    donor = bsa.ca_coordinates(structure_file, donor_chain_id) / 10.0
    n_donor = len(donor)
    if n_donor < 2:
        return empty

    seq_a = bsa.chain_sequence(structure_file, acceptor_chain_id)
    seq_d = bsa.chain_sequence(structure_file, donor_chain_id)
    lam_a, lam_d = (residues.loc[list(s), "lambdas"].to_numpy(float) for s in (seq_a, seq_d))
    sig_a, sig_d = (residues.loc[list(s), "sigmas"].to_numpy(float) for s in (seq_a, seq_d))
    q_a, q_d = (residues.loc[list(s), "q"].to_numpy(float) for s in (seq_a, seq_d))
    eps_yu, k_yu = genParamsDH(conditions["Temperature"], conditions["Ionic strength"])

    # Cross-chain combining rules, and the Gō map read off the crystal geometry.
    lam_x = 0.5 * (lam_a[:, None] + lam_d[None, :])
    sig_x = 0.5 * (sig_a[:, None] + sig_d[None, :])
    qq_x = q_a[:, None] * q_d[None, :]
    crystal = np.linalg.norm(acceptor[:, None] - donor[None, :], axis=-1)
    go_mask = crystal < RELAX_GO_CUTOFF
    n_go = int(go_mask.sum())
    if n_go == 0:
        return empty
    r0_go = crystal[go_mask]

    # Intra-donor non-bonded pairs (bonded neighbours excluded).
    left, right = np.triu_indices(n_donor, k=2)
    lam_dd = 0.5 * (lam_d[left] + lam_d[right])
    sig_dd = 0.5 * (sig_d[left] + sig_d[right])
    qq_dd = q_d[left] * q_d[right]

    def energy(flat):
        """Total energy and its analytic gradient w.r.t. the donor coordinates."""
        x = flat.reshape(n_donor, 3)
        delta = x[None, :, :] - acceptor[:, None, :]
        cross = np.maximum(np.linalg.norm(delta, axis=-1), 1e-3)
        e_ah, d_ah = _ah_pair(cross, sig_x, lam_x)
        e_dh, d_dh = _dh_pair(cross, qq_x, eps_yu, k_yu)
        dedr = d_ah + d_dh
        e_go, d_go = _go_pair(cross[go_mask], r0_go)
        dedr[go_mask] += eps_go * d_go
        total = eps_go * e_go.sum() + e_ah.sum() + e_dh.sum()
        grad = ((dedr / cross)[:, :, None] * delta).sum(axis=0)

        pair_vec = x[left] - x[right]
        intra = np.maximum(np.linalg.norm(pair_vec, axis=-1), 1e-3)
        e_ah_i, d_ah_i = _ah_pair(intra, sig_dd, lam_dd)
        e_dh_i, d_dh_i = _dh_pair(intra, qq_dd, eps_yu, k_yu)
        total += e_ah_i.sum() + e_dh_i.sum()
        pair_grad = ((d_ah_i + d_dh_i) / intra)[:, None] * pair_vec
        np.add.at(grad, left, pair_grad)
        np.add.at(grad, right, -pair_grad)

        bond_vec = np.diff(x, axis=0)
        bond = np.maximum(np.linalg.norm(bond_vec, axis=-1), 1e-9)
        total += 0.5 * BOND_K * ((bond - BOND_R0) ** 2).sum()
        bond_grad = (BOND_K * (bond - BOND_R0) / bond)[:, None] * bond_vec
        grad[1:] += bond_grad
        grad[:-1] -= bond_grad
        return total, grad.ravel()

    e_start = energy(donor.ravel())[0]
    result = minimize(energy, donor.ravel(), method="L-BFGS-B", jac=True, options=RELAX_OPTIONS)
    relaxed = result.x.reshape(n_donor, 3)

    cross_min = np.maximum(np.linalg.norm(acceptor[:, None] - relaxed[None, :], axis=-1), 1e-3)
    g_min = float(_go_pair(cross_min[go_mask], r0_go)[0].sum())
    # A contact counts as still formed within 1.2*r0 (the Gō well is short-ranged).
    formed = int((cross_min[go_mask] < 1.2 * r0_go).sum())
    bond_crystal = np.linalg.norm(np.diff(donor, axis=0), axis=-1)

    return {
        "relax_n_go": n_go,                       # cross-chain contacts in the CG crystal geometry
        "relax_E_start": float(e_start),          # energy of the unrelaxed crystal pose (kJ/mol)
        "relax_E_min": float(result.fun),         # energy after relaxation (kJ/mol)
        "relax_dE": float(e_start - result.fun),  # energy released - how strained the crystal pose is
        "relax_G_min": g_min,                     # relaxed Gō CV = −(effective contact number)
        "relax_f_kept": float(-g_min / n_go),     # fraction of the map the force field holds at once
        "relax_f_formed": float(formed / n_go),   # fraction of contacts still within 1.2*r0
        "relax_rmsd": float(np.sqrt(((relaxed - donor) ** 2).sum(-1).mean())),  # donor displacement (nm)
        "relax_bond_strain": float(np.abs(bond_crystal - BOND_R0).mean()),  # mean |Cα-Cα − 0.38| (nm); curation check
    }


#······································································#
# The interface surface, in the coarse-grained frame
#······································································#

#······································································#
# 10. Interface structure  (buried area + 3-D compactness; needs the complex)
#······································································#

# Minimum per-residue buried surface (Å²) to count a residue as interfacial -
# below this is Shrake-Rupley sampling noise, not real burial.
IFACE_BSA_MIN = 1.0


def _patch_shape(points: np.ndarray) -> dict:
    """Gyration-tensor shape descriptors of a 3-D point cloud (the interface patch).

    Motivated by cooperativity - a tight, globular patch should bind as one unit
    (steep transition), an extended one more piecewise (shallow). Empirically the
    extent terms (``iface_Rg``, ``iface_dmax``) turn out to carry location
    (first-mode) signal, behaving as another interface-size descriptor, while the
    pure shape terms (anisotropy, flatness, density) carry little of either -
    so treat this group as a size/extent descriptor, not a steepness one.

    Parameters
    ----------
    points : numpy.ndarray
        Interface residue centroids, shape ``(P, 3)``, Å.

    Returns
    -------
    dict
        Radius of gyration, patch diameter, residue density and the gyration
        anisotropy/flatness shares (``nan`` if fewer than 3 points).
    """
    if len(points) < 3:
        return {"iface_Rg": np.nan, "iface_dmax": np.nan, "iface_density": np.nan,
                "iface_aniso": np.nan, "iface_flat": np.nan}
    centred = points - points.mean(axis=0)
    gyration = centred.T @ centred / len(points)
    ev = np.sort(np.linalg.eigvalsh(gyration))[::-1]      # descending principal components
    rg2 = float(ev.sum())
    rg = float(np.sqrt(rg2))
    dmax = float(np.sqrt(((points[:, None] - points[None, :])**2).sum(-1)).max())
    return {
        "iface_Rg": rg,                           # spatial extent of the interface patch (Å) - small = tight cluster
        "iface_dmax": dmax,                       # interface diameter (Å) - largest residue-residue separation
        "iface_density": len(points) / rg**3 if rg > 0 else np.nan,  # interface residues per Å³ (3-D packing density)
        "iface_aniso": float(ev[0] / rg2) if rg2 > 0 else np.nan,    # elongation (1/3 = globular blob, →1 = extended strip)
        "iface_flat": float(ev[2] / rg2) if rg2 > 0 else np.nan,     # flatness (→0 = planar patch, 1/3 = globular)
    }


def _cg_beads(
        structure_file: str,
        chain_id: str,
        residues: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Cα bead centres (Å) and CALVADOS bead radii (σ/2, Å) of one chain.

    Indexed over the chain's resolved residues, matching
    :func:`src.analysis.bsa.ca_coordinates` and :func:`~src.analysis.bsa.chain_sequence`.
    """
    positions = bsa.ca_coordinates(structure_file, chain_id)
    sequence = bsa.chain_sequence(structure_file, chain_id)
    # residues.csv stores sigma in nm; the bsa machinery works in Å.
    radii = residues.loc[list(sequence), "sigmas"].to_numpy(float) * 10.0 / 2.0
    return positions, radii


def interface_area_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
    ) -> dict:
    """Buried-surface and 3-D shape descriptors of the acceptor-donor interface.

    Computed in the coarse-grained frame: one bead per residue at its Cα,
    each carrying its CALVADOS σ, so the occlusion reported is occlusion the force
    field can feel. An all-atom Shrake-Rupley surface would describe side-chain
    packing that a one-bead model has no representation of. Where ``N_C`` counts
    contacts, buried area measures how much bead surface is actually occluded, and
    the 3-D terms describe the extent/shape of the patch. The extent terms
    (``iface_Rg``, ``iface_dmax``) empirically behave as further interface-size
    descriptors rather than the steepness descriptors they were motivated as.

    Parameters
    ----------
    structure_file : str
        Path to the complex (``.pdb``/``.cif``), e.g. a SLiM entry's ``PDB`` field.
    acceptor_chain_id, donor_chain_id : str
        Author chain ids of the acceptor and donor in that structure.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code; supplies ``sigmas``.

    Returns
    -------
    dict
        Buried-area descriptors (size, engagement, packing, anchor dominance)
        plus the 3-D patch-shape descriptors of :func:`_patch_shape`.
    """
    acc_xyz, acc_rad = _cg_beads(structure_file, acceptor_chain_id, residues)
    don_xyz, don_rad = _cg_beads(structure_file, donor_chain_id, residues)
    bsa_map = bsa.cg_buried_surface_area(acc_xyz, acc_rad, don_xyz, don_rad)
    acc_burial = bsa_map.sum(axis=1)   # per-acceptor-bead involvement
    don_burial = bsa_map.sum(axis=0)   # per-donor-bead involvement
    acc_is_iface = acc_burial > IFACE_BSA_MIN
    don_is_iface = don_burial > IFACE_BSA_MIN
    n_iface_a = int(acc_is_iface.sum())
    n_iface_d = int(don_is_iface.sum())
    total = float(bsa_map.sum())
    n_iface = n_iface_a + n_iface_d
    don_sum = float(don_burial.sum())

    # 3-D shape of the interface patch, from the buried beads' own centres.
    patch = np.vstack([acc_xyz[acc_is_iface], don_xyz[don_is_iface]]) \
        if n_iface else np.empty((0, 3))

    return {
        # Buried area: interface size and packing.
        "BSA_total": total,                       # total buried surface area (Å²) - structural interface size
        "N_iface_A": n_iface_a,                   # acceptor residues at the interface (by burial)
        "N_iface_D": n_iface_d,                   # donor residues at the interface (by burial)
        "f_iface_D": n_iface_d / bsa_map.shape[1] if bsa_map.shape[1] else 0.0,  # fraction of the donor buried (engagement)
        "BSA_per_iface": total / n_iface if n_iface else 0.0,  # mean burial per interface residue (packing tightness)
        "BSA_frac_topD": float(don_burial.max() / don_sum) if don_sum > 0 else 0.0,  # burial share of the top donor residue (anchor dominance)
        # 3-D patch shape: cooperativity, hence transition steepness.
        **_patch_shape(patch),
    }


#······································································#
# 11. Relative burial  (how DEEPLY the SLiM is engulfed, not how widely)
#······································································#
# Every other interface descriptor is a breadth measure: absolute area, or a thresholded
# count of engaged residues. None divides by the surface a bead actually has, so none
# separates a motif whose beads each shed a fifth of their surface from one that is
# engulfed. Note that the contact map is defined from the crystal, so at that geometry
# every interface is buried by construction; what varies between systems is how much of it
# the thermal ensemble holds.

# Relative solvent accessibility below which a bead counts as buried - the
# conventional structural-biology cut, applied to the coarse-grained surface.
RSA_BURIED = 0.25

# Shell thickness, Å, above the closest acceptor bead that still counts as part of
# the binding site when locating it on the domain.
CG_SITE_SHELL = 4.0


def relative_burial_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
    ) -> dict:
    """Depth of donor burial, normalised by the surface each bead actually has.

    Divides the coarse-grained burial map by the donor's isolated-chain bead SASA
    (:func:`src.analysis.bsa.cg_residue_sasa`), giving the fraction of available
    surface that binding removes. Normalising over the interface beads rather
    than the whole chain keeps this independent of flank length, which varies
    widely across the library.

    Parameters
    ----------
    structure_file : str
        Path to the complex.
    acceptor_chain_id, donor_chain_id : str
        Author chain ids of the acceptor and donor.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code; supplies ``sigmas``.

    Returns
    -------
    dict
        Chain- and interface-level burial fractions, the relative-accessibility
        distribution over interface beads, and anchor depth.
    """
    acc_xyz, acc_rad = _cg_beads(structure_file, acceptor_chain_id, residues)
    don_xyz, don_rad = _cg_beads(structure_file, donor_chain_id, residues)
    buried = bsa.cg_buried_surface_area(acc_xyz, acc_rad, don_xyz, don_rad).sum(axis=0)
    free = bsa.cg_residue_sasa(don_xyz, don_rad)

    iface = buried > IFACE_BSA_MIN
    n_iface = int(iface.sum())
    if not n_iface or free.sum() <= 0:
        return {k: 0.0 for k in
                ("f_burial_D", "f_burial_iface", "rsa_bound_iface", "drsa_mean_iface",
                 "drsa_max", "f_deep_D", "n_deep_D", "burial_gini_D")}

    # Relative accessibility remaining after binding, per interface bead. The
    # denominator is the bead's own free surface, so a fully engulfed bead scores 0.
    free_iface = np.maximum(free[iface], 1e-9)
    rsa_bound = np.clip(free_iface - buried[iface], 0.0, None) / free_iface
    share = buried[iface] / buried[iface].sum()
    gini = float((2.0 * np.arange(1, n_iface + 1) - n_iface - 1)
                 @ np.sort(share) / n_iface)

    return {
        "f_burial_D": float(buried.sum() / free.sum()),   # chain-level burial fraction (confounded by flank length)
        "f_burial_iface": float(buried[iface].sum() / free_iface.sum()),  # over engaged beads only - length-invariant
        "rsa_bound_iface": float(rsa_bound.mean()),       # accessibility left after binding (low ⇒ engulfed)
        "drsa_mean_iface": float((1.0 - rsa_bound).mean()),  # mean relative surface lost
        "drsa_max": float((buried[iface] / free_iface).max()),  # deepest single bead (anchor depth)
        "f_deep_D": float((rsa_bound < RSA_BURIED).mean()),  # fraction of the footprint engulfed
        "n_deep_D": float((rsa_bound < RSA_BURIED).sum()),   # count of engulfed beads
        "burial_gini_D": gini,                            # burial concentration (0 even, →1 one dominant anchor)
    }


#······································································#
# Background energy: CALVADOS at eps_go = 0
#······································································#

#······································································#
# 12. Interface charge and hydrophobicity  (per-contact residue parameters)
#······································································#

def interface_features(contacts_with_params: pd.DataFrame) -> dict:
    """Charge and hydrophobicity features of the contacting interface.

    Built from the per-contact residue parameters: electrostatic complementarity
    (the sign of each contact's charge product) and the summed/mean Ashbaugh-
    Hatch hydrophobicity that drives the ``eps_go``-independent background
    attraction.

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
        "Qint_A": float(q_int_a),                 # net charge of the acceptor interface residues
        "Qint_D": float(q_int_d),                 # net charge of the donor interface residues
        "Qint_prod": float(q_int_a * q_int_d),    # charge complementarity (negative ⇒ attractive)
        "q_pair_sum": float(q_ij.sum()),          # summed per-contact charge products
        "N_attr": int((q_ij < 0).sum()),          # oppositely-charged (attractive) contacts
        "N_rep": int((q_ij > 0).sum()),           # like-charged (repulsive) contacts
        "sum_lam": float(lam_ij.sum()),           # total interface hydrophobicity (drives background attraction)
        "mean_lam": float(lam_ij.mean()) if lam_ij.size else 0.0,  # mean per-contact hydrophobicity
    }


#······································································#
# 13. CALVADOS background energetics  (the dG intercept at eps_go = 0)
#······································································#

def background_energy_features(
        contacts_with_params: pd.DataFrame,
        conditions: dict = DEFAULT_CONDITIONS,
        eps_lj: float = EPS_LJ,
        cutoff_lj: float = CUTOFF_LJ,
        cutoff_yu: float = CUTOFF_YU,
    ) -> dict:
    """``eps_go``-independent CALVADOS background energy over the native contacts.

    Evaluates the Ashbaugh-Hatch and Debye-Hückel (Yukawa) potentials at each
    contact's equilibrium distance with the same forms and constants as the
    simulations, and sums them. This is the generic attraction already present at
    ``eps_go = 0`` - an estimate of the intercept of the ``dG``-vs-``eps_go``
    relationship (a contact-geometry enthalpy proxy, not a calibrated free
    energy).

    Parameters
    ----------
    contacts_with_params : pandas.DataFrame
        Output of :func:`_contact_residue_params` (provides ``lam_ij``,
        ``sig_ij``, ``q_ij`` and ``Equilibrium distance``).
    conditions : dict, optional
        Thermodynamic conditions (``Temperature``, ``Ionic strength``); by
        default :data:`DEFAULT_CONDITIONS`.
    eps_lj, cutoff_lj, cutoff_yu : float, optional
        Ashbaugh-Hatch prefactor (kJ/mol) and LJ/Yukawa cutoffs (nm); by default
        the CALVADOS values :data:`EPS_LJ`, :data:`CUTOFF_LJ`, :data:`CUTOFF_YU`.

    Returns
    -------
    dict
        Ashbaugh-Hatch, Debye-Hückel and total background energy (kJ/mol).
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
    return {
        "E_AH": float(e_ah),                      # Ashbaugh-Hatch (hydrophobic/vdW) background energy (kJ/mol)
        "E_DH": float(e_dh),                      # Debye-Hückel (electrostatic) background energy (kJ/mol)
        "E_bg": float(e_ah + e_dh),               # total eps_go-independent background attraction (dG intercept)
    }


#······································································#
# 14. Screened electrostatics and charge patterning
#······································································#

def electrostatic_features(
        acceptor_seq: str,
        donor_seq: str,
        row: dict,
        residues: pd.DataFrame,
        conditions: dict = DEFAULT_CONDITIONS,
    ) -> dict:
    """Long-range screened electrostatics and donor charge patterning.

    :func:`interface_features` sums charge products over native contacts, i.e.
    at contact range - but CALVADOS' Debye-Hückel term is long-ranged, screened
    over ``κ⁻¹`` (~8 Å at 150 mM). At that range the chains interact as net
    charges, so the monopole product and how the interface charge compares with
    the rest of the chain both matter, independently of the contact-level detail.

    Charge patterning is added for the donor: a blocky, highly charged SLiM
    behaves very differently from a uniformly mixed one of the same net charge,
    both in how expanded it is when free and in how it approaches the domain. The
    measure is the Sawle-Ghosh sequence charge decoration (SCD).

    Parameters
    ----------
    acceptor_seq, donor_seq : str
        Full one-letter chain sequences.
    row : dict
        Feature row so far; uses ``Q_A``/``Q_D`` (net charges) and ``Qint_A``/
        ``Qint_D`` (interface charges).
    residues : pandas.DataFrame
        Residue-parameter table (see :func:`load_residue_params`).
    conditions : dict, optional
        Thermodynamic conditions; by default :data:`DEFAULT_CONDITIONS`.

    Returns
    -------
    dict
        Debye screening length, the screened monopole interaction, the
        interface-versus-bulk charge segregation of both chains, and the donor's
        charge patterning and off-footprint charge.
    """
    _eps_yu, k_yu = genParamsDH(conditions["Temperature"], conditions["Ionic strength"])
    debye = 1.0 / k_yu                             # screening length, nm

    q_a, q_d = row["Q_A"], row["Q_D"]
    charges = residues.loc[list(donor_seq), "q"].to_numpy(dtype=float)
    n_don = len(charges)
    # Sawle-Ghosh SCD: charge products weighted by sqrt of sequence separation.
    # Negative = attractive patterning (compacting), positive = repulsive/blocky.
    separation = np.sqrt(np.abs(np.subtract.outer(np.arange(n_don), np.arange(n_don))))
    pair_charge = np.outer(charges, charges)
    upper = np.triu_indices(n_don, k=1)
    scd = float((pair_charge[upper] * separation[upper]).sum() / n_don) if n_don > 1 else np.nan

    n_iface_d = row.get("N_RCD", 0)
    # Charge the donor carries outside its binding footprint: this part cannot
    # help the specific site, but it does set how the free peptide behaves.
    q_outside_d = q_d - row["Qint_D"]

    return {
        "debye_nm": float(debye),                  # Debye screening length (nm) at the simulated ionic strength
        "QAQD_screened": float(q_a * q_d * np.exp(-1.0)),  # net-charge product damped over one screening length
        "Q_seg_A": float(row["Qint_A"] - q_a * row["N_RCA"] / max(len(acceptor_seq), 1)),  # acceptor interface charge minus its per-residue share (patch segregation)
        "Q_seg_D": float(row["Qint_D"] - q_d * n_iface_d / max(n_don, 1)),  # donor interface charge segregation
        "SCD_D": scd,                              # donor charge patterning (blockiness; <0 compacting, >0 expanding)
        "Q_outside_D": float(q_outside_d),         # donor charge outside the binding footprint (flank electrostatics)
        "q_asym": float(abs(q_a) - abs(q_d)),      # charge-magnitude asymmetry between the chains
    }


#······································································#
# 15. System size and non-specific competition
#······································································#

def competition_features(row: dict) -> dict:
    """Bulk "stickiness" of the chains that competes with the specific site.

    Beyond the native contacts, every residue of the acceptor carries a generic
    CALVADOS attraction, so a large, sticky domain offers non-native surface that
    can transiently hold the donor and compete with specific binding. The
    background energy :func:`background_energy_features` deliberately sums only
    over the native contacts, so that competing bulk is not otherwise represented
    - these composites add it (all products/ratios of already-computed features,
    which a linear score model cannot form on its own).

    Parameters
    ----------
    row : dict
        The partially-assembled feature row; must already contain ``L_A``,
        ``L_D``, ``lam_mean_A``, ``lam_mean_D``, ``Q_A``, ``Q_D``, ``N_RCA``,
        ``N_C`` and ``E_bg``.

    Returns
    -------
    dict
        Total and non-interface stickiness capacities, competition ratios against
        the specific site, and the non-specific electrostatic magnitude.

    Notes
    -----
    Empirically these carry little location signal beyond the plain size terms
    already present (``L_A``, ``lam_mean_A``) and slightly degrade the operational
    error, but ``compete_NC`` is one of the few features with genuine steepness
    (second-mode) signal - competition per specific contact plausibly acts on
    cooperativity rather than on where the transition sits.
    """
    sticky_a = row["L_A"] * row["lam_mean_A"]
    sticky_nonif_a = max(row["L_A"] - row["N_RCA"], 0) * row["lam_mean_A"]
    n_c = row["N_C"]
    e_bg = abs(row["E_bg"])
    return {
        "sticky_A": sticky_a,                     # total acceptor stickiness capacity (size × hydrophobicity)
        "sticky_D": row["L_D"] * row["lam_mean_D"],   # donor stickiness capacity
        "sticky_nonif_A": sticky_nonif_a,         # non-interface acceptor surface competing for the donor
        "compete_NC": sticky_nonif_a / n_c if n_c else np.nan,      # competing bulk per specific contact
        "compete_Ebg": sticky_nonif_a / e_bg if e_bg > 1e-6 else np.nan,  # non-specific vs specific attraction
        "absQAQD": abs(row["Q_A"] * row["Q_D"]),  # non-specific electrostatic magnitude between the chains
    }


#······································································#
# 16. Non-specific competition for the SLiM  (who else is pulling on it)
#······································································#
# The Go potential acts only on native pairs, but Ashbaugh-Hatch and Debye-Huckel act
# between every pair, so sticky or oppositely-charged surface outside the interface competes
# for the donor and raises the ``eps_go`` needed. Three competitors the rest of the
# vocabulary cannot separate:
#
#   1. The acceptor's unrestrained tails. ``PDB Domains`` fixes what the run restrains; the
#      rest is free chain that can reach and wrap. ``sticky_nonif_A`` lumps it together with
#      folded surface through ``L_A``.
#   2. The donor's own non-footprint residues, the mirror of (1).
#   3. Charge patterning, which the mean charge ``Q`` cannot see.
#
# The sub-families are kept separable so they can be scored apart.

# CALVADOS Ashbaugh-Hatch λ above which a residue is counted as sticky enough to
# compete for the donor; 0.5 sits just above the library-wide mean.
LAM_STICKY = 0.5


def _sequence_charges(seq: str, residues: pd.DataFrame) -> np.ndarray:
    """Per-residue formal charges of a one-letter sequence."""
    return residues.loc[list(seq), "q"].to_numpy(float)


def _scd(charges: np.ndarray) -> float:
    """Sawle-Ghosh sequence charge decoration.

    Charge products weighted by the square root of sequence separation. Negative
    values indicate attractive (compacting) patterning, positive repulsive.

    Parameters
    ----------
    charges : numpy.ndarray
        Per-residue charges in sequence order.

    Returns
    -------
    float
        SCD; ``0.0`` for sequences shorter than two residues.
    """
    n = charges.size
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float((charges[i] * charges[j] * np.sqrt(j - i)).sum() / n)


def _charge_blockiness(charges: np.ndarray, window: int = 5) -> float:
    """Unnormalised Das-Pappu charge asymmetry ``δ``.

    Mean squared deviation of each window's charge asymmetry from the whole
    sequence's, so segregated charge blocks score high and well-mixed sequences
    near zero. This is ``δ``, not ``κ``: normalising needs ``δ_max`` from a
    permutation search, which would add a localCIDER dependency for a quantity of
    the same rank ordering.

    Parameters
    ----------
    charges : numpy.ndarray
        Per-residue charges in sequence order.
    window : int, optional
        Sliding-window length, by default ``5``.

    Returns
    -------
    float
        Charge asymmetry; ``0.0`` for sequences shorter than the window or
        carrying no charge.
    """
    n = charges.size
    if n < window:
        return 0.0
    pos, neg = np.maximum(charges, 0.0), np.maximum(-charges, 0.0)
    total = pos.sum() + neg.sum()
    if total <= 0:
        return 0.0
    sigma = (pos.sum() - neg.sum()) ** 2 / total
    deviations = []
    for start in range(n - window + 1):
        p, m = pos[start:start + window].sum(), neg[start:start + window].sum()
        t = p + m
        deviations.append(((p - m) ** 2 / t - sigma) ** 2 if t > 0 else sigma ** 2)
    return float(np.mean(deviations))


def _unrestrained_mask(chain: dict) -> np.ndarray:
    """Boolean mask of the acceptor residues the simulation leaves free.

    ``PDB Domains`` is in author numbering and is shifted into the chain's own
    construct frame exactly as the curation step does when writing the restraint
    YAML, so the mask matches the residues actually restrained in the run.

    Parameters
    ----------
    chain : dict
        A chain entry from ``systems.json``.

    Returns
    -------
    numpy.ndarray
        ``True`` where the residue is unrestrained; all ``True`` when the chain
        carries no domain annotation, since then nothing is restrained.
    """
    n = len(chain["Sequence"])
    free = np.ones(n, bool)
    shift = chain["Construct"]["start"] - 1
    for domain in chain.get("PDB Domains") or []:
        lo, hi = max(domain["start"] - shift, 1), min(domain["end"] - shift, n)
        if hi >= lo:
            free[lo - 1:hi] = False
    return free


def slim_competition_features(
        acceptor_chain: dict,
        donor_chain: dict,
        specific_interactions: pd.DataFrame,
        residues: pd.DataFrame,
    ) -> dict:
    """Non-specific competitors for the donor, separated by who they are.

    Splits the acceptor's unrestrained tails from its folded surface, mirrors the
    split on the donor, and adds charge patterning for both chains.

    Parameters
    ----------
    acceptor_chain, donor_chain : dict
        Chain entries from ``systems.json``; the acceptor's ``PDB Domains`` and
        ``Construct`` fields define the restrained region.
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses both residue-index columns.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code.

    Returns
    -------
    dict
        Tail extent, stickiness and reach; the tail/domain and
        footprint/non-footprint stickiness splits; and charge patterning.
    """
    acceptor_seq, donor_seq = acceptor_chain["Sequence"], donor_chain["Sequence"]
    lam_a = residues.loc[list(acceptor_seq), "lambdas"].to_numpy(float)
    lam_d = residues.loc[list(donor_seq), "lambdas"].to_numpy(float)
    q_a = _sequence_charges(acceptor_seq, residues)
    q_d = _sequence_charges(donor_seq, residues)

    free = _unrestrained_mask(acceptor_chain)
    n_a, n_d, n_free = len(acceptor_seq), len(donor_seq), int(free.sum())
    # contiguous free stretches at each terminus - the arms that can actually reach
    tail_n = int(np.argmin(free)) if free[0] else 0
    tail_c = int(np.argmin(free[::-1])) if free[-1] else 0

    acc_res = specific_interactions["Acceptor residue"].to_numpy()
    don_res = specific_interactions["Donor residue"].to_numpy()
    iface_a, iface_d = np.zeros(n_a, bool), np.zeros(n_d, bool)
    if acc_res.size:
        iface_a[np.unique(acc_res) - 1] = True
        iface_d[np.unique(don_res) - 1] = True

    # A tail anchored beside the groove competes; one at the far end cannot reach.
    if acc_res.size and n_free:
        reach = int(np.abs(np.subtract.outer(np.flatnonzero(free),
                                             np.flatnonzero(iface_a))).min())
    else:
        reach = -1

    domain_nonif = (~free) & (~iface_a)
    sticky_tail = float(lam_a[free].sum())
    sticky_domain = float(lam_a[domain_nonif].sum())
    return {
        # Acceptor unrestrained tails.
        "L_tail_A": n_free,                       # unrestrained acceptor residues (free, flexible chain)
        "L_tailN_A": tail_n,                      # contiguous free N-terminal arm
        "L_tailC_A": tail_c,                      # contiguous free C-terminal arm
        "f_tail_A": n_free / n_a,                 # unrestrained fraction of the acceptor
        "lam_tail_A": float(lam_a[free].mean()) if n_free else 0.0,   # stickiness of the free chain
        "sticky_tail_A": sticky_tail,             # competing stickiness carried by the tails
        "Q_tail_A": float(q_a[free].sum()),       # net charge of the free chain
        "f_sticky_tail_A": float((lam_a[free] > LAM_STICKY).mean()) if n_free else 0.0,  # sticky fraction of the tail
        "d_tail_iface_A": reach,                  # sequence distance from the nearest free residue to the footprint
        # Splitting the folded surface off from the tails.
        "sticky_domnonif_A": sticky_domain,       # restrained non-interface surface (cannot reach)
        "tail_frac_nonif_A": (sticky_tail / (sticky_tail + sticky_domain)
                              if sticky_tail + sticky_domain > 0 else 0.0),  # mobile share of competing surface
        # Donor mirror of the same split; overlaps the rejected λ family.
        "sticky_nonif_D": float(lam_d[~iface_d].sum()),   # donor surface outside its own footprint
        "lam_go_D": float(lam_d[iface_d].mean()) if iface_d.any() else 0.0,       # stickiness of contact-carrying residues
        "lam_nongo_D": float(lam_d[~iface_d].mean()) if (~iface_d).any() else 0.0,  # stickiness of the rest of the donor
        "Q_nonif_D": float(q_d[~iface_d].sum()),  # charge outside the donor footprint
        # Charge patterning; the existing SCD covers the donor only.
        "SCD_A": _scd(q_a),                       # acceptor charge patterning (blockiness)
        "SCD_tail_A": _scd(q_a[free]) if n_free > 1 else 0.0,  # patterning of the free chain specifically
        "chg_block_A": _charge_blockiness(q_a),   # acceptor charge segregation (unnormalised Das-Pappu δ)
        "chg_block_D": _charge_blockiness(q_d),   # donor charge segregation
        "NCPR_A": float(q_a.sum() / n_a),         # acceptor net charge per residue
        "NCPR_D": float(q_d.sum() / n_d),         # donor net charge per residue
        "NCPR_prod": float(q_a.sum() / n_a) * float(q_d.sum() / n_d),  # signed per-residue charge complementarity
    }


#······································································#
# 17. Regional sequence patterning  (charge and hydrophobicity, per region)
#······································································#
# How charge and stickiness are arranged, rather than how much there is. Mean composition
# cannot see arrangement, and averaging across regions cancels it, so each property is
# computed per region:
#
#   donor     whole chain | SLiM span | N-flank | C-flank
#   acceptor  free N-arm | restrained domain | free C-arm
#
# Two orthogonal properties, each in a decoration and a blockiness form: charge (SCD,
# Das-Pappu delta) and hydrophobicity (SHD, windowed lambda variance).


def _shd(lambdas: np.ndarray, beta: float = 1.0) -> float:
    """Sequence hydropathy decoration.

    The hydrophobicity analogue of :func:`_scd`: pairwise λ sums weighted by
    inverse sequence separation, so contiguous sticky blocks score above the same
    residues scattered.

    Parameters
    ----------
    lambdas : numpy.ndarray
        Per-residue CALVADOS λ in sequence order.
    beta : float, optional
        Separation exponent, by default ``1.0``.

    Returns
    -------
    float
        SHD; ``0.0`` for sequences shorter than two residues.
    """
    n = lambdas.size
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    return float(((lambdas[i] + lambdas[j]) * (j - i) ** -beta).sum() / n)


def _blockiness(values: np.ndarray, window: int = 5) -> float:
    """Windowed-mean variance of a per-residue property.

    Variance of the sliding-window means, so a sequence whose property is
    segregated into blocks scores above one where it is evenly mixed.

    Parameters
    ----------
    values : numpy.ndarray
        Per-residue property in sequence order.
    window : int, optional
        Sliding-window length, by default ``5``.

    Returns
    -------
    float
        Variance of the window means; ``0.0`` if the sequence is shorter than the
        window.
    """
    if values.size < window:
        return 0.0
    means = np.convolve(values, np.ones(window) / window, mode="valid")
    return float(means.var())


def _region_patterning(seq: str, residues: pd.DataFrame, tag: str) -> dict:
    """Charge and hydrophobicity patterning of one sequence region."""
    n = len(seq)
    if n == 0:
        return {f"{k}_{tag}": 0.0 for k in
                ("L", "lam_mean", "SHD", "lam_block", "SCD", "chg_block", "NCPR", "FCR")}
    lam = residues.loc[list(seq), "lambdas"].to_numpy(float)
    q = residues.loc[list(seq), "q"].to_numpy(float)
    return {
        f"L_{tag}": float(n),                          # region length
        f"lam_mean_{tag}": float(lam.mean()),          # mean stickiness
        f"SHD_{tag}": _shd(lam),                       # hydrophobicity decoration (sticky blocks)
        f"lam_block_{tag}": _blockiness(lam),          # hydrophobicity segregation
        f"SCD_{tag}": _scd(q),                         # charge decoration (<0 compacting)
        f"chg_block_{tag}": _charge_blockiness(q),     # charge segregation
        f"NCPR_{tag}": float(q.sum() / n),             # net charge per residue
        f"FCR_{tag}": float(np.count_nonzero(q) / n),  # fraction of charged residues
    }


def regional_patterning_features(
        acceptor_chain: dict,
        donor_chain: dict,
        specific_interactions: pd.DataFrame,
        residues: pd.DataFrame,
    ) -> dict:
    """Charge and hydrophobicity patterning across seven sequence regions.

    Four on the disordered donor (whole chain, the contact-footprint span, and
    each flank) and three on the acceptor (the two unrestrained tails and the
    restrained domain, using the same split as :func:`slim_competition_features`).

    Parameters
    ----------
    acceptor_chain, donor_chain : dict
        Chain entries from ``systems.json``.
    specific_interactions : pandas.DataFrame
        Parsed contacts; the donor residue indices delimit the SLiM span.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code.

    Returns
    -------
    dict
        Eight patterning statistics for each of the seven regions.
    """
    donor_seq = donor_chain["Sequence"]
    acceptor_seq = acceptor_chain["Sequence"]
    don_res = specific_interactions["Donor residue"].to_numpy()
    if don_res.size:
        lo, hi = int(don_res.min()), int(don_res.max())
    else:
        lo, hi = 1, len(donor_seq)

    free = _unrestrained_mask(acceptor_chain)
    tail_n = int(np.argmin(free)) if free[0] else 0
    tail_c = int(np.argmin(free[::-1])) if free[-1] else 0

    out = {}
    out.update(_region_patterning(donor_seq, residues, "don"))            # whole IDP
    out.update(_region_patterning(donor_seq[lo - 1:hi], residues, "slim"))  # SLiM span
    out.update(_region_patterning(donor_seq[:lo - 1], residues, "nflank"))  # IDP N-terminal flank
    out.update(_region_patterning(donor_seq[hi:], residues, "cflank"))      # IDP C-terminal flank
    out.update(_region_patterning(acceptor_seq[:tail_n], residues, "ntail"))  # acceptor free N-arm
    domain_seq = "".join(np.array(list(acceptor_seq))[~free])
    out.update(_region_patterning(domain_seq, residues, "dom"))             # restrained domain
    out.update(_region_patterning(
        acceptor_seq[len(acceptor_seq) - tail_c:] if tail_c else "", residues, "ctail"))  # acceptor free C-arm
    return out


#······································································#
# The acceptor as an object: excluded volume and site placement
#······································································#

#······································································#
# 18. Acceptor domain dimensions  (the excluded volume the donor navigates)
#······································································#
# The acceptor enters the binding equilibrium as a large impenetrable object in a small box.
# Its size sets how much non-specific surface competes for the donor (group 10) and how much
# box volume is denied to it, which shifts the effective concentration behind
# ``Kd = C_box(1-f)^2/f``, and its shape sets whether the site is an exposed face or a
# recessed groove.
#
# Measured on the restrained beads only: the unrestrained tails are floppy and their crystal
# coordinates say little about the ensemble. Every quantity is Calpha + sigma, so it lives in
# the frame the simulation integrates.

def _unrestrained_mask_structure(
        chain: dict,
        structure_file: str,
        chain_id: str,
    ) -> np.ndarray:
    """Unrestrained mask over the structure's resolved residues.

    ``PDB Domains`` is in author numbering and so are
    :func:`src.analysis.bsa.residue_numbers`, so the two are compared directly.
    The sequence-space :func:`_unrestrained_mask` cannot be truncated onto the
    structure instead: unresolved residues make the two indexings differ wherever a
    chain has gaps.
    """
    resids = bsa.residue_numbers(structure_file, chain_id)
    free = np.ones(resids.size, bool)
    for domain in chain.get("PDB Domains") or []:
        free &= ~((resids >= domain["start"]) & (resids <= domain["end"]))
    return free


def domain_dimension_features(
        structure_file: str,
        acceptor_chain: dict,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
        system: dict,
    ) -> dict:
    """Size, shape and packing of the folded acceptor, and where the site sits on it.

    Parameters
    ----------
    structure_file : str
        Path to the complex.
    acceptor_chain : dict
        Acceptor chain entry from ``systems.json``; its ``PDB Domains`` select the
        restrained beads.
    acceptor_chain_id, donor_chain_id : str
        Author chain ids in the structure.
    residues : pandas.DataFrame
        Residue-parameter table indexed by one-letter code.
    system : dict
        System entry, for the box volume.

    Returns
    -------
    dict
        Gyration, extent, bead and hull volume, packing, asphericity, surface-to-
        volume ratio, box occupancy, and the depth/eccentricity of the site.
    """
    xyz, radii = _cg_beads(structure_file, acceptor_chain_id, residues)
    don_xyz, don_rad = _cg_beads(structure_file, donor_chain_id, residues)
    # Restrict to the restrained core; the structure is indexed over resolved
    # residues, so the annotation mask is truncated to match.
    free = _unrestrained_mask_structure(acceptor_chain, structure_file, acceptor_chain_id)
    core = xyz[~free] if (~free).any() else xyz
    core_radii = radii[~free] if (~free).any() else radii
    if core.shape[0] < 4:
        return {k: np.nan for k in
                ("Rg_A", "R_max_A", "V_bead_A", "V_hull_A", "packing_A", "aspher_A",
                 "SASA_cg_A", "SA_V_A", "f_box_A", "site_depth_A", "site_ecc_A")}

    centred = core - core.mean(axis=0)
    gyration = centred.T @ centred / core.shape[0]
    eigenvalues = np.sort(np.linalg.eigvalsh(gyration))[::-1]
    r_g = float(np.sqrt(eigenvalues.sum()))
    radial = np.linalg.norm(centred, axis=1)
    v_bead = float((4.0 / 3.0 * np.pi * core_radii ** 3).sum())
    try:
        v_hull = float(ConvexHull(core).volume)
    except Exception:                                     # degenerate / near-planar
        v_hull = np.nan
    sasa = float(bsa.cg_residue_sasa(core, core_radii).sum())

    # Where the donor binds, relative to the acceptor's own centre: a site on the
    # surface has depth ~ R_max, a recessed groove noticeably less.
    contact = np.linalg.norm(don_xyz[:, None, :] - core[None, :, :], axis=-1).min(axis=0)
    site = core[contact < contact.min() + CG_SITE_SHELL]
    site_r = float(np.linalg.norm(site.mean(axis=0) - core.mean(axis=0))) if len(site) else np.nan

    return {
        # Extent.
        "Rg_A": r_g,                              # radius of gyration of the folded core (Å)
        "R_max_A": float(radial.max()),           # bounding radius from the core centre (Å)
        # Volume and packing.
        "V_bead_A": v_bead,                       # summed bead volume (Å³) - the matter present
        "V_hull_A": v_hull,                       # convex-hull volume (Å³) - the space claimed
        "packing_A": v_bead / v_hull if v_hull and np.isfinite(v_hull) else np.nan,  # how densely the core packs
        "aspher_A": float(1.5 * eigenvalues[0] / eigenvalues.sum() - 0.5),  # asphericity (0 sphere, 1 rod)
        # Surface the donor can meet.
        "SASA_cg_A": sasa,                        # coarse-grained surface of the folded core (Å²)
        "SA_V_A": sasa / v_hull if v_hull and np.isfinite(v_hull) else np.nan,  # surface per unit claimed volume
        "f_box_A": (v_hull / (system["Box sides"][0] ** 3 * 1000.0)
                    if v_hull and np.isfinite(v_hull) else np.nan),  # fraction of the box the acceptor denies the donor
        # Where the site sits on the object.
        "site_depth_A": (float(radial.max()) - site_r) if np.isfinite(site_r) else np.nan,  # recess below the bounding radius (Å)
        "site_ecc_A": site_r / r_g if np.isfinite(site_r) and r_g > 0 else np.nan,  # site eccentricity in units of Rg
    }


#······································································#
# Properties the crystal pose does not describe
#······································································#
# The groups above describe the bound pose. Reaching it is a separate matter: a disordered
# donor that is doing something else when free, an acceptor with other sticky places to go,
# and a register that must be found among near-copies. None of that is recoverable from the
# native contact map.


#······································································#
# 23. Chain self-interaction  (what each chain does on its own)
#······································································#
# The Go term is inter-chain, but Ashbaugh-Hatch and Debye-Huckel also act within a chain,
# so a donor that satisfies itself has less to gain from binding, and an acceptor tail that
# folds onto its own site occludes it. Both are relaxations under the force field of group
# 9, run without the partner. Self-satisfaction is local and geometric, so the relaxed
# geometry is measured rather than the sequence averaged.

def _chain_energy(
        coords: np.ndarray,
        lam: np.ndarray,
        sig: np.ndarray,
        charge: np.ndarray,
        eps_yu: float,
        k_yu: float,
        fixed: np.ndarray | None = None,
    ):
    """Intra-chain CALVADOS energy and analytic gradient of one coarse-grained chain.

    Ashbaugh-Hatch plus Debye-Hückel over non-bonded pairs (``|i-j| >= 2``) and the
    harmonic backbone, matching :func:`relaxation_features`. When ``fixed`` is
    given, those beads contribute energy but receive no gradient, so a rigid core
    can be held while the rest relaxes.

    Parameters
    ----------
    coords : numpy.ndarray
        Bead positions, shape ``(n, 3)``, nm.
    lam, sig, charge : numpy.ndarray
        Per-bead CALVADOS λ, σ (nm) and formal charge.
    eps_yu, k_yu : float
        Debye-Hückel prefactor and inverse screening length from ``genParamsDH``.
    fixed : numpy.ndarray or None, optional
        Boolean mask of beads to hold, by default ``None``.

    Returns
    -------
    callable
        ``energy(flat) -> (float, numpy.ndarray)`` over the flattened coordinates.
    """
    n = coords.shape[0]
    left, right = np.triu_indices(n, k=2)
    lam_p = 0.5 * (lam[left] + lam[right])
    sig_p = 0.5 * (sig[left] + sig[right])
    qq_p = charge[left] * charge[right]
    hold = np.zeros(n, bool) if fixed is None else fixed

    def energy(flat):
        x = flat.reshape(n, 3)
        vec = x[left] - x[right]
        dist = np.maximum(np.linalg.norm(vec, axis=-1), 1e-3)
        e_ah, d_ah = _ah_pair(dist, sig_p, lam_p)
        e_dh, d_dh = _dh_pair(dist, qq_p, eps_yu, k_yu)
        total = e_ah.sum() + e_dh.sum()
        grad = np.zeros((n, 3))
        pair_grad = ((d_ah + d_dh) / dist)[:, None] * vec
        np.add.at(grad, left, pair_grad)
        np.add.at(grad, right, -pair_grad)

        bond_vec = np.diff(x, axis=0)
        bond = np.maximum(np.linalg.norm(bond_vec, axis=-1), 1e-9)
        total += 0.5 * BOND_K * ((bond - BOND_R0) ** 2).sum()
        bond_grad = (BOND_K * (bond - BOND_R0) / bond)[:, None] * bond_vec
        grad[1:] += bond_grad
        grad[:-1] -= bond_grad
        grad[hold] = 0.0
        return float(total), grad.ravel()

    return energy


def _bead_params(seq: str, residues: pd.DataFrame):
    """Per-bead ``(lambda, sigma_nm, charge)`` for a one-letter sequence."""
    return (residues.loc[list(seq), "lambdas"].to_numpy(float),
            residues.loc[list(seq), "sigmas"].to_numpy(float),
            residues.loc[list(seq), "q"].to_numpy(float))


def calvados_topology(system_id: str, sim_root: str = SIM_ROOT) -> str | None:
    """Path to the CALVADOS-generated starting topology for a system, if present.

    ``top.pdb`` is what the setup step builds from ``config.yaml`` /
    ``components.yaml``: complete chains over the authored sequence, where a
    crystal resolves only part of it (for ``ywhab-mlxipl``, 246 acceptor residues
    against 229 resolved). It is also the configuration the run actually starts
    from, so any single-chain probe launched from it starts where the simulation
    does rather than from a pose the simulation never visits.

    Parameters
    ----------
    system_id : str
        System ``ID`` from ``systems.json``.
    sim_root : str, optional
        Root of the simulation tree, by default :data:`SIM_ROOT`.

    Returns
    -------
    str or None
        Path to the lowest-``eps_go`` topology, or ``None`` if none exists.
    """
    matches = sorted(glob.glob(f"{sim_root}/*/{system_id}/eps_go=*/scan/rep*/top.pdb"))
    return matches[0] if matches else None


def _topology_chain(topology_file: str, index: int) -> np.ndarray:
    """Cα coordinates (nm) of one chain of a CALVADOS topology, in segment order."""
    universe = utils.load_structure(topology_file)
    return universe.segments[index].atoms.positions / 10.0


def self_interaction_features(
        structure_file: str,
        acceptor_chain: dict,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
        conditions: dict = DEFAULT_CONDITIONS,
        topology_file: str | None = None,
    ) -> dict:
    """How much each chain satisfies itself without the partner.

    Relaxes the donor alone from its bound geometry under the group-9 force field,
    with no partner and no Gō term.

    Returns
    -------
    dict
        Donor self-collapse energy, radius of gyration, compaction relative to the
        bound pose, and self-contact count.
    """
    empty = {"self_dE_D": np.nan, "self_Rg_D": np.nan, "self_compact_D": np.nan,
             "self_ncontact_D": np.nan}
    donor = bsa.ca_coordinates(structure_file, donor_chain_id) / 10.0
    acceptor_xtal = bsa.ca_coordinates(structure_file, acceptor_chain_id) / 10.0
    if donor.shape[0] < 4:
        return empty
    seq_d = bsa.chain_sequence(structure_file, donor_chain_id)
    lam_d, sig_d, q_d = _bead_params(seq_d, residues)
    eps_yu, k_yu = genParamsDH(conditions["Temperature"], conditions["Ionic strength"])

    # The donor on its own, started from the bound conformation.
    fn = _chain_energy(donor, lam_d, sig_d, q_d, eps_yu, k_yu)
    e0 = fn(donor.ravel())[0]
    res = minimize(fn, donor.ravel(), method="L-BFGS-B", jac=True, options=RELAX_OPTIONS)
    relaxed = res.x.reshape(donor.shape)
    rg_bound = float(np.sqrt(((donor - donor.mean(0)) ** 2).sum(1).mean()))
    rg_self = float(np.sqrt(((relaxed - relaxed.mean(0)) ** 2).sum(1).mean()))
    sep = np.abs(np.subtract.outer(np.arange(len(relaxed)), np.arange(len(relaxed))))
    dist = np.linalg.norm(relaxed[:, None] - relaxed[None, :], axis=-1)
    n_self = int(((dist < SELF_CONTACT_NM) & (sep > 2)).sum() / 2)

    out = {
        "self_dE_D": float(e0 - res.fun),         # energy the donor releases by satisfying itself
        "self_Rg_D": rg_self,                     # radius of gyration of the collapsed donor (nm)
        "self_compact_D": rg_self / rg_bound if rg_bound > 0 else np.nan,  # collapse relative to the bound pose
        "self_ncontact_D": float(n_self),         # intra-chain contacts it forms alone
    }

    # The acceptor's free tails against a rigid core.
    # The unrestrained tails are unresolved in every structure in the library, so
    # free_struct is empty when working from a crystal. The CALVADOS topology spans the
    # authored sequence and does carry them, so this probe runs only with a topology.
    if topology_file is None:
        return {**out, "tail_dE_A": np.nan, "tail_selfdock_A": np.nan,
                "tail_d_site_A": np.nan}
    acceptor = _topology_chain(topology_file, 0)
    seq_full = acceptor_chain["Sequence"]
    if acceptor.shape[0] != len(seq_full):
        return {**out, "tail_dE_A": np.nan, "tail_selfdock_A": np.nan,
                "tail_d_site_A": np.nan}
    lam_a, sig_a, q_a = _bead_params(seq_full, residues)
    free = _unrestrained_mask(acceptor_chain)          # sequence space matches the topology
    if free.sum() < 3 or (~free).sum() < 4:
        return {**out, "tail_dE_A": 0.0, "tail_selfdock_A": 0.0, "tail_d_site_A": np.nan}
    fn_a = _chain_energy(acceptor, lam_a, sig_a, q_a, eps_yu, k_yu, fixed=~free)
    e0_a = fn_a(acceptor.ravel())[0]
    res_a = minimize(fn_a, acceptor.ravel(), method="L-BFGS-B", jac=True, options=RELAX_OPTIONS)
    tails = res_a.x.reshape(acceptor.shape)[free]
    # The site is the core beads the donor touches in the bound crystal pose,
    # mapped onto the topology through the resolved residues' author numbering.
    resids = bsa.residue_numbers(structure_file, acceptor_chain_id)
    touched = (np.linalg.norm(donor[:, None] - acceptor_xtal[None, :], axis=-1).min(0)
               < RELAX_GO_CUTOFF)
    shift = acceptor_chain["Construct"]["start"] - 1
    site_idx = np.array([r - shift - 1 for r, t in zip(resids, touched)
                         if t and 0 <= r - shift - 1 < len(seq_full)])
    if not site_idx.size:
        return {**out, "tail_dE_A": float(e0_a - res_a.fun),
                "tail_selfdock_A": 0.0, "tail_d_site_A": np.nan}
    to_site = np.linalg.norm(tails[:, None] - acceptor[site_idx][None, :], axis=-1).min(1)
    return {
        **out,
        "tail_dE_A": float(e0_a - res_a.fun),     # energy released by the tails relaxing onto the core
        "tail_selfdock_A": float((to_site < SELF_CONTACT_NM).mean()),  # tail beads that end up ON the site
        "tail_d_site_A": float(to_site.min()),    # closest approach of any tail bead to the site (nm)
    }


#······································································#
# 24. Free-donor ensemble  (the entropy reference we never measured)
#······································································#
# The entropy paid on binding is S_free - S_bound, and every entropy feature above is
# measured on the crystal, which describes only the bound state. A short BAOAB Langevin
# run of the donor alone in the same numpy force field characterises the free state: no
# partner, no Go term and no rare event to sample, so it converges quickly.

def free_ensemble_features(
        sequence: str,
        residues: pd.DataFrame,
        conditions: dict = DEFAULT_CONDITIONS,
        topology_file: str | None = None,
        chain_index: int = 1,
        n_steps: int = FREE_STEPS,
        n_seeds: int = FREE_SEEDS,
    ) -> dict:
    """Dimensions of the free donor from a short single-chain CALVADOS run.

    BAOAB Langevin at CALVADOS conditions, averaged over seeds. Equilibrium moments
    are integrator-independent, so this samples the free-chain ensemble.

    Started from the CALVADOS-generated topology when one is given: that chain
    spans the authored sequence and is the configuration the production run itself
    starts from. The fallback is a minimised random walk, which is only needed for
    a system that has not been set up yet.

    Returns
    -------
    dict
        Free radius of gyration and end-to-end distance, the Flory exponent from
        internal scaling, and the intra-chain contact density.
    """
    n = len(sequence)
    if n < 6:
        return {k: np.nan for k in ("Rg_free_D", "Ree_free_D", "nu_D", "n_intra_free_D",
                                    "Rg_free_per_N", "swell_D")}
    lam, sig, charge = _bead_params(sequence, residues)
    eps_yu, k_yu = genParamsDH(conditions["Temperature"], conditions["Ionic strength"])
    mass = residues.loc[list(sequence), "MW"].to_numpy(float) if "MW" in residues else \
        np.full(n, 110.0)
    fn = _chain_energy(np.zeros((n, 3)), lam, sig, charge, eps_yu, k_yu)
    start = None
    if topology_file:
        candidate = _topology_chain(topology_file, chain_index)
        if candidate.shape[0] == n:               # authored sequence, so it should match
            start = candidate - candidate.mean(0)
    kt = KB_KJ * conditions["Temperature"]
    gamma, dt = FREE_FRICTION, FREE_DT
    c1, c2 = np.exp(-gamma * dt), np.sqrt(1.0 - np.exp(-2.0 * gamma * dt))
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    far = sep > 2

    rg, ree, nu, ncon = [], [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        # Self-avoiding-ish start: a random walk with 0.38 nm steps.
        if start is not None:
            # Perturb the shared starting conformation per seed so the seeds are
            # not replicas of one trajectory.
            x = start + rng.normal(scale=START_JITTER_NM, size=start.shape)
        else:
            step = rng.normal(size=(n - 1, 3))
            step /= np.linalg.norm(step, axis=1, keepdims=True)
            x = np.vstack([np.zeros(3), np.cumsum(step * BOND_R0, axis=0)])
        # A random walk self-intersects, so relax the clashes out before any
        # dynamics: an unrelaxed overlap makes the first Ashbaugh-Hatch force
        # arbitrarily large and the trajectory diverges immediately.
        x = minimize(fn, x.ravel(), method="L-BFGS-B", jac=True,
                     options={"maxiter": 2000}).x.reshape(n, 3)
        v = rng.normal(size=(n, 3)) * np.sqrt(kt / mass[:, None])
        grad = fn(x.ravel())[1].reshape(n, 3)
        snaps = []
        for step_i in range(n_steps):
            v -= 0.5 * dt * grad / mass[:, None]
            x += 0.5 * dt * v
            v = c1 * v + c2 * np.sqrt(kt / mass[:, None]) * rng.normal(size=(n, 3))
            x += 0.5 * dt * v
            grad = fn(x.ravel())[1].reshape(n, 3)
            v -= 0.5 * dt * grad / mass[:, None]
            if not np.isfinite(x).all():          # diverged; discard this seed
                snaps = []
                break
            if step_i >= n_steps // 2 and step_i % FREE_STRIDE == 0:
                snaps.append(x.copy())
        if not snaps:
            continue
        snaps = np.array(snaps)
        centred = snaps - snaps.mean(1, keepdims=True)
        rg.append(np.sqrt((centred ** 2).sum(-1).mean(-1)).mean())
        ree.append(np.linalg.norm(snaps[:, -1] - snaps[:, 0], axis=-1).mean())
        pair = np.linalg.norm(snaps[:, :, None, :] - snaps[:, None, :, :], axis=-1).mean(0)
        # Internal scaling <R_ij> ~ |i-j|^nu, fitted over the resolvable separations.
        seps = np.arange(3, max(4, n // 2))
        means = np.array([pair[sep == s].mean() for s in seps])
        ok = np.isfinite(means) & (means > 0)
        nu.append(np.polyfit(np.log(seps[ok]), np.log(means[ok]), 1)[0] if ok.sum() > 2
                  else np.nan)
        ncon.append(((pair < SELF_CONTACT_NM) & far).sum() / 2.0 / n)

    if not rg:
        return {k: np.nan for k in ("Rg_free_D", "Ree_free_D", "nu_D", "n_intra_free_D",
                                    "Rg_free_per_N", "swell_D")}
    rg_mean = float(np.mean(rg))
    # nu needs >=3 resolvable separations, so ``seps`` is too short to fit below
    # n = 12 and every seed returns NaN. That is the right answer for a chain that
    # short, but np.nanmean warns on an all-NaN slice, so the empty case is explicit.
    nu_arr = np.asarray(nu, dtype=float)
    return {
        "Rg_free_D": rg_mean,                     # free-state radius of gyration (nm)
        "Ree_free_D": float(np.mean(ree)),        # free-state end-to-end distance (nm)
        "nu_D": float(np.nanmean(nu_arr)) if np.isfinite(nu_arr).any() else np.nan,
                                                  # Flory exponent (0.33 globule → 0.6 swollen)
        "n_intra_free_D": float(np.mean(ncon)),   # intra-chain contacts per residue when free
        "Rg_free_per_N": rg_mean / n ** 0.5,      # Rg against the ideal-chain expectation
        "swell_D": rg_mean / (BOND_R0 * n ** 0.5),  # swelling relative to a random walk
    }


#······································································#
# 25. Register degeneracy  (how many other ways the SLiM could bind)
#······································································#
# The groups above describe the native contact map. A coarse-grained donor is a sticky
# polymer and can register wrong: bind with its footprint shifted along its own sequence,
# satisfying some contacts and not others. A donor with many near-degenerate registers
# spends time in non-native modes, so a deeper well is needed for the native one to
# dominate. This acts on the location of the transition rather than its steepness.

def register_features(
        specific_interactions: pd.DataFrame,
        donor_seq: str,
        residues: pd.DataFrame,
    ) -> dict:
    """Uniqueness of the native binding register along the donor sequence.

    Slides the contact footprint along the donor and scores each shift by how well
    the substituted residues match the native ones in λ and charge, so a
    compositionally uniform donor scores many near-equivalent registers.

    Returns
    -------
    dict
        Alternative-register count, the native-to-best-alternative margin, and the
        λ / charge autocorrelation of the donor.
    """
    don_res = specific_interactions["Donor residue"].to_numpy()
    n = len(donor_seq)
    if don_res.size == 0 or n < 6:
        return {k: np.nan for k in ("n_registers_D", "register_margin_D",
                                    "register_best_D", "lam_autocorr_D", "q_autocorr_D")}
    lam, _sig, charge = _bead_params(donor_seq, residues)
    idx = np.unique(don_res) - 1
    span = idx.max() - idx.min()
    native_lam, native_q = lam[idx], charge[idx]

    scores = []
    for shift in range(-(n - 1), n):
        moved = idx + shift
        if moved.min() < 0 or moved.max() >= n or shift == 0:
            continue
        # Match in the two properties the force field actually distinguishes.
        d_lam = np.abs(lam[moved] - native_lam).mean()
        d_q = np.abs(charge[moved] - native_q).mean()
        scores.append(np.exp(-(d_lam / REG_LAM_SCALE + d_q / REG_Q_SCALE)))
    if not scores:
        return {k: np.nan for k in ("n_registers_D", "register_margin_D",
                                    "register_best_D", "lam_autocorr_D", "q_autocorr_D")}
    scores = np.array(scores)
    ac = lambda v: float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 3 and v.std() > 0 else 0.0
    return {
        "n_registers_D": float((scores > REG_ALT_CUT).sum()),  # shifted registers scoring near-native
        "register_best_D": float(scores.max()),   # similarity of the best alternative to the native
        "register_margin_D": float(1.0 - scores.max()),  # how uniquely the native register is defined
        "lam_autocorr_D": ac(lam),                # λ repetitiveness along the donor
        "q_autocorr_D": ac(charge),               # charge repetitiveness along the donor
    }


#······································································#
# 26. Acceptor surface patches  (resolved decoys, not summed stickiness)
#······································································#
# ``sticky_nonif_A`` sums lambda over non-interface surface, which cannot distinguish one
# large greasy patch from the same stickiness spread thinly. The summed form is largely
# redundant with domain size (group 18); a patch census measures competition instead.

def surface_patch_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
    ) -> dict:
    """Contiguous sticky patches on the exposed acceptor surface, and the site's rank.

    Returns
    -------
    dict
        Patch count and size, the stickiness of the best decoy relative to the
        native site, and how far away it lies.
    """
    empty = {k: np.nan for k in ("n_patch_A", "S_patch_max_A", "lam_patch_max_A",
                                 "patch_ratio_A", "d_patch_site_A")}
    acc_xyz, acc_rad = _cg_beads(structure_file, acceptor_chain_id, residues)
    don_xyz, _don_rad = _cg_beads(structure_file, donor_chain_id, residues)
    if acc_xyz.shape[0] < 6:
        return empty
    seq_a = bsa.chain_sequence(structure_file, acceptor_chain_id)
    lam_a = residues.loc[list(seq_a), "lambdas"].to_numpy(float)
    exposed = bsa.cg_residue_sasa(acc_xyz, acc_rad) > PATCH_SASA_MIN
    site = np.linalg.norm(don_xyz[:, None] - acc_xyz[None, :], axis=-1).min(0) < PATCH_SITE_A

    sticky = exposed & (lam_a > LAM_STICKY) & (~site)
    if sticky.sum() < 2:
        return {**empty, "n_patch_A": 0.0, "S_patch_max_A": 0.0, "lam_patch_max_A": 0.0,
                "patch_ratio_A": 0.0}
    # Single-linkage clustering in 3-D: beads within PATCH_LINK_A are one patch.
    members = np.flatnonzero(sticky)
    close = np.linalg.norm(acc_xyz[members][:, None] - acc_xyz[members][None, :],
                           axis=-1) < PATCH_LINK_A
    label = -np.ones(len(members), int)
    current = 0
    for start in range(len(members)):
        if label[start] >= 0:
            continue
        stack, label[start] = [start], current
        while stack:
            node = stack.pop()
            for nb in np.flatnonzero(close[node] & (label < 0)):
                label[nb] = current
                stack.append(nb)
        current += 1
    sizes = np.array([(label == c).sum() for c in range(current)])
    strength = np.array([lam_a[members[label == c]].sum() for c in range(current)])
    best = int(np.argmax(strength))
    site_strength = float(lam_a[site].sum()) if site.any() else np.nan
    centre = acc_xyz[members[label == best]].mean(0)
    site_centre = acc_xyz[site].mean(0) if site.any() else np.full(3, np.nan)
    return {
        "n_patch_A": float(current),              # contiguous sticky non-site patches
        "S_patch_max_A": float(sizes.max()),      # beads in the largest decoy patch
        "lam_patch_max_A": float(strength[best]),  # summed λ of the strongest decoy
        "patch_ratio_A": (float(strength[best] / site_strength)
                          if site_strength and np.isfinite(site_strength) and site_strength > 0
                          else np.nan),           # decoy strength relative to the real site
        "d_patch_site_A": float(np.linalg.norm(centre - site_centre)),  # decoy-to-site distance (Å)
    }


#······································································#
# 27. Site orientation and electrostatic steering
#······································································#
# Group 18 locates the site; it does not describe how precisely the donor must be oriented
# to engage it, which is a rotational-entropy cost. A deep narrow groove and a shallow wide
# one can share ``site_depth_A`` while demanding very different orientations. The dipole
# terms extend the monopole-only ``QAQD_screened``: a domain whose dipole points at its own
# site concentrates a complementary donor there.

def site_orientation_features(
        structure_file: str,
        acceptor_chain_id: str,
        donor_chain_id: str,
        residues: pd.DataFrame,
    ) -> dict:
    """How much orientational freedom the site leaves, and whether charge steers to it.

    Returns
    -------
    dict
        Site solid angle and aspect ratio, spread of the contact directions, and
        the acceptor dipole with its alignment to the site.
    """
    empty = {k: np.nan for k in ("site_solid_angle_A", "groove_aspect_A", "contact_aniso",
                                 "dipole_A", "dipole_site_align_A")}
    acc_xyz, acc_rad = _cg_beads(structure_file, acceptor_chain_id, residues)
    don_xyz, _r = _cg_beads(structure_file, donor_chain_id, residues)
    if acc_xyz.shape[0] < 6 or don_xyz.shape[0] < 2:
        return empty
    seq_a = bsa.chain_sequence(structure_file, acceptor_chain_id)
    q_a = residues.loc[list(seq_a), "q"].to_numpy(float)
    sasa = bsa.cg_residue_sasa(acc_xyz, acc_rad)
    site = np.linalg.norm(don_xyz[:, None] - acc_xyz[None, :], axis=-1).min(0) < PATCH_SITE_A
    if not site.any() or sasa.sum() <= 0:
        return empty

    centre = acc_xyz.mean(0)
    site_xyz = acc_xyz[site]
    # Shape of the site patch from its own gyration tensor.
    centred = site_xyz - site_xyz.mean(0)
    eig = np.sort(np.linalg.eigvalsh(centred.T @ centred / max(len(site_xyz), 1)))[::-1]
    # Directions from the acceptor centre to each contacted bead: a tight cone means
    # the donor has to arrive one way.
    direction = site_xyz - centre
    norm = np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    unit = direction / norm
    resultant = float(np.linalg.norm(unit.mean(0)))
    dipole_vec = (q_a[:, None] * (acc_xyz - centre)).sum(0)
    dipole = float(np.linalg.norm(dipole_vec))
    site_dir = site_xyz.mean(0) - centre
    align = float(dipole_vec @ site_dir /
                  max(dipole * np.linalg.norm(site_dir), 1e-9)) if dipole > 1e-9 else 0.0
    return {
        "SASA_site": float(sasa[site].sum()),     # SASA of the site
        "site_solid_angle_A": float(sasa[site].sum() / sasa.sum()),  # share of the surface that is the site
        "groove_aspect_A": float(np.sqrt(eig[0] / max(eig[-1], 1e-9))),  # elongation of the site patch
        "contact_aniso": resultant,               # 1 ⇒ all contacts face one way (tight cone)
        "dipole_A": dipole,                       # acceptor dipole magnitude (e·Å)
        "dipole_site_align_A": align,             # cosine between the dipole and the site direction
    }


#······································································#
# 28. Rich contact-graph descriptors  (when the graph is the only variable)
#······································································#
# Groups 3-7 describe the contact graph with counts, degree statistics and one spectral
# cut. That is sufficient while systems also differ in sequence, box and structure, but it
# is the whole description when the graph is edited and everything else held fixed. Two
# graphs with identical ``N_C``, ``N_RCA`` and ``N_RCD`` can still differ by several
# kJ/mol in the ``eps_go`` they require, so this group is weighted towards intensive
# quantities that separate graphs at fixed counts.
#
# Everything is a function of the contact list alone (endpoint indices and ``r0``), in
# nine themes: degree distribution on both sides; the core left after peeling; fragility
# to single-edge deletion; four-cycles and neighbourhood overlap; path structure; the
# spectrum beyond ``lambda2``; the ``r0``-weighted graph; and the sequence register that
# the bipartite abstraction discards by forgetting that residues are ordered along a
# chain.

# Normalised-Laplacian eigenvalues below this count as a near-independent module
# (0 exactly for a genuinely disconnected one).
NEAR_MODULE_EIGENVALUE = 0.1

# Sequence periods (residues) probed in the donor contact pattern: an α-helical
# SLiM presents one face every ~3.6 residues, an extended one every ~2.
PERIOD_HELIX = 3.6
PERIOD_STRAND = 2.0

GRAPH_RICH_FEATURES = (
    # degree distribution, both sides
    "deg_gini_A", "deg_gini_D", "deg_cv_A", "deg_cv_D", "deg_branch_A",
    "deg_branch_D", "deg_hub_A", "deg_hub_D", "deg_assort", "f_leaf",
    # load-bearing core and fragility
    "k_core_max", "f_edges_2core", "f_bridges", "n_articulation", "f_pendant",
    # cycles and redundancy
    "n_squares", "sq_per_edge", "bip_clustering", "cyc_per_node",
    "codeg_max_A", "codeg_max_D",
    # neighbourhood overlap / nestedness
    "jaccard_A", "jaccard_D", "nest_A", "nest_D",
    # path structure
    "path_mean", "diameter_graph", "efficiency",
    # spectrum beyond lambda2
    "adj_lambda1", "adj_gap", "graph_energy", "lambda2_un", "n_near_modules",
    "spec_entropy",
    # the r0-weighted graph
    "w_deg_gini_A", "w_deg_gini_D", "w_lambda2", "w_hub_A", "w_hub_D",
    "w_comp_max", "hub_tightness",
    # register structure (residues are ordered; the bipartite graph forgets)
    "reg_corr", "reg_slope", "reg_scatter", "don_gap_mean", "don_gap_max",
    "acc_gap_mean", "L_SA", "rho_CA", "N_sites_A", "S_site_A",
    #"per_helix_D", "per_strand_D",
)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, →1 = one holds all)."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n, total = ordered.size, ordered.sum()
    if n < 2 or total <= 0:
        return 0.0
    return float(((2 * np.arange(1, n + 1) - n - 1) @ ordered) / (n * total))


def _bridges_and_articulation(n_nodes: int, edges: np.ndarray) -> tuple[int, int]:
    """Count bridge edges and articulation nodes by iterative Tarjan low-link.

    A bridge is an edge whose removal disconnects its component, an
    articulation point a node whose removal does. Both measure how much of
    the interface hangs off a single contact.

    Parameters
    ----------
    n_nodes : int
        Number of nodes; edges index into ``range(n_nodes)``.
    edges : numpy.ndarray
        ``(n_edges, 2)`` integer endpoint pairs.

    Returns
    -------
    tuple[int, int]
        ``(n_bridges, n_articulation_points)``.
    """
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for edge_id, (u, v) in enumerate(edges):
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))

    discovered = np.full(n_nodes, -1)
    low = np.zeros(n_nodes, dtype=int)
    articulation = np.zeros(n_nodes, dtype=bool)
    n_bridges, timer = 0, 0

    for root in range(n_nodes):
        if discovered[root] >= 0:
            continue
        discovered[root] = low[root] = timer
        timer += 1
        # (node, edge we arrived by, un-consumed neighbour iterator)
        stack = [(root, -1, iter(adjacency[root]))]
        root_children = 0
        while stack:
            node, arrival, neighbours = stack[-1]
            descended = False
            for neighbour, edge_id in neighbours:
                if edge_id == arrival:            # never re-cross the tree edge
                    continue
                if discovered[neighbour] < 0:
                    discovered[neighbour] = low[neighbour] = timer
                    timer += 1
                    stack.append((neighbour, edge_id, iter(adjacency[neighbour])))
                    descended = True
                    break
                low[node] = min(low[node], discovered[neighbour])
            if descended:
                continue
            stack.pop()
            if not stack:
                continue
            parent = stack[-1][0]
            low[parent] = min(low[parent], low[node])
            if low[node] > discovered[parent]:
                n_bridges += 1
            if stack[-1][1] == -1:                # the parent is the DFS root
                root_children += 1
            elif low[node] >= discovered[parent]:
                articulation[parent] = True
        articulation[root] = root_children > 1
    return n_bridges, int(articulation.sum())


def _core_numbers(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """k-core number of every node, by repeated peeling of low-degree nodes."""
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    degree = np.zeros(n_nodes, dtype=int)
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
        degree[u] += 1
        degree[v] += 1

    alive = np.ones(n_nodes, dtype=bool)
    core = np.zeros(n_nodes, dtype=int)
    k = 0
    while alive.any():
        k += 1
        peeled = True
        while peeled:
            peeled = False
            for node in np.flatnonzero(alive & (degree < k)):
                alive[node] = False
                core[node] = k - 1
                for neighbour in adjacency[node]:
                    if alive[neighbour]:
                        degree[neighbour] -= 1
                peeled = True
    return core


def _pair_overlap(codegree: np.ndarray, degree: np.ndarray) -> tuple[float, float, float]:
    """Neighbourhood overlap among same-side residues.

    Two acceptor residues contacting the same donors are substitutable: losing
    one leaves the other covering its partners, so overlap is what makes an
    interface survive edge deletion.

    Returns
    -------
    tuple[float, float, float]
        Mean Jaccard overlap over pairs sharing a partner, the NODF-style
        nestedness (mean overlap normalised by the smaller degree, over pairs
        of unequal degree), and the largest number of shared partners.
    """
    n = degree.size
    if n < 2:
        return np.nan, np.nan, np.nan
    upper = np.triu_indices(n, 1)
    shared = codegree[upper]
    union = degree[upper[0]] + degree[upper[1]] - shared
    jaccard = np.divide(shared, union, out=np.zeros_like(shared, dtype=float),
                        where=union > 0)
    overlapping = shared > 0
    smaller = np.minimum(degree[upper[0]], degree[upper[1]])
    # NODF only counts pairs of differing degree: equal-degree pairs cannot be
    # nested, only coincident.
    nested = (degree[upper[0]] != degree[upper[1]]) & (smaller > 0)
    return (float(jaccard[overlapping].mean()) if overlapping.any() else 0.0,
            float((shared[nested] / smaller[nested]).mean()) if nested.any() else np.nan,
            float(shared.max()))


def _sequence_periodicity(residue_indices: np.ndarray, period: float) -> float:
    """Share of the contact pattern's spectral power sitting at one sequence period.

    Projects the binary contact indicator over the footprint span onto a single
    Fourier mode, normalised so the value is the fraction of total power at that
    period and therefore lies in ``[0, 1]``.
    """
    lo, hi = int(residue_indices.min()), int(residue_indices.max())
    length = hi - lo + 1
    if length < max(6, 2 * period):
        return np.nan                             # too short to resolve the period
    indicator = np.zeros(length)
    indicator[np.unique(residue_indices) - lo] = 1.0
    centred = indicator - indicator.mean()
    power = length * (centred ** 2).sum()
    if power <= 0:
        return 0.0                                # every residue contacts: no modulation at all
    mode = np.exp(-2j * np.pi * np.arange(length) / period)
    return float(np.abs(centred @ mode) ** 2 / power)


def contact_graph_rich_features(specific_interactions: pd.DataFrame) -> dict:
    """Wiring descriptors of the contact graph, at fixed contact count.

    Complements groups 3-7, which are dominated by how many contacts there are.
    These are chosen to separate graphs that agree on the counts: how unequally
    coordination is shared, what survives peeling, how much of the interface
    hangs off a single edge, how redundantly the two footprints are cross-linked,
    and - because the bipartite abstraction forgets that residues are ordered -
    how regularly the contacts march along the two sequences.

    Parameters
    ----------
    specific_interactions : pandas.DataFrame
        Parsed contacts; uses the acceptor/donor residue indices and the
        ``Equilibrium distance`` column (nm) for the weighted graph.

    Returns
    -------
    dict
        The 53 features named in :data:`GRAPH_RICH_FEATURES`; all-``NaN`` for
        fewer than three contacts, where none of them is defined.
    """
    acceptor = specific_interactions["Acceptor residue"].to_numpy()
    donor = specific_interactions["Donor residue"].to_numpy()
    r0 = specific_interactions["Equilibrium distance"].to_numpy(dtype=float)
    n_c = len(acceptor)
    if n_c < 3:
        return {name: np.nan for name in GRAPH_RICH_FEATURES}

    acc_residues, acc_index = np.unique(acceptor, return_inverse=True)
    don_residues, don_index = np.unique(donor, return_inverse=True)
    n_a, n_d = acc_residues.size, don_residues.size
    n_nodes = n_a + n_d

    # Bipartite incidence, then the full node-by-node adjacency (acceptors first).
    incidence = np.zeros((n_a, n_d))
    np.add.at(incidence, (acc_index, don_index), 1.0)
    incidence = (incidence > 0).astype(float)
    deg_a, deg_d = incidence.sum(1), incidence.sum(0)
    adjacency = np.zeros((n_nodes, n_nodes))
    adjacency[:n_a, n_a:] = incidence
    adjacency[n_a:, :n_a] = incidence.T
    degree = adjacency.sum(1)
    edges = np.column_stack([acc_index, don_index + n_a])

    # Degree distribution.
    # ⟨k²⟩/⟨k⟩ is the mean degree seen from a random edge end: it governs how far
    # a perturbation propagates, and separates a few anchors + many singletons
    # from an evenly coordinated interface of the same mean degree.
    branch_a = float((deg_a ** 2).mean() / deg_a.mean()) if deg_a.mean() > 0 else np.nan
    branch_d = float((deg_d ** 2).mean() / deg_d.mean()) if deg_d.mean() > 0 else np.nan
    end_a, end_d = deg_a[acc_index], deg_d[don_index]
    assortativity = (float(np.corrcoef(end_a, end_d)[0, 1])
                     if end_a.std() > 0 and end_d.std() > 0 else 0.0)
    # Edges hanging off a singly-coordinated residue: the first to be lost.
    f_leaf = float(((end_a == 1) | (end_d == 1)).mean())

    # Core and fragility.
    core = _core_numbers(n_nodes, edges)
    in_2core = (core[edges[:, 0]] >= 2) & (core[edges[:, 1]] >= 2)
    n_bridges, n_articulation = _bridges_and_articulation(n_nodes, edges)
    n_components, labels = connected_components(adjacency, directed=False)

    # Cycles and overlap.
    codegree_a = incidence @ incidence.T
    codegree_d = incidence.T @ incidence
    upper_a = np.triu_indices(n_a, 1)
    shared_a = codegree_a[upper_a]
    # In a bipartite graph the shortest cycle is a square: two acceptors both
    # contacting the same two donors - the elementary redundant cross-link.
    n_squares = float((shared_a * (shared_a - 1) / 2).sum()) if shared_a.size else 0.0
    paths_3 = float(((deg_a[acc_index] - 1) * (deg_d[don_index] - 1)).sum())
    jaccard_a, nest_a, codeg_max_a = _pair_overlap(codegree_a, deg_a)
    jaccard_d, nest_d, codeg_max_d = _pair_overlap(codegree_d, deg_d)
    n_cycles = n_c - n_nodes + n_components

    # Path structure.
    distance = shortest_path(adjacency, method="D", unweighted=True, directed=False)
    reachable = np.isfinite(distance) & (distance > 0)
    inverse = np.zeros_like(distance)
    inverse[reachable] = 1.0 / distance[reachable]

    # Spectrum beyond lambda2.
    adjacency_eigenvalues = np.linalg.eigvalsh(adjacency)
    laplacian_eigenvalues = np.linalg.eigvalsh(np.diag(degree) - adjacency)
    share = np.clip(laplacian_eigenvalues, 0.0, None)
    share = share / share.sum() if share.sum() > 0 else share
    nonzero = share[share > 0]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalised = np.eye(n_nodes) - adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    normalised_eigenvalues = np.linalg.eigvalsh(normalised)

    # The weighted graph.
    # Same weighting convention as ``N_eff``: a contact counts for how readily it
    # forms, so the weighted graph is the one the Gō term actually sees.
    weight = np.exp(-r0 / r0.mean())
    weighted_incidence = np.zeros((n_a, n_d))
    np.add.at(weighted_incidence, (acc_index, don_index), weight)
    weighted = np.zeros((n_nodes, n_nodes))
    weighted[:n_a, n_a:] = weighted_incidence
    weighted[n_a:, :n_a] = weighted_incidence.T
    weighted_degree = weighted.sum(1)
    inv_sqrt_w = 1.0 / np.sqrt(np.maximum(weighted_degree, 1e-12))
    weighted_normalised = (np.eye(n_nodes)
                           - weighted * inv_sqrt_w[:, None] * inv_sqrt_w[None, :])
    weight_per_component = np.bincount(labels[acc_index], weights=weight,
                                       minlength=n_components)
    # Mean r0 of the contacts each residue makes, against how many it makes.
    r0_per_node = np.zeros(n_nodes)
    np.add.at(r0_per_node, acc_index, r0)
    np.add.at(r0_per_node, don_index + n_a, r0)
    r0_per_node /= np.maximum(degree, 1.0)
    tightness = (-float(np.corrcoef(degree, r0_per_node)[0, 1])
                 if degree.std() > 0 and r0_per_node.std() > 0 else np.nan)

    # Register structure.
    # Where the bipartite graph sees an unordered edge set, the chains are
    # sequences: a peptide threading a groove walks monotonically along both, a
    # looped or bipartite-mode SLiM does not.
    register = (float(np.corrcoef(acceptor, donor)[0, 1])
                if acceptor.std() > 0 and donor.std() > 0 else np.nan)
    span_acceptor = int(acc_residues.max() - acc_residues.min() + 1)
    span_donor = int(don_residues.max() - don_residues.min() + 1)
    if donor.std() > 0:
        slope, intercept = np.polyfit(donor, acceptor, 1)
        scatter = float((acceptor - (slope * donor + intercept)).std() / span_acceptor)
        # Expressed per footprint span, so it is the dimensionless pitch (±1 =
        # the two footprints are traversed together) rather than an unbounded
        # ratio that blows up whenever the donor barely moves.
        slope = float(slope * span_donor / span_acceptor)
    else:
        slope, scatter = np.nan, np.nan
    gaps_d = np.diff(don_residues)
    gaps_a = np.diff(acc_residues)
    n_sites_a, s_site_a = _count_clusters(acceptor)

    return {
        # degree distribution - how unequally coordination is shared out
        "deg_gini_A": _gini(deg_a),               # inequality of acceptor coordination (0 = flat)
        "deg_gini_D": _gini(deg_d),               # inequality of donor coordination
        "deg_cv_A": float(deg_a.std() / deg_a.mean()) if deg_a.mean() > 0 else np.nan,  # scale-free spread of acceptor degrees
        "deg_cv_D": float(deg_d.std() / deg_d.mean()) if deg_d.mean() > 0 else np.nan,  # scale-free spread of donor degrees
        "deg_branch_A": branch_a,                 # ⟨k²⟩/⟨k⟩ acceptor side - how far perturbations propagate
        "deg_branch_D": branch_d,                 # ⟨k²⟩/⟨k⟩ donor side
        "deg_hub_A": float(deg_a.max() / n_c),    # share of contacts on the single busiest acceptor residue
        "deg_hub_D": float(deg_d.max() / n_c),    # share of contacts on the single busiest donor residue
        "deg_assort": assortativity,              # do busy acceptors pair with busy donors? (+ = hub-to-hub)
        "f_leaf": f_leaf,                         # contacts hanging off a singly-coordinated residue

        # the load-bearing core, and how fragile the rest is
        "k_core_max": float(core.max()),          # densest mutually-supported core (peeling depth)
        "f_edges_2core": float(in_2core.mean()),  # contacts surviving removal of all dangling residues
        "f_bridges": float(n_bridges / n_c),      # contacts whose loss splits the interface
        "n_articulation": float(n_articulation),  # residues whose loss splits the interface
        "f_pendant": float((degree == 1).mean()),  # interface residues held by a single contact

        # redundancy: four-cycles and shared partners
        "n_squares": n_squares,                   # elementary redundant cross-links (2 acceptors × 2 donors)
        "sq_per_edge": float(n_squares / n_c),    # cross-linking per contact (intensive)
        "bip_clustering": float(4 * n_squares / paths_3) if paths_3 > 0 else np.nan,  # bipartite clustering coefficient
        "cyc_per_node": float(n_cycles / n_nodes),  # cyclomatic number per interface residue (intensive f_redundant)
        "codeg_max_A": codeg_max_a,               # most donors shared by any two acceptor residues
        "codeg_max_D": codeg_max_d,               # most acceptors shared by any two donor residues

        # neighbourhood overlap - which contacts are substitutable
        "jaccard_A": jaccard_a,                   # mean partner-set overlap between acceptor residues
        "jaccard_D": jaccard_d,                   # mean partner-set overlap between donor residues
        "nest_A": nest_a,                         # nestedness: do weak acceptors contact a subset of the strong ones' partners?
        "nest_D": nest_d,                         # nestedness, donor side (1 = perfect core-periphery)

        # path structure - long and thin, or compact and interlocked
        "path_mean": float(distance[reachable].mean()),  # mean contact-graph separation of interface residues
        "diameter_graph": float(distance[reachable].max()),  # longest such separation
        "efficiency": float(inverse.sum() / (n_nodes * (n_nodes - 1))),  # global efficiency (1 = all-to-all)

        # spectrum beyond lambda2
        "adj_lambda1": float(adjacency_eigenvalues[-1]),  # spectral radius - dominant coordination mode
        "adj_gap": float(adjacency_eigenvalues[-1] - adjacency_eigenvalues[-2]),  # how far that mode dominates the next
        "graph_energy": float(np.abs(adjacency_eigenvalues).sum() / n_c),  # graph energy per contact
        "lambda2_un": float(max(laplacian_eigenvalues[1], 0.0)),  # unnormalised algebraic connectivity (size-carrying twin of lambda2)
        "n_near_modules": float((normalised_eigenvalues < NEAR_MODULE_EIGENVALUE).sum()),  # near-independent sub-modules (soft N_comp)
        "spec_entropy": float(-(nonzero * np.log(nonzero)).sum() / np.log(n_nodes)) if nonzero.size else np.nan,  # spectral evenness (1 = no dominant mode)

        # the r0-weighted graph - a contact counts for how readily it forms
        "w_deg_gini_A": _gini(weighted_incidence.sum(1)),  # inequality of acceptor coordination, weighted by contact tightness
        "w_deg_gini_D": _gini(weighted_incidence.sum(0)),  # same, donor side
        "w_lambda2": float(max(np.linalg.eigvalsh(weighted_normalised)[1], 0.0)),  # algebraic connectivity of the weighted graph
        "w_hub_A": float(weighted_incidence.sum(1).max() / weight.sum()),  # share of the binding weight on one acceptor residue
        "w_hub_D": float(weighted_incidence.sum(0).max() / weight.sum()),  # share of the binding weight on one donor residue
        "w_comp_max": float(weight_per_component.max() / weight.sum()),  # share of the binding weight in the largest module
        "hub_tightness": tightness,               # + ⇒ the most-coordinated residues also carry the tightest contacts

        # register structure - the ordering the bipartite graph throws away
        "reg_corr": register,                     # acceptor/donor index correlation (±1 = a fixed threading register)
        "reg_slope": float(slope),                # acceptor residues traversed per donor residue (the pitch)
        "reg_scatter": scatter,                   # RMS departure from that register, per acceptor footprint
        "don_gap_mean": float(gaps_d.mean()) if gaps_d.size else np.nan,  # mean sequence gap between contacting donor residues
        "don_gap_max": float(gaps_d.max()) if gaps_d.size else np.nan,  # largest such gap (the linker between sub-sites)
        "acc_gap_mean": float(gaps_a.mean()) if gaps_a.size else np.nan,  # mean sequence gap between contacting acceptor residues
        "L_SA": float(span_acceptor),             # acceptor footprint span (mirror of L_SD)
        "rho_CA": float(n_c / span_acceptor),     # contact density along the acceptor footprint (mirror of rho_C)
        "N_sites_A": float(n_sites_a),            # contiguous acceptor contact clusters (mirror of N_sites_D)
        "S_site_A": float(s_site_a),              # residues in the largest acceptor cluster (mirror of S_site_D)
        #"per_helix_D": _sequence_periodicity(donor, PERIOD_HELIX),  # power at the α-helical period (one face of a helix)
        #"per_strand_D": _sequence_periodicity(donor, PERIOD_STRAND),  # power at the extended-peptide period (alternating side chains)
    }


#······································································#
# Entropic cost of confinement on binding
#······································································#

#······································································#
# 19. Confinement / chain entropy  (disordered-flank tethering)
#······································································#

def confinement_features(
        specific_interactions: pd.DataFrame,
        len_donor: int,
    ) -> dict:
    """Donor tethering features that set the entropic cost of binding.

    Disordered donor residues flanking the contact footprint must be confined on
    binding; longer flanks raise the entropic penalty and the ``eps_go`` needed
    to reach a given ``dG``.

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
        "L_flankN_D": flank_n,                    # N-terminal disordered flank (residues confined on binding)
        "L_flankC_D": flank_c,                    # C-terminal disordered flank length
        "f_flank_D": (flank_n + flank_c) / len_donor,  # flank fraction of the donor (entropic-cost proxy)
    }


#······································································#
# 20. Bound-state chain strain  (how stretched the donor must be)
#······································································#

# Cα-Cα contour length per residue (Å) - the CALVADOS bond length, and the
# hard limit on how far apart two donor beads can be pulled.
CONTOUR_PER_RESIDUE = 3.8

# Sequence separations below this are dominated by local backbone geometry
# rather than chain statistics, so extension ratios over them are meaningless.
MIN_SEPARATION = 3

# Extension ratio above which a segment counts as near-fully stretched.
EXTENDED_RATIO = 0.7


def chain_strain_features(structure_file: str, donor_chain_id: str) -> dict:
    """How extended the donor peptide is in the bound state.

    Thermodynamic perturbation theory gives ``∂dG/∂eps_go = -<N_formed>``, so the
    transition location is set by how many native contacts the bound ensemble can
    actually satisfy, not by ``N_C``. A donor whose native contacts require it
    to lie near-fully stretched pays a large configurational-entropy price: such
    conformations are a vanishing fraction of the chain's ensemble, so those
    contacts form only rarely and a deeper well is needed for the same ``dG``.

    Measured directly on the bound structure: for every donor residue pair
    separated by at least :data:`MIN_SEPARATION` residues, the extension ratio is
    the Cα-Cα distance over the contour length available between them,
    ``d_ij / (3.8*|i-j|)``, which is 1 for a fully stretched segment and ~0.3-0.5
    for a random coil.

    Parameters
    ----------
    structure_file : str
        Path to the full-atom complex (``.pdb``/``.cif``).
    donor_chain_id : str
        Author chain id of the donor (SLiM) chain.

    Returns
    -------
    dict
        Mean/max/upper-percentile extension ratios, the stretched-segment
        fraction, the end-to-end extension and the bound peptide's compaction
        relative to an ideal chain (``nan`` if the donor is too short).
    """
    ca = bsa.ca_coordinates(structure_file, donor_chain_id)
    n_res = len(ca)
    empty = {"ext_mean": np.nan, "ext_max": np.nan, "ext_p90": np.nan,
             "f_extended": np.nan, "ext_endtoend": np.nan, "Rg_bound_D": np.nan,
             "compaction_D": np.nan}
    if n_res <= MIN_SEPARATION:
        return empty

    separation = np.abs(np.subtract.outer(np.arange(n_res), np.arange(n_res)))
    distance = np.sqrt(((ca[:, None] - ca[None, :]) ** 2).sum(-1))
    usable = separation >= MIN_SEPARATION
    ratio = distance[usable] / (CONTOUR_PER_RESIDUE * separation[usable])

    rg = float(np.sqrt(((ca - ca.mean(axis=0)) ** 2).sum(-1).mean()))
    # Ideal (Gaussian) chain of the same length, as the reference for compaction.
    rg_ideal = CONTOUR_PER_RESIDUE * np.sqrt(n_res / 6.0)

    return {
        "ext_mean": float(ratio.mean()),           # mean segment extension (0.3-0.5 coil-like, →1 stretched)
        "ext_max": float(ratio.max()),             # most stretched segment in the bound peptide
        "ext_p90": float(np.percentile(ratio, 90)),  # upper-decile extension, how taut
        "f_extended": float((ratio > EXTENDED_RATIO).mean()),  # fraction of segments near-fully stretched
        "ext_endtoend": float(distance[0, -1] / (CONTOUR_PER_RESIDUE * (n_res - 1))),  # end-to-end extension of the whole peptide
        "Rg_bound_D": rg,                          # radius of gyration of the bound donor (Å)
        "compaction_D": float(rg / rg_ideal),      # bound Rg vs ideal-chain Rg (<1 compact, >1 extended)
    }


#······································································#
# Composites
#······································································#

#······································································#
# 21. Physics-structured composites  (target-free midpoint proxies)
#······································································#

# Confinement penalty per flanking donor residue (kJ/mol). Sets only the scale of
# ``eps0_hat``; a fitted model rescales it, so the value matters less than the ratio it
# introduces, which is not otherwise formable from the raw features.
KAPPA_FLANK = 1.0


def composite_features(row: dict, kappa_flank: float = KAPPA_FLANK) -> dict:
    """Target-free composites the transition midpoint should scale with.

    A linear score model cannot form ratios such as ``E_bg / N_C`` from the raw
    features, yet the midpoint of ``dG(eps_go)`` is inherently a per-contact
    quantity: the Gō term ``-eps_go*N`` must offset the background attraction and
    the confinement penalty, i.e. ``eps_go_mid ≈ (E_conf - E_bg) / N``. These
    composites hand that structure to the model directly.

    Parameters
    ----------
    row : dict
        The partially-assembled feature row; must already contain ``N_C``,
        ``N_eff``, ``E_bg``, ``sum_lam``, ``L_flankN_D`` and ``L_flankC_D``.
    kappa_flank : float, optional
        Per-flank-residue confinement penalty (kJ/mol); by default
        :data:`KAPPA_FLANK`.

    Returns
    -------
    dict
        Per-contact / per-effective-contact energies, per-contact flank load and
        the heuristic midpoint estimate ``eps0_hat`` (``nan`` for an empty
        contact set).
    """
    n_c = row["N_C"]
    n_eff = row["N_eff"]
    flank_tot = row["L_flankN_D"] + row["L_flankC_D"]
    e_bg = row["E_bg"]
    if not n_c or not n_eff:
        return {"E_bg_per_NC": np.nan, "E_bg_per_Neff": np.nan, "lam_per_Neff": np.nan,
                "flank_per_NC": np.nan, "eps0_hat": np.nan}
    return {
        "E_bg_per_NC": e_bg / n_c,                # background energy per contact
        "E_bg_per_Neff": e_bg / n_eff,            # background energy per effective (tightness-weighted) contact
        "lam_per_Neff": row["sum_lam"] / n_eff,   # interface hydrophobicity per effective contact
        "flank_per_NC": flank_tot / n_c,          # flank residues per contact (entropy-to-enthalpy ratio)
        # eps_go at which the Gō term offsets confinement minus background
        # attraction - a first-principles guess at the transition midpoint.
        "eps0_hat": (kappa_flank * flank_tot - e_bg) / n_eff,  # first-principles midpoint estimate
    }


#······································································#
# 22. Target-conditioned composites  (need the target dG)
#······································································#

def target_features(
        dG_target: float,
        n_contacts: int,
        background_energy: float,
        conditions: dict = DEFAULT_CONDITIONS,
    ) -> dict:
    """Composite features conditioning on the target free energy.

    Combines the target with the contact count and the ``eps_go = 0`` background
    into physically-motivated predictors. ``eps_go_estimate`` is a crude first-
    principles guess from ``dG_target ≈ E_bg - eps_go*N_C`` (each bound contact
    adds ``-eps_go``); an uncalibrated heuristic, not a substitute for the fitted
    model.

    Parameters
    ----------
    dG_target : float
        Target binding free energy (kJ/mol).
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
        The target ``Kd``, per-contact target energy, background-to-target gap
        and the heuristic ``eps_go_estimate``.
    """
    rt = constants.R * conditions["Temperature"]
    gap = background_energy - dG_target
    return {
        "dG_target": dG_target,                   # requested binding free energy (kJ/mol)
        "Kd_target": float(np.exp(dG_target / rt)),  # requested dissociation constant (M)
        "dG_per_contact": dG_target / n_contacts if n_contacts else np.nan,  # target energy per contact
        "dG_gap": gap,                            # background-to-target energy gap (E_bg − dG_target)
        "eps_go_estimate": max(0.0, gap / n_contacts) if n_contacts else np.nan,  # crude first-principles eps_go guess
    }


#······································································#
# The recommended core set
#······································································#

# Groups excluded from the core, by the function that produces them.
CORE_EXCLUDED = (
    "relative_burial_features",
    "slim_competition_features",
    "regional_patterning_features",
    "self_interaction_features",
    "free_ensemble_features",
    "register_features",
    "surface_patch_features",
# Group 28 is held out of the core so that ``core_columns`` keeps the composition it was
# benchmarked with. Remove this entry to fold the rich graph features in.
    "contact_graph_rich_features",
)

# Explicit membership of the excluded groups, so the core can be selected from a
# cached table without re-running the featuriser.
CORE_EXCLUDED_FEATURES = (
    # 11 relative burial
    "f_burial_D", "f_burial_iface", "rsa_bound_iface", "drsa_mean_iface", "drsa_max",
    "f_deep_D", "n_deep_D", "burial_gini_D",
    # 16 SLiM competition
    "L_tail_A", "L_tailN_A", "L_tailC_A", "f_tail_A", "lam_tail_A", "sticky_tail_A",
    "Q_tail_A", "f_sticky_tail_A", "d_tail_iface_A", "sticky_domnonif_A",
    "tail_frac_nonif_A", "SCD_tail_A", "sticky_nonif_D", "lam_go_D", "lam_nongo_D",
    "Q_nonif_D", "SCD_A", "chg_block_A", "chg_block_D", "NCPR_A", "NCPR_D", "NCPR_prod",
    # 17 regional patterning
    *(f"{stat}_{region}"
      for region in ("don", "slim", "nflank", "cflank", "ntail", "dom", "ctail")
      for stat in ("L", "lam_mean", "SHD", "lam_block", "SCD", "chg_block", "NCPR", "FCR")),
    # 23 chain self-interaction
    "self_dE_D", "self_Rg_D", "self_compact_D", "self_ncontact_D", "tail_dE_A",
    "tail_selfdock_A", "tail_d_site_A",
    # 24 free-donor ensemble
    "Rg_free_D", "Ree_free_D", "nu_D", "n_intra_free_D", "Rg_free_per_N", "swell_D",
    # 25 register degeneracy
    "n_registers_D", "register_best_D", "register_margin_D", "lam_autocorr_D",
    "q_autocorr_D",
    # 26 surface patches
    "n_patch_A", "S_patch_max_A", "lam_patch_max_A", "patch_ratio_A", "d_patch_site_A",
    # 28 rich contact graph (provisional - see CORE_EXCLUDED)
    *GRAPH_RICH_FEATURES,
)

# Every feature computable from the contact list alone (groups 3-7 and 28). This is the
# set that varies between graph-edited variants of one parent, where sequence, structure
# and box are held fixed.
CONTACT_GRAPH_FEATURES = (
    # 3 count and topology
    "N_C", "N_RC", "N_RCA", "N_RCD", "N_RCmin", "N_RCmax", "val_A", "val_D",
    "L_SD", "rho_C", "f_int_A", "f_int_D", "N_sites_D", "S_site_D", "deg_max_D",
    "deg_std_D",
    # 4 geometry
    "r0_mean", "r0_std", "r0_min", "r0_max", "N_eff", "f_short",
    # 5 cooperativity
    "N_comp", "S_comp_max", "f_comp_max", "graph_density", "deg_mean", "deg_max_A",
    "deg_std_A", "N_cycles", "f_redundant",
    # 6 native-distance heterogeneity
    "r0_iqr", "r0_skew", "r0_cv", "r0_p90", "f_loose", "w_participation", "w_entropy",
    # 7 spectra
    "lambda2", "f_cut", "cut_balance", "deg_entropy",
    # 28 rich graph descriptors
    *GRAPH_RICH_FEATURES,
)

my_little_features =[
    # Conditions
    'C_box',

    # Basin depth
    'N_RCA',
    'N_RCD',
    'N_eff',
    'relax_G_min',

    # Unbinding barrier
    'graph_density',
    'f_redundant',
    'val_A',
    'val_D',
    'deg_max_D',
    'deg_std_D',
    'r0_mean',
    'w_participation',
    'n_deep_D',

    # Cooperativity and multistate
    'S_comp_max',
    'f_comp_max',

    # Spatial search
    'BSA_total',
    'f_burial_iface',
    'q_pair_sum',
    'mean_lam',
    'E_bg',
    'L_don',
    'L_slim',
    'Rg_A',
    'SASA_cg_A',
    #'SASA_site',

    # Orientational search
    #'dipole_A',
    #'dipole_site_align_A',

    # Conformational search
    'SCD_D',
    'lam_nongo_D',
    'Q_nonif_D',
    'ext_mean',

    # Secondary retention
    #'sticky_A',
    #'Qint_prod',

    # Competition
    #'L_flankN_D',
    #'L_flankC_D',
    #'f_flank_D',

    # Friend's suggestion
    'relax_n_go',
    'relax_E_min',
    'S_site_D',
    'lambda2',
    'k_core_max',      # unbinding barrier - cooperative failure of the load-bearing core
    'hub_tightness',   # basin depth - are the anchors also the tightest contacts
    'reg_corr',        # conformational search - one threading register, or scattered
    'reg_scatter',     # conformational search - how loosely that register is defined
    'deg_cv_A',        # unbinding barrier, acceptor side (mirrors deg_std_D)
    'S_site_A',        # cooperativity - one acceptor groove or several patches
    'deg_assort',      # cooperativity - hub-to-hub wiring
]


def core_columns(feature_names: list[str]) -> list[str]:
    """Select the recommended core from a full :func:`featurize` row.

    Drops the seven groups in :data:`CORE_EXCLUDED` and keeps everything else, in
    the order given. On the 273-system library this leaves 128 of 237 numeric
    features and scores RMSE 6.934 against 7.292 for the full set, under strict
    leave-one-system-out with the production GP. Selecting further within the
    core buys at most ~0.4 kJ/mol and does not hold up across folds, while
    selecting from the full feature table is worse than the core at every budget.

    Selection is by name only, so constant and non-numeric columns pass through;
    filter those separately. On the cached ``data/modelling/features.csv`` the
    raw count is 130 of 239.

    Parameters
    ----------
    feature_names : list of str
        Column names of a full feature table.

    Returns
    -------
    list of str
        The core subset, order preserved.
    """
    excluded = set(CORE_EXCLUDED_FEATURES)
    return [name for name in feature_names if name not in excluded]


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
    """Assemble the default feature row for a single SLiM of a system.

    Resolves the acceptor/donor chains, parses the contacts and attaches their
    residue parameters once, then concatenates every default feature group
    (groups 1–11 and 13–28 above, including the structure-based interface,
    relaxation and diffuseness descriptors) into a single flat row of ~290
    numeric features. The target-conditioned composites (group 12) are added only
    when ``dG_target`` is given. The optional entropy / ESM groups are not
    included here - call them explicitly if wanted.

    Parameters
    ----------
    system : dict
        A single system entry from ``systems.json``.
    specific_interactions_txt : str
        Path to the system's ``specific_interactions.txt``.
    residues : pandas.DataFrame
        Residue-parameter table (see :func:`load_residue_params`).
    dG_target : float or None, optional
        Target binding free energy (kJ/mol); when given, adds the
        target-conditioned composites, by default ``None``.
    slim_index : int, optional
        Index into the system's ``SLiMs`` list, by default ``0``.
    conditions : dict, optional
        Thermodynamic conditions; by default :data:`DEFAULT_CONDITIONS`.

    Returns
    -------
    dict
        One flat ``{name: value}`` row, prefixed by the system ``ID``.

    Raises
    ------
    KeyError
        If the SLiM entry lacks the ``PDB`` / ``PDB Acceptor chain ID`` /
        ``PDB Donor chain ID`` fields needed for the interface-area features.
    """
    slim_entry, acceptor_chain, donor_chain = resolve_slim(system, slim_index)
    acceptor_seq = acceptor_chain["Sequence"]
    donor_seq = donor_chain["Sequence"]

    topology = calvados_topology(system["ID"])
    contacts = utils.parse_specific_interactions(specific_interactions_txt)
    contacts = _contact_residue_params(contacts, acceptor_seq, donor_seq, residues)

    row = {"System": system["ID"]}
    row.update(box_features(system))
    row.update(chain_features(acceptor_seq, donor_seq, residues))
    row.update(contact_topology_features(contacts, len(acceptor_seq), len(donor_seq)))
    row.update(contact_geometry_features(contacts))
    row.update(contact_graph_features(contacts))
    row.update(confinement_features(contacts, len(donor_seq)))
    row.update(interface_features(contacts))
    row.update(background_energy_features(contacts, conditions))
    row.update(contact_heterogeneity_features(contacts))
    row.update(contact_spectral_features(contacts))
    row.update(contact_graph_rich_features(contacts))
    row.update(competition_features(row))         # needs L_A/L_D/lam_mean/Q/N_RCA/N_C/E_bg from the groups above
    row.update(electrostatic_features(acceptor_seq, donor_seq, row, residues, conditions))
    row.update(slim_competition_features(acceptor_chain, donor_chain, contacts, residues))
    row.update(regional_patterning_features(acceptor_chain, donor_chain, contacts, residues))
    row.update(register_features(contacts, donor_seq, residues))
    row.update(free_ensemble_features(donor_seq, residues, conditions, topology))
    row.update(composite_features(row))           # needs N_C/N_eff/E_bg/sum_lam/flanks from the groups above

    structure, acc_id, don_id = _slim_structure(slim_entry)
    row.update(interface_area_features(structure, acc_id, don_id, residues))
    row.update(relative_burial_features(structure, acc_id, don_id, residues))
    row.update(domain_dimension_features(structure, acceptor_chain, acc_id, don_id, residues, system))
    row.update(self_interaction_features(structure, acceptor_chain, acc_id, don_id,
                                        residues, conditions, topology))
    row.update(surface_patch_features(structure, acc_id, don_id, residues))
    row.update(site_orientation_features(structure, acc_id, don_id, residues))
    row.update(chain_strain_features(structure, don_id))
    row.update(interface_diffuseness_features(structure, acc_id, don_id))
    row.update(relaxation_features(structure, acc_id, don_id, residues, conditions=conditions))

    if dG_target is not None:
        row.update(target_features(dG_target, row["N_C"], row["E_bg"], conditions))

    return row


def _slim_structure(slim_entry: dict) -> tuple[str, str, str]:
    """Structure path and acceptor/donor chain ids for the interface features."""
    try:
        return (slim_entry["PDB"],
                slim_entry["PDB Acceptor chain ID"],
                slim_entry["PDB Donor chain ID"])
    except KeyError as exc:
        raise KeyError(
            f"SLiM '{slim_entry['ID']}' lacks structure field {exc} needed for "
            f"the interface-area features."
        ) from exc


#······································································#
# Optional - configurational entropy  (sparrow ν/Rg + localCIDER descriptors)
#······································································#
# Sequence-property predictors of chain expansion, used as a zero-shot proxy for the
# binding conformational entropy. Not part of :func:`featurize`: these need optional
# dependencies and did not improve prediction.

# Default Flory scaling exponent when the ν predictor is unavailable (θ-solvent).
_DEFAULT_NU = 0.5

try:
    from sparrow import Protein as _SparrowProtein
    _HAS_SPARROW = True
except ImportError:
    _HAS_SPARROW = False
try:
    from localcider.sequenceParameters import SequenceParameters as _CiderParams
    _HAS_CIDER = True
except ImportError:
    _HAS_CIDER = False


def _sequence_descriptors(seq: str) -> dict:
    """Chain-expansion / charge-patterning descriptors of a single sequence.

    Uses the ALBATROSS/Tesei models via ``sparrow`` for the Flory scaling
    exponent ``nu`` and radius of gyration ``rg`` (a more expanded chain has more
    conformational entropy to lose on binding), and ``localCIDER`` for charge-
    patterning / hydropathy descriptors. Returns only the descriptors whose
    predictor is installed; a too-short or non-standard sequence yields ``{}``.

    Parameters
    ----------
    seq : str
        One-letter sequence (non-standard residues are dropped first).

    Returns
    -------
    dict
        Any of ``{'nu', 'rg', 'kappa', 'FCR', 'SCD', 'hydro', 'f_disorder'}``.
    """
    seq = "".join(c for c in seq if c in _STANDARD_AA)
    if len(seq) < 5:
        return {}
    out = {}
    if _HAS_CIDER:
        try:
            cider = _CiderParams(seq)
            kappa = cider.get_kappa()
            out["kappa"] = float(kappa) if kappa >= 0 else np.nan  # charge segregation (−1 ⇒ undefined, no charges)
            out["FCR"] = float(cider.get_FCR())                    # fraction of charged residues
            out["SCD"] = float(cider.get_SCD())                    # sequence charge decoration (charge patterning)
            out["hydro"] = float(cider.get_mean_hydropathy())      # mean hydropathy
            out["f_disorder"] = float(cider.get_fraction_disorder_promoting())  # disorder-promoting fraction
        except Exception:
            pass
    if _HAS_SPARROW:
        try:
            protein = _SparrowProtein(seq)
            out["nu"] = float(protein.predictor.scaling_exponent())     # Flory scaling exponent (chain expansion)
            out["rg"] = float(protein.predictor.radius_of_gyration())   # radius of gyration (Å)
        except Exception:
            pass
    return out


def entropy_features(donor_seq: str, specific_interactions: pd.DataFrame) -> dict:
    """Configurational-entropy features of the (disordered) donor SLiM.

    Estimates the donor's chain expansion (:func:`_sequence_descriptors`) for the
    whole donor, the contacting motif, and the confined flanks, plus a
    polymer-physics composite ``S_conf`` (more expanded flanks lose more entropy
    per residue) and ``nu_gap`` (how much more compact the motif is than the
    whole chain).

    Parameters
    ----------
    donor_seq : str
        Full one-letter donor (SLiM) chain sequence.
    specific_interactions : pandas.DataFrame
        Parsed contacts; the donor residue indices set the motif/flank split.

    Returns
    -------
    dict
        Expansion descriptors for the donor (``*_D``), motif (``*_motif``) and
        flanks (``*_flank``), plus composites ``S_conf`` and ``nu_gap``. Keys are
        present only for installed predictors (empty regions are skipped).
    """
    don = specific_interactions["Donor residue"].to_numpy()
    n = len(donor_seq)
    lo, hi = (int(don.min()), int(don.max())) if don.size else (1, n)
    motif = donor_seq[lo - 1:hi]                  # contacting span (the SLiM motif)
    flanks = donor_seq[:lo - 1] + donor_seq[hi:]  # confined disordered flanks

    row = {}
    for tag, seq in (("D", donor_seq), ("motif", motif), ("flank", flanks)):
        for key, value in _sequence_descriptors(seq).items():
            row[f"{key}_{tag}"] = value

    nu_flank = row.get("nu_flank", row.get("nu_D", _DEFAULT_NU))
    row["S_conf"] = len(flanks) * nu_flank         # confinement-entropy proxy (flank length × expansion)
    if "nu_D" in row and "nu_motif" in row:
        row["nu_gap"] = row["nu_D"] - row["nu_motif"]  # motif-vs-chain compaction
    return row


#······································································#
# Optional - ESM-2 sequence embedding  (fair-esm + torch)
#······································································#
# Learned per-sequence descriptor. NOT part of :func:`featurize`; returns a
# vector (reduce with PCA inside each CV fold before feeding a model).

# Lazily-loaded ESM-2 models, keyed by checkpoint name.
_ESM_CACHE: dict = {}


def _load_esm(model_name: str = "esm2_t12_35M_UR50D"):
    """Load and cache an ESM-2 model, alphabet and batch converter."""
    if model_name not in _ESM_CACHE:
        import esm  # optional dependency
        model, alphabet = getattr(esm.pretrained, model_name)()
        model.eval()
        _ESM_CACHE[model_name] = (model, alphabet, alphabet.get_batch_converter())
    return _ESM_CACHE[model_name]


def esm_embedding(seq: str, model_name: str = "esm2_t12_35M_UR50D",
                  repr_layer: int | None = None) -> "np.ndarray":
    """Mean-pooled ESM-2 residue embedding of a sequence.

    Returns the per-residue final-layer representation averaged over the sequence
    - a fixed-length learned descriptor capturing sequence/biophysical context
    beyond the hand-crafted features. High-dimensional (e.g. 480 for
    ``esm2_t12_35M``): reduce with PCA (fit inside each CV fold) before feeding a
    model. Requires the optional ``fair-esm`` + ``torch`` dependencies.

    Parameters
    ----------
    seq : str
        One-letter sequence (non-standard residues dropped first).
    model_name : str, optional
        ESM-2 checkpoint, by default ``"esm2_t12_35M_UR50D"`` (480-dim).
    repr_layer : int or None, optional
        Representation layer to pool; defaults to the model's final layer.

    Returns
    -------
    numpy.ndarray
        Mean-pooled embedding, shape ``(embed_dim,)``.
    """
    import torch
    model, _alphabet, batch_converter = _load_esm(model_name)
    layer = model.num_layers if repr_layer is None else repr_layer
    seq = "".join(c for c in seq if c in _STANDARD_AA)
    _, _, tokens = batch_converter([("seq", seq)])
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer])
    rep = out["representations"][layer][0, 1:len(seq) + 1]   # drop BOS/EOS tokens
    return rep.mean(0).numpy()


#······································································#
# Transform expansion  (systematic nonlinear pool for embedded selection)
#······································································#

def expand_transforms(
        frame: pd.DataFrame,
        columns: list[str] | None = None,
        eps: float = 1e-9,
    ) -> pd.DataFrame:
    """Append systematic nonlinear transforms of numeric feature columns.

    Adds ``1/x`` (strictly-positive columns), ``log`` and ``sqrt`` (non-negative
    columns) so a linear score model can still exploit the physically-expected
    ``1/N``-type nonlinearities (the ``1/N`` transform was the proof this pays
    off). Meant to feed an embedded selector (ElasticNet / stability selection),
    which prunes the enlarged pool - do not use unregularised on the result.

    Parameters
    ----------
    frame : pandas.DataFrame
        A numeric feature table (non-numeric columns are ignored).
    columns : list of str, optional
        Which columns to transform; by default every numeric column.
    eps : float, optional
        Guard threshold on the column minimum for ``1/x``/``log``/``sqrt``
        eligibility, by default ``1e-9``.

    Returns
    -------
    pandas.DataFrame
        ``frame`` with added ``inv_*`` / ``log_*`` / ``sqrt_*`` columns for the
        eligible inputs. Columns already prefixed with a transform are skipped.
    """
    numeric = frame.select_dtypes("number")
    columns = list(numeric.columns) if columns is None else columns
    out = frame.copy()
    for col in columns:
        if col.startswith(("inv_", "log_", "sqrt_")):
            continue
        x = numeric[col].to_numpy(dtype=float)
        col_min = np.nanmin(x)
        if col_min > eps:
            out[f"inv_{col}"] = 1.0 / x
            out[f"log_{col}"] = np.log(x)
        if col_min >= 0.0:
            out[f"sqrt_{col}"] = np.sqrt(x)
    return out
