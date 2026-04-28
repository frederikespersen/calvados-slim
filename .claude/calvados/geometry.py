"""
src.calvados.geometry
---------------------
Initial geometry construction: spiral IDPs, PDB extraction, multi-chain placement.
"""

############################################################
# Imports
############################################################

import copy
from typing import Optional

import MDAnalysis as mda
import mdtraj as md
import numpy as np
import yaml
from Bio import SeqIO


############################################################
# Spiral initial configuration
############################################################

def _polar_to_cartesian(r: float, phi: float) -> tuple[float, float]:
    """Convert polar coordinates to Cartesian (x, y)."""
    return r * np.cos(phi), r * np.sin(phi)


def xy_spiral_array(
    n: int,
    delta: float = 0.0,
    arc: float = 0.38,
    separation: float = 0.7,
) -> np.ndarray:
    """Place *n* beads on an Archimedes spiral in the xy-plane.

    Gives a compact, non-overlapping initial configuration for IDPs that
    avoids large gradients at the start of minimisation.

    Parameters
    ----------
    n : int
        Number of beads (residues).
    delta : float, optional
        Offset added to all coordinates (nm).
    arc : float, optional
        Arc length between consecutive points (nm).
    separation : float, optional
        Radial distance between successive turns (nm).

    Returns
    -------
    np.ndarray
        Shape ``(n, 3)`` array of positions in nm; z-coordinate is zero.
    """
    r = arc
    b = separation / (2 * np.pi)
    phi = r / b
    coords: list[list[float]] = []
    for _ in range(n):
        x, y = _polar_to_cartesian(r, phi)
        coords.append([x, y, 0.0])
        phi += arc / r
        r = b * phi
    return np.array(coords) + delta


############################################################
# PDB geometry extraction
############################################################

def get_ssdomains(name: str, fdomains: str, verbose: bool = True) -> list[list[int]]:
    """Parse secondary-structure domain residue indices from a YAML file.

    Parameters
    ----------
    name : str
        Protein name key in the YAML.
    fdomains : str
        Path to the domains YAML file.
    verbose : bool, optional
        Print the parsed domains.

    Returns
    -------
    list[list[int]]
        Each inner list contains 1-based residue indices belonging to one domain.
    """
    with open(fdomains, "r") as fh:
        domainbib = yaml.safe_load(fh.read())

    domains = domainbib[name]
    if verbose:
        print(f"Using domains {domains}")

    ssdomains: list[list[int]] = []
    for domain in domains:
        if isinstance(domain[0], int):
            ssdomains.append(list(range(domain[0], domain[1] + 1)))
        else:
            # Non-contiguous domain: union of sub-ranges
            merged: list[int] = []
            for subdomain in domain:
                merged += list(range(subdomain[0], subdomain[1] + 1))
            ssdomains.append(merged)
    return ssdomains


def geometry_from_pdb(
    pdb2predic: str,
    pdb_ini: str,
    compact_ini: bool,
    coarse_grained: str = "CA",
    ssdomains: Optional[list[list[int]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract bead positions from a PDB file.

    For CA coarse-graining every residue is represented by its Cα atom.
    For COM coarse-graining, residues inside ``ssdomains`` use their
    heavy-atom centre of mass; all others fall back to Cα.

    Parameters
    ----------
    pdb2predic : str
        Path to the reference PDB (used to define equilibrium geometry).
    pdb_ini : str
        Path to the initial-configuration PDB (used only when
        ``compact_ini=True``).
    compact_ini : bool
        If ``True``, load ``pdb_ini`` as the starting configuration;
        otherwise use the reference geometry for both.
    coarse_grained : str, optional
        Coarse-graining scheme: ``"CA"`` or ``"COM"``.
    ssdomains : list[list[int]], optional
        1-based residue index lists for COM domains (required when
        ``coarse_grained != "CA"``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(ini_pos, pos, center_of_mass)`` all in nm.
        ``pos`` is the reference geometry; ``ini_pos`` may differ when
        ``compact_ini=True``.
    """
    u = mda.Universe(pdb2predic)

    if coarse_grained == "CA":
        cas = u.select_atoms("name CA")
        center_of_mass = cas.center_of_mass()
        cas.translate(-center_of_mass)
        pos = cas.positions / 10.0  # Å → nm
    else:
        assert ssdomains is not None, "ssdomains required for COM coarse-graining"
        ss_residues: list[int] = sum(ssdomains, [])
        pos_list: list[np.ndarray] = []
        for i, res in enumerate(u.residues):
            if i + 1 in ss_residues:
                pos_list.append(res.atoms.center_of_mass())
            else:
                pos_list.append(res.atoms.select_atoms("name CA").positions.squeeze())
        center_of_mass = u.atoms.center_of_mass()
        pos = (np.array(pos_list) - center_of_mass) / 10.0

    if compact_ini:
        ini_conformation = md.load_pdb(pdb_ini)
        ini_pos = np.squeeze(ini_conformation.xyz)
        cog = ini_pos.mean(axis=0)
        ini_pos -= cog
        center_of_mass = cog * 10.0  # nm → Å for consistency with MDA
    else:
        ini_pos = pos

    return ini_pos, pos, center_of_mass / 10.0  # all in nm


############################################################
# Multi-chain placement
############################################################

def place_chains(
    L: float,
    components: dict,
    names_list: list[str],
    default_lz_idp: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Arrange all protein chains on a regular grid inside the simulation box.

    Each component is tiled in the xy-plane; different components are stacked
    along z.  The final array is centred at the origin.

    Parameters
    ----------
    L : float
        Box side length in xy (nm).
    components : dict
        Component metadata; each entry must contain ``ini_pos``, ``chains``,
        and ``isIDP``.
    names_list : list[str]
        Ordered list of component names.
    default_lz_idp : float, optional
        Maximum allowed z-extent for IDP stacking (nm).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(pos_chains, centre_of_geometry)`` in nm.
    """
    comps = copy.deepcopy(components)

    for name in names_list:
        pos = comps[name]["ini_pos"]
        chains = comps[name]["chains"]
        is_idp = comps[name]["isIDP"]

        x_span = np.abs(pos[:, 0].max() - pos[:, 0].min()) + 1
        y_span = np.abs(pos[:, 1].max() - pos[:, 1].min()) + 1
        x_slots = int(L // x_span)
        y_slots = int(L // y_span)
        per_plane = x_slots * y_slots

        if is_idp:
            z_step = 1.47  # fixed IDP layer spacing (nm)
            if x_span > L or y_span > L:
                raise ValueError("Box too small in xy for IDP placement")
            if z_step * (chains + 1) > default_lz_idp:
                raise ValueError("Box too short in z for IDP stacking")
        else:
            z_step = np.ceil(np.abs(pos[:, 2].max() - pos[:, 2].min())) + 1

        placed: list[list[float]] = []
        added = 0
        for zp in range(chains // per_plane + 1):
            for yp in range(y_slots):
                for xp in range(x_slots):
                    if added < chains:
                        offset = np.array([x_span * xp, y_span * yp, z_step * zp])
                        placed.extend((pos + offset).tolist())
                        added += 1

        arr = np.array(placed)
        cog = np.array([
            (arr[:, 0].max() + arr[:, 0].min()) / 2,
            (arr[:, 1].max() + arr[:, 1].min()) / 2,
            (arr[:, 2].max() + arr[:, 2].min()) / 2,
        ])
        comps[name]["pos_chains_tmp"] = arr - cog

    # Stack components along z with a 1 nm gap between adjacent groups
    all_pos: list[np.ndarray] = []
    for idx, name in enumerate(names_list):
        if idx == 0:
            z_offset = np.zeros(3)
        else:
            prev = comps[names_list[idx - 1]]["pos_chains_tmp"]
            curr = comps[name]["pos_chains_tmp"]
            z_offset = np.array([0, 0, np.abs(prev[:, 2]).max() + np.abs(curr[:, 2]).max() + 1])
        comps[name]["pos_chains_tmp"] = comps[name]["pos_chains_tmp"] + z_offset
        all_pos.append(comps[name]["pos_chains_tmp"])

    pos_chains = np.vstack(all_pos)
    cog_total = np.array([
        (pos_chains[:, 0].max() + pos_chains[:, 0].min()) / 2,
        (pos_chains[:, 1].max() + pos_chains[:, 1].min()) / 2,
        (pos_chains[:, 2].max() + pos_chains[:, 2].min()) / 2,
    ])
    pos_chains -= cog_total
    return pos_chains, cog_total


############################################################
# Pairwise distance matrix
############################################################

def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix between two sets of points.

    Parameters
    ----------
    a : np.ndarray
        Shape ``(m, d)`` — first set of points.
    b : np.ndarray
        Shape ``(n, d)`` — second set of points.

    Returns
    -------
    np.ndarray
        Shape ``(m, n)`` distance matrix.
    """
    d1 = -2 * np.dot(a, b.T)
    d2 = np.sum(np.square(a), axis=1, keepdims=True)
    d3 = np.sum(np.square(b), axis=1)
    return np.sqrt(d1 + d2 + d3)


############################################################
# Box validation
############################################################

def validate_box(
    L: float,
    Lz: float,
    components: dict,
    names_list: list[str],
) -> None:
    """Raise if the initial configuration does not fit inside the simulation box.

    Parameters
    ----------
    L : float
        Box side in xy (nm).
    Lz : float
        Box length in z (nm).
    components : dict
        Component metadata with ``ini_pos`` already populated.
    names_list : list[str]
        Ordered component names.
    """
    pos_all, _ = place_chains(L, components, names_list)
    z_extent = np.abs(pos_all[:, 2].max() - pos_all[:, 2].min())
    print(f"Initial state z-extent: {z_extent:.2f} nm  (box Lz = {Lz} nm)")
    if z_extent + 1 > Lz:
        raise ValueError(f"Lz={Lz} nm is too small; need at least {z_extent + 1:.1f} nm")
