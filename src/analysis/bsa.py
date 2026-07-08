"""
src.analysis.bsa
------------------------------------------------------------------------
Buried-surface-area (BSA) contact map from a full-atom structure.

Inter-chain residue contacts are read off the solvent-
accessible surface each chain loses to the other. A residue pair counts as a
contact only if it clears two independent bars:

1. **Meaningful burial**: The surface buried by a single partner residue
   is a real fraction of at least one of the two residues' own isolated
   SASA, not just a sliver of it (see :func:`buried_contact_map`).
2. **Direct contact**: At least one heavy-atom pair between the two
   residues sits within van der Waals contact distance.

Burial alone can flag residues separated by a pinched-off solvent channel
with no atoms actually touching; proximity alone can flag residues that
brush past each other without meaningfully occluding either one's surface.
Requiring both keeps the map to contacts that are substantial and physical.

Lengths and areas are in Å / Å² throughout (MDAnalysis' native unit).
"""


import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from src import utils, constants


#······································································#
# Parameters
#······································································#

# Solvent probe radius, Å (~1.4 Å, the standard Shrake-Rupley/Lee-Richards value).
_PROBE_RADIUS = 1.4

# Sphere test points per atom; more points smooth the per-atom area estimate at
# quadratic-in-neighbours cost. 960 matches common atomistic SASA tools.
_N_SPHERE_POINTS = 960

# van der Waals radius, Å, for any element absent from
# constants.element.vdw_radius; carbon, the commonest protein atom.
_DEFAULT_VDW = 1.70

# Minimum fraction of a residue's own isolated SASA that a single partner
# residue must bury to count as a meaningful (not just measurable) burial.
_RELATIVE_BURIAL_CUTOFF = 0.10

# Maximum heavy-atom-heavy-atom distance, Å, counted as direct contact -- the
# common CAPRI-style cutoff for an atomic interface contact.
_CONTACT_DISTANCE = 5.0


#······································································#
# Sphere sampling
#······································································#

def _fibonacci_sphere(n_points: int = _N_SPHERE_POINTS) -> np.ndarray:
    """Deterministic, near-uniform points on the unit sphere.

    Parameters
    ----------
    n_points : int, optional
        Number of points to place, by default ``960``.

    Returns
    -------
    numpy.ndarray
        Unit vectors, shape ``(n_points, 3)``.
    """
    i = np.arange(n_points)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_points)
    theta = golden_angle * i
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


#······································································#
# Chain atoms
#······································································#

def _element(atom_name: str) -> str:
    """Guess an atom's element symbol from its (standard-residue) PDB name.

    Standard amino-acid atoms are only C, N, O, S and H, whose element is the
    first alphabetic character of the name (leading hydrogen-numbering digits,
    e.g. ``"1HB"``, are stripped first). This sidesteps the structure's element
    column, which readers populate inconsistently.

    Parameters
    ----------
    atom_name : str
        PDB atom name (e.g. ``"CA"``, ``"ND1"``, ``"1HB"``).

    Returns
    -------
    str
        Single-letter element symbol, or ``""`` for an empty name.
    """
    return atom_name.lstrip("0123456789")[:1].upper()

#······································································#

def _chain_atoms(universe: mda.Universe, chain_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Heavy atoms of a chain's resolved standard residues, in structure order.

    Water, ligands and non-standard residues are dropped (as SMOG's cleanup
    does), and hydrogens excluded so the surface is scored over heavy atoms
    only. The residue ordering matches curation's PDB author-residue ordering
    (:meth:`src.curation.System._pdb_author_resids`), so the resulting map
    aligns index-for-index with the SMOG shadow map.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        The loaded structure.
    chain_id : str
        Author chain id to extract.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, int]
        ``(positions, radii, res_index, n_residues)``: heavy-atom coordinates
        ``(M, 3)`` Å, per-atom van der Waals radii ``(M,)`` Å, the 0-based
        residue ordinal of each atom ``(M,)``, and the residue count ``N``.

    Raises
    ------
    ValueError
        If the chain has no resolved standard amino-acid residues.
    """
    residues = [res for res in utils.select_chain(universe, chain_id).residues
                if res.resname in constants.residue.name3to1]
    if not residues:
        raise ValueError(f"Chain '{chain_id}' has no resolved standard amino-acid residues.")

    positions, radii, res_index = [], [], []
    for i, res in enumerate(residues):
        for atom in res.atoms:
            element = _element(atom.name)
            if element == "H":
                continue
            positions.append(atom.position)
            radii.append(constants.element.vdw_radius.get(element, _DEFAULT_VDW))
            res_index.append(i)

    return (np.array(positions), np.array(radii), np.array(res_index), len(residues))

#······································································#

def _combine_chains(
        pos1: np.ndarray, rad1: np.ndarray, res1: np.ndarray,
        pos2: np.ndarray, rad2: np.ndarray, res2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack two chains' atoms into one array set, labelled by chain.

    Parameters
    ----------
    pos1, rad1, res1 : numpy.ndarray
        Chain 1's per-atom positions, radii and residue ordinals, as returned
        by :func:`_chain_atoms`.
    pos2, rad2, res2 : numpy.ndarray
        Chain 2's equivalents.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(positions, radii, chain_labels, res_labels)``: combined ``(M, 3)``
        positions and ``(M,)`` radii, plus ``(M,)`` chain labels (``0``/``1``)
        and each atom's residue ordinal within its own chain.
    """
    positions = np.vstack([pos1, pos2])
    radii = np.concatenate([rad1, rad2])
    chain_labels = np.concatenate([np.zeros(len(pos1), dtype=int), np.ones(len(pos2), dtype=int)])
    res_labels = np.concatenate([res1, res2])
    return positions, radii, chain_labels, res_labels


#······································································#
# Isolated per-residue SASA
#······································································#

def _atom_sasa(positions: np.ndarray, radii: np.ndarray, probe_radius: float, sphere_points: np.ndarray) -> np.ndarray:
    """Per-atom SASA of a standalone atom set, via Shrake-Rupley.

    Unlike :func:`_buried_area_matrix`, every atom is scored regardless of
    whether it has any burial -- this is the isolated-chain SASA used as the
    denominator for relative burial in :func:`buried_contact_map`, so buried
    core atoms (contributing zero) must be included, not skipped.

    Parameters
    ----------
    positions : numpy.ndarray
        Atom coordinates, shape ``(M, 3)``, Å.
    radii : numpy.ndarray
        Per-atom van der Waals radius, shape ``(M,)``, Å.
    probe_radius : float
        Solvent probe radius, Å.
    sphere_points : numpy.ndarray
        Unit-sphere test points, shape ``(n_points, 3)``.

    Returns
    -------
    numpy.ndarray
        Per-atom SASA, shape ``(M,)``, Å².
    """
    inflated = radii + probe_radius
    point_area = 4.0 * np.pi * inflated**2 / sphere_points.shape[0]

    dists = distance_array(positions, positions)
    np.fill_diagonal(dists, np.inf)
    neighbour_mask = dists < (inflated[:, None] + inflated[None, :])

    sasa = np.empty(positions.shape[0])
    for i, neighbours in enumerate(neighbour_mask):
        neighbours = np.flatnonzero(neighbours)
        if neighbours.size == 0:
            sasa[i] = point_area[i] * sphere_points.shape[0]
            continue
        surface = positions[i] + sphere_points * inflated[i]
        point_dists = distance_array(surface, positions[neighbours])
        accessible = ~(point_dists < inflated[neighbours]).any(axis=1)
        sasa[i] = accessible.sum() * point_area[i]

    return sasa

#······································································#

def _residue_sasa(
        positions: np.ndarray,
        radii: np.ndarray,
        res_labels: np.ndarray,
        n_res: int,
        probe_radius: float,
        sphere_points: np.ndarray,
    ) -> np.ndarray:
    """Per-residue isolated SASA of a single chain, summed over its atoms.

    Parameters
    ----------
    positions : numpy.ndarray
        Chain atom coordinates, shape ``(M, 3)``, Å.
    radii : numpy.ndarray
        Per-atom van der Waals radius, shape ``(M,)``, Å.
    res_labels : numpy.ndarray
        0-based residue ordinal of each atom, shape ``(M,)``.
    n_res : int
        Number of residues in the chain.
    probe_radius : float
        Solvent probe radius, Å.
    sphere_points : numpy.ndarray
        Unit-sphere test points, shape ``(n_points, 3)``.

    Returns
    -------
    numpy.ndarray
        Per-residue SASA of the chain scored on its own, shape ``(n_res,)``, Å².
    """
    atom_sasa = _atom_sasa(positions, radii, probe_radius, sphere_points)
    residue_sasa = np.zeros(n_res)
    np.add.at(residue_sasa, res_labels, atom_sasa)
    return residue_sasa


#······································································#
# Buried surface area
#······································································#

def _buried_area_matrix(
        positions: np.ndarray,
        radii: np.ndarray,
        chain_labels: np.ndarray,
        res_labels: np.ndarray,
        n1: int,
        n2: int,
        probe_radius: float,
        sphere_points: np.ndarray,
    ) -> np.ndarray:
    """Reduce a full-atom Shrake-Rupley pass to an inter-chain residue BSA map.

    For every interface atom, its inflated surface is sampled and the points
    that are accessible with only its own chain present but occluded once the
    partner chain is added are found; each such point's area is attributed to
    the nearest occluding partner atom's residue. Summed over all atoms of both
    chains, this is the interface ``dSASA`` (``SASA_A + SASA_B - SASA_complex``)
    decomposed per residue pair, counting burial in both directions.

    Parameters
    ----------
    positions : numpy.ndarray
        Heavy-atom coordinates of both chains, shape ``(M, 3)``, Å.
    radii : numpy.ndarray
        Per-atom van der Waals radius, shape ``(M,)``, Å.
    chain_labels : numpy.ndarray
        Chain of each atom, ``0`` for chain 1 and ``1`` for chain 2, shape
        ``(M,)``.
    res_labels : numpy.ndarray
        0-based residue ordinal of each atom within its own chain, shape
        ``(M,)``.
    n1, n2 : int
        Residue counts of chain 1 and chain 2.
    probe_radius : float
        Solvent probe radius, Å.
    sphere_points : numpy.ndarray
        Unit-sphere test points, shape ``(n_points, 3)``.

    Returns
    -------
    numpy.ndarray
        Buried surface area per residue pair, shape ``(n1, n2)``, Å²; entry
        ``[i, j]`` is the surface buried between residue ``i`` of chain 1 and
        residue ``j`` of chain 2.
    """
    inflated = radii + probe_radius
    point_area = 4.0 * np.pi * inflated**2 / sphere_points.shape[0]

    dists = distance_array(positions, positions)
    np.fill_diagonal(dists, np.inf)
    reachable = dists < (inflated[:, None] + inflated[None, :])
    partner = chain_labels[:, None] != chain_labels[None, :]

    bsa = np.zeros((n1, n2))
    for a in np.flatnonzero((reachable & partner).any(axis=1)):
        neighbours = np.flatnonzero(reachable[a])
        surface = positions[a] + sphere_points * inflated[a]
        point_dists = distance_array(surface, positions[neighbours])
        inside = point_dists < inflated[neighbours]

        same_chain = chain_labels[neighbours] == chain_labels[a]
        accessible_isolated = ~inside[:, same_chain].any(axis=1)

        partner_cols = ~same_chain
        partner_atoms = neighbours[partner_cols]
        inside_partner = inside[:, partner_cols]
        buried = accessible_isolated & inside_partner.any(axis=1)
        if not buried.any():
            continue

        # Attribute each buried point to the nearest partner atom containing it.
        containing_dists = np.where(inside_partner, point_dists[:, partner_cols], np.inf)
        nearest = partner_atoms[np.argmin(containing_dists, axis=1)]

        a_res, b_res = res_labels[a], res_labels[nearest[buried]]
        if chain_labels[a] == 0:
            np.add.at(bsa, (a_res, b_res), point_area[a])
        else:
            np.add.at(bsa, (b_res, a_res), point_area[a])

    return bsa


#······································································#
# Direct atomic contact
#······································································#

def _direct_contact_matrix(
        pos1: np.ndarray,
        pos2: np.ndarray,
        res1: np.ndarray,
        res2: np.ndarray,
        n1: int,
        n2: int,
        contact_distance: float,
    ) -> np.ndarray:
    """Flag residue pairs with at least one heavy-atom pair within contact distance.

    Parameters
    ----------
    pos1, pos2 : numpy.ndarray
        Heavy-atom coordinates of chain 1 and chain 2, shape ``(M1, 3)`` /
        ``(M2, 3)``, Å.
    res1, res2 : numpy.ndarray
        0-based residue ordinal of each atom, shape ``(M1,)`` / ``(M2,)``.
    n1, n2 : int
        Residue counts of chain 1 and chain 2.
    contact_distance : float
        Maximum heavy-atom-heavy-atom distance counted as contact, Å.

    Returns
    -------
    numpy.ndarray
        Boolean ``(n1, n2)`` map, ``True`` where any atom of residue ``i``
        (chain 1) lies within `contact_distance` of any atom of residue ``j``
        (chain 2).
    """
    dists = distance_array(pos1, pos2)
    atom_i, atom_j = np.nonzero(dists < contact_distance)
    contact = np.zeros((n1, n2), dtype=bool)
    contact[res1[atom_i], res2[atom_j]] = True
    return contact


#······································································#
# Public interface
#······································································#

def buried_surface_area(
        structure_file: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        probe_radius: float = _PROBE_RADIUS,
        n_points: int = _N_SPHERE_POINTS,
    ) -> np.ndarray:
    """Inter-chain residue buried-surface-area map from a full-atom structure.

    Scores the two chains' heavy-atom solvent-accessible surface [1]_ with a
    Shrake-Rupley rolling ball [2]_, each atom taking its Bondi van der Waals
    radius [3]_, and returns, per residue pair, the surface buried between
    them when the complex forms. Only resolved standard amino acids are
    considered; missing loops are not reconstructed, since burial cannot be
    inferred for unresolved regions.

    Parameters
    ----------
    structure_file : os.PathLike
        Path to the input structure (``.pdb`` or ``.cif``).
    chain1_id : str
        Author chain id of the first chain.
    chain2_id : str
        Author chain id of the second chain.
    probe_radius : float, optional
        Solvent probe radius, Å, by default ``1.4``.
    n_points : int, optional
        Sphere test points per atom, by default ``960``.

    Returns
    -------
    numpy.ndarray
        Buried surface area of shape ``(N1, N2)``, Å², where entry ``[i, j]``
        is the surface buried between residue ``i`` of ``chain1_id`` and
        residue ``j`` of ``chain2_id``.

    References
    ----------
    .. [1] Lee B., Richards F.M. (1971) The interpretation of protein
       structures: estimation of static accessibility. Journal of Molecular
       Biology, 55(3), 379-400. https://doi.org/10.1016/0022-2836(71)90324-X
    .. [2] Shrake A., Rupley J.A. (1973) Environment and exposure to solvent
       of protein atoms. Lysozyme and insulin. Journal of Molecular Biology,
       79(2), 351-371. https://doi.org/10.1016/0022-2836(73)90011-9
    .. [3] Bondi A. (1964) van der Waals volumes and radii. The Journal of
       Physical Chemistry, 68(3), 441-451.
       https://doi.org/10.1021/j100785a001
    """
    universe = utils.load_structure(structure_file)
    pos1, rad1, res1, n1 = _chain_atoms(universe, chain1_id)
    pos2, rad2, res2, n2 = _chain_atoms(universe, chain2_id)
    positions, radii, chain_labels, res_labels = _combine_chains(pos1, rad1, res1, pos2, rad2, res2)
    sphere_points = _fibonacci_sphere(n_points)

    return _buried_area_matrix(
        positions, radii, chain_labels, res_labels, n1, n2, probe_radius, sphere_points,
    )

#······································································#

def buried_contact_map(
        structure_file: os.PathLike,
        chain1_id: str,
        chain2_id: str,
        relative_cutoff: float = _RELATIVE_BURIAL_CUTOFF,
        contact_distance: float = _CONTACT_DISTANCE,
        probe_radius: float = _PROBE_RADIUS,
        n_points: int = _N_SPHERE_POINTS,
    ) -> np.ndarray:
    """Compute the inter-chain buried-surface contact map between two chains.

    A residue pair is a contact only if it is both meaningfully buried and
    directly touching (see the module docstring for the rationale): the
    burial from :func:`buried_surface_area` must exceed `relative_cutoff` of
    at least one of the two residues' own isolated SASA (from
    :func:`_residue_sasa`), *and* at least one heavy-atom pair between them
    must lie within `contact_distance`. Mirrors
    :func:`src.analysis.smog.shadow_contact_map` in signature and output, and
    can be used as a self-contained (no external SMOG2 binary) drop-in for it.

    Parameters
    ----------
    structure_file : os.PathLike
        Path to the input structure (``.pdb`` or ``.cif``).
    chain1_id : str
        Author chain id of the first chain.
    chain2_id : str
        Author chain id of the second chain.
    relative_cutoff : float, optional
        Minimum fraction of a residue's own isolated SASA that a partner
        residue must bury, by default ``0.10`` -- large enough to exclude a
        glancing, negligible loss of surface, without requiring the
        (arbitrarily large) absolute burial a bulky residue could offer.
    contact_distance : float, optional
        Maximum heavy-atom-heavy-atom distance counted as direct contact, Å,
        by default ``5.0`` (the common CAPRI-style atomic-contact cutoff).
    probe_radius : float, optional
        Solvent probe radius, Å, by default ``1.4``.
    n_points : int, optional
        Sphere test points per atom, by default ``960``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(N1, N2)`` where entry ``[i, j]`` is ``True``
        if residue ``i`` of ``chain1_id`` contacts residue ``j`` of
        ``chain2_id``.
    """
    universe = utils.load_structure(structure_file)
    pos1, rad1, res1, n1 = _chain_atoms(universe, chain1_id)
    pos2, rad2, res2, n2 = _chain_atoms(universe, chain2_id)
    positions, radii, chain_labels, res_labels = _combine_chains(pos1, rad1, res1, pos2, rad2, res2)
    sphere_points = _fibonacci_sphere(n_points)

    bsa = _buried_area_matrix(positions, radii, chain_labels, res_labels, n1, n2, probe_radius, sphere_points)
    isolated_1 = _residue_sasa(pos1, rad1, res1, n1, probe_radius, sphere_points)
    isolated_2 = _residue_sasa(pos2, rad2, res2, n2, probe_radius, sphere_points)

    with np.errstate(divide='ignore', invalid='ignore'):
        relative_1 = np.where(isolated_1[:, None] > 0, bsa / isolated_1[:, None], 0.0)
        relative_2 = np.where(isolated_2[None, :] > 0, bsa / isolated_2[None, :], 0.0)
    meaningful = (relative_1 > relative_cutoff) | (relative_2 > relative_cutoff)

    direct = _direct_contact_matrix(pos1, pos2, res1, res2, n1, n2, contact_distance)

    return meaningful & direct
