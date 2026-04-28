"""
src.calvados.forces
-------------------
OpenMM force objects for CALVADOS coarse-grained simulations.

Includes Ashbaugh-Hatch Lennard-Jones, Debye-Hückel Yukawa, harmonic backbone
bonds, elastic network for folded domains, Go-type native-contact potentials,
and equilibration centre restraints.
"""

############################################################
# Imports
############################################################

import numpy as np
import pandas as pd
from openmm import openmm, unit

from .geometry import euclidean, get_ssdomains


############################################################
# Primary interaction forces
############################################################

def set_interactions(
    system,
    residues: pd.DataFrame,
    components: dict,
    components_order: list[str],
    lj_eps: float,
    cutoff: float,
    yukawa_df: pd.DataFrame,
    exclusion_count: int,
    coarse_grained: str = "CA",
    fdomains: str | None = None,
) -> tuple:
    """Build harmonic-bond, Ashbaugh-Hatch LJ, and Yukawa forces.

    The Ashbaugh-Hatch potential smoothly interpolates between a WCA repulsion
    (λ=0) and full LJ (λ=1), controlled per residue by the CALVADOS lambda
    parameter.

    Parameters
    ----------
    system : openmm.System
        The system being constructed (unused here but kept for API consistency).
    residues : pd.DataFrame
        Per-residue parameters (``sigmas``, ``lambdas``).
    components : dict
        Component metadata including ``seq``, ``chains``, and (for folded
        domains) ``dmap``.
    components_order : list[str]
        Ordered component names.
    lj_eps : float
        LJ well depth in kJ/mol.
    cutoff : float
        LJ cutoff distance in nm.
    yukawa_df : pd.DataFrame
        Pre-computed ``yukawa_eps`` and ``yukawa_kappa`` per component.
    exclusion_count : int
        Running total of added exclusions (updated in-place and returned).
    coarse_grained : str, optional
        ``"CA"`` or ``"COM"`` — determines backbone bond length.
    fdomains : str, optional
        Path to domains YAML, required for COM bonding of folded residues.

    Returns
    -------
    tuple
        ``(hb, yu, ah, exclusion_count)`` — the three force objects and the
        updated exclusion counter.
    """
    hb = openmm.HarmonicBondForce()

    # Ashbaugh-Hatch: WCA below r* = 2^(1/6) σ, attractive above
    ah_expr = (
        "select(step(r-2^(1/6)*s),"
        "4*eps*l*((s/r)^12-(s/r)^6-shift),"
        "4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))"
        "; s=0.5*(s1+s2); l=0.5*(l1+l2)"
        "; shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6"
    )
    ah = openmm.CustomNonbondedForce(ah_expr)
    ah.addGlobalParameter("eps", lj_eps * unit.kilojoules_per_mole)
    ah.addGlobalParameter("rc", float(cutoff) * unit.nanometer)
    ah.addPerParticleParameter("s")
    ah.addPerParticleParameter("l")

    yukawa_kappa = yukawa_df.loc[list(yukawa_df.index)[0], "yukawa_kappa"]
    yu = openmm.CustomNonbondedForce("q*(exp(-kappa*r)/r-shift); q=q1*q2")
    yu.addGlobalParameter("kappa", yukawa_kappa / unit.nanometer)
    yu.addGlobalParameter("shift", np.exp(-yukawa_kappa * 4.0) / 4.0 / unit.nanometer)
    yu.addPerParticleParameter("q")

    total_particles = 0
    for name in components_order:
        chains = components[name]["chains"]
        N = len(components[name]["seq"])
        ssdomains = None
        ss_residues: list[int] = []

        if coarse_grained != "CA" and not components[name]["isIDP"]:
            assert fdomains is not None
            ssdomains = get_ssdomains(name, fdomains, verbose=False)
            ss_residues = sum(ssdomains, [])

        for j in range(chains):
            begin = j * N + total_particles
            end = begin + N
            for a, eps_yu in zip(components[name]["seq"], yukawa_df.loc[name, "yukawa_eps"]):
                yu.addParticle([eps_yu * unit.nanometer * unit.kilojoules_per_mole])
                ah.addParticle([
                    residues.loc[a, "sigmas"] * unit.nanometer,
                    residues.loc[a, "lambdas"] * unit.dimensionless,
                ])

            for i in range(begin, end - 1):
                res_i = i - begin
                if coarse_grained == "CA" or components[name]["isIDP"]:
                    hb.addBond(i, i + 1, 0.38 * unit.nanometer,
                               8033.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                else:
                    # Folded residues use native equilibrium distances from dmap
                    if (res_i + 1) in ss_residues or (res_i + 2) in ss_residues:
                        d0 = components[name]["dmap"][res_i, res_i + 1]
                        hb.addBond(i, i + 1, d0 * unit.nanometer,
                                   8033.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                    else:
                        hb.addBond(i, i + 1, 0.38 * unit.nanometer,
                                   8033.0 * unit.kilojoules_per_mole / unit.nanometer**2)
                yu.addExclusion(i, i + 1)
                ah.addExclusion(i, i + 1)
                exclusion_count += 2

        total_particles += chains * N

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4.0 * unit.nanometer)
    ah.setCutoffDistance(cutoff * unit.nanometer)

    return hb, yu, ah, exclusion_count


############################################################
# Elastic network (folded-domain restraints)
############################################################

def set_harmonic_network(
    components: dict,
    components_order: list[str],
    yu,
    ah,
    go_basic,
    fdomains: str,
    exclusion_count: int,
    cs_cutoff: float = 0.9,
    k_restraint: float = 700.0,
) -> tuple:
    """Add harmonic bonds between nearby residues in structured domains (with Go).

    Only residue pairs that are both inside a structured domain and separated
    by less than ``cs_cutoff`` nm receive elastic-network bonds.  The same
    pairs are excluded from non-bonded forces to avoid double-counting.

    Parameters
    ----------
    components : dict
        Must contain ``dmap`` (distance matrix, nm) for folded components.
    components_order : list[str]
        Ordered component names.
    yu, ah, go_basic :
        Force objects that receive matching exclusions.
    fdomains : str
        Path to domains YAML.
    exclusion_count : int
        Running exclusion counter.
    cs_cutoff : float, optional
        Maximum native distance to include in the network (nm).
    k_restraint : float, optional
        Spring constant in kJ/mol/nm².

    Returns
    -------
    tuple
        ``(cs, yu, ah, go_basic, exclusion_count)``
    """
    cs = openmm.HarmonicBondForce()
    total_particles = 0

    for name in components_order:
        chains = components[name]["chains"]
        N = len(components[name]["seq"])

        if not components[name]["isIDP"]:
            ssdomains = get_ssdomains(name, fdomains)
            for n_chain in range(chains):
                begin = n_chain * N + total_particles
                end = begin + N
                for i in range(begin, end - 2):
                    for j in range(i + 2, end):
                        ri, rj = i - begin, j - begin
                        in_ss = any(
                            (ri + 1) in dom and (rj + 1) in dom for dom in ssdomains
                        )
                        if in_ss and components[name]["dmap"][ri, rj] < cs_cutoff:
                            d0 = components[name]["dmap"][ri, rj]
                            cs.addBond(i, j, d0 * unit.nanometer,
                                       k_restraint * unit.kilojoules_per_mole / unit.nanometer**2)
                            yu.addExclusion(i, j)
                            ah.addExclusion(i, j)
                            go_basic.addExclusion(i, j)
                            exclusion_count += 3

        total_particles += chains * N

    cs.setUsesPeriodicBoundaryConditions(True)
    return cs, yu, ah, go_basic, exclusion_count


def set_harmonic_network_no_go(
    components: dict,
    components_order: list[str],
    yu,
    ah,
    fdomains: str,
    exclusion_count: int,
    cs_cutoff: float = 0.9,
    k_restraint: float = 700.0,
) -> tuple:
    """Add harmonic elastic-network bonds without a Go potential.

    Identical logic to :func:`set_harmonic_network` but without exclusions
    on a Go force object.

    Parameters
    ----------
    components : dict
        Must contain ``dmap`` for folded components.
    components_order : list[str]
        Ordered component names.
    yu, ah :
        Yukawa and Ashbaugh-Hatch force objects.
    fdomains : str
        Path to domains YAML.
    exclusion_count : int
        Running exclusion counter.
    cs_cutoff : float, optional
        Maximum native distance (nm).
    k_restraint : float, optional
        Spring constant in kJ/mol/nm².

    Returns
    -------
    tuple
        ``(cs, yu, ah, exclusion_count)``
    """
    cs = openmm.HarmonicBondForce()
    total_particles = 0

    for name in components_order:
        chains = components[name]["chains"]
        N = len(components[name]["seq"])

        if not components[name]["isIDP"]:
            ssdomains = get_ssdomains(name, fdomains)
            for n_chain in range(chains):
                begin = n_chain * N + total_particles
                end = begin + N
                for i in range(begin, end - 2):
                    for j in range(i + 2, end):
                        ri, rj = i - begin, j - begin
                        in_ss = any(
                            (ri + 1) in dom and (rj + 1) in dom for dom in ssdomains
                        )
                        if in_ss and components[name]["dmap"][ri, rj] < cs_cutoff:
                            d0 = components[name]["dmap"][ri, rj]
                            cs.addBond(i, j, d0 * unit.nanometer,
                                       k_restraint * unit.kilojoules_per_mole / unit.nanometer**2)
                            yu.addExclusion(i, j)
                            ah.addExclusion(i, j)
                            exclusion_count += 2

        total_particles += chains * N

    cs.setUsesPeriodicBoundaryConditions(True)
    return cs, yu, ah, exclusion_count


############################################################
# Go-type native-contact potentials
############################################################

def go_basic_bond_force(
    gorestraint: float,
    go_cutoff: float,
    pairs: np.ndarray,
    pair_distances: np.ndarray,
    ah,
    yu,
    add_exclusions: bool,
    exclusion_count: int,
) -> tuple:
    """Tabulated Go 12-10 potential implemented as a CustomBondForce.

    Uses a 12-10 Lennard-Jones form with a hard cutoff at ``go_cutoff`` nm.

    Parameters
    ----------
    gorestraint : float
        Go well depth in kJ/mol.
    go_cutoff : float
        Cutoff distance in nm; contacts beyond this are set to zero.
    pairs : np.ndarray
        Shape ``(M, 2)`` array of atom index pairs.
    pair_distances : np.ndarray
        Shape ``(M,)`` equilibrium distances in nm.
    ah, yu :
        Force objects that receive matching exclusions.
    add_exclusions : bool
        Whether to add exclusions to ``ah`` and ``yu``.
    exclusion_count : int
        Running exclusion counter.

    Returns
    -------
    tuple
        ``(go_force, ah, yu, exclusion_count)``
    """
    go = openmm.CustomBondForce(
        f"select(step(r-{go_cutoff}), 0, {gorestraint}*(5*(s/r)^12-6*(s/r)^10))"
    )
    go.addPerBondParameter("s")
    go.setUsesPeriodicBoundaryConditions(True)
    go.setName("Go_basic")

    assert len(pairs) == len(pair_distances)
    for (i, j), d0 in zip(pairs, pair_distances):
        go.addBond(int(i), int(j), [d0 * unit.nanometer])
        if add_exclusions:
            ah.addExclusion(int(i), int(j))
            yu.addExclusion(int(i), int(j))
            exclusion_count += 2

    return go, ah, yu, exclusion_count


def go_basic_tabulated(
    gorestraint: float,
    go_cutoff: float,
    one_chain_location: np.ndarray,
    one_chain_dis_location: np.ndarray,
    components: dict,
    components_order: list[str],
):
    """Tabulated Go potential using a 2D sigma lookup table.

    Allows different equilibrium distances for each residue-type pair
    by encoding them in an OpenMM ``Discrete2DFunction``.

    Parameters
    ----------
    gorestraint : float
        Go well depth in kJ/mol.
    go_cutoff : float
        Cutoff radius in nm.
    one_chain_location : np.ndarray
        Shape ``(M, 2)`` — residue index pairs for a single chain.
    one_chain_dis_location : np.ndarray
        Shape ``(M,)`` — equilibrium distances for those pairs (nm).
    components : dict
        Component metadata with ``seq`` and ``chains``.
    components_order : list[str]
        Ordered component names (exactly two components required).

    Returns
    -------
    openmm.CustomNonbondedForce
        The Go force with tabulated sigma function.
    """
    assert len(components_order) == 2, "Tabulated Go requires exactly two components"
    seq_a = len(components[components_order[0]]["seq"])
    seq_b = len(components[components_order[1]]["seq"])
    table_len = seq_a + seq_b

    sigma_table = np.zeros((table_len, table_len))
    for (ri, rj), d0 in zip(one_chain_location, one_chain_dis_location):
        sigma_table[ri, rj] = d0
        sigma_table[rj, ri] = d0

    go = openmm.CustomNonbondedForce(
        f"select(step(r-{go_cutoff}), 0, "
        f"{gorestraint}*(5*(sigma(type1,type2)/r)^12 - 6*(sigma(type1,type2)/r)^10))"
    )
    go.addPerParticleParameter("type")

    total_particles = 0
    for seq_idx, name in enumerate(components_order):
        seq_len = len(components[name]["seq"])
        offset = 0 if seq_idx == 0 else seq_a
        for j in range(components[name]["chains"]):
            begin = j * seq_len + total_particles
            end = begin + seq_len
            for res_idx in range(begin, end):
                go.addParticle([res_idx - begin + offset])
                if res_idx != end - 1:
                    go.addExclusion(res_idx, res_idx + 1)
        total_particles += components[name]["chains"] * seq_len

    go.addTabulatedFunction(
        "sigma",
        openmm.Discrete2DFunction(table_len, table_len, sigma_table.flatten(order="F")),
    )
    go.setForceGroup(2)
    go.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    go.setCutoffDistance(go_cutoff * unit.nanometer)
    go.setName("Go_basic")
    return go


############################################################
# Equilibration restraint
############################################################

def force_to_center(box: list[float], k: float):
    """Create a harmonic restraint pulling each particle toward the box centre.

    Used during equilibration to prevent chains from drifting to box edges
    before the slab has formed.

    Parameters
    ----------
    box : list[float]
        Box dimensions ``[Lx, Ly, Lz]`` in nm.
    k : float
        Spring constant in kJ/mol/nm.

    Returns
    -------
    openmm.CustomExternalForce
        The restraint force (particles not yet added).
    """
    rcent = openmm.CustomExternalForce("k*abs(periodicdistance(x,y,z,x,y,z0))")
    rcent.addGlobalParameter("k", k * unit.kilojoules_per_mole / unit.nanometer)
    rcent.addGlobalParameter("z0", box[2] / 2.0 * unit.nanometer)
    rcent.setName("rcent")
    return rcent
