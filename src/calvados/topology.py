"""
src.calvados.topology
---------------------
Building coarse-grained mdtraj and OpenMM topologies and particle lists.
"""

############################################################
# Imports
############################################################

import numpy as np
import pandas as pd
import mdtraj as md
from openmm import app, unit


############################################################
# Topology construction
############################################################

def build_topology(
    components: dict,
    components_order: list[str],
) -> tuple[md.Topology, app.Topology]:
    """Create mdtraj and OpenMM topologies from component sequences.

    Each chain in each component becomes a separate chain object.  Consecutive
    residues are bonded.  All atoms are represented as carbon for mass/radius
    purposes; actual parameters are applied by the force setup.

    Parameters
    ----------
    components : dict
        Component metadata; each entry requires ``seq`` (list of one-letter
        codes) and ``chains`` (int).
    components_order : list[str]
        Ordered list of component names.

    Returns
    -------
    tuple[md.Topology, app.Topology]
        ``(mdtraj_topology, openmm_topology)``
    """
    _LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    top = md.Topology()
    openmm_top = app.Topology()
    chain_count = 0

    for name in components_order:
        for _ in range(components[name]["chains"]):
            label = _LABELS[chain_count % len(_LABELS)]
            chain_count += 1

            md_chain = top.add_chain(label)
            omm_chain = openmm_top.addChain(label)

            for resname in components[name]["seq"]:
                md_res = top.add_residue(resname, md_chain)
                omm_res = openmm_top.addResidue(resname, omm_chain)
                top.add_atom(resname, element=md.element.carbon, residue=md_res)
                openmm_top.addAtom(resname, app.element.Element.getBySymbol("C"), omm_res)

            # Bond consecutive atoms within the chain
            omm_atoms = list(omm_chain.atoms())
            for i in range(md_chain.n_atoms - 1):
                top.add_bond(md_chain.atom(i), md_chain.atom(i + 1))
                openmm_top.addBond(omm_atoms[i], omm_atoms[i + 1])

    return top, openmm_top


def build_box(Lx: float, Ly: float, Lz: float) -> tuple:
    """Construct orthogonal periodic box vectors.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions in nm.

    Returns
    -------
    tuple
        Three OpenMM ``unit.Quantity`` vectors ``(a, b, c)``.
    """
    a = unit.Quantity(np.zeros(3), unit.nanometers)
    b = unit.Quantity(np.zeros(3), unit.nanometers)
    c = unit.Quantity(np.zeros(3), unit.nanometers)
    a[0] = Lx * unit.nanometers
    b[1] = Ly * unit.nanometers
    c[2] = Lz * unit.nanometers
    return a, b, c


############################################################
# Particle mass assignment
############################################################

def add_particles(
    system,
    residues: pd.DataFrame,
    components: dict,
    components_order: list[str],
):
    """Add particles to an OpenMM System with correct masses.

    Terminal residues carry modified masses: N-terminus +2 Da (NH₂ cap),
    C-terminus +16 Da (OH cap).

    Parameters
    ----------
    system : openmm.System
        The system to add particles to.
    residues : pd.DataFrame
        Residue parameters indexed by one-letter code, containing ``MW``.
    components : dict
        Component metadata with ``seq`` and ``chains``.
    components_order : list[str]
        Ordered component names.

    Returns
    -------
    openmm.System
        The same system with particles added.
    """
    for name in components_order:
        fasta = list(components[name]["seq"])
        for _ in range(components[name]["chains"]):
            system.addParticle((residues.loc[fasta[0], "MW"] + 2) * unit.amu)
            for aa in fasta[1:-1]:
                system.addParticle(residues.loc[aa, "MW"] * unit.amu)
            system.addParticle((residues.loc[fasta[-1], "MW"] + 16) * unit.amu)
    return system
