"""
src.calvados.trajectory
-----------------------
Trajectory I/O, slab-centring, density-profile accumulation, and post-processing.
"""

############################################################
# Imports
############################################################

import os
from typing import Optional

import MDAnalysis
import MDAnalysis as mda
import mdtraj as md
import numpy as np
import pandas as pd
import yaml
from MDAnalysis import transformations


############################################################
# Trajectory pooling
############################################################

def backup_traj(traj_pool: str, traj_new: str, ref_top: md.Trajectory) -> None:
    """Append a newly completed cycle DCD to the cumulative trajectory pool.

    If the pool does not yet exist the new file is simply copied; otherwise
    the two trajectories are concatenated and the pool is overwritten.

    Parameters
    ----------
    traj_pool : str
        Path to the cumulative DCD pool file.
    traj_new : str
        Path to the newly produced DCD file.
    ref_top : md.Trajectory
        Reference trajectory supplying topology, unit-cell angles, and
        unit-cell lengths for mdtraj loading.
    """
    def _load_with_unitcell(path: str) -> md.Trajectory:
        traj = md.load_dcd(path, ref_top.top)
        traj.unitcell_angles = ref_top.unitcell_angles.repeat(len(traj), axis=0)
        traj.unitcell_lengths = ref_top.unitcell_lengths.repeat(len(traj), axis=0)
        traj.unitcell_vectors = ref_top.unitcell_vectors.repeat(len(traj), axis=0)
        return traj

    if os.path.isfile(traj_pool):
        merged = md.join([_load_with_unitcell(traj_pool), _load_with_unitcell(traj_new)])
        merged.save_dcd(traj_pool)
    else:
        os.system(f"cp {traj_new} {traj_pool}")


############################################################
# Slab centring
############################################################

def _calc_zpatch(
    z: np.ndarray,
    h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the largest contiguous high-density window along z.

    Scans the histogram ``h`` to locate the widest run of bins above the
    minimum value (i.e. the condensed slab).

    Parameters
    ----------
    z : np.ndarray
        Bin centres in Å.
    h : np.ndarray
        Atom counts per bin.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(z_patch, h_patch)`` — centres and counts of the dominant slab.
    """
    cutoff = h.min()
    best_count = 0.0
    running_count = 0.0
    z_window: list[float] = []
    h_window: list[float] = []
    best_z: list[float] = []
    best_h: list[float] = []

    for ix, x in enumerate(h):
        if x > cutoff:
            running_count += x
            z_window.append(z[ix])
            h_window.append(x)
        else:
            if running_count > best_count:
                best_count = running_count
                best_z = z_window.copy()
                best_h = h_window.copy()
            running_count = 0.0
            z_window = []
            h_window = []

    # Handle slab ending at box boundary
    if running_count > best_count:
        best_z = z_window
        best_h = h_window

    return np.array(best_z), np.array(best_h)


def center_slab(
    path: str,
    record: str,
    cycle: int,
    replica: int,
    ref_atoms: str = "all",
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Align every frame so the condensed slab sits at the box centre.

    Translates the slab centre of density to z = Lz/2 and writes a
    molecule-whole (unwrapped) DCD.

    Parameters
    ----------
    path : str
        Dataset directory (contains ``{record}/`` sub-folder).
    record : str
        Simulation record identifier.
    cycle : int
        Simulation cycle index.
    replica : int
        Replica index.
    ref_atoms : str, optional
        MDAnalysis selection string for the density reference.
    start, end, step : int, optional
        Trajectory slice parameters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(histograms, z_centres)`` — per-frame density histograms and bin
        centres in nm.
    """
    top_pdb = f"{path}/{record}/{cycle}/top_{replica}_noCONECT.pdb"
    dcd = f"{path}/{record}/traj_pool.dcd"
    u = MDAnalysis.Universe(top_pdb, dcd, in_memory=True)

    ag_ref = u.select_atoms(ref_atoms)
    ag = u.select_atoms("all")

    # Bonds must be present for make-whole transformations
    bonds = [(i, i + 1) for seg in u.segments for i in seg.atoms.indices[:-1]]
    u.add_TopologyAttr("bonds", bonds)

    lz = u.dimensions[2]  # Å
    edges = np.arange(0, lz + 1, 1)
    dz = (edges[1] - edges[0]) / 2.0
    z = edges[:-1] + dz
    n_frames = len(u.trajectory[start:end:step])
    hs = np.zeros((n_frames, len(z)))

    with MDAnalysis.Writer(f"{path}/{record}/unwrapped.dcd", ag.n_atoms) as W:
        for t, ts in enumerate(u.trajectory[start:end:step]):
            zpos = ag_ref.positions[:, 2]
            h, _ = np.histogram(zpos, bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0, 0, -zmax + 0.5 * lz]))
            ts = transformations.wrap(ag)(ts)

            # Fine-tune: align slab patch centre exactly to Lz/2
            zpos = ag_ref.positions[:, 2]
            h, _ = np.histogram(zpos, bins=edges)
            zpatch, hpatch = _calc_zpatch(z, h)
            zmid = np.average(zpatch, weights=hpatch)
            ag.translate(np.array([0, 0, -zmid + 0.5 * lz]))
            ts = transformations.wrap(ag)(ts)

            zpos = ag_ref.positions[:, 2]
            h, _ = np.histogram(zpos, bins=edges)
            hs[t] = h

            ts = transformations.unwrap(ag)(ts)
            W.write(ag)

    return hs, z / 10.0  # Å → nm


############################################################
# Density-profile computation
############################################################

def calc_slab_profiles(
    path: str,
    record: str,
    cycle: int,
    replica: int,
    ref_atoms: str,
    components: dict,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
) -> None:
    """Compute and save per-component 1-D density profiles along z.

    Calls :func:`center_slab` first, then bins the centred trajectory for
    each component selection defined in ``components[name]['sel_atoms']``.

    Parameters
    ----------
    path : str
        Dataset directory.
    record : str
        Simulation record identifier.
    cycle : int
        Simulation cycle.
    replica : int
        Replica index.
    ref_atoms : str
        Selection for slab centring.
    components : dict
        Must include ``sel_atoms`` for each component key.
    start, end, step : int, optional
        Trajectory slice parameters.
    """
    h_ref, z = center_slab(path, record, cycle, replica, ref_atoms)
    np.save(f"{path}/{record}/z.npy", z)
    np.save(f"{path}/{record}/h_ref.npy", h_ref)

    top_pdb = f"{path}/{record}/{cycle}/top_{replica}_noCONECT.pdb"
    u = MDAnalysis.Universe(top_pdb, f"{path}/{record}/unwrapped.dcd", in_memory=True)
    bonds = [(i, i + 1) for seg in u.segments for i in seg.atoms.indices[:-1]]
    u.add_TopologyAttr("bonds", bonds)

    n_frames = len(u.trajectory[start:end:step])
    h_ref = h_ref[start:end:step]
    bin_width = 1  # Å
    volume = u.dimensions[0] * u.dimensions[1] * bin_width / 1e3  # nm³
    edges = np.arange(0, z.size + bin_width, bin_width)
    n_bins = edges.size - 1

    np.save(f"{path}/{record}/{record}_ref_profile.npy", h_ref / volume)
    h_ref_mean = h_ref.mean(axis=0) / volume
    all_profiles = np.c_[z, h_ref_mean]

    for name, comp in components.items():
        ag_sel = u.select_atoms(comp["sel_atoms"])
        h_sel = np.zeros((n_frames, n_bins))
        for t, ts in enumerate(u.trajectory[start:end:step]):
            ts = transformations.wrap(ag_sel)(ts)
            zpos = ag_sel.positions[:, 2]
            h, _ = np.histogram(zpos, bins=edges)
            h_sel[t] = h
        np.save(f"{path}/{record}/{record}_sel_profile_{name}.npy", h_sel / volume)
        h_sel_mean = h_sel.mean(axis=0) / volume
        all_profiles = np.c_[all_profiles, h_sel_mean]

    np.save(f"{path}/{record}/{record}_profiles.npy", all_profiles)


############################################################
# Post-processing utilities
############################################################

def postprocess(
    cwd: str,
    dataset: str,
    record: str,
    cycle: int,
    top: md.Topology,
    fdomains: str,
) -> None:
    """Centre the trajectory on a folded anchor and save aligned DCDs.

    Produces two outputs:
    - ``traj_centered.dcd`` — image-molecules centred on the first folded chain
    - ``traj_only4visualization.dcd`` — superposed onto the first frame (for
      VMD rendering; distances should be computed from ``traj_centered.dcd``).

    Parameters
    ----------
    cwd : str
        Root working directory.
    dataset : str
        Dataset name.
    record : str
        Record identifier.
    cycle : int
        Simulation cycle.
    top : md.Topology
        mdtraj topology object.
    fdomains : str
        Path to domains YAML for secondary-structure domain lookup.
    """
    from .geometry import get_ssdomains

    traj = md.load_dcd(f"{cwd}/{dataset}/{record}/traj_pool.dcd", top)
    config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml"))

    anchor_offset = 0
    anchor_chain_idx = None
    anchor_molecules: list[int] = []

    for idx, name in enumerate(config["components_order"]):
        if not config["components"][name]["isIDP"]:
            ssdomains = get_ssdomains(name, fdomains)
            anchor_molecules = [i + anchor_offset - 1 for i in sum(ssdomains, [])]
            anchor_chain_idx = idx
            break
        anchor_offset += len(config["components"][name]["seq"])

    anchor_set = set(traj.top.chain(anchor_chain_idx).atoms)
    traj = traj.image_molecules(inplace=False, anchor_molecules=[anchor_set], make_whole=True)
    traj[0].save_pdb(f"{cwd}/{dataset}/{record}/traj_centered.pdb")
    traj.save_dcd(f"{cwd}/{dataset}/{record}/traj_centered.dcd")

    traj = traj.superpose(traj[0], atom_indices=anchor_molecules)
    traj[0].save_pdb(f"{cwd}/{dataset}/{record}/traj_only4visualization.pdb")
    traj.save_dcd(f"{cwd}/{dataset}/{record}/traj_only4visualization.dcd")


def calc_com_trajectory(
    path: str,
    name: str,
    residues_file: str,
    mdtraj_top: md.Topology,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
    input_pdb: str = "top.pdb",
) -> md.Trajectory:
    """Compute per-chain centre-of-mass trajectories and radius of gyration.

    Saves a COM DCD and per-chain Rg arrays alongside the source trajectory.

    Parameters
    ----------
    path : str
        Directory containing ``traj_pool.dcd`` and the input PDB.
    name : str
        Identifier used for output filenames.
    residues_file : str
        Path to residue parameter CSV providing ``MW`` column.
    mdtraj_top : md.Topology
        mdtraj topology object.
    start, end, step : int, optional
        Trajectory slice parameters.
    input_pdb : str, optional
        Filename of the topology PDB inside ``path``.

    Returns
    -------
    md.Trajectory
        COM trajectory with one bead per chain.
    """
    _LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10)
    u = MDAnalysis.Universe(f"{path}/{input_pdb}", f"{path}/traj_pool.dcd", in_memory=True)
    ag = u.select_atoms("all")
    bonds = [(i, i + 1) for seg in u.segments for i in seg.atoms.indices[:-1]]
    u.add_TopologyAttr("bonds", bonds)

    with MDAnalysis.Writer(f"{path}/traj.dcd", ag.n_atoms) as W:
        for _, ts in enumerate(u.trajectory[start:end:step]):
            ts = transformations.unwrap(ag)(ts)
            W.write(ag)

    traj = md.load_dcd(f"{path}/traj.dcd", top=mdtraj_top)
    traj.xyz -= traj.unitcell_lengths[0] / 2.0

    residues = pd.read_csv(residues_file).set_index("one")
    chain_props: dict = {}

    for chain, label in zip(traj.top.chains, _LABELS):
        seq = [res.name for res in chain.residues]
        masses = residues.loc[seq, "MW"].values.copy()
        masses[0] += 2
        masses[-1] += 16
        chain_props[f"{chain.index}{label}"] = {"masses": masses, "rgs": []}

    cm_top = md.Topology()
    xyz = np.empty((traj.n_frames, traj.n_chains, 3))

    for chain, label in zip(traj.top.chains, _LABELS):
        mws = chain_props[f"{chain.index}{label}"]["masses"]
        new_chain = cm_top.add_chain()
        res = cm_top.add_residue("COM", new_chain, resSeq=chain.index)
        cm_top.add_atom(label, element=traj.top.atom(0).element, residue=res)

        t_chain = traj.atom_slice(traj.top.select(f"chainid {chain.index}"))
        com = np.sum(t_chain.xyz * mws[np.newaxis, :, np.newaxis], axis=1) / mws.sum()
        si = np.linalg.norm(t_chain.xyz - com[:, np.newaxis, :], axis=2)
        rg = np.sqrt(np.sum(si**2 * mws, axis=1) / mws.sum())
        chain_props[f"{chain.index}{label}"]["rgs"].append(rg.tolist())
        xyz[:, chain.index] = com

    cm_traj = md.Trajectory(xyz, cm_top, traj.time, traj.unitcell_lengths, traj.unitcell_angles)

    for key, props in chain_props.items():
        np.save(f"{path}/{name}_rg_prot_{key}.npy", np.array(props["rgs"]).flatten())

    cm_traj[0].save_pdb(f"{path}/com_top.pdb")
    cm_traj.save_dcd(f"{path}/com_traj.dcd")
    return cm_traj
