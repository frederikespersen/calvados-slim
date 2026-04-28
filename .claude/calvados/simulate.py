"""
src.calvados.simulate
---------------------
Main OpenMM simulation driver for CALVADOS coarse-grained MD.

Handles topology construction, force assembly, equilibration and production
runs, trajectory wrapping, and optional slab density-profile analysis.
"""

############################################################
# Imports
############################################################

import copy
import os
import time
from typing import Any

import MDAnalysis
import mdtraj as md
import numpy as np
import pandas as pd
import yaml
from MDAnalysis import transformations
from openmm import XmlSerializer, app, openmm, unit

from .forces import (
    force_to_center,
    go_basic_tabulated,
    set_harmonic_network,
    set_harmonic_network_no_go,
    set_interactions,
)
from .geometry import euclidean, geometry_from_pdb, get_ssdomains, place_chains, xy_spiral_array
from .parameters import gen_params_dh, load_parameters
from .topology import add_particles, build_box, build_topology
from .trajectory import backup_traj, calc_slab_profiles


############################################################
# Simulation driver
############################################################

def simulate(config: dict[str, Any]) -> None:
    """Run a single CALVADOS equilibration or production simulation cycle.

    Reads all settings from ``config``, constructs the OpenMM system, runs
    the integrator, writes trajectory files, and (for production slab runs)
    computes density profiles.

    Parameters
    ----------
    config : dict
        Flat dictionary of simulation parameters.  Required keys:

        - ``record`` — run identifier string
        - ``dataset`` — dataset subdirectory name
        - ``temp`` — temperature (K)
        - ``ionic`` — ionic strength (M)
        - ``pH`` — solution pH
        - ``cutoff`` — LJ cutoff (nm)
        - ``steps`` — number of integration steps
        - ``wfreq`` — reporter write frequency (steps)
        - ``L`` — box xy side (nm)
        - ``Lz`` — box z length (nm)
        - ``k_eq`` — equilibration restraint constant (kJ/mol/nm)
        - ``lj_eps`` — LJ well depth (kJ/mol)
        - ``CoarseGrained`` — ``"CA"`` or ``"COM"``
        - ``Go_potential_bool`` — bool
        - ``Go_model_type`` — e.g. ``"Go_basic"``
        - ``Go_cutoff`` — Go cutoff (nm)
        - ``Gorestraint`` — Go well depth (kJ/mol)
        - ``gpu`` — use CUDA if True, CPU otherwise
        - ``Threads`` — CPU thread count
        - ``cycle`` — cycle index
        - ``replica`` — replica index
        - ``equilibrium`` — bool
        - ``notslab`` — bool (disable slab post-processing)
        - ``do_postprocess`` — bool
        - ``components`` — per-component metadata dict
        - ``components_order`` — ordered list of component names
        - ``fdomains`` — path to domains YAML
        - ``path2pdbdir`` — directory containing initial PDB files
        - ``initial_type`` — CALVADOS variant
        - ``k_restraint`` — elastic-network spring constant (kJ/mol/nm²)
        - ``cwd`` — root working directory
    """
    # ── Unpack config ─────────────────────────────────────
    record         = config["record"]
    dataset        = config["dataset"]
    temp           = config["temp"]
    ionic          = config["ionic"]
    pH             = config["pH"]
    cutoff         = config["cutoff"]
    steps          = config["steps"]
    wfreq          = config["wfreq"]
    L              = config["L"]
    Lz             = config["Lz"]
    k_eq           = config["k_eq"]
    lj_eps         = config["lj_eps"]
    CoarseGrained  = config["CoarseGrained"]
    Go_bool        = config["Go_potential_bool"]
    Go_type        = config["Go_model_type"]
    Go_cutoff      = config["Go_cutoff"]
    Gorestraint    = config["Gorestraint"]
    gpu            = config["gpu"]
    Threads        = config["Threads"]
    cycle          = config["cycle"]
    replica        = config["replica"]
    equilibrium    = config["equilibrium"]
    notslab        = config["notslab"]
    do_postprocess = config["do_postprocess"]
    components     = config["components"]
    comp_order     = config["components_order"]
    fdomains       = config["fdomains"]
    path2pdbdir    = config["path2pdbdir"]
    initial_type   = config["initial_type"]
    cwd            = config["cwd"]
    run_dir        = f"{cwd}/{dataset}/{record}/{cycle}"

    os.system(f"uname -n > {run_dir}/{replica}_uname.txt")

    # ── Residue parameters ────────────────────────────────
    residues = load_parameters(cwd, dataset, initial_type)

    # ── Yukawa parameters ─────────────────────────────────
    yukawa_df = pd.DataFrame(columns=["fasta", "yukawa_eps", "yukawa_kappa"], dtype=object)
    for name, comp in components.items():
        eps, kappa = gen_params_dh(residues, list(comp["seq"]), temp, pH, ionic)
        yukawa_df.loc[name] = {"fasta": list(comp["seq"]), "yukawa_eps": eps, "yukawa_kappa": kappa}
    yukawa_df.to_pickle(f"{run_dir}/yukawa_df.pkl")

    # ── Initial positions ─────────────────────────────────
    comps_sc = copy.deepcopy(components)
    one_chain_loc  = np.load(f"{run_dir}/OneChain_location.npy")
    one_chain_dis  = np.load(f"{run_dir}/OneChainDis_location.npy")

    for name, comp in comps_sc.items():
        if comp["isIDP"]:
            comp["ini_pos"] = xy_spiral_array(len(comp["seq"]))
        else:
            pdb_path = (
                f"{path2pdbdir}/{name}_{CoarseGrained}_ini.pdb"
                if comp["compact_ini"] else
                f"{path2pdbdir}/{name}_rank0_relax.pdb"
            )
            ini_pos, pos, com = geometry_from_pdb(
                pdb2predic=f"{cwd}/extract_relax/{name}_rank0_relax.pdb",
                pdb_ini=pdb_path,
                compact_ini=comp["compact_ini"],
                coarse_grained=CoarseGrained,
                ssdomains=get_ssdomains(name, f"{cwd}/domains.yaml"),
            )
            comp["ini_pos"] = ini_pos
            comp["pos"]     = pos
        components[name]["ini_pos"] = comp["ini_pos"]
        if not comp["isIDP"]:
            components[name]["pos"] = comp["pos"]

    ini_pos, com_total = place_chains(L, comps_sc, comp_order)

    # ── Topology ──────────────────────────────────────────
    top, omm_top = build_topology(components, comp_order)
    print(f"Lz: {Lz}")

    ref_traj = md.Trajectory(
        ini_pos + np.array([L / 2, L / 2, Lz / 2]), top, 0, [L, L, Lz], [90, 90, 90]
    )
    ini_traj = md.Trajectory(
        ini_pos + com_total, top, 0, [L, L, Lz], [90, 90, 90]
    )
    ini_traj.save_pdb(f"{run_dir}/ini_beads.pdb", force_overwrite=True)
    ref_traj.save_pdb(f"{run_dir}/top_{replica}.pdb", force_overwrite=True)
    ref_traj.save_gro(f"{run_dir}/top_{replica}.gro", force_overwrite=True)

    # ── System ────────────────────────────────────────────
    print("Building system...")
    system = openmm.System()
    a, b, c = build_box(L, L, Lz)
    system.setDefaultPeriodicBoxVectors(a, b, c)
    omm_top.setPeriodicBoxVectors([a, b, c])

    pdb = app.PDBFile(f"{run_dir}/top_{replica}.pdb")
    system = add_particles(system, residues, components, comp_order)

    # Distance maps for folded components
    for name, comp in components.items():
        if comp["isIDP"]:
            comp["dmap"] = None
        else:
            comp["dmap"] = euclidean(comp["pos"], comp["pos"])
            np.save(f"{run_dir}/dmap_{name}.npy", comp["dmap"])

    # ── Forces ────────────────────────────────────────────
    exclusion_count = 0
    print("Setting interactions...")

    hb, yu, ah, exclusion_count = set_interactions(
        system, residues, components, comp_order,
        lj_eps, cutoff, yukawa_df, exclusion_count,
        coarse_grained=CoarseGrained, fdomains=fdomains,
    )

    has_folded = any(not comp["isIDP"] for comp in components.values())

    if Go_type == "Go_basic":
        go = go_basic_tabulated(
            Gorestraint, Go_cutoff, one_chain_loc, one_chain_dis, components, comp_order
        )
        if has_folded:
            cs, yu, ah, go, exclusion_count = set_harmonic_network(
                components, comp_order, yu, ah, go, fdomains, exclusion_count
            )
            system.addForce(cs)
        system.addForce(go)
    elif has_folded:
        cs, yu, ah, exclusion_count = set_harmonic_network_no_go(
            components, comp_order, yu, ah, fdomains, exclusion_count
        )
        system.addForce(cs)

    system.addForce(hb)
    system.addForce(ah)
    system.addForce(yu)
    print(f"Total exclusions: {exclusion_count}")

    if equilibrium:
        rcent = force_to_center([L, L, Lz], k_eq)
        for i in range(len(ini_pos)):
            rcent.addParticle(i)
        system.addForce(rcent)

    # ── Integrator & simulation ───────────────────────────
    integrator = openmm.LangevinIntegrator(
        temp * unit.kelvin, 0.01 / unit.picosecond, 0.01 * unit.picosecond
    )
    xml_tag = "equilibrium" if equilibrium else "production"
    with open(f"{run_dir}/{xml_tag}_{cycle}_system.xml", "w") as fh:
        fh.write(XmlSerializer.serialize(system))

    if gpu:
        platform = openmm.Platform.getPlatformByName("CUDA")
        props = {"CudaPrecision": "mixed"}
    else:
        platform = openmm.Platform.getPlatformByName("CPU")
        props = {"Threads": str(Threads)}

    simulation = app.Simulation(omm_top, system, integrator, platform, props)

    # ── Initial positions ─────────────────────────────────
    if not equilibrium:
        if notslab:
            simulation.context.setPositions(pdb.positions)
        elif cycle == 0:
            simulation.context.setPositions(
                app.PDBFile(f"{run_dir}/equilibrium_lf.pdb").positions
            )
        else:
            prev_run = f"{cwd}/{dataset}/{record}/{cycle - 1}/production_lf.pdb"
            simulation.context.setPositions(app.PDBFile(prev_run).positions)
    else:
        simulation.context.setPositions(pdb.positions)

    # ── Reporters ─────────────────────────────────────────
    dcd_path = f"{run_dir}/{xml_tag}_{cycle}.dcd"
    log_path = f"{run_dir}/statedata_{cycle}{'_equi' if equilibrium else ''}.log"

    simulation.reporters.append(
        app.DCDReporter(dcd_path, wfreq, enforcePeriodicBox=True, append=False)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            log_path, int(wfreq),
            step=True, speed=True, elapsedTime=True,
            progress=True, remainingTime=True,
            totalSteps=steps, separator="\t",
        )
    )

    # ── Run ───────────────────────────────────────────────
    print(f"Running {steps} steps...")
    t0 = time.time()
    simulation.step(steps)
    elapsed = time.time() - t0
    print(
        f"{record} finished: "
        f"{int(elapsed // 3600)}h {int((elapsed // 60) % 60)}min "
        f"{elapsed % 60:.2f}s"
    )

    # ── Post-run: wrap and save last frame ─────────────────
    u = MDAnalysis.Universe(dcd_path)
    pbc_path = f"{run_dir}/{xml_tag}_{cycle}_pbc.dcd"
    with MDAnalysis.Writer(pbc_path, u.atoms.n_atoms) as W:
        while True:
            transformations.wrap(u.atoms)(u.trajectory.ts)
            W.write(u.atoms)
            try:
                u.trajectory.next()
            except Exception:
                break

    traj = md.load_dcd(pbc_path, top)
    traj[-1].save_pdb(f"{run_dir}/{xml_tag}_lf.pdb")
    os.remove(dcd_path)

    if not equilibrium:
        pool_path = f"{cwd}/{dataset}/{record}/traj_pool.dcd"
        backup_traj(pool_path, pbc_path, ref_traj)
        os.system(
            f"grep -v CONECT {run_dir}/top_{replica}.pdb "
            f"> {run_dir}/top_{replica}_noCONECT.pdb"
        )

        if not notslab:
            atom_start = 0
            for name in comp_order:
                n = len(components[name]["seq"]) * components[name]["chains"]
                components[name]["sel_atoms"] = f"index {atom_start}:{atom_start + n - 1}"
                atom_start += n
            calc_slab_profiles(
                f"{cwd}/{dataset}", record, cycle, replica, "all", components
            )
            from .analysis_bridge import calc_profile
            calc_profile(cwd, dataset, record, plot=True)

    if do_postprocess:
        from .trajectory import postprocess, calc_com_trajectory
        postprocess(cwd, dataset, record, cycle, top, f"{cwd}/domains.yaml")
        calc_com_trajectory(
            path=f"{cwd}/{dataset}/{record}",
            name=record,
            residues_file=f"{cwd}/{dataset}/residues_pub.csv",
            mdtraj_top=top,
            input_pdb=f"{cycle}/top_{replica}_noCONECT.pdb",
        )


############################################################
# CLI entry point
############################################################

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", default="config_sim.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as fh:
        simulate(yaml.safe_load(fh))
