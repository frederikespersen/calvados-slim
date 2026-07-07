# calvados-slim

Coarse-grained molecular-dynamics modelling of **SLiM** (Short Linear Motif)
interactions on top of a modified [CALVADOS](https://github.com/KULL-Centre/CALVADOS)
engine.

---

## Summary

Many protein–protein interactions are mediated by short linear motifs in
intrinsically disordered regions binding to folded domains. `calvados-slim`
provides a pipeline to model these interactions with the CALVADOS coarse-grained
force field, augmented with **specific inter-molecular ("Go") contacts** that
encode a known SLiM interface.

The pipeline has three stages:

1. **Curation** — turn a structural specification of a system (chains + SLiMs,
   backed by PDB and/or AlphaFold structures) into ready-to-run CALVADOS input.
   SLiM interfaces are derived from the *buried solvent-accessible surface area*
   between the chains, optionally combined with AlphaFold confidence metrics
   (PAE and the P<sub>inc</sub> interface score), then assigned equilibrium
   distances from the structure.
2. **Simulation** — run the curated systems with the `slim` branch of a CALVADOS
   fork, scanning the Go interaction strength `k`.
3. **Analysis** — estimate binding affinities (K<sub>d</sub>, ΔG) from the
   trajectories, with blocking-based error bars.

The four numbered notebooks in [`notebooks/`](notebooks/) walk through the
workflow end-to-end.

---

## Installation

### 1. Clone `calvados-slim`

```bash
git clone https://github.com/frederikespersen/calvados-slim.git
cd calvados-slim
```

### 2. Clone the CALVADOS fork into `src/calvados`

The pipeline depends on the `slim` branch of a CALVADOS fork, which adds the
specific inter-molecular interactions used to model SLiMs. It must live at
`src/calvados` (the code imports it as `src.calvados.calvados`):

```bash
git clone -b slim https://github.com/frederikespersen/CALVADOS.git src/calvados
```

### 3. Create the conda environment

```bash
conda env create -f env.yaml
conda activate calvados
```

This provides Python 3.10 together with OpenMM, MDAnalysis, MDTraj, BioPython,
PDBFixer, gemmi, and the rest of the stack.

### 4. Install the CALVADOS fork as a module in the environment

```bash
pip install -e src/CALVADOS
```

### 5. Install SMOG2 (for curation)

Deriving SLiM contacts requires [SMOG2](https://smog-server.org/smog2/). Install
it separately and point the `SMOGBIN` environment variable at its `bin`
directory:

```bash
export SMOGBIN="$HOME/software/smog-2.5/bin"
```

> SMOG2 is only needed for the **curation** stage. Running and analysing existing
> simulations does not require it.

### 6. Run from the repository root

The package is imported as `src.…`, and the CALVADOS fork as
`src.calvados.calvados.…`, so commands and notebooks should be run from the
repository root (with it on `PYTHONPATH`).

---

## Example of use

A minimal end-to-end run for the bundled systems. See the
[`notebooks/`](notebooks/) for the full, annotated version.

### Curate inputs

```python
import json, os
from src import curation

os.environ["SMOGBIN"] = os.path.expanduser("~/software/smog-2.5/bin")

with open("data/curation/systems.json") as f:
    systems = json.load(f)

# Writes per-system CALVADOS inputs into data/curation/systems/<ID>/
for system in systems:
    curation.System.curate(system, output_dir="data/curation/systems")
```

### Generate a Go-strength scan of simulations

```python
import numpy as np
from src import curation

with open("data/configurations/conditions.json") as f:
    conditions = json.load(f)["Default"]
with open("data/configurations/sampling.json") as f:
    sampling = json.load(f)["Scan"]
with open("data/configurations/go_scans.json") as f:
    scan = json.load(f)["Thorough limited"]

ks = np.arange(scan["Lower"], scan["Upper"] + scan["Step"], scan["Step"])

for system in systems:
    system_dir = os.path.join("data/curation/systems", system["ID"])
    for k in ks:
        sim_dir = os.path.join("sim/scan", system["ID"], f"k={k:.2f}")
        os.makedirs(sim_dir, exist_ok=True)
        curation.Simulation.build(
            system=system,
            system_dir=system_dir,
            conditions=conditions,
            sampling=sampling,
            slim_go_k=float(k),
            sim_dir=sim_dir,
        )
```

Each `sim_dir` now contains a CALVADOS config, components file, the per-run
`specific_interactions.txt`, and Slurm submission scripts. Submit the generated
scripts (e.g. `sbatch`) to run the simulations.

### Estimate binding affinity from a trajectory

```python
from src.analysis.slim import TwoChain

Kd, Kd_lower, Kd_upper = TwoChain.get_kd_sem(
    top_pdb="sim/scan/sh3-prm/k=3.00/top.pdb",
    traj_dcd="sim/scan/sh3-prm/k=3.00/sh3-prm.dcd",
    specific_interactions_txt="sim/scan/sh3-prm/k=3.00/specific_interactions.txt",
)
print(f"Kd = {Kd:.2e} M  (−1 SEM {Kd_lower:.2e}, +1 SEM {Kd_upper:.2e})")
```

---

## Citation

This work builds on CALVADOS. If you use it, please cite the upstream CALVADOS
papers ([KULL-Centre/CALVADOS](https://github.com/KULL-Centre/CALVADOS)) alongside
the SMOG2 and AlphaFold/P<sub>inc</sub> methods referenced in
`src/analysis/alphafold.py`.
