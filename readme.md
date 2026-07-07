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

### 5. Run from the repository root

The package is imported as `src.…`, and the CALVADOS fork as
`src.calvados.calvados.…`, so commands and notebooks should be run from the
repository root (with it on `PYTHONPATH`).

### 6. (Optional) Install SMOG2

Curation derives SLiM contacts from buried surface area
([`src.analysis.sasa`](src/analysis/sasa.py)) and needs no external tools.
SMOG2 is only required if you want to cross-check those contacts against a
[SMOG2](https://smog-server.org/smog2/) *Shadow Contact Map*
([`src.analysis.smog`](src/analysis/smog.py)); if so, install it separately and
pass its `bin` directory to `shadow_contact_map` via the `smog_bin` argument:

```python
import os
from src.analysis import smog

contacts = smog.shadow_contact_map(
    structure_file="data/pdb/2MPS.cif",
    chain1_id="A",
    chain2_id="B",
    smog_bin=os.path.expanduser("~/software/smog-2.5/bin"),
)
```


---

## Example of use

A minimal end-to-end run for the bundled systems. See the
[`notebooks/`](notebooks/) for the full, annotated version.

[To-do]

---

## Citation

This work builds on CALVADOS. If you use it, please cite the upstream CALVADOS
papers ([KULL-Centre/CALVADOS](https://github.com/KULL-Centre/CALVADOS)) alongside
the AlphaFold/P<sub>inc</sub> methods referenced in
`src/analysis/alphafold.py`.
