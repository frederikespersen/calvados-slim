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


---

## Citation

This work builds on CALVADOS. If you use it, please cite the upstream CALVADOS
papers ([KULL-Centre/CALVADOS](https://github.com/KULL-Centre/CALVADOS)) alongside
the AlphaFold/P<sub>inc</sub> methods referenced in
`src/analysis/alphafold.py`.
