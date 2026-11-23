# calvados-slim

Code, curated inputs and simulation data for the MSc thesis *Calibrating SLiM interactions in CALVADOS* (Frederik Espersen Knudsen, University of Copenhagen, August 2026; supervised by Kresten Lindorff-Larsen).

This folder is an archive of the thesis work for the Lindorff-Larsen group. It is not a package to install, and it is not maintained as software. It is kept so that the reported calibration curves can be traced back to their inputs and trajectories, and so that the contact definitions and analysis code can be reused.

## Background

Short linear motifs (SLiMs) in disordered regions bind folded domains with affinities that coarse-grained force fields such as CALVADOS do not capture, as the atomic chemistry behind the specificity is averaged out.

In this work, SLiM binding is enforced with native-contact Gō potentials between the motif and its domain. The contacts are taken from experimental or AlphaFold structures, and a single well depth, `eps_go`, sets their strength. The relationship between `eps_go` and the binding free energy, ΔG°, was characterised for 276 curated SLiM-domain systems by simulating each over a range of `eps_go` and reweighting the trajectories into monotonic `eps_go`(ΔG°) curves. Models were then trained to predict these curves from features available before any simulation is run.

The thesis (`Masters_Thesis_handin.pdf`) gives the full methods and results. This README only describes where things are.

## Layout

```
calvados-slim/
├── readme.md
├── env.yaml                   conda environment used for the analyses
├── src/
│   ├── calvados/              CALVADOS fork, branch `slim` (Gō specific interactions)
│   ├── curation.py            building CALVADOS systems from curated entries
│   ├── model.py               eps_go(ΔG°) regression models
│   ├── utils.py, constants.py, visualisation.py
│   └── analysis/
│       ├── slim.py            bound fraction, Kd and ΔG° from trajectories
│       ├── mbar.py            MBAR reweighting along eps_go
│       ├── preprocessing.py   cleaning of scan free energies before modelling
│       ├── features.py        pre-simulation features for the models
│       ├── bsa.py, smog.py    contact maps (buried surface area, SMOG2 shadow map)
│       ├── alphafold.py       AlphaFold 3 confidence metrics
│       └── coarsegrain.py
├── notebooks/                 the workflow, numbered in order (see below)
├── data/
│   ├── configurations/        conditions, sampling settings, residue parameters, cluster job templates
│   ├── curation/              curated systems and per-system simulation inputs
│   ├── pdb/                   PDB entries used by the curated systems
│   ├── alphafold3/            AlphaFold 3 models used where no PDB entry was suitable
│   ├── analysis/              Q-bound cutoffs and ΔG° per system and eps_go
│   ├── modelling/             feature table, ablation results and the final model
│   └── synthetic/             contact-graph variants and their reweighted ΔG°
├── sim/                       trajectories (see below)
└── docs/                      reference notes on the feature set and the model
```

## Workflow

The notebooks are meant to be read in order. Each starts with a short description of what it reads and writes. All are run from the repository root.

| Notebook | Step | Main outputs |
| --- | --- | --- |
| `1_curation` | Builds simulation inputs for each entry in `systems.json` | `data/curation/1dp-idp/systems/<system>/` |
| `2_characterisation` | Describes the curated set (sizes, contacts, motif span) | Figures |
| `3_simulation` | Sets up speed tests, `eps_go` scans and refinement runs, and submits them to the cluster | `sim/1dp-idp/` |
| `4_analysis` | Calibrates the bound-state cutoff Q<sub>bound</sub> per system and computes K<sub>d</sub> and ΔG° per run | `data/analysis/Q_bounds_calibrated.csv`, `dGs.csv` |
| `5_reweighting` | Reweights each scan along `eps_go` with MBAR | `data/analysis/dGs_reweighted_*.csv` |
| `6_modelling` | Functional PCA of the curves and models predicting them from features | `data/modelling/` |
| `7_experiments` | Test case against experiment: PRPF40A WW domains binding SF1 | `sim/exp/`, figures |
| `8_figures` | Thesis figures | Figures |
| `9_ablation` | Ablation study of the final model | `data/modelling/ablation/` |

## Curated systems

`data/curation/1dp-idp/systems.json` holds the 276 systems. Each entry gives the chains (sequence, construct, structure file and folded domains), the SLiM (acceptor, donor, structure of the complex and experimental affinity) and the simulation box. The empty template is `src/curation_scheme.json`. Candidates that were rejected are kept in `unsuited_systems.json`.

For every system, `1_curation` writes a folder under `data/curation/1dp-idp/systems/` containing the coarse-grained chains, `specific_interactions.txt` and a ChimeraX script for visual checks of the contacts. `specific_interactions.txt` is the contact definition used by the CALVADOS fork; each line reads

```
<chain A> <residue i> | <chain B> <residue j> | <native distance / nm> <eps_go / kJ mol⁻¹>
```

`data/curation/exp/` holds the PRPF40A-SF1 system and its variants, and `data/curation/mdp/` holds full-length p53 (monomer and tetramer) as groundwork for motifs in multidomain proteins.

## Simulation data

Trajectories follow the path

```
sim/<set>/<system>/eps_go=<value>/<mode>/rep<n>/
```

where `<set>` is `1dp-idp` (one folded domain and one disordered partner), `exp` (the PRPF40A-SF1 test case) or `mdp` (p53). `<mode>` is `scan` for production runs at the grid values 0 to 5 and at refined values between them, or `visualisation` for short-stride runs. Sampling settings are in `data/configurations/sampling.json`; a scan run is 5 × 10⁸ steps saved every 5 × 10⁴ steps (10,000 frames).

Each run folder contains

| File | Content |
| --- | --- |
| `<system>.dcd` | Trajectory, one bead per residue |
| `top.pdb` | Topology and starting coordinates |
| `config.yaml`, `components.yaml` | CALVADOS input |
| `specific_interactions.txt` | Gō contacts at this `eps_go` |
| `bonds_*.txt`, `restr_*.txt` | Bonds and elastic-network restraints of folded domains |
| `<system>.xml` | Serialised OpenMM system |
| `<system>.log`, `logs/` | Energies, speed and job output |

Restart checkpoints, superseded trajectories and per-run job scripts were removed before archiving. The job templates are in `data/configurations/slurm/`.

## Software

The analyses were run with the conda environment in `env.yaml` (Python 3.10, OpenMM 8.2). Three further pieces are needed.

1. The CALVADOS fork in `src/calvados` (branch `slim` of https://github.com/frederikespersen/CALVADOS), installed with `pip install -e src/calvados --no-deps`. The code imports it as `src.calvados.calvados`.
2. BLOCKING (https://github.com/fpesceKU/BLOCKING) on `PYTHONPATH`, as the fork's analysis module imports `BlockAnalysis` from it.
3. SMOG2, only for the shadow contact maps in `src/analysis/smog.py`.

The ESM-2 and `sparrow` feature families in `src/analysis/features.py` are optional; their dependencies are commented out in `env.yaml`.

## Contact

Frederik Espersen Knudsen (frederikespersen@icloud.com). The CALVADOS papers and the methods cited in the thesis should be cited alongside any use of this material.
