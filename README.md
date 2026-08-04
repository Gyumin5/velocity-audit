# velocity-audit

Companion code for **"A Two-Stage Provenance and Consistency Audit of Pose-Derived Speed Channels in Public Driving Datasets."**

Public driving datasets publish an ego-speed channel that downstream calibration, learning, and localization systems consume as ground truth. That channel is not a direct measurement — it is the terminal output of a multi-stage sensing chain, so its noise, latency, and physical meaning are properties of the fusion pipeline rather than of the platform's motion. This repository contains the audit protocol, the probe implementation, and the analysis scripts used to produce every number and figure in the paper.

## What the audit does

The audit runs in two stages, and the separation between them is the point.

**Stage 1 — instrument characterization (truth known).** A fixed local least-squares polynomial probe is characterized without consulting any published velocity. Two sources fix its behavior: a closed-form noise-propagation expression whose variance ratio depends only on window geometry, and a synthetic sweep over trajectories with an analytic speed.

**Stage 2 — release measurement (truth unknown).** That characterized probe is applied unchanged — `W=5`, `p=3`, no per-dataset retuning — to each public release. The residual against the release's published velocity is read as a measurement *of the release*, never as a score for the probe and never as an accuracy verdict. Where pose and velocity share a smoother, the audit reports the case as undecidable rather than as agreement.

No step treats a published velocity as ground truth. None of the audited releases ships an absolute speed truth, which is why this is a consistency audit and not an accuracy evaluation.

## Layout

```
src/velref/       probe, metrics, dataset readers, audit core
scripts/          per-dataset processing and figure generation (26 scripts)
results/          the analysis outputs the paper reports
tests/            synthetic truth-known regression test
```

## Install

```bash
python -m pip install -e ".[dev]"
pytest
```

Requires Python 3.11+.

## Reproducing

The synthetic stage needs no external data:

```bash
python scripts/run_synthetic_l10.py
```

The release stage needs the public datasets, which we do not redistribute — obtain each from its own maintainers under its own license. Scripts expect dataset roots under `/mnt/Data/velref/<dataset>` by default; the argparse-based ones accept `--root` to point elsewhere, and the rest have the default at the top of the file.

Audited releases: HeLiPR, Oxford RobotCar, nuScenes, KITTI raw, KITTI-360, Boreas, Pit30M, and KAIST Complex Urban (the non-inertial reference-physics boundary case).

Analysis outputs already committed under `results/` correspond to the tables and figures in the paper, so the reported numbers can be checked without re-downloading the source datasets. `results/audited_sequences.csv` is the complete list of the 285 sequence/run/scene IDs the audit ran on (115 across the seven main releases and KAIST Complex Urban, plus the three pose-only check sets), with the sample count for each.

The per-frame streams behind those aggregates (pose, published velocity, and each estimator's output per sequence) are **not** redistributed here: they contain the source releases' own pose and velocity content, and several of those releases are distributed under non-commercial or no-redistribution terms that this repository's MIT licence cannot cover. The scripts regenerate them from each dataset's own copy.

## Scope and limitations

The empirical statements are about the seven releases examined. The protocol is stated so that others can run it, but its conclusions are not claimed to extend to releases we did not audit. Provenance labels are assigned by reading each release's documentation before the probe is run; where documentation does not support a label, the protocol returns `unclassified` rather than guessing, and that release is excluded from the coupling comparison.

## AI assistance

Claude (Anthropic) was used as a coding assistant in implementing portions of the processing and analysis scripts in this repository, and as a writing aid for parts of the manuscript text. The research questions, study design, audit methodology, experimental execution, and technical conclusions were directed by the authors, and all AI-assisted code was inspected, tested, and executed by the authors against the underlying data. The same disclosure appears in the acknowledgment section of the paper.

## License

MIT (see `LICENSE`). The datasets themselves are not covered by this license and remain subject to their own terms.
