# velocity-audit

Companion code for **"A Two-Stage Provenance and Consistency Audit of Pose-Derived Speed Channels in Public Driving Datasets."**

Public driving datasets publish an ego-speed channel that downstream calibration, learning, and localization systems consume as ground truth. That channel is not a direct measurement — it is the terminal output of a multi-stage sensing chain, so its noise, latency, and physical meaning are properties of the fusion pipeline rather than of the platform's motion. This repository contains the audit protocol, the probe implementation, and the analysis scripts used to produce every number and figure in the paper.

## What the audit does

The audit runs in two stages, and the separation between them is the point.

**Stage 1 — instrument characterization (truth known).** A fixed local least-squares polynomial probe is characterized without consulting any published velocity. Two sources fix its behavior: a closed-form noise-propagation expression whose variance ratio depends only on window geometry, and a synthetic sweep over trajectories with an analytic speed.

**Stage 2 — release measurement (truth unknown).** That characterized probe is applied unchanged — `W=5`, `p=3`, no per-dataset retuning — to each public release. The residual against the release's published velocity is read as a measurement *of the release*, never as a score for the probe and never as an accuracy verdict. Where pose and velocity share a smoother, the audit reports the case as undecidable rather than as agreement.

"Unchanged" fixes the window's **duration**, not its sample count: `W=5` spans 1.0 s at the 10 Hz analysis cadence, and a release published at a higher rate is decimated to that cadence before the probe runs. Pit30M publishes at 100 Hz, where the same `W=5` would span 0.1 s and leave the probe barely distinguishable from central differencing; `scripts/pit30m_build_10hz.py` puts it on the analysis cadence, and `scripts/pit30m_stride_offset_sensitivity.py` shows the choice of decimation phase moves the result by 5.2 percentage points and the smoothness ratio not at all.

No step treats a published velocity as ground truth. None of the audited releases ships an absolute speed truth, which is why this is a consistency audit and not an accuracy evaluation.

## Layout

```
src/velref/       probe, metrics, dataset readers, audit core
scripts/          per-dataset processing, table generation, and figures (38 scripts)
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

The generators for the paper's main tables are:

| script | produces | backs |
| --- | --- | --- |
| `build_crossds_tables.py` | `crossds_recomputed.csv`, `multimetric_recomputed.csv`, `latency_sweep.csv` | the cross-dataset table, the multi-metric table, the latency paragraph |
| `curvature_bins_all_datasets.py` | `curvature_bins_all_datasets.csv` | the curvature-bin figure |
| `degradation_stress_all.py` | `degradation_stress.csv` | the pose-degradation sweep |
| `integrated_drift.py` | `integrated_drift.csv` | the residual-versus-drift paragraph |
| `pit30m_build_10hz.py` | `results/pit30m_10hz/` | Pit30M's analysis series |
| `pit30m_native_window_control.py` | `pit30m_native_control.csv` | the no-decimation control on Pit30M |
| `per_frame_manifest.py` | `per_frame_manifest.csv` | content digests of every stream the analysis consumed |
| `verify_crossds_table.py` | `crossds_verify.csv` | an independent recomputation of all seven rows |

These consume the per-frame streams under `results/<release>/`, which are the stage this reproduction path starts from — see the note on redistribution below. Earlier tags of this repository shipped the aggregate CSVs without their generators, and its Pit30M script ran at an analysis cadence that did not match the paper's window-duration rule; both are fixed here, and the Pit30M numbers changed as a result.

The per-frame streams behind those aggregates (pose, published velocity, and each estimator's output per sequence) are **not** redistributed here: they contain the source releases' own pose and velocity content, and several of those releases are distributed under non-commercial or no-redistribution terms that this repository's MIT licence cannot cover. The scripts regenerate them from each dataset's own copy.

One boundary is worth stating plainly. The reproduction path exercised for this revision runs from per-frame stream to published number, not from source archive to published number: we no longer hold local copies of the source releases, and the stage before it — reading each release and emitting its per-frame stream — is unchanged from earlier tags and has been checked only against the aggregates it produced then. Anyone holding the source releases can exercise the full path.

So that boundary is checkable rather than merely declared, `results/per_frame_manifest.csv` records a content digest of all 159 per-frame streams the analysis consumed. The digest is taken over the numeric columns — `t`, `x`, `y`, and the published velocity for the input side, the estimator outputs for the derived side — rather than over the file bytes, so it does not depend on the parquet writer or its version, and values are rounded to 1e-9 first so a different BLAS build does not register as a mismatch. Regenerate a stream from your own copy of a release, run `scripts/per_frame_manifest.py`, and compare: if the input digest matches, any downstream disagreement is in the analysis rather than in the extraction.

## Scope and limitations

The empirical statements are about the seven releases examined. The protocol is stated so that others can run it, but its conclusions are not claimed to extend to releases we did not audit. Provenance labels are assigned by reading each release's documentation before the probe is run; where documentation does not support a label, the protocol returns `unclassified` rather than guessing, and that release is excluded from the coupling comparison.

## AI assistance

Claude (Anthropic) was used as a coding assistant in implementing portions of the processing and analysis scripts in this repository, and as a writing aid for parts of the manuscript text. The research questions, study design, audit methodology, experimental execution, and technical conclusions were directed by the authors, and all AI-assisted code was inspected, tested, and executed by the authors against the underlying data. The same disclosure appears in the acknowledgment section of the paper.

## License

MIT (see `LICENSE`). The datasets themselves are not covered by this license and remain subject to their own terms.
