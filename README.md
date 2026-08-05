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
scripts/          per-dataset processing, table generation, and figures (41 scripts)
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

The release stage needs the public datasets, which we do not redistribute — obtain each from its own maintainers under its own license. Scripts expect dataset roots under `/mnt/Data/velref/<dataset>`. That path is our own machine's layout, not a requirement: the argparse-based scripts take `--root`, and the rest carry the default as a constant at the top of the file. Nothing else in the repository depends on it.

Audited releases: HeLiPR, Oxford RobotCar, nuScenes, KITTI raw, KITTI-360, Boreas, Pit30M, and KAIST Complex Urban (the non-inertial reference-physics boundary case).

Analysis outputs already committed under `results/` correspond to the tables and figures in the paper, so the reported numbers can be checked without re-downloading the source datasets. `results/audited_sequences.csv` is the complete list of the 285 sequence/run/scene IDs the audit ran on (115 across the seven main releases and KAIST Complex Urban, plus the three pose-only check sets), with the sample count for each.

The generators for the paper's main tables are:

| script | produces | backs |
| --- | --- | --- |
| `build_crossds_tables.py` | `crossds_recomputed.csv`, `multimetric_recomputed.csv`, `latency_sweep.csv` | the cross-dataset table, the multi-metric table, the latency paragraph |
| `curvature_bins_all_datasets.py` | `curvature_bins_all_datasets.csv` | the curvature-bin figure |
| `degradation_stress_all.py` | `degradation_stress.csv` | the pose-degradation sweep |
| `integrated_drift.py` | `integrated_drift.csv` | the residual-versus-drift paragraph |
| `pit30m_build_10hz.py` | `results/pit30m_10hz/` (local only, see below) | Pit30M's analysis series |
| `pit30m_native_window_control.py` | `pit30m_native_control.csv` | the no-decimation control on Pit30M |
| `native_window_control.py` | `boreas_native_control.csv`, `nuscenes_x20_native_control.csv` | the same control on Boreas and nuScenes, plus the source-to-stream reproduction check |
| `per_frame_manifest.py` | `per_frame_manifest.csv` | content digests of every stream the analysis consumed |
| `verify_crossds_table.py` | `crossds_verify.csv` | an independent recomputation of all seven rows |
| `make_two_stage_figure.py` | `fig_two_stage.pdf` | Figure 1, the two-stage schematic |
| `make_curvature_fig.py` | `fig_curvature_bins.pdf` | Figure 2, the curvature-binned response |
| `make_regime_figure.py` | `fig_regime_overview.pdf` | Figure 3, the regime overview |

Every figure in the paper is drawn by one of the last three, and each of those
reads its values from the CSVs above rather than carrying a transcribed copy, so
a figure cannot drift out of step with the table it illustrates.

One row in that table means something different from the rest. Every `produces`
entry above is a file committed here except `results/pit30m_10hz/`, which is a
per-frame stream and therefore not redistributed (see below). The script writes
it into your own working copy; what is committed for it is its digest, in
`results/per_frame_manifest.csv`.

### Getting from a release to a per-frame stream

The table above starts at the per-frame stream. These are the scripts that
produce that stream from each release's own files, which is the stage you run
first if you are working from the source datasets:

| release | script |
| --- | --- |
| HeLiPR | `run_helipr_layer23.py` |
| Oxford RobotCar | `download_oxford.py` (fetch), `run_oxford_layer23.py` |
| nuScenes | `run_nuscenes_layer23.py` |
| KITTI raw | `run_kitti_layer23.py` |
| KITTI-360 | `process_kitti360.py` |
| Boreas | `run_boreas_layer23.py` |
| Pit30M | `process_pit30m.py`, then `pit30m_build_10hz.py` |
| KAIST Complex Urban | `process_kaist_cu_multi.py` |
| Argoverse 2, DurLAR | `process_av2_durlar.py` |
| Newer College | `process_ncd.py` |

Three pairs of scripts look like duplicates and are not. `process_kaist_cu.py`
handles the single `urban08` sequence and `process_kaist_cu_multi.py` runs all
of them and aggregates — the paper's nine KAIST CU sequences come from the
latter. `spectral_check.py` prints the illustrative `roundabout01` case;
`spectral_check_all.py` produces the six-sequence median and IQR the paper
reports. `run_dr_experiment.py` covers HeLiPR alone and `run_dr_all.py` covers
all seven releases; both write into `results/dr/`, and the drift paragraph in
the paper uses `integrated_drift.py`.

`scripts/decimation_threat_control.py` is kept for the record and is not behind
any claim in the paper: it pushes the aliasing threat further instead of
removing it, and the direct control in `native_window_control.py` superseded it
at v1.3-access. Its own docstring carries the caveat that makes its low-`W` rows
uninterpretable.

### What is under `results/`

- Top level — the aggregate CSVs the paper's tables and figures are computed
  from, plus `audited_sequences.csv` and `per_frame_manifest.csv`.
- `l10/`, `l10_sg_sweep/` — Stage 1. The synthetic truth-known sweep and the
  Savitzky–Golay comparison sweep, both reproducible with no external data.
- `helipr/` — the in-depth separated-regime case study, including the spectral
  and degradation checks.
- `av2/`, `durlar/`, `ncd/` — the three pose-only external checks reported as
  supplementary evidence for the smoothness band.
- `kaist_cu/` — the non-inertial reference-physics boundary case.
- `dr/` — dead-reckoning drift, the downstream-consequence check.
- `oxford/`, `oxford_x11/`, `nuscenes_x20/`, `boreas/`, `kitti360/`, `pit30m/`
  — per-release summaries. The `_x11` and `_x20` suffixes mark the runs at those
  releases' native rates rather than at the 10 Hz analysis cadence.

These consume the per-frame streams under `results/<release>/`, which are the stage this reproduction path starts from — see the note on redistribution below. Earlier tags of this repository shipped the aggregate CSVs without their generators, and its Pit30M script ran at an analysis cadence that did not match the paper's window-duration rule; both are fixed here, and the Pit30M numbers changed as a result.

The per-frame streams behind those aggregates (pose, published velocity, and each estimator's output per sequence) are **not** redistributed here: they contain the source releases' own pose and velocity content, and several of those releases are distributed under non-commercial or no-redistribution terms that this repository's MIT licence cannot cover. The scripts regenerate them from each dataset's own copy.

One boundary is worth stating plainly, and it has moved since the last tag. For **Boreas and nuScenes** the full path is now exercised: both were re-obtained from their public buckets and re-extracted, and the result reproduces the committed per-frame streams exactly — timestamps, position, and published velocity agree bit for bit on all 23 sequences. `scripts/native_window_control.py` prints that comparison before it runs its control, so it is a check you can repeat rather than a claim you have to accept. For the **other five releases** we no longer hold local copies, so there the exercised path still runs from per-frame stream to published number; the stage before it is unchanged from earlier tags and has been checked only against the aggregates it produced then. Anyone holding those releases can exercise the full path.

So that boundary is checkable rather than merely declared, `results/per_frame_manifest.csv` records a content digest of all 159 per-frame streams the analysis consumed. The digest is taken over the numeric columns — `t`, `x`, `y`, and the published velocity for the input side, the estimator outputs for the derived side — rather than over the file bytes, so it does not depend on the parquet writer or its version, and values are rounded to 1e-9 first so a different BLAS build does not register as a mismatch. Regenerate a stream from your own copy of a release, run `scripts/per_frame_manifest.py`, and compare: if the input digest matches, any downstream disagreement is in the analysis rather than in the extraction.

## Scope and limitations

The empirical statements are about the seven releases examined. The protocol is stated so that others can run it, but its conclusions are not claimed to extend to releases we did not audit. Provenance labels are assigned by reading each release's documentation before the probe is run; where documentation does not support a label, the protocol returns `unclassified` rather than guessing, and that release is excluded from the coupling comparison.

## AI assistance

Claude (Anthropic) was used as a coding assistant in implementing portions of the processing and analysis scripts in this repository, and as a writing aid for parts of the manuscript text. The research questions, study design, audit methodology, experimental execution, and technical conclusions were directed by the authors, and all AI-assisted code was inspected, tested, and executed by the authors against the underlying data. The paper carries the corresponding disclosure in its acknowledgment section.

## License

MIT (see `LICENSE`), and it covers the code in this repository only.

The audited datasets are not covered by it and are not redistributed here. Each
remains subject to its own licence and terms of use, which take precedence, and
so does anything you derive from your own copy of one. The aggregate CSVs
committed under `results/` are summary statistics computed from those releases
rather than extracts of them; if your intended use goes beyond checking the
numbers in the paper, check the terms of the release the numbers came from.
