# Changelog

## v1.3-access

The no-decimation control now covers all three releases published above the
analysis cadence, not just Pit30M, and the reproduction path for two of them
runs from the source release rather than from our committed stream.

### The control that removes decimation, on Boreas and nuScenes

`scripts/native_window_control.py` re-obtains Boreas (200 Hz) and nuScenes
(50 Hz) from their public buckets, reproduces the 10 Hz analysis series, and
then reaches the same 1.0 s window without decimating at all — `W=100` and
`W=25` respectively. The qualitative reading survives in both cases; the
magnitude does not.

| release | decimated (paper) | native | latency behaviour | curvature bins |
| --- | --- | --- | --- | --- |
| Boreas | +102% | +129.7% (200 Hz, W=100) | collapses to ≤1.8 pp at every non-zero shift, both settings | all five increase, both settings |
| nuScenes | −5.6% | −16.2% (50 Hz, W=25) | does not collapse, same shape, ~2.9x the scale | four of five decrease at 10 Hz (the highest-curvature bin is +7%); all five decrease natively |
| Pit30M | +77% | +140% (100 Hz, W=50) | collapses to ≤0.5 pp | all five increase |

So stride decimation is not what produces any of the three readings. What the
control does establish is that the reported *magnitudes* — and the smoothness
ratio especially, which reaches 11x on Boreas and 32x on nuScenes natively
against the 1.7x–4.1x band at 10 Hz — are properties of the analysis cadence
rather than of the releases alone. The paper now says so, and states the band
for the 10 Hz cadence explicitly.

### Reproduction path: source release to published number, for two releases

The same script checks the freshly extracted 10 Hz series against the committed
per-frame streams before it runs anything else. On all 23 Boreas and nuScenes
sequences the timestamps, position, and published velocity agree exactly — max
absolute difference 0.0, not merely within tolerance. The reproduction boundary
described in the README therefore now applies to five releases rather than
seven.

### Superseded

`scripts/decimation_threat_control.py` was written when those two source
releases were believed unavailable: it pushes the aliasing threat further
(10 → 5 → 3.33 Hz, every phase) instead of removing it. The direct control
above supersedes it for that question. It is kept because the sweep is still
informative, with one caveat now documented in the script and visible in its
output: at `p=3` a setting with `W<3` leaves a single degree of freedom, so the
probe interpolates rather than smooths, and those rows — identifiable by a
smoothness ratio below one — say nothing about the release.

## v1.2-access

This revision fixes a protocol violation in the Pit30M pipeline and, in the
course of tracing it, makes several previously implicit conventions explicit.
Numbers that changed are listed with their cause, so a reader comparing against
an earlier tag can tell a correction from a convention change from a re-draw.

### Pit30M ran at the wrong analysis cadence (correction)

The audit fixes the probe's window by **duration**, not by sample count: `W=5`
spans 1.0 s at the 10 Hz analysis cadence. Pit30M publishes its per-frame stream
at 100 Hz, and the aggregates behind the earlier tag were computed on that native
stream, where the same `W=5` spans 0.1 s. At that width the probe is barely
distinguishable from central differencing, which is why Pit30M previously read as
a null result.

`scripts/pit30m_build_10hz.py` puts the release on the analysis cadence and is
committed, so the series behind the table can now be regenerated. Everything
downstream of it was regenerated from that series.

| quantity | was | now |
| --- | --- | --- |
| M4 change vs central | +0.01% | +76.9% |
| smoothness ratio | 1.05x | 1.69x |
| median M4, central / probe | 0.00364 / 0.00364 | 0.00532 / 0.00942 |

Two controls bound the correction. `scripts/pit30m_stride_offset_sensitivity.py`
sweeps all ten decimation phases: the change spans +72% to +77% and the
smoothness ratio 1.68x to 1.70x, so the phase is not what produces the reading.
`scripts/pit30m_native_window_control.py` reaches the same 1.0 s window without
decimating at all, by keeping 100 Hz and widening the probe to `W=50`: the sign
(+140%), the collapse under any non-zero latency shift (within 0.5 percentage
points of zero), and the direction of the per-curvature-bin change (+102% to
+947%) all reproduce. The magnitudes differ between the two settings; the
qualitative reading does not.

### Pit30M source-log count (erratum)

The 53 analysis segments come from **12** distinct source logs. Earlier text said
16. The segment set itself is unchanged.

### KITTI-360 low-speed metric (convention unification)

`M2` is the RMS of the estimated speed on samples where the published speed is
below 0.3 m/s. That rule is now applied identically to all seven releases in a
single generator, with sequences carrying fewer than five in-mask samples skipped
rather than propagated as NaN. KITTI-360's earlier value had been computed under a
different, undocumented rule.

| quantity | was | now |
| --- | --- | --- |
| KITTI-360 M2, central / probe | 0.4180 / 0.4171 | 0.1097 / 0.1082 |

The other six releases are unaffected. The claim this metric supports — that `M2`
moves by no more than a few percent on six releases and by 11% on nuScenes, whose
low-speed mask is set by a CAN channel — holds under the unified rule.

### Cross-dataset point estimates (unchanged)

`crossds_recomputed.csv`, `multimetric_recomputed.csv`, and `latency_sweep.csv`
now come from one generator, `scripts/build_crossds_tables.py`, where previously
they were committed as aggregates without it. On the six releases that did not
change cadence, every `M3` and `M4` point estimate reproduces the earlier values
to full double precision; `M2` is the one column that moved, on KITTI-360 only,
for the reason given above.

The paired-bootstrap spread column moved slightly on two rows — nuScenes
2.5–97.5% upper bound −2.81% to −3.01%, KITTI-360 lower bound −0.81% to −0.74%.
The seed is the same; the single generator draws in a different order than the
scripts it replaced. That column is a within-dataset consistency band, not a
dataset-level confidence interval, and the shift is a re-draw rather than a
change in method.

### Pose-degradation sweep (now a median over realizations)

The sweep injects random position noise and timestamp jitter, so any single seed
is one realization. On the shorter scenes that matters: at ds=5 the nuScenes
reduction spans 21%–33% across draws, which is wider than the differences the
table is read for, while the other six releases move by at most 8 percentage
points. Each cell is now the median over 20 independent realizations, with the
5th–95th percentile across those realizations written alongside it in
`results/degradation_stress.csv`.

The correction changes one statement in the paper. nuScenes sits at 27% at ds=5,
below the 30% expectation, so that expectation is now reported as holding at
ds≤2 on all seven releases and at ds=5 on six — with the reason stated, since
nuScenes scenes are the shortest in the corpus at 19.5 s and the ds=5 series
retains about 39 samples.

`scripts/degradation_convention_check.py` additionally reports the sweep scored
on sequence interiors with the reference re-interpolated onto the jittered clock
— the convention in `run_analytics.py` — next to the full-series convention used
in the table, so the choice between them is visible rather than assumed.

### Figures are no longer transcribed by hand

`make_regime_figure.py` previously carried a hand-copied table of the values it
plotted. It now reads them from the committed CSVs. Two entries had drifted out
of step with the paper's tables and are corrected by the change: Pit30M (which
still showed the pre-correction 0% and 1.1x) and the KAIST CU smoothness ratio,
which the figure gave as 2.5x — a median of per-sequence ratios — where the text
and every other entry use the ratio of medians, 2.3x.

The observed smoothness band follows from the same CSVs rather than being typed
in, and is 1.7x–4.1x across the seven releases.

### New: reproduction boundary is now checkable

`results/per_frame_manifest.csv` records a content digest of all 159 per-frame
streams the analysis consumes. The per-frame streams are not redistributed here,
so this is what lets someone who regenerates them from their own copy of a
release confirm they match ours before comparing any downstream number. See the
README for the digest convention.
