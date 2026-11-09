# Local pruning correction and verification

This follows [the implementation review](LOCAL-PRUNING-IMPLEMENTATION-REVIEW.md)
and the local objective in [the method definition](tangent-pruning-methods.md),
section 5. All verification below ran on CPU with CUDA hidden. Existing
experiment outputs and GPU jobs were left intact.

## Corrections

`rescale.row_scales` now computes
`sqrt((H0diag / Hprimediag) * (1 + gamma * G))` when a distinct shared
curvature is supplied. The original formula omitted the ambient ratio.
The ordinary `Hprime=H0` branch retains its formula.

`gsm8k_pruning` and `gsm8k_sequential` now use corrected curvature-trace
normalization unconditionally. Alpha is the actual added geometric/ambient
weight-curvature trace ratio on supported coordinates, before damping. This
uses paired projector/input moments, as does downstream trace normalization.
The rank-based branch and `--local-normalization` selector have been removed.
The general `RescaleSparseGPT` adapter also normalizes against its actual
shared curvature diagonal, rather than mean chart rank.

Configs record `local_pruning_version = "curvature-trace-v2"` and the rescaling
source hash. Immutable provenance prevents mixing new stages with old output
directories. Stage validation and scheduled local evaluation reject legacy
local candidates; comparison reports exclude them. The corrected smoke gate
uses `qwen-9b-gsm8k-sequential-local-smoke-v2`, so the legacy smoke cannot enable
another broad sweep. Old raw artifacts are retained for traceability, not as
valid evidence about corrected local pruning.

All nine active legacy local jobs were cancelled: 80, 84, 85, 101, 102, 105,
106, 109, and 110. The cancellation and invalidated artifact list are recorded
in `results/tangent/gsm8k/local-legacy-invalidated-v1.json`. No replacement GPU
jobs were launched as part of this correction.

Candidate diagnostics include both output-metric and weight-curvature trace
ratios, gamma and averaging convention, ambient/readout counts, rank range,
supported-readout fraction, unsupported input coordinates and excluded energy,
exact scale extrema, and scale quantiles from at most 65,536 deterministically
strided entries. The runners also persist atlas coverage and unsupported
factors. Unsupported features have unit scale. A nonzero projector with zero
geometric input energy disables the trace-normalized addition; an atlas with
no supported geometry remains an error.

## Controlled reference

`gsm8k_exactness.compare_fixed_mask` evaluates four candidates on each ordered
SparseGPT mask: full coupled exact refit, row-diagonal exact refit, congruence
exact refit, and the ordered SparseGPT result. Every reference uses the same
original-coordinate ridge `delta * scale²` as the scaled SparseGPT solve,
where `delta = .01 * mean(diag(H0))`. This controls damping while separating:

- output coupling: row-exact minus full-exact loss under the full metric;
- row congruence: congruence-exact minus row-exact loss under the row metric;
- ordered compensation: ordered minus congruence-exact loss under the
  congruence metric.

These gaps use different evaluation metrics and should not be added together.
The report includes every candidate's losses under all three metrics, the
unregularized full metric, and an isotropically damped full metric. It also
persists masks, free-coordinate normal-equation residuals, scale diagnostics,
ambient mask overlap, and rank-versus-trace mask overlap. Cropped output
metrics remain principal submatrices without reorthogonalization. Strength
normalization is recomputed within each crop, so crop alpha is not the
full-layer gamma restricted to those coordinates.

## Recorded-capture results

The current default implementation was audited in
`results/tangent/gsm8k/local-audit-controlled-v3/{config,report}.json`, using
`capture-v1`, `geometry-seed0-v2/atlas.npz`, block 15 `mlp.down_proj`, three
seeds, 8-by-16 coordinate crops, and alpha in {0.3, 1, 3}: nine comparisons.
The report hashes the atlas and sources.

On the complete recorded layer at alpha=1, corrected normalization gives gamma
14725.15538 and added curvature ratio **1.0**. The removed rank formula would
give gamma 39475.80235 and ratio **2.68084114**. This historical scalar diagnostic
uses paired readout ranks and input energies without constructing a full weight
Hessian; it is not a selectable pruning method.

For the corrected implementation, ordered/full-exact loss ratios were
**1.03126–1.41573** with matched ridge. The maximum free normal-equation residual
was **1.69e-18**. All three isolated approximation gaps were positive. The
previous 18-comparison diagnostic remains in `local-audit-controlled-v2` as a
historical snapshot of the normalization investigation; its source hashes
refer to that earlier implementation.

Five of eight fitted factors remain unsupported. Active-factor chart coverage
on this capture is 1.0, which only measures the factors actually assigned to
these samples. Neither that coverage nor these numerical checks demonstrates
semantic locality or manifold quality. These crop checks do not establish
behavioral retention.

## Reproduction

Run from the repository root with the installed environment, without invoking
Pixi resolution:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .pixi/envs/default/bin/python -m pytest -q \
  tests/test_local_pruning_audit.py tests/test_gsm8k_exactness.py \
  tests/test_variant_b.py tests/test_gsm8k_pruning.py \
  tests/test_gsm8k_sequential.py tests/test_sample_metric.py \
  tests/test_downstream_tangent.py tests/test_ladder.py tests/test_tangent_techniques.py \
  tests/test_gsm8k_sweep.py tests/test_gsm8k_original.py tests/test_gsm8k_report.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 .pixi/envs/default/bin/python \
  -m sparsifier.eval.gsm8k_exactness \
  --capture results/tangent/gsm8k/capture-v1 \
  --atlas results/tangent/gsm8k/geometry-seed0-v2/atlas.npz \
  --output /tmp/local-audit-controlled-v3
```

The focused suite passed **73 tests**, including the formerly expected failure.
It checks trace ratios against independently assembled quadratic diagonals,
sample pairing, uniform activation scaling, unsupported/zero-energy cases,
independent default candidate agreement, alpha-zero parity, exact free-coordinate
stationarity, rejection of legacy local plans/checkpoints/smoke, and exclusion
of invalid results from reports. Future behavioral comparisons require fresh
versioned gflow jobs and the site/sample limitations in the original review.
