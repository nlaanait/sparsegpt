# Local pruning implementation review — 2026-11-08

## Verdict

The GSM8K `local` arm implements the **row-diagonal congruence approximation** in tangent-pruning-methods.md §5, not an exact sample-dependent geometric solve. Independent CPU checks confirm local alpha-zero parity and the compensating update against a constrained, regularized quadratic. No reversed scaling, missing unscale, or score-only compensation bug was found in this path. This does not establish that the approximation or fitted geometry is useful.

## Findings

1. **Rank normalization is not curvature-trace normalization.** `gsm8k_pruning.prune_candidate` sets gamma = alpha * output_dim * ambient_token_count / sum(readout_ranks). With geometry zero outside readouts this normalizes the average output-metric trace. The actual geometric weight-curvature trace also depends on readout input energy. Recomputing from `capture-v1` and `geometry-seed0-v2` gives an added/ambient curvature-trace ratio of **2.68084095 at alpha=1**, not 1. This is compatible with a declared rank-normalized local metric, but not with interpreting alpha as the trace ratio used by downstream pruning. Comparisons at the same nominal alpha are therefore not equal-strength comparisons. Log both ratios; introduce explicitly versioned trace-normalized runs if that is the intended comparison. Do not silently relabel or modify old results.

2. **A real generalized-helper bug, outside the current local experiment.** `rescale.row_scales` with `h0_diag` and `hprime_diag` computes sqrt(1 + gamma*G*H0diag/Hprimediag). Section 5.2 requires sqrt((H0diag/Hprimediag)*(1 + gamma*G)) when H^(p)_qq = H0_qq*(1+gamma*G_pq). The ambient term is missing its ratio. The default Hprime=H0 branch and the bespoke GSM8K `prune_candidate` path do not have this bug. An explicit expected-failure regression documents it. Fix and test this branch before using a different shared curvature; it cannot explain the current local results.

3. **Substantial approximation error is already measured.** The stored exactness audit found congruence/exact fixed-mask objective ratios about 1.04–2.11 on small coordinate crops. This combines removal of output coupling, row-congruence error, SparseGPT's ordered compensation, and differing damping objectives. It does not isolate any one approximation or measure full-model error. A negative local result cannot cleanly reject the ideal geometric objective on that evidence. Compare full coupled, row-diagonal exact, congruence exact and ordered SparseGPT solutions under controlled masks and explicitly matched damping before attributing the loss.

4. **The protected samples and sites are narrower than task behavior.** The local objective uses final-question-token readouts; other calibration tokens contribute ambient curvature only. Sequential runs fit local geometry at each selected gate/up/down projection output. This is internally aligned but gate/up outputs are preactivations, not residual-write sites. Local geometry there is not downstream task sensitivity. Generated reasoning tokens are not geometric calibration samples. These are design limitations rather than incorrect tensor pairing.

5. **Factor-local charts are implemented, but their semantic/manifold quality is unproven.** `FactorAtlas` fits a sparse block dictionary on geometry-fit samples, allows co-active factors, clusters within each factor, and maps centered local PCA directions through the block's QR frame. It orthonormalizes the active tangent sum before extracting projector diagonals. Sequential collection refreshes task/C4 activations after prior pruning and refits the atlas. This matches the stated approximate factor-local construction; it is not the published BSF estimator. The pilot had five unsupported blocks out of eight and no demonstrated behavioral locality advantage. Unsupported factors abstain rather than inventing directions.

## Verified contracts

- Scales are computed from paired readout input squares and corresponding projector diagonals, not separately averaged factors.
- Weights are scaled before SparseGPT and unscaled afterward; compensation therefore includes the scale ratio.
- Both baseline and geometric GSM8K arms use the same top-k mask path, ambient token budget, and requested damping.
- Damping acts in scaled coordinates, inducing a scale-dependent ridge in original coordinates, as the document explicitly warns.
- Local alpha=0 reproduces equal-data ambient SparseGPT in the tested supported case.
- A two-input, two-output single-deletion case agrees with an independently constructed constrained quadratic including scaled-coordinate damping.
- Focused audit invocation: **13 passed, 1 expected failure**. Expected failure is the unused generalized-Hprime helper bug above.

## Recommended next diagnostic

Keep existing experiment provenance intact. Before another broad local sweep, compare rank-normalized and actual-curvature-trace-normalized strengths, initially at direct residual-write projections; persist scale quantiles, actual added trace ratios, active-factor coverage, ranks, mask overlap, and fixed-mask objective gaps. Use fresh, versioned runs. This separates excessive weighting, poor geometry, and solver approximation from a genuine lack of task-retention benefit.

This review changed no pruning implementation, running job, or queued experiment. Tests are in `tests/test_local_pruning_audit.py`.

## Follow-up implementation and CPU verification

The findings above describe the original audit. A subsequent fix corrects the
generalized `row_scales` ambient ratio and replaces the expected failure with a
passing regression. The GSM8K single-layer and sequential runners now use
curvature-trace normalization unconditionally. The incorrect rank-based default
and its selector flag have been removed. Config hashes bind the implementation
version and rescaling source;
new diagnostics record both strength ratios, scale quantiles, unsupported
energy, ranks, and atlas coverage. All nine active legacy local jobs were
cancelled; legacy candidates are rejected on resume/evaluation and excluded
from new comparison reports. Raw historical artifacts remain on disk.

The extended exactness audit compares full coupled, row-diagonal, congruence,
and ordered solutions on a common mask with matched scaled-coordinate damping.
The saved down-projection capture confirms a rank-normalized curvature ratio
of 2.68084114 at alpha=1. See [the follow-up verification](LOCAL-PRUNING-VERIFICATION.md)
for reproducible commands, controlled approximation gaps, and limitations.
