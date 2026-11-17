# Full-metric CG smoke investigation

The representative-calibration full-metric smoke (gflow 207) rejected its
candidate because CG did not meet `rtol=1e-4` in 50 iterations. The expanded
atlas / surrogate-refit smoke (208) succeeded. No failed full-metric candidate
was written or applied.

Three guarded direct diagnostics reproduced the block-15 setup: Qwen3.5-9B,
30% neuron sparsity, `alpha=input_alpha=0.5`, four backward-calibration
questions, two probes, and a fixed mask. Each saved the same 16 replay-factor
entries and calibration provenance, and each reported RHS norm 585.506 with
35,233,792 retained weight coordinates. The ambient Hessian and mask were not
saved, so those could not be compared bitwise across processes.

| CG variant | Relative true residual at 50 | At 200 | At 400 | Result |
|---|---:|---:|---:|---|
| Current diagonal preconditioner; restart every 25 | 0.0358 | 0.0195 | 0.00724 | Did not converge |
| Same diagonal; no periodic restart | 0.0127 | 0.00980 | 0.0162 | Did not converge |
| Shared retained-column curvature Cholesky; no periodic restart | 0.0100 | — | — | Stopped at 140; residual 0.045 |

The residual is `||b - A x|| / ||b||`; the requested threshold was 0.0001.
Values need not decrease monotonically in the Euclidean norm. Recomputed and
recursive residuals agreed closely, including at 50 and 400 iterations, so
these observations do not indicate false convergence or appreciable residual
drift. Removing periodic restarts alone did not solve convergence. The one
shared-curvature preconditioner tried did not show a useful improvement through
140 iterations; it was stopped to release the GPU for queued work. This does
not rule out other preconditioners or longer convergence paths.

The 400-iteration current-preconditioner run reported relative stationarity
0.000535, but that statistic has a different denominator from the CG residual
and is not the requested stopping test. The solver correctly rejected it.
The diagnostics measured peak CUDA allocated memory about 21.82 GiB for the
diagonal runs and 22.10 GiB for the shared-curvature trial. The latter's peak
system-wide unified-memory usage was 34.88 GiB, on a 121.69 GiB host. CUDA
allocation and system memory overlap and must not be added.

The evidence points to difficult conditioning and insufficient effectiveness
of the tested preconditioners on this configuration, rather than GPU memory
exhaustion. It does not establish a matrix/operator correctness defect; the
existing small explicit-Hessian tests and CUDA checks cover different, much
smaller cases. No production solver or tolerance was changed.

Useful next checks are to persist the actual fixed mask, ambient Hessian,
weights, damping, and RHS for offline repeatable comparisons; log the true
residual and iteration count even on failure; and evaluate preconditioners
that account for both shared curvature and the sample-conditioned geometric
term. A successful method should meet the existing true-residual threshold,
retain the same mask/objective/damping, and be checked against explicit small
references before a larger smoke. The frozen replay factors for these direct
diagnostics remain in their respective `qwen-9b-full-metric-*-v1` result
directories; temporary tracing scripts were under `/tmp/full-metric-investigation`.
