# Whole-neuron pruning implementation

`whole-neuron-v1` is a separate block-level technique. It deletes static SwiGLU gate/up/down triplets and compensates only in surviving down columns. It does not change the entrywise pruner registry or reinterpret N:M flags. The implementation follows the supplied whole-neuron implementation specification; the historical entrywise experiments remain separate.

## Solver and metrics

`prune/whole_neuron.py` validates bias-free dense `nn.Linear` triplets and traces the actual MLP computation to reject extra transformations, tied parameters, other activations and unsupported layouts. It accepts the installed Qwen3.5 dense SwiGLU MLP. One explicit boolean mask controls all three projections. Surviving gate/up rows are copied bitwise; no recovery training, bias correction, or additional entrywise pruning is applied.

The primary selector is shared-column block OBS with deterministic index tie-breaking and exact per-block quotas. The requested count is `floor(sparsity * original_width)`, retaining at least one neuron. Unsupported ambient columns are protected by default, with quotas allocated across supported original indices. Requests that cannot meet the exact deletion count fail.

After selection, an independent Cholesky fixed-mask refit solves surviving down columns against the original reference. Output rows are solved in chunks. One damping owner supplies the same effective regularized curvature to selection and refit; a retained-factor failure restarts selection with stronger damping. The block candidate and final refit objectives, free-gradient residual, retries, timing, mask and counts are reported. Zero sparsity returns original tensors bitwise.

Metrics are `ambient`, `local`, `downstream_activation_diagonal`, and `downstream_weight_diagonal`. Local geometry pairs actual down-input readouts with down-output factor-local frames. Downstream targets the down output and reuses next-block residual anchors, the existing probe estimator, and its sequence/probe conventions. The new unsquared probe callback streams full neuron-direction scores as well as weight diagonals without requiring a probe cache. `weight_diagonal` retains token interference; the whole-neuron direct score additionally sums output-coordinate contributions before squaring.

All curvature statistics use ambient token means. Each geometric metric normalizes its own supported weight-curvature trace; material geometry outside ambient support causes an error. Scaling uses the shared rescaling helper in FP32 production and an FP64 reference path. Alpha zero bypasses atlas fitting and backward calibration, following the identical ambient solve. Shared scaled-coordinate damping induces an anisotropic original-coordinate ridge; it is recorded explicitly.

`whole_neuron_metric.py` provides full local/frozen-probe curvature operators, direct neuron scores, FP64 explicit fixed-mask references capped at 2,048 down coordinates, and optional preconditioned CG with convergence diagnostics. Retained caches have an explicit size limit checked before copying. `down_only_bridge` compares equal deleted down-coordinate counts using independently solved fixed masks and identical full curvature/ridge. It does not claim equal triplet storage. Full group scores and full refits can differ from the diagonal surrogate; no full coupled OBS selector is claimed.

```python
from sparsifier.prune import SwiGLUTriplet, WholeNeuronConfig, prune_triplet
from sparsifier.prune.whole_neuron import apply_candidate

triplet = SwiGLUTriplet(mlp)
config = WholeNeuronConfig(sparsity=0.3, metric="ambient", alpha=0)
# Actual, unpadded down-projection input activations: [ambient_tokens, width].
candidate = prune_triplet(triplet, down_inputs, config)
apply_candidate(triplet, candidate)
```

For local/downstream geometry, pass `geometric_diagonal=K`, where K uses the same ambient token denominator. `full_metric` supplies full-direction selection/reference access. Available selectors are `block_obs`, `direct_write`, `direct_full_metric`, `random`, and the bounded `greedy_reference`. Compensation is `surrogate_refit`, `none`, or bounded `full_metric_refit`. Group cardinality and support are explicit; hardware-aligned rounding is not implemented.

## Sequential experiments and artifacts

`python -m sparsifier.eval.gsm8k_whole_neuron --help` exposes the existing manifest/pilot/C4/output/block/seed/evaluation flags plus the metric, selector, compensation, blocksize, damping and support policy. `--modules` must name the complete triplet. Ambient defaults alpha to zero; other metrics default to one. An explicit nonzero ambient alpha is rejected.

Each stage is `(block, "mlp_triplet")`, with all three candidate tensors and mask saved atomically in one file. Original/candidate tensor hashes, original shapes, ordered neuron indices, config and upstream stage hashes, source hashes keyed by repository-relative paths, calibration IDs/positions, atlas/probe data, package/backend identities and resource diagnostics accompany it. A sidecar hash permits recovery after a complete stage save but before report commit; uncommitted temporary files and orphan files without a hash are not accepted. All tensors are validated before application, and any copy failure restores the entire original triplet.

Calibration refreshes the current partially pruned model before every block. Ambient exposure retains the historical 4,096 C4 + 4,096 math-token convention, including both fit and pruning-calibration readouts; atlas fitting is restricted to fit rows and probes to pruning-calibration rows. A deployment-cast finite-output check uses a held-out diagnostic prompt. No final-test data is used. The driver records architectural removed slots separately from original-nonzero removal, incidental zeros, eligible denominators and all-model denominators.

The new driver strictly resolves the installed optional-kernel wrappers to named reference functions and records their identities for all its arms. Before each downstream block it checks an actual-backend signed down-weight perturbation in the deployed dtype against the down-output VJP, using the actual cast perturbation. A non-measurable or failed check stops the run. Tiny CPU Qwen tests cover linear- and full-attention paths. This does not establish full-size bf16 GPU parity. The historical backend helper and its audit xfail remain unchanged.

GPU execution must use the repository's guarded gflow workflow. This implementation does not submit experiments. The solver reports a buffer estimate before model load; actual peak memory and the retained-width cubic refit cost still require a scheduled pilot. Production full-metric refits with more than 2,048 weight coordinates are deliberately rejected by this driver until deterministic replay is integrated; the operator/CG API is available for bounded studies with a declared cache budget.

`gsm8k_whole_neuron_evaluate` loads completed triplet chains for original GSM8K training-validation evaluation using the existing behavioral and general-text language helpers. `gsm8k_whole_neuron_report` validates paired evaluation provenance and labels actual removal-budget matches. Its prompt bootstrap is exploratory and conditions on the fitted calibration; controlled family data needs clustered uncertainty analysis. Entrywise and whole-neuron comparisons concern complete pipelines, with the down-only bridge available to study the reconstruction confound.

## Compact export and runtime

```bash
python -m sparsifier.eval.gsm8k_whole_neuron_export --source RUN --output ARTIFACT
```

Export consumes verified, completed triplet stages without recalibration. It saves every remaining model tensor plus explicit per-layer widths and original neuron indices. It does not rely on a single global `intermediate_size` or a `save_pretrained` roundtrip. `load_compact(ARTIFACT)` reconstructs the HF skeleton, replaces each relevant MLP with a compact adapter, and strictly loads the self-contained weights. Toy clean-process tests and a tiny HF Qwen test verify mixed-width forward/generation parity. Resume verifies the existing artifact and provenance hashes before accepting it.

`python -m sparsifier.eval.whole_neuron_runtime --artifact ARTIFACT --device cpu` measures the loaded compact model, reporting prefill and cached decode timings separately, with dtype/device/prompt/batch/backend/package metadata and actual artifact bytes. CUDA measurements require guarded scheduling. Width reduction alone is not a latency claim; comparable entrywise runtime measurements must use their actual execution representation. Dense masked stage files are diagnostic artifacts, not compressed models.

## Validation boundary

CPU tests cover fixed-mask algebra and stationarity, shared quotas and ties, support guards, scaled independent row references, alpha-zero parity, greedy conditional costs, block elimination, damping restarts, rollback, full directional cross-coordinate cancellation, downstream shared-weight contractions, full local/downstream quadratic actions, bounded refits and CG nonconvergence, refreshed sequential calibration, stale/corrupt stage rejection, mixed-width compact reload and generation, and tiny actual-backend finite changes.

No whole-neuron model experiment or GPU runtime benchmark has been launched for this implementation. A scheduled production-precision single-block pilot is still required before the development grid. Quiet-context selectivity, reasoning robustness, semantic factor recovery, and accuracy retention remain empirical questions. Thinking/long-generation evaluation would require a separately versioned protocol and dense control; the reused GSM8K prompt currently disables thinking.
