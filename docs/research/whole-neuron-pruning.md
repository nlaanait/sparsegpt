# Whole-neuron pruning implementation

`whole-neuron-v2` is a separate block-level technique. It deletes static SwiGLU gate/up/down triplets and compensates only in surviving down columns. It does not change the entrywise pruner registry or reinterpret N:M flags. The implementation follows the supplied whole-neuron implementation specification; the historical entrywise experiments remain separate. V2 corrects the regressions documented in [the implementation review](WHOLE-NEURON-IMPLEMENTATION-REVIEW.md); v1 stage/export artifacts are deliberately rejected rather than silently reinterpreted.

## Solver and metrics

`prune/whole_neuron.py` validates bias-free dense `nn.Linear` triplets and traces the MLP computation to reject extra transformations, tied parameters, other activations and unsupported layouts. Because FX leaves can hide hook transformations, validation rejects forward/pre-forward hooks on the MLP, all its submodules, and the corresponding global hook registries, including observational hooks without a separately verified contract. It accepts the unmodified installed Qwen3.5 dense SwiGLU MLP. One explicit boolean mask controls all three projections. Surviving gate/up rows are copied bitwise; no recovery training, bias correction, or additional entrywise pruning is applied.

The primary selector is shared-column block OBS with deterministic index tie-breaking and exact per-block quotas. The requested count is `floor(sparsity * original_width)`, retaining at least one neuron. Unsupported ambient columns are protected by default, with quotas allocated across supported original indices. Requests that cannot meet the exact deletion count fail.

After selection, an independent Cholesky fixed-mask refit solves surviving down columns against the original reference. Output rows are solved in chunks. One damping owner supplies the effective delta: selection and surrogate refitting share the same regularized curvature. A full-metric refit changes the geometric operator but preserves the induced original-coordinate ridge `delta * scale.square()`. Factorization failure in either refit restarts selection with stronger damping. `surrogate_pre_refit_objective`, `surrogate_post_refit_objective`, and `surrogate_relative_stationarity` describe the congruence surrogate; `full_metric_*` fields describe the full reference. Zero sparsity returns original tensors bitwise.

Metrics are `ambient`, `local`, `downstream_activation_diagonal`, and `downstream_weight_diagonal`. Local geometry pairs actual down-input readouts with down-output factor-local frames. Downstream targets the down output and reuses next-block residual anchors, the existing probe estimator, and its sequence/probe conventions. The new unsquared probe callback streams full neuron-direction scores as well as weight diagonals without requiring a probe cache. `weight_diagonal` retains token interference; the whole-neuron direct score additionally sums output-coordinate contributions before squaring.

All curvature statistics use ambient token means. Each geometric metric normalizes its own supported weight-curvature trace; material geometry outside ambient support causes an error. Scaling uses the shared rescaling helper in FP32 production and an FP64 reference path. Alpha zero bypasses atlas fitting and backward calibration, following the identical ambient solve, and ignores any supplied previously fitted full metric even during direct selection. Shared scaled-coordinate damping induces an anisotropic original-coordinate ridge; it is recorded explicitly and retained by the full-refit solver path. An isotropic full-reference ridge remains available as an explicit library-level control, not an implicit change in the pruning pipeline.

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

For local/downstream geometry, pass `geometric_diagonal=K`, where K uses the same ambient token denominator. For two-sided coupling, pass `{"base": K, "input": K_input}` and set `input_alpha`; each supported trace is normalized separately. `full_metric` supplies full-direction selection/refit access. Available selectors are `block_obs`, `direct_write`, `direct_full_metric`, `random`, and the bounded `greedy_reference`. Compensation is `surrogate_refit`, `none`, or `full_metric_refit`. Full refitting supports replay-backed CG at production dimensions; the explicit matrix remains a bounded reference. Group cardinality and support are explicit; hardware-aligned rounding is not implemented.

## Sequential experiments and artifacts

The default calibration manifest now contains 512 original training questions
for atlas fitting, 128 separate pruning-calibration questions, and 64 independent
engineering diagnostic questions. Every parent is disjoint from the full
reserved evaluation pool. Controlled variants remain an explicit diagnostic
source. Representative submissions use separate `repcal-v1` names and a matching
dense reference. [Data, sampling, and evaluation identities](representative-gsm8k-calibration.md).

`python -m sparsifier.eval.gsm8k_whole_neuron --help` exposes the existing manifest/pilot/C4/output/block/seed/evaluation flags plus the metric, selector, compensation, blocksize, damping and support policy. `--modules` must name the complete triplet. Ambient defaults alpha to zero; other metrics default to one. An explicit nonzero ambient alpha is rejected.

`--skip-behavior` creates a candidate-only phase for a separately shared behavioral evaluation. It preserves the held-out finite-output check and records its diagnostic IDs, but scores no separate controlled subset. The decision64/light128 pilot uses this mode; see [the GSM8K status](GSM8K-EXPERIMENT-STATUS.md) for its fixed seeded original-question manifest and generation protocol.

Each stage is `(block, "mlp_triplet")`, with all three candidate tensors and mask saved atomically in one file. Original/candidate tensor hashes, original shapes, ordered neuron indices, config and upstream stage hashes, source hashes keyed by repository-relative paths, calibration IDs/positions, atlas/probe data, package/backend identities and resource diagnostics accompany it. A sidecar hash permits recovery after a complete stage save but before report commit; uncommitted temporary files and orphan files without a hash are not accepted. All tensors are validated before application, and any copy failure restores the entire original triplet.

Calibration refreshes the current partially pruned model before every block. Ambient exposure retains the historical 4,096 C4 + 4,096 math-token convention, including both fit and pruning-calibration readouts; atlas fitting is restricted to fit rows and probes to pruning-calibration rows. A deployment-cast finite-output check uses a held-out diagnostic prompt. No final-test data is used. The driver records architectural removed slots separately from original-nonzero removal, incidental zeros, eligible denominators and all-model denominators.

The new driver strictly resolves the installed optional-kernel wrappers to named reference functions and records their identities for all its arms. Before each downstream block it checks an actual-backend signed down-weight perturbation in the deployed dtype against the down-output VJP, using the actual cast perturbation. A non-measurable or failed check stops the run. Tiny CPU Qwen tests cover linear- and full-attention paths. This does not establish full-size bf16 GPU parity. The historical backend helper and its audit xfail remain unchanged.

GPU execution must use the repository's guarded gflow workflow. This implementation does not submit experiments. The solver reports a buffer estimate before model load; actual peak memory and refit cost still require a scheduled pilot. `--compensation full_metric_refit` now stores frozen low-rank factors in content-addressed replay and uses matrix-free CG above 2,048 coordinates. `--full-metric-solver`, `--full-metric-rtol`, `--full-metric-max-iterations`, replay disk/cache budgets, and `--replay-chunk-size` control the solve. The cache budget excludes solver workspaces and the current factor. Nonconvergence stops the transaction before publication. `--input-alpha` and `--input-structure` enable two-sided coupling, fitting input charts on geometry-fit data only. [Math, replay guarantees, and a gflow example](two-sided-full-metric-implementation.md).

`gsm8k_whole_neuron_evaluate` loads completed triplet chains for original GSM8K training-validation evaluation using the existing behavioral and general-text language helpers. It requires `--calibration-manifest` as well as `--manifest`: the source manifest hash must match the pruning config, all recorded calibration IDs must be present, and original parent `source_index` sets must be disjoint from evaluation. Controlled variant IDs do not bypass their parent-source exclusion. Missing or inconsistent parent provenance is an error. This check runs before CUDA configuration or model loading and records its outcome in evaluation provenance. It establishes parent-source separation, not semantic nonduplication between different source questions. `gsm8k_whole_neuron_report` validates paired evaluation provenance and labels actual removal-budget matches. Its prompt bootstrap is exploratory and conditions on the fitted calibration; controlled family data needs clustered uncertainty analysis. Entrywise and whole-neuron comparisons concern complete pipelines, with the down-only bridge available to study the reconstruction confound.

## Compact export and runtime

```bash
python -m sparsifier.eval.gsm8k_whole_neuron_export --source RUN --output ARTIFACT
```

Export consumes verified, completed triplet stages without recalibration. It saves every remaining model tensor plus explicit per-layer widths and original neuron indices. It does not rely on a single global `intermediate_size` or a `save_pretrained` roundtrip. `load_compact(ARTIFACT)` reconstructs the HF skeleton, replaces each relevant MLP with a compact adapter, and strictly loads the self-contained weights. Toy clean-process tests and a tiny HF Qwen test verify mixed-width forward/generation parity. Resume verifies the existing artifact and provenance hashes before accepting it.

`python -m sparsifier.eval.whole_neuron_runtime --artifact ARTIFACT --device cpu` measures the loaded compact model, reporting prefill and cached decode timings separately, with dtype/device/prompt/batch/backend/package metadata and actual artifact bytes. CUDA measurements require guarded scheduling. Width reduction alone is not a latency claim; comparable entrywise runtime measurements must use their actual execution representation. Dense masked stage files are diagnostic artifacts, not compressed models.

## Validation boundary

The reused GSM8K behavioral evaluator had a separate stopping bug: it omitted
tokenizer EOS `<|im_end|>`, continued generating, and could reject a completed
correct answer when the token budget was exhausted. A frozen entrywise
checkpoint reproduces 5/32 under that policy and 22/32 with both EOS IDs,
with the strict parser unchanged and matching token prefixes. This identifies
an evaluator artifact, not a whole-neuron accuracy measurement. Historical
whole-neuron sample-zero scores share that evaluator and require the same
caveat; corrected whole-neuron scores have not been measured in this audit.
See the [live GSM8K status](GSM8K-EXPERIMENT-STATUS.md).

CPU tests cover fixed-mask algebra and stationarity, shared quotas and ties, support guards, scaled independent row references, alpha-zero parity, greedy conditional costs, block elimination, damping restarts, rollback, full directional cross-coordinate cancellation, downstream shared-weight contractions, full local/downstream quadratic actions, bounded refits and CG nonconvergence, refreshed sequential calibration, stale/corrupt stage rejection, mixed-width compact reload and generation, and tiny actual-backend finite changes.

Historical whole-neuron pilots and model-wide screens are recorded with their source versions in the [GSM8K status](GSM8K-EXPERIMENT-STATUS.md). Those measurements do not establish accuracy for every full-metric selector or refit variant in the current code. New solver variants require a scheduled production-precision single-block pilot before a development grid. CPU export/runtime parity does not establish GPU speed. Quiet-context selectivity, reasoning robustness, semantic factor recovery, and accuracy retention remain empirical questions. Thinking-enabled or longer-budget evaluation requires a separately versioned protocol and dense control; the reused GSM8K prompt currently disables thinking.
