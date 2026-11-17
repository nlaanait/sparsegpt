# Whole-neuron implementation review

Reviewed 2026-11-12 against `whole-neuron-pruning.md`. Scope includes the triplet solver, full-metric operators, probe integration, sequential driver, stage persistence, evaluation/reporting, compact export/load, and runtime measurement. No production code or experiment jobs were changed.

## Conclusion

The default hook-free `block_obs` + `surrogate_refit` path follows the documented algebra in the cases tested. The implementation does **not strictly satisfy the note across all advertised API paths**. Three numerical/architectural regressions and an evaluation-separation gap remain. Passing toy tests is not evidence of full-size bf16 accuracy retention or runtime improvement.

## Findings

### [P2] Alpha zero does not disable a supplied full metric in direct selection

`src/sparsifier/prune/whole_neuron.py:291–297`

The geometry-normalization branch is skipped at alpha zero, but `direct_full_metric` still calls the supplied metric's `direct_scores`. Its existing gamma remains active. For a down weight `[1,2]`, identity calibration inputs, and streamed group energies `[100,0]`, the ambient control deletes neuron 0 while the alpha-zero call with that metric deletes neuron 1.

This violates the note's promise that alpha zero follows the identical ambient solve. The CLI's calibration routine returns no metric at alpha zero, so the demonstrated defect is in the public solver API when callers reuse a fitted metric. Fix by explicitly disabling geometric selection whenever alpha is zero, regardless of the supplied object. Test mask and refit equality with a nonempty full metric, not only with `full_metric=None`.

### [P2] Full refitting changes the damping geometry and lacks the documented restart path

`src/sparsifier/prune/whole_neuron.py:313–328`

Selection uses transformed weights `V = W ⊙ scale` and `H0 + delta I`. In original coordinates this induces the ridge

\[
\delta\sum_{pq}\mathrm{scale}_{pq}^{2}(W-\widehat W)_{pq}^{2}.
\]

`full_metric_refit` instead receives scalar `delta` and solves with an isotropic original-coordinate ridge. The audit verifies that its result is stationary for that implemented isotropic problem but fails stationarity when the selection ridge `delta * scale.square()` is used with the same full geometric operator. Changing the geometric operator for the full reference is intentional; changing the ridge additionally confounds that comparison.

The code labels the isotropic choice in diagnostics, so this is a specification discrepancy rather than an undisclosed algebraic error. The note's unqualified claim of one owner supplying the same effective regularized curvature to selection/refit does not cover what this path actually does. Preserve the induced anisotropic ridge in the full reference, or explicitly define/version this as a separate isotropic-ridge control and narrow the note's claim. The full-refit call is also outside the `LinAlgError` handler that restarts selection, so the restart promise currently applies only to surrogate refitting.

`pre_refit_objective`, `post_refit_objective`, and `relative_stationarity` continue to describe the surrogate even on this path; `full_metric_*` describe the full solve. Name the former fields explicitly as surrogate diagnostics to prevent misleading comparisons.

The driver restricts this path to bounded references; this finding does not invalidate the default production surrogate-refit algebra.

### [P2] Architecture validation misses transformations implemented by hooks

`src/sparsifier/prune/whole_neuron.py:68–82`

FX treats `nn.Linear` projections as leaves. A registered down-projection forward hook returning `output + 1` is accepted by `SwiGLUTriplet`, although the actual module is no longer `down(silu(gate(x))*up(x))`. Compact replacement and the functional deployment-output check omit that hook, so the accepted model's forward semantics can change on export even with otherwise correct tensors.

The note explicitly claims validation of the actual computation and rejection of extra transformations. Reject unsupported forward/pre-forward hooks on the MLP and its relevant submodules, or introduce an explicit verified contract for observational hooks. Preserve the existing support for the unmodified Qwen MLP. The reproduction is synthetic; no claim is made that the current Qwen checkpoint has such a hook.

### [P2] Original-validation evaluation does not establish source separation

`src/sparsifier/eval/gsm8k_whole_neuron_evaluate.py:29–43`

The evaluator verifies that supplied rows are labeled validation, are from training, and use original GSM8K IDs. It does not compare them with the pruning run's calibration sources. An independently hashed manifest can therefore label a previously used calibration question as validation and pass these checks. Hashes establish artifact identity, not disjointness.

The existing entrywise sweep planner checks `source_index` intersections, but this standalone whole-neuron entry point does not invoke that guard. Persist calibration source IDs/indices (including parent sources for controlled variants), or require the source calibration manifest and verify its stored hash; reject intersections before loading the model. This finding follows from control-flow inspection, not from observing contaminated results. No existing experiment is alleged to be contaminated.

## Verified correspondence

| Note requirement | Review result |
|---|---|
| Static gate/up/down triplet mask; unchanged surviving gate/up rows | Implemented; tensor validation and tests cover it |
| Exact floor deletion count, one survivor, protected ambient support | Implemented in primary selector; unsupported requests fail |
| Stable ties and per-block quotas | Existing tests cover uneven blocks and independent elimination |
| Independent fixed-mask down refit | Correct sign and stationarity; matches explicit row-wise reference |
| Shared damping and selection restart | Verified for surrogate path; full-refit exception above |
| Ambient token-mean geometry and supported trace normalization | Implemented; material excluded geometry is rejected |
| Alpha-zero and zero-sparsity parity | Default path passes; supplied full-metric exception above |
| Local factor-specific charts | Driver uses Gram dictionary plus per-active-factor local atlas frames |
| Downstream token interference and full neuron-direction scores | Probe callback contracts `g.T @ X` per sequence before squaring; group statistic also sums output coordinates before squaring |
| Full operators, bounded direct solve, optional CG | Existing tests compare explicit operators, stationarity, and nonconvergence diagnostics |
| Equal down-coordinate bridge | Independently refits both masks using the same full curvature/ridge; does not claim equal triplet savings |
| Sequential refresh and atomic stages | Current model is recollected; stage hashes bind original weights, config and predecessor; all tensors validated before application |
| Transaction rollback and stale/corrupt stage rejection | Existing tests pass; hashless orphan files are rejected rather than guessed |
| New reference backend and signed finite-change gate | Strict wrapper resolver exists separately from historical helper; tiny linear/full-attention tests pass |
| Compact mixed-width representation and reload | Existing toy clean-process and tiny HF tests pass; default exporter saves all state tensors |
| Runtime reporting | Measures the loaded representation, separates prefill/decode, labels lack of warmup; no speedup conclusion justified |

The direct-write/random selectors are ablations, not implementations of geometric block OBS. The full reference is a coupled fixed-mask solve, not a full coupled OBS selector. Those distinctions agree with the note.

## Validation and limits

Existing focused suite: **31 passed**. Added `tests/test_whole_neuron_audit.py`: **3 strict expected failures**, each recording an unresolved specification regression rather than hiding a passing test. Convert these to ordinary passing tests when correcting the associated behavior.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .pixi/envs/default/bin/python -m pytest -q \
  tests/test_whole_neuron.py tests/test_whole_neuron_metric.py \
  tests/test_gsm8k_whole_neuron.py tests/test_whole_neuron_export.py \
  tests/test_whole_neuron_audit.py
```

This review makes no claim that the whole-neuron method retains GSM8K performance. Production-scale memory, bf16 numerical behavior, retained-width cubic refit cost, and actual compact-model latency still need the guarded scheduled pilot specified in the note. The note was left unchanged so the discrepancies remain visible against the requested contract.
