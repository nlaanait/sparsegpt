# Downstream pruning implementation review

Reviewed 2026-11-12 against `tangent-pruning-methods.md`, especially Sections 6.1–6.5 and Appendix E. Scope: `prune/downstream.py`, the GSM8K downstream driver, factor-atlas construction, shared rescaling/SparseGPT, and the installed Qwen execution path. This is an implementation audit, not a new GSM8K result. Production code and experiment jobs were left unchanged.

## Verdict

The core implementation matches the **activation-diagonal, row-congruence surrogate** described in Section 6.4. I found no demonstrated transpose, VJP-direction, sample-pairing, trace-normalization, or compensation error in that path. It is not an implementation of the full shared-weight downstream quadratic. The strongest scientific concern is the approximation made before weight aggregation, not a reversed chain rule.

Two actionable implementation issues remain: the claimed PyTorch backend is not enforced with the installed kernel wrappers, and the experiment's nonzero-geometry guard runs before unsupported geometry is excluded.

## Findings

### 1. [P1] The reference-backend helper leaves optional kernel dispatch active

Location: `src/sparsifier/models/qwen.py:29–45`; invoked by `gsm8k_downstream.main` before model construction.

`_unwrap_torch` follows only `__wrapped__`. In this environment the Qwen functions are `kernels.layer` `Func` module instances, whose `forward` closes over the decorated function. Those module instances do not expose the expected outer `__wrapped__` chain. The helper consequently returns the original dispatcher and prints `-> pytorch` anyway. The installed Transformers fallback wrapper can select the installed FLA implementation even when `use_kernels=False`; that flag does not remove optional-package fallback dispatch.

Evidence: a tiny CPU Qwen forward after the helper reached FLA's `ChunkGatedDeltaRuleFunction` and Triton, then failed because its input tensors were on CPU. The audit contains a strict expected-failure regression for the helper. This demonstrates incorrect backend selection/logging; it does **not** demonstrate incorrect FLA gradients or establish which backend historical jobs used.

Correction: resolve the supported module/function wrapper forms, verify the selected callable rather than printing an unconditional claim, and test the reference path with optional packages installed. Record actual forward/backward backend identities and dependency versions in experiment provenance. The driver currently does not hash the imported Qwen helper in its source list. Validate production-precision forward and backward parity before attributing differences to pruning geometry.

The architecture finite-difference tests use an explicit **test-only** extraction of the reference function to isolate mathematical propagation. This bypass is not a production fix.

### 2. [P2] The geometric-arm guard can accept an entirely discarded metric

Location: `src/sparsifier/eval/gsm8k_downstream.py:148–149`, together with `src/sparsifier/prune/downstream.py:265–277`.

The driver checks the raw sum of `geometry_energy`. Finalization subsequently zeros all columns without ambient support. Thus a positive raw trace can become zero, set gamma to zero, and produce unit scales while the run remains labeled downstream. For example, ambient inputs `[1, 0]`, geometric inputs `[0, 1]`, and sensitivity `[1]` pass the driver condition but lose all geometry during finalization. The new audit reproduces this exactly.

The generic solver's zero-trace fallback is documented and need not be removed. The experiment, however, explicitly intends to reject a geometric arm without supported geometry. Finalize before checking the supported trace (when alpha is positive), and record raw trace, excluded trace, supported trace, and the achieved added/ambient ratio. Merely reporting the number of unsupported input coordinates does not quantify discarded geometric mass. This edge case is demonstrated synthetically; no claim is made that an existing Qwen stage encountered it.

## Scientific limitations, not implementation bugs

### Token interference is discarded before the weight-space diagonal is formed

At `downstream.py:156`, each VJP activation gradient is squared. At line 235, the result is multiplied by squared inputs. Consequently the statistic is

\[
K^{\mathrm{actdiag}}_{pq}=\mathbb E_r\sum_t g_{t,p}^{,2}\phi_{t,q}^{,2},
\]

whereas the exact shared-weight diagonal for the same frozen metric is

\[
K^{\mathrm{weightdiag}}_{pq}=\mathbb E_r\left(\sum_t g_{t,p}\phi_{t,q}\right)^2.
\]

The VJP itself traverses the network's token-mixing paths correctly. The information is lost when its entries are squared before token aggregation. More probes cannot restore it. The tests establish:

| Scalar anchor; two unit inputs | Exact geometric curvature | Implemented surrogate |
|---|---:|---:|
| `y1 - y2` | 0 | 2 |
| `y1 + y2` | 4 | 2 |

The discrepancy can reverse coordinate importance, not merely change a common normalization. For input columns `(1,1)` and `(0.5,-0.5)` and anchor `y1-y2`, exact weight diagonals are `(0,1)` while surrogate diagonals are `(2,0.5)`. These examples concern raw geometric curvature, not a claim about a particular compensated mask.

Recommended next implementation variant: accumulate `Z = g.T @ X` within each independent sequence/probe, then accumulate `Z.square()`. Average probes and sum sequences/weighted anchors with the declared convention. Do not sum independent examples before squaring. Normalize this new statistic by its own supported trace and retain the current variant as a named ablation. Using the new diagonal with row congruence would still approximate the full coupled compensation problem; compare both against a small full-quadratic fixed-mask reference.

### The downstream experiment retains factor-local geometry, but protects a bounded readout

`prune_site` uses `FactorAtlas` with the Gram block dictionary; it does not use the generic whole-activation `ChartTangentBasis`. For each sample, `FactorAtlas.frames(mode='local')` selects one local chart within each active factor and orthonormalizes their combined span. The union-of-local-factor-manifolds idea therefore survives in this driver. The dictionary remains an approximate estimator, not a validated recovery of semantic factors or the published BSF estimator.

The metric protects only the last prompt token at the next block's residual output; the final block uses its own residual output. Geometry is refreshed on the current sequentially pruned model. It does not directly protect later generated reasoning states, logits, or a dense-model reference trajectory. These are explicit experimental choices, not missing autograd paths. Eight probes and 32 backward-calibration problems are defaults, not demonstrated convergence criteria. Report rank/coverage, probe variability, and mask stability before interpreting a weak retention result as failure of the general hypothesis.

## What was verified

- Exact basis enumeration agrees with an explicit Jacobian, including token mixing; random probes are reproducible and converge in the existing small tests.
- Frozen target outputs become gradient leaves without severing already differentiable earlier targets. Multi-target/multi-anchor tests pass, and parameter gradient buffers are preserved.
- Anchor-prefix early exit preserves the captured graph and removes hooks.
- Sample-paired moments, optional input-basis coupling, Conv1D layout, and zero-rank handling pass the existing checks.
- Separate ambient and backward budgets do not double-count ambient inputs. An additional nonzero-alpha audit verifies the requested supported trace ratio with unequal budgets.
- Shared scale/prune/unscale compensation and alpha-zero integration parity pass. SparseGPT's batch-count normalization differs from the raw token sum by a common scalar; this does not change the undamped scale ratios and scales relative damping consistently.
- With the test-only reference-backend bypass, tiny float32 Qwen models give agreement between a next-block VJP and central weight finite differences through both linear-attention and full-attention anchors. These checks use a down-projection target and a fixed scalar readout to isolate propagation; they do not validate the full-size bf16 path, learned atlas quality, or finite high-sparsity perturbations.

## Reproduction and disposition

Added `tests/test_downstream_pruning_audit.py`. Run without GPU visibility:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. .pixi/envs/default/bin/python -m pytest -q \
  tests/test_downstream_tangent.py tests/test_gsm8k_downstream.py \
  tests/test_local_pruning_audit.py tests/test_downstream_pruning_audit.py
```

Result: **40 passed, 1 expected failure**. Reviewed runtime: torch `2.12.0+cu130`, transformers `5.20.0`, kernels `0.17.2`, flash-linear-attention `0.5.2`.

The backend regression is marked `xfail(strict=True)` to record the verified unresolved defect; it must become a normal passing regression when fixed. No production solver, checkpoint, methods manuscript, or queued job was changed by this review.
