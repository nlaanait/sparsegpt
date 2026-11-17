# GSM8K pruning experiments: results by technique

**Work intentionally stopped by the user.** Seven of nine requested grid
points are verified; all seven are nonzero. Local-65 has 25/96 pruning sites
saved and verified, with no accuracy score yet. Downstream-65, corrected
precision/calibration control evaluations, and the matched SparseGPT-70
baseline remain unfinished. The saved checkpoints, exact worker commands
and restart sequence are recorded in the
[resume status](GSM8K-EXPERIMENT-STATUS.md#saved-state-and-how-to-resume).
Resume only on an explicit user request.

**Evaluation correction, Nov 16:** the shared GSM8K evaluator omitted the
tokenizer's end-of-message EOS. Exact reproduction of one frozen SparseGPT-60
batch gives 5/32 under the old policy and 22/32 when generation stops at either
declared EOS, with the strict parser unchanged and all token prefixes matching.
The full corrected local-70 screen now scores **13/128 (10.16%)**, versus
**0/128** historically and dense **123/128**, with 128 matching decoded prefixes
and identical answer NLLs. Corrected downstream-70 is **12/128 (9.38%)**, also
with full raw-token verification and matching historical prefixes. The historical scores below retain the old policy; literal zero scores must
not be interpreted as absence of correct mathematical answers. Corrected dense is **123/128**. At 55%, corrected SparseGPT is **96/128**,
local is **89/128**, and downstream is **98/128**. Corrected SparseGPT-60 is
**95/128**, versus 21/128 historically; corrected local-60 is **72/128**.
Corrected downstream-60 is **77/128 (60.16%)**, with 10 truncated; all 128
raw-token and gold-input checks and all 96 source-tensor checks pass. All six
55–60% points are nonzero. Corrected SparseGPT-65 is **54/128 (42.19%)**,
with 27 truncated; raw-token, gold-input and checkpoint audits pass. Seven of
nine requested points are complete, all nonzero. Local-65 pruning 262 was
intentionally stopped at 25/96 saved sites; downstream-65 remains unstarted.
See the [live status and raw-token evidence](GSM8K-EXPERIMENT-STATUS.md).

In the historical tables below, “corrected local” means the geometry
normalization fix (v2). Those evaluations still used the old EOS policy.

**Scope:** Qwen3.5-9B, exploratory GSM8K training-set validation, seed 0. The official test set was not used. Sparsity refers to eligible MLP weights, not all model weights. Results below distinguish complete model-wide evaluations, screens, and one-projection or one-block pilots.

| Technique | Additional information | Computational change | What we tested |
|---|---|---|---|
| Ambient SparseGPT | Local input second moment | Shared curvature and compensating updates | Task-calibrated entrywise pruning; whole-neuron ambient pipeline |
| Input-side augmentation | Paired differences or input tangent probes | Add shared input curvature | Paired-difference augmentation and controls, one projection only |
| Local output geometry | Sample-conditioned tangent/presence projectors | Approximate row curvatures; rescale weights and compensation | Corrected factor-local output geometry, entrywise model-wide pruning |
| Downstream pullbacks | Downstream geometry and intervening VJPs | Backward calibration, then the same approximate row solve | Entrywise activation-diagonal and weight-diagonal variants; whole-neuron weight-diagonal pipeline |

## Model-wide: completed original 512-question validation

All MLP gate, up, and down projections were pruned entrywise across the model; generation limit 2,048 tokens. Dense and task SparseGPT used the earlier default environment; local and downstream used GB10, so small cross-method differences are not conclusive. The downstream column here is the **activation-diagonal** variant, not weight-diagonal.

| Eligible MLP sparsity | Whole model removed | Dense | Ambient task SparseGPT | Corrected local | Downstream activation-diagonal |
|---:|---:|---:|---:|---:|---:|
| 30% | 16.19% | 494/512 (96.48%) | [493/512 (96.29%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s30-seed0-v1-original512/report.json) | [495/512 (96.68%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s30-seed0-v2-original512/report.json) | [494/512 (96.48%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-downstream-s30-seed0-v1-original512/report.json) |
| 50% | 26.98% | 494/512 (96.48%) | [445/512 (86.91%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s50-seed0-v1-original512/report.json) | [436/512 (85.16%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s50-seed0-v2-original512-batch8-v1/report.json) | [416/512 (81.25%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-downstream-s50-seed0-v1-original512/report.json) |

At 30%, these observations are close to dense; at 50%, all three completed candidates lose accuracy. They do not establish a geometry advantage. Legacy local v1 results are invalid after a normalization fix; only corrected v2 appears here. [Correction details](LOCAL-PRUNING-VERIFICATION.md).

## Model-wide: additional screens and decision evaluations

Scores in each table use the stated reference and question set. A screen is a compute-triage rule, not a retention guarantee; rejected runs stopped early.

| Entrywise technique | Sparsity | Result | Protocol |
|---|---:|---:|---|
| Ambient task SparseGPT | 70%, 80%, 90%, 99% | 0/128 at each | Earlier 128-question screen; dense 125/128 |
| Corrected local | 50% | 105/128 (82.03%) | Batch-one 64→128; dense 123/128 |
| Corrected local | 70% | 0/64 | Batch-one screen; stopped after 64 |
| Downstream activation-diagonal | 70%, 80% | 8/128, 0/128 | Earlier 128-question screen; dense 125/128 |
| Downstream weight-diagonal | 30%, 50% | 124/128, 107/128 | Batch-eight 64→128; dense 123/128 |
| Downstream weight-diagonal | 70% | 0/64 | Batch-eight screen; stopped after 64 |

Corrected local 50% was evaluated at batch one, while weight-diagonal used batch eight. These results do not constitute a controlled ranking between the methods. [Status dashboard](GSM8K-EXPERIMENT-STATUS.md) has partial and canceled runs.

## Whole-neuron pipelines

Whole-neuron pruning deletes gate/up/down neuron triplets and compensates surviving down columns. Its compensation differs from entrywise pruning; each row compares an entire pipeline. On the same batch-eight 128-question screen, dense scored **123/128**.

| Model-wide eligible sparsity | Ambient | Downstream weight-diagonal |
|---:|---:|---:|
| 30% | 42/128; 49 truncated | 69/128; 21 truncated |
| 50% | 0/128; all truncated | 0/128; all truncated |
| 60% | 0/128; all truncated | Canceled after 30/32 blocks; no evaluation |
| 70%, 80%, 90%, 99% | Canceled | Canceled |

All five evaluated candidates failed the 15-point screen and stopped without a 512-question result. At 30%, whole-neuron pruning removed about 16.19% of model weights. A separate 30% **block-15-only** pilot scored 124/128 ambient and 122/128 downstream; it did not predict full-model retention.

## Input-side augmentation: one-projection development only

The paired-difference and control experiments pruned only block 15's `mlp.down_proj` to 50% sparsity, removing approximately 0.28% of model weights. They used 32 controlled examples and a 512-token cap; each listed error was a truncated output.

| Candidate | Correct / 32 |
|---|---:|
| Dense | 28 |
| Ambient task SparseGPT | 29 |
| Paired-difference augmentation, α = 0.3 / 1 / 3 | 29 / 28 / 29 |
| Repaired-pair control, α = 1 | 27 |
| Rotated-input control, α = 1 | 26 |

[Development comparison](../../results/tangent/gsm8k/development-comparison.json). This small, truncation-sensitive pilot does not show a method advantage, and no model-wide input-augmentation result is available. The longer 2,048-token reevaluation covered dense, task SparseGPT, and legacy local only; it did not reevaluate input augmentation.

## Interpretation

- **The original 512-question comparisons retain the old EOS policy and different environments.** Ambient task SparseGPT scored 493/512 at 30% and 445/512 at 50%; local scored 495/512 and 436/512. These results do not establish a controlled method ranking or a final retention claim.
- **Corrected stopping still leaves substantial loss at 70%:** local is 13/128 and downstream weight-diagonal is 12/128, versus dense 123/128. The literal historical zeros were artifacts. The matched SparseGPT-70 baseline and calibration/precision controls remain pending; the 55–65% grid is incomplete.
- **The block-15 whole-neuron pilot did not validate full-model retention.** Both historical 30% full-model runs were rejected by the old-policy screen. Those evaluations share the EOS bug and have no corrected measurements in this audit. Canceled higher-sparsity candidates have no accuracy result.
- **Input-side augmentation remains untested at model-wide scope.** Its one-projection pilot is too small to locate a compression frontier.

Detailed provenance, historical job states, and diagnostics: [history](GSM8K-EXPERIMENT-HISTORY.md) · [status](GSM8K-EXPERIMENT-STATUS.md) · [method definitions](tangent-pruning-methods.md).
