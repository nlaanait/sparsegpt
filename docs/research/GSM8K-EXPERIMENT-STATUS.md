# GSM8K experiment status

**Calibration protocol update:** future runs default to 512 original training
questions for atlas fitting, 128 separate pruning-calibration questions, and
64 independent engineering diagnostics. All 704 parents are disjoint from the
full reserved 512-question evaluation pool and prior diagnostic sources.
Representative-calibration results and saved checkpoints are recorded below. Earlier results
retain their original controlled-template calibration and measure transfer
from that narrower fitting population. [Protocol details](representative-gsm8k-calibration.md).

**Snapshot: Nov 16, 2026, 9:31 pm. Central. GPU work for this investigation is intentionally stopped by the user.** Seven of nine corrected grid points are verified. At 55%: **SparseGPT 96/128**, **local 89/128**, **downstream 98/128**. At 60%: **SparseGPT 95/128**, **local 72/128**, **downstream 77/128**. At 65%: **SparseGPT 54/128 (42.19%)**, with 27 truncated. All seven raw-token grades, checkpoint links and gold-input audits pass. **Local-65 has 25/96 verified pruning sites saved; its accuracy and downstream-65 remain unmeasured.** The corrected precision/calibration controls and matched SparseGPT-70 baseline also remain unfinished. Corrected 70% anchors are local 13/128 and downstream 12/128, versus historical 0/128 and dense 123/128. Qwen3.5-9B; exploratory validation, not official test. Resume only on an explicit user request.

## Saved state and how to resume

The user intentionally cancelled this investigation's nine unfinished jobs
**262–265 and 273–277** at **8:59 pm Central on Nov 16**. The local-65 worker
is inactive, and its 25 saved sites are intact. These cancelled phases have
no new measured accuracy. The CPU collector exited after observing the
cancellations; its **10 verified evaluations out of 16 planned** remain saved
(dense, seven grid points, and the two geometry-70 anchors).

| Work remaining | Saved state | Cancelled job IDs |
|---|---|---|
| Local 65% | 25/96 pruning sites verified; accuracy evaluation unstarted | 262 → 263 |
| Downstream weight-diagonal 65% | Pruning and evaluation unstarted | 264 → 265 |
| Prompt-high local 70% control | All 96 pruning sites complete; corrected EOS evaluation unstarted | 273 |
| Reasoning-high local 70% control | All 96 pruning sites complete; corrected EOS evaluation unstarted | 274 |
| FP64 local 70% control | All 96 pruning sites complete; corrected EOS evaluation unstarted | 275 |
| Matched task SparseGPT 70% baseline | Pruning and evaluation unstarted | 276 → 277 |

The local-65 checkpoint is
[`qwen-9b-gsm8k-mlp-local-s65-seed0-batch32-repcal-v1`](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s65-seed0-batch32-repcal-v1).
Its [25-site verification](../../results/tangent/gsm8k/eos-corrected-v1/cancellations/20261117T025928Z/local-s65-prefix-verification.json)
checks every saved tensor's hash, upstream chain, finite values and sparsity,
plus the saved execution policy and source hashes. The next site is
**stage index 25 (the 26th site), block 8 `mlp.up_proj`**, with zero-based
block indexing. This is a partial pruning checkpoint, so it adds no accuracy
point or completed model to the audit counts.

The [resume plan](../../results/tangent/gsm8k/eos-corrected-v1/cancellations/20261117T025928Z/resume-plan.json)
contains the exact corrected worker arguments, output folders, priorities,
resource caps, and dependency mapping for all nine unfinished phases. The
[queue snapshot and cancellation evidence](../../results/tangent/gsm8k/eos-corrected-v1/cancellations/20261117T025928Z/queue-state.json)
and [submission journal snapshot](../../results/tangent/gsm8k/eos-corrected-v1/cancellations/20261117T025928Z/submissions-before-stop.json)
retain the original job records. Nothing was resubmitted.

When the user requests a resume:

1. Recheck the saved prefix with the CPU-only command below. Preserve the
   checkpoint folders, calibration manifests, `capture-v1`, dense pilot and
   corrected dense reference. The current source and batch policy match the
   saved run. If they change later, use the
   [matching frozen-source archive](../../results/tangent/gsm8k/pruning-artifact-audit-v1/frozen-source-v1/manifest.json)
   in an isolated implementation and verify configuration compatibility.
2. Submit only the nine unfinished phases from the resume plan through
   **gflow**, with fresh unique names and job IDs. Use the same output paths
   so local-65 restores its 25 sites. Recorded caps are **1 exclusive GPU,
   64G GPU memory, 80G host memory, 06:00:00 per phase**, project
   `qwen-sparsegpt`. Keep completed dependencies; replace dependencies on
   cancelled jobs with the corresponding new IDs. The plan records the exact
   dependency split and priorities. All seven completed grid points and the
   already complete control checkpoints are reusable.
3. Update the six cancelled evaluation IDs **in their existing label
   records** in
   [`eos-corrected-v1/submissions.json`](../../results/tangent/gsm8k/eos-corrected-v1/submissions.json),
   then restart the CPU collector using the second command below. Keep its
   10 verified results. The resume plan lists the six old IDs to replace;
   each label must remain unique. The corrected worker arguments retain
   dense reference **266** and the versioned EOS adapter.
4. After the remaining evaluations finish, repeat the generation-input and
   common-input parity audits and regenerate the sparsity chart. Keep the
   prepared native EOS patch **unapplied while resuming frozen-source
   work**, since changing production source hashes would invalidate the
   partial pruning configuration. Its 83-test validation and patch remain
   saved for later integration.

Run from the repository root; both commands below are CPU-only. The collector
command is for use after replacement job IDs have been registered.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=src:. \
  .pixi/envs/gb10/bin/python \
  results/tangent/gsm8k/eos-corrected-v1/cancellations/20261117T025928Z/verify-prefix.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=src:. \
  .pixi/envs/gb10/bin/python scripts/collect_gsm8k_eos_results.py \
  --snapshot /home/nlaanait/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --watch
```

**Confirmed evaluation artifact:** the pinned model config and CPU empty-model
generation config use EOS **248044** (`<|endoftext|>`), while the tokenizer's
EOS is **248046** (`<|im_end|>`). No generation-config file is cached, and the
live worker uses the same default cache. The evaluator uses only the model's
generation EOS and rejects every budget-truncated answer. All **53** additional
strict correct SparseGPT-60 answers have newline-terminated answer markers;
they are not guesses from an unfinished number. The
[evidence](../../results/tangent/gsm8k/pruning-artifact-audit-v1/stop-token-evidence.json)
records the exact config and answer hashes. The 74/128 rescore is a sensitivity
analysis, not a corrected EOS evaluation. Diagnostic 255 exactly reproduced
**all 32 historical texts** in the original first batch. The model-only EOS
policy scores **5/32**, with 25 truncated; stopping those identical raw token
streams at either model or tokenizer EOS scores **22/32**, with one truncated.
**31/32** emitted the tokenizer EOS. Thus the evaluator continued beyond an
explicit end-of-message token and rejected correct completed answers at the
budget limit. The [raw-token proof](../../results/tangent/gsm8k/pruning-artifact-audit-v1/stop-token-first-batch-proof.json)
records this comparison; fresh generation with the corrected stop list also scores **22/32**, and
all 32 token prefixes agree through the corrected stopping point. Historical scores and production pruning code remain unchanged.
Literal old sample zeros cannot establish absence of correct mathematical
answers.

The [paired-score figure](../../results/tangent/gsm8k/eos-corrected-v1/figures/eos-effect.html)
compares six fully audited frozen-checkpoint conditions under the two stopping
rules. Its [data and provenance](../../results/tangent/gsm8k/eos-corrected-v1/figures/eos-effect.data.json)
retain exact counts and source hashes. This descriptive comparison is not a
fit of the incomplete sparsity frontier.

The [sparsity chart](../../results/tangent/gsm8k/eos-corrected-v1/figures/sparsity-bracket.html)
shows the corrected common-128 measurements across 55–70% eligible MLP sparsity.
Unmeasured levels are blank and lines do not cross missing levels. Its
[data and provenance](../../results/tangent/gsm8k/eos-corrected-v1/figures/sparsity-bracket.data.json)
bind every point to the raw-token grade, independently reconstructed inputs,
and the audited 96-stage checkpoint. Regenerate it after new input audits
with `python scripts/plot_gsm8k_sparsity_bracket.py`. It does not fit or infer
a zero-accuracy threshold.
The [independent generation-input audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/generation-inputs-summary.json)
reconstructs prompts and left-padded token IDs from the bound gold manifests,
then compares them with saved packed-input hashes and per-answer prompt
metadata. All ten completed common-128 evaluations pass. Two in-memory
corruption controls confirm that wrong packed
input hashes and wrong prompt hashes are rejected. This audit checks inputs;
raw-token scoring is verified separately. It will be repeated on all final
outputs before completion.

The [cross-evaluation input comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/common128-input-parity.json)
also verifies that those ten evaluations use exactly the dense run's prompt
hashes, packed-token hashes, batch membership, and gold answers on the common
128 questions. Its three corruption controls reject changed prompts, packed
inputs, and gold answers. The
[reproducible checker](../../results/tangent/gsm8k/pruning-artifact-audit-v1/common-inputs-audit.py)
will be rerun after the remaining evaluations finish.



The [pinned upstream tree](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)
also lacks `generation_config.json`; the missing file is not solely a local
download omission. An [upstream community proposal](https://huggingface.co/Qwen/Qwen3.5-9B/blob/refs%2Fpr%2F32/generation_config.json)
supplies both EOS IDs, but that proposal is not part of our pinned revision.
The evaluation adapter sets both IDs explicitly without changing the model pin
or cache.

The tested correction is
[`scripts/run_gsm8k_eos_control.py`](../../scripts/run_gsm8k_eos_control.py):
use EOS **[248044, 248046]**, retain the strict parser and token budget, save
raw generation IDs, tag the immutable configuration before hashing, and reject
an old-policy dense reference. **28 focused tests pass**, including the existing
decision-screen tests. Corrected dense reference **266 completed at 123/128** (one truncated), with
[all 128 answers independently audited](../../results/tangent/gsm8k/eos-corrected-v1/audits/dense.json)
and [per-question agreement with historical dense](../../results/tangent/gsm8k/eos-corrected-v1/audits/dense-historical-comparison.json).
Local-70 **271 completed at 13/128**, and downstream-70 **272 completed at 12/128**; the nine grid
evaluations are recorded in
[`eos-corrected-v1`](../../results/tangent/gsm8k/eos-corrected-v1/submissions.json).
Completed pruning checkpoints are reused; old scores are retained. Current
measurements use the versioned adapter so resumable pruning workers retain
matching source hashes. After the remaining frozen-source work is resumed and completed, the shared
evaluator will receive the same EOS-union rule and a stopping-policy change
in its execution metadata. The exact current source is preserved in a
[hashed archive](../../results/tangent/gsm8k/pruning-artifact-audit-v1/frozen-source-v1/manifest.json)
for reproduction. The
[real-config resume check](../../results/tangent/gsm8k/eos-corrected-v1/audits/legacy-resume-guard.json)
confirms corrected evaluation rejects an old-policy cached configuration
without changing either the original file or its temporary copy.

The permanent shared fix is now [prepared and CPU-verified](../../results/tangent/gsm8k/pruning-artifact-audit-v1/native-eos-pending/plan.json)
in an isolated source copy: **83 tests pass**, including 20 native EOS/cache
regression cases. Real tiny-Qwen generation confirms stopping at the tokenizer
EOS, including EOS exactly at the token budget. Both decision drivers reject a
stale completed report before loading a model. The pending patch explicitly
passes the model/tokenizer EOS union, preserves token IDs, and binds stopping,
behavior, prompt, and parser provenance into the execution policy. It has
**not been applied to the working source**, preserving the saved partial
pruning configuration for a later resume.
The older `gsm8k_screened` wrapper also needed explicit policy checks before
its saved-screen early return. With only the two shared-file changes, five
CPU regressions reproduced acceptance of stale baseline/candidate metadata
or rows, including a cached 0/128 rejection. The pending three-file patch now
rejects all five cases and accepts a current-policy reference. The versioned grid uses `gsm8k_decision_full` and is unaffected by this legacy
wrapper gap. [Coverage and source hashes](../../results/tangent/gsm8k/pruning-artifact-audit-v1/native-eos-pending/consumer-coverage.json).

A [CPU restoration check](../../results/tangent/gsm8k/pruning-artifact-audit-v1/native-eos-pending/frozen-checkpoint-restore-compatibility.json)
also validates the actual 96-site frozen local-70 checkpoint through the
isolated native sources. Legacy pruning metadata remains acceptable for
restoring weights; the new execution policy rejects legacy evaluation caches.
No additional full model or GPU allocation was used for this check.

The [real GPU comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s60-direct-eos-equivalence.json)
also verifies that explicit EOS keywords (255) and the adapter (270) produce
identical raw token matrices on the same 32 SparseGPT-60 questions, including
all padding after each EOS. This supports the pending native generation call;
it is scoped to that matched batch, not a new full benchmark.


The [historical-policy common-128 sensitivity audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/completion-censoring-sensitivity.json)
keeps the strict numeric-marker parser and changes only the blanket rejection
of truncated output. These are distinct metrics, not interchangeable scores:

| Candidate | Finished and correct | Correct strict marker, ignoring truncation |
|---|---:|---:|
| Dense | 123/128 | 123/128 |
| SparseGPT 60% | 21/128 | 74/128 |
| Local 70%, original calibration | 0/128 | 10/128 |
| Downstream 70%, original calibration | 0/128 | 4/128 |
| Local 70%, reasoning calibration | 34/128 | 42/128 |
| Local 70%, FP64 solver | 0/128 | 9/128 |

Thus a literal zero depends on the completion requirement even if the EOS
comparison eventually finds no rescue. The substantial degradation at 70%
remains visible under either scoring rule.

## Representative-calibration entrywise runs

This table preserves the **historical model-only EOS policy**. The corrected
70% results are recorded above and in the corrected-results section below.

All-model MLP pruning uses 512 original questions for tangent atlas fitting and 128 separate pruning-calibration questions. The 128-question screen permits a 15-point dense-relative drop; passing candidates continue to 512. Sparsity refers to eligible MLP weights.

| Sparsity | Local | Downstream weight-diagonal |
|---:|---|---|
| 50% | **Complete · 210/214:** [452/512 (88.28%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s50-seed0-batch8-repcal-v1-decision512/report.json); 10 truncated. Screen **114/128** vs dense **123/128**. Batch 8. | **Complete · 215/216:** [447/512 (87.30%)](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-downstream-weightdiag-s50-seed0-batch32-repcal-v1-decision512/report.json); 10 truncated. Screen **109/128** vs dense **123/128**. Batch 32. |
| 70% | **222/223 complete; failed screen:** 96/96 pruning sites saved; **0/128**, all 128 truncated. Batch 32. No 512-question result. | **227/225 complete; failed screen:** 96/96 sites; **0/128**, all 128 truncated. Batch 32. No 512-question result. |

The 50% local pruning worker completed all 96 stages and the four-question engineering check, but gflow marked job 210 failed because of a wrapper shell error after the worker succeeded. Job 214 reused the completed checkpoint. The batch-32 dense reference is complete at **123/128**, with one truncated answer. The 70% evaluations depend on their pruning jobs and completed reference job 221. The local 70% screen rejected the candidate after 128 questions. Restarting the idle gflow daemon cleared the scheduler resource hold and allowed job 222 to start. The local and downstream 50% results use different batch sizes and do not establish a controlled method ranking. The screen is a compute triage rule, not a retention guarantee. The earlier controlled-template results below use different fitting data.

## 55–65% sparsity bracket: stopped by user; 7/9 measured

Nine additional Qwen3.5-9B candidates use the original representative
calibration (4,096 C4 plus 4,096 task-prompt tokens), seed 0, all 96 MLP sites,
and production FP32 pruning. Downstream means the corrected shared-weight
diagonal pullback. They use the same batch-32 128-question screen. Original
evaluations used dense reference 221; corrected evaluations use 266. The
existing 15-point rule continues passing candidates to
512. Compare the common first 128 for the zero frontier, and report any full
512 result separately. A measured 0/128 does not establish zero population
accuracy. Sparsity refers to eligible MLP weights.

The EOS diagnostic **255 completed**; corrected dense **266 completed**, and
local-70 **271 completed**, with downstream-70 **272 completed**. Local-60
pruning **256 completed** from its preserved checkpoint; corrected evaluation
**257 completed at 72/128 (56.25%)**, with 16 truncated. The [full CPU audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-cpu.json)
verifies all 96 tensors, finite weights, intended 60% sparsity, and no all-zero
rows or columns. Its geometry/ambient trace ratio is 1 within rounding. The
[matched configuration check](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-pruning-config-comparison.json)
finds only sparsity differs from local-55; seed, calibration, code, and batch
policy match. Sixteen distinct full sources now have 1,536 audited site
records. The producer's four-question engineering score is not the corrected
common-128 benchmark result. A scheduler dependency
update retained the old dependency on diagnostic 253, causing queued jobs
244–254 to auto-cancel when checkpointing job 243. All ten remaining bracket
phases were rebuilt with identical commands and output paths as jobs
256–265; their exact dependency sets are independently verified in the
[recovery journal](../../results/tangent/gsm8k/pruning-artifact-audit-v1/stop-token-queue-recovery.json).
These cancellations are scheduling events, not measured method failures.
The immutable original plan remains the protocol record.
The [resume-RNG audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-resume-rng.json)
also checks that the local pipeline does not rely on the process-global RNG
history skipped by a resume. Code inspection finds fresh explicitly seeded
generators for per-example capture, token selection, dictionary initialization,
and chart fitting. A CPU test with changed NumPy/Torch global RNG histories
reproduces all 160 sampled-position arrays, 8,192-token calibration inputs,
76 saved atlas arrays, and projector diagonals exactly. The source hashes
match the active local-60 run. This is a sampling/fitting check on a small CPU
model and synthetic atlas data, not proof of bitwise full-model GPU equality.


| Method | 55% | 60% | 65% |
|---|---|---|---|
| Task SparseGPT | [95/128 · 74.22%](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s55-screen.json), 4 truncated | [21/128 · 16.41%](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s60-screen.json), 99 truncated; 74 strict markers ignoring truncation | No historical-policy run; corrected evaluation below |
| Local | [83/128 · 64.84%](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/local-s55-screen.json), 19 truncated | No historical-policy run; corrected result below | No historical-policy run; corrected evaluation below |
| Downstream weight-diagonal | [98/128 · 76.56%](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/downstream-weightdiag-s55-screen.json), 4 truncated | No historical-policy run; corrected evaluation below | No historical-policy run; corrected evaluation below |

The numbers in the preceding table are **historical stopping-policy scores**.
They are not the corrected frontier. The remaining evaluation commands now
write separate corrected outputs and depend on corrected dense reference 266.
Pruning commands, calibration, source weights, and the strict parser are
unchanged. The [CPU collector](../../scripts/collect_gsm8k_eos_results.py)
watches completed gflow evaluations and writes independent checkpoint/raw-token
audits to [its own progress file](../../results/tangent/gsm8k/eos-corrected-v1/collector/progress.json).
It checks terminal scheduler state, complete common-128 data, source lineage,
raw-token grades, and historical parity where an old evaluation exists. It
leaves the GPU queue and the main result journal unchanged. The corrected evaluation records are:

| Method | 55% evaluation | 60% evaluation | 65% evaluation |
|---|---|---|---|
| Task SparseGPT | [96/128 · 75.00%](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s55.json), 3 truncated | [95/128 · 74.22%](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s60.json), 5 truncated | [54/128 · 42.19%](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65.json), 27 truncated |
| Local | [89/128 · 69.53%](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s55.json), 7 truncated | [72/128 · 56.25%](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60.json), 16 truncated | Stopped: 25/96 sites saved (262); evaluation 263 unmeasured |
| Downstream weight-diagonal | [98/128 · 76.56%](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s55.json), 4 truncated | [77/128 · 60.16%](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60.json), 10 truncated | Stopped before pruning (264/265) |

All use batch 32, the same first 128 question IDs, 2,048 new tokens, and the
existing 15-point screen. Corrected SparseGPT-55 **267 completed at
96/128 (75.00%)**, with 3 truncated, versus historical 95/128 (4 truncated).
The [paired comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s55-historical-comparison.json)
verifies all 128 decoded prefixes match and answer NLL is unchanged. The
[checkpoint link](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s55-checkpoint-score-link.json)
binds the score to all 96 audited source stages. Its 21.09375-point drop
from dense 123/128 fails the screen, so there is no 512-question extension.
Corrected local-55 **268 completed at 89/128 (69.53%)**, with 7 truncated,
versus historical 83/128 (19 truncated). Its
[paired comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s55-historical-comparison.json)
verifies all 128 decoded prefixes match, identical prompt and batching metadata,
zero answer-NLL difference, and no lost correct answers. The
[checkpoint link](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s55-checkpoint-score-link.json)
binds it to all 96 audited source stages. The 26.5625-point dense-relative drop
also rejects the 512-question extension. Corrected downstream-55 **269 completed
at 98/128 (76.56%)**, with 4 truncated, identical to its historical score.
Corrected SparseGPT-60 **270 completed at 95/128 (74.22%)**, with 5 truncated,
versus historical 21/128 with 99 truncated. Their
[downstream55](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s55-historical-comparison.json)
and [SparseGPT60](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s60-historical-comparison.json)
comparisons verify all 128 decoded prefixes, identical input provenance, and
zero answer-NLL differences. Both scores are bound to their 96 audited source
stages. SparseGPT-60 gains 74 correct answers and loses none. The
[rescue breakdown](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s60-rescue-breakdown.json)
shows 53 were rejected only by budget truncation, 20 had post-EOS text that
made the strict parser return no answer, and one had a wrong parsed number
after continuing beyond chat EOS despite eventually reaching model EOS.
Thus the old text-only truncation sensitivity score (74/128) underestimated
the corrected stopping result (95/128). Corrected local-60 **257 completed at 72/128 (56.25%)**, with 16 truncated.
Its [raw-token audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60.json),
[checkpoint linkage](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-checkpoint-score-link.json),
and [gold-input audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s60-generation-inputs.json)
all pass. Only the bound checkpoint hashes differ from local-55 in the
evaluation configuration, while only sparsity differs in pruning. There is
no historical-policy common-128 evaluation of this new 60% checkpoint. Its
39.84375-point dense-relative drop rejects the 512-question extension. Seven of nine
corrected grid points are now complete, all nonzero. Their dense-relative
drops reject the 512-question extensions. Corrected results are recorded in
[`eos-corrected-v1/progress.json`](../../results/tangent/gsm8k/eos-corrected-v1/progress.json).

Downstream-60 pruning **258 completed all 96 sites**. Its
[CPU checkpoint audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60-cpu.json)
verifies finite weights, matching hashes and no all-zero rows or columns. The
[full scaling check](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60-trace.json)
binds the reported trace ratio to this checkpoint and verifies every actual
zero count against per-block integer mask selection. Geometry/ambient scaling
is 1 within floating-point error. This is eligible MLP sparsity
(59.999932% of eligible weights; 32.378416% of all model weights removed).
Corrected evaluation **259 completed at 77/128 (60.16%)**, with 10 truncated.
The [raw-token grade](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60.json),
[checkpoint link](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60-checkpoint-score-link.json),
and [all 128 gold-input checks](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60-generation-inputs.json)
pass. Its 35.9375-point dense-relative drop rejects the 512-question extension.
There is no historical-policy common-128 evaluation of this new checkpoint.
Only sparsity differs from downstream-55 in pruning; only the source hashes
differ in evaluation ([immutable config check](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s60-matched-configuration.json)).

SparseGPT-65 pruning **260 completed all 96 sites**. Its
[CPU checkpoint audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65-cpu.json)
verifies finite weights, matching hashes and sparsity counts, with no all-zero
rows or columns. Its
[immutable config comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65-matched-configuration.json)
confirms that only sparsity differs from SparseGPT-55 in pruning, and only the
source hashes differ in corrected evaluation. Evaluation **261 completed at
54/128 (42.19%)**, with 27 truncated. Its
[raw-token audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65.json),
[checkpoint linkage](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65-checkpoint-score-link.json),
and [gold-input audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/sparsegpt-task-s65-generation-inputs.json)
pass. Its 53.90625-point dense-relative drop rejects the 512-question extension.
This is the first common-128 evaluation of the new checkpoint; there is no
historical-policy counterpart. Local-65 pruning **262 was intentionally
stopped with 25/96 sites saved**; evaluation 263 remains unmeasured.

Completed cells show the common-128 score; other cells list prune/evaluation
job IDs. The control pruning checkpoints completed before this bracket;
their corrected EOS evaluations remain unfinished. Each new
prune/evaluation pair follows the preceding evaluation, in ascending sparsity
and SparseGPT/local/downstream order. All jobs are exclusive with 64G GPU /
80G host caps and six-hour phase limits, using the existing guarded,
checkpoint-resumable workers. The
[immutable plan](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/plan.json),
[submission journal](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/submissions.json),
and [live queue verification](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/queue-verification.json)
record all 18 commands, IDs, dependencies, resources, and matching production
source hashes. `scripts/submit_gsm8k_sparsity_bracket.py` records the
historical sweep. For the interrupted corrected-EOS queue, use the resume
plan above with its current worker arguments and corrected dense reference. Four of nine historical-policy common-128 measurements
are complete; all three historical 55% scores are nonzero. The corrected
grid currently has seven completed points, as shown above. The
[verified checkpoint-to-score links](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/verified-links.json)
bind these scores to their exact source configs, final pruning reports, CPU
audits, and saved answer files.

SparseGPT-55 pruning **235 completed** all 96 sites. The
[independent CPU audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s55-cpu.json)
verified hashes/chains, finite weights, 54.9999237–54.9999873% zeros, no
all-zero rows/columns, norm ratios 0.9687–1.0314 relative to dense, and
first-attempt 0.01 damping at every site. Its four-question engineering check
independently rescores to 4/4 with no truncation. Matched evaluation **236
completed at 95/128 (74.22%)**, with **4 truncated**, versus dense **123/128**.
The [independent full-screen audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s55-screen.json)
finds no scoring mismatches; ignoring truncation still gives 95 correct.
The 21.875-point dense-relative drop fails the quality screen, so there is no
512-question extension. This is a nonzero accuracy point, irrespective of
that quality-screen decision. Local-55 pruning **237 completed** all 96 sites,
with configuration differing from the audited local-70 baseline only in
sparsity and unchanged production source hashes. Its
[independent CPU audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/local-s55-cpu.json)
verified hashes/chains, finite values, 54.9999237–54.9999873% zeros, no all-zero
rows/columns, norm ratios 0.9638–1.0285 relative to dense, first-attempt 0.01
damping, and geometry/ambient trace ratios equal to 1 within rounding. Its
engineering check is independently verified at 4/4, with no truncation.
Matched evaluation **238 completed at 83/128 (64.84%)**, with **19 truncated**,
versus dense 123/128 on exactly the same question IDs. The
[independent full-screen audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/local-s55-screen.json)
finds no scoring mismatches; ignoring truncation gives 85 strict correct
markers. Evaluation settings match local-70 except for source config/report
hashes. The 31.25-point dense-relative drop rejects the quality screen, so
there is no 512-question extension. This is also a nonzero 55% point.

Downstream-55 pruning **239 completed** all 96 sites. Its config differs from the audited
downstream-70 baseline only in sparsity, with unchanged source hashes,
`pullback.metric=weight_diagonal`, and full-metric refitting disabled.
The [independent CPU audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/downstream-weightdiag-s55-cpu.json)
verified hashes/chains, finite values, 54.9999237–54.9999873% zeros, no all-zero
rows/columns, norm ratios 0.9624–1.0298 relative to dense, first-attempt 0.01
damping, and engineering 4/4 with no truncation. The downstream-specific
[trace check](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/downstream-weightdiag-s55-trace.json)
verifies all 96 recorded and recomputed geometry/ambient ratios equal to 1
within rounding. This uses `achieved_geometry_to_ambient_ratio`; the generic
local-solver trace field is absent. Matched evaluation **240 completed at
98/128 (76.56%)**, with **4 truncated**, versus dense 123/128 on the same IDs.
The [independent full-screen audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/downstream-weightdiag-s55-screen.json)
finds no scoring mismatches; ignoring truncation still gives 98 correct.
Settings match downstream-70 except source config/report hashes. The
19.53125-point dense-relative drop rejects the quality screen, so there is no
512-question extension. SparseGPT-60 pruning **241 completed** all 96 sites,
changing only sparsity from its verified 55% configuration, with unchanged
source hashes. Its [independent CPU audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s60-cpu.json)
verified hashes/chains, finite values, 59.9998474–59.9999746% zeros, no all-zero
rows/columns, norm ratios 0.9544–1.0347 relative to dense, and first-attempt
0.01 damping at every site. Its engineering score is 0/4, all truncated
(one correct strict marker when truncation is ignored). Matched evaluation
**242 completed at 21/128 (16.41%)**, with 99 truncated, and failed the quality
screen. Settings match SparseGPT-55 except source config/report hashes. The
[independent screen audit](../../results/tangent/gsm8k/sparsity-bracket-repcal-v1/audits/sparsegpt-task-s60-screen.json)
finds no disagreement with the implemented parser, but removing truncation
rejection gives 74 strict correct final markers. This large dependence on
termination policy triggered diagnostic 255, described above; it prevents
interpreting the score drop as a clean mathematical-accuracy frontier.

Endpoint caveat: historical SparseGPT 50/70% uses the earlier calibration and
runtime protocol; representative local 50% uses batch 8. These endpoints are
context rather than fully matched anchors for the new nine batch-32 runs.
The representative downstream 50/70% endpoints already use batch 32.

## 70% artifact audit

**Completed corrected local-70 check:** **13/128 (10.16%)**, with **95
truncated**, versus historical **0/128**, all truncated, and corrected dense
**123/128**. Job **271 completed** the common-128 screen and rejected the
candidate under the 15-point rule; there is no 512-question extension. The
[independent raw-token audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s70.json)
finds zero scoring mismatches. All 128 decoded prefixes match the historical
outputs, and teacher-forced answer NLL differs by exactly zero in the
[paired comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s70-historical-comparison.json).
The [checkpoint-to-score link](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s70-checkpoint-score-link.json)
binds the result to all 96 previously audited tensors and the corrected dense
reference. The old literal zero is an evaluation artifact; substantial
generation damage remains.

The [error analysis](../../results/tangent/gsm8k/eos-corrected-v1/audits/local-s70-error-analysis.json)
records 11 completed wrong numeric answers and 9 completed unparseable outputs.
No rejected completed output has the correct number on an explicit final-marker
line. The 95 truncated generations all lack either EOS in their 2,048-token
streams and repeatedly emit the same eight-token sequences (median maximum
frequency 118). Examples also contain arithmetic errors such as `3 × 30 = 30`.
One additional truncated response contains a correct newline-terminated marker,
so the diagnostic score ignoring truncation is **14/128**; the unchanged primary
finished-answer score remains **13/128**. Corrected downstream-70 **272 completed at 12/128 (9.38%)**, with 80 truncated.

The full corrected downstream-70 result also passes its
[independent raw-token audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s70.json)
and [paired historical comparison](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s70-historical-comparison.json):
all 128 decoded prefixes match and answer NLL is identical. Its
[checkpoint-to-score link](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s70-checkpoint-score-link.json)
binds the score to all 96 independently audited weight-diagonal stages.
There are 30 wrong numeric completions, 6 EOS-terminated outputs without a
valid final answer, and 80 budget exhaustions. The
[error inspection](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s70-error-analysis.json)
finds no additional correct number on the rejected outputs' explicit marker
lines; ignoring truncation still gives 12/128. The source uses weight-diagonal
curvature with full-metric refitting disabled. Its
[96-stage trace check](../../results/tangent/gsm8k/eos-corrected-v1/audits/downstream-weightdiag-s70-trace.json)
confirms saved budget arithmetic equals alpha 1 within rounding; it does not
independently remeasure the complete curvature tensor.

| Corrected common-128 result | Finished and correct | Truncated | Strict correct marker, ignoring truncation |
|---|---:|---:|---:|
| Dense | 123/128 | 1 | 123/128 |
| Local 70% | 13/128 | 95 | 14/128 |
| Downstream weight-diagonal 70% | 12/128 | 80 | 12/128 |

Both historical zeros were stopping artifacts. The corrected generation
protocol still shows a large quality loss. The one-answer difference does
not establish a method advantage. A measured zero would not establish zero
population accuracy.
Both jobs completed their 128-question screens and were rejected by the
15-point rule, so neither has a 512-question extension.

The matched representative-calibration task SparseGPT-70 baseline remains
unstarted. Its planned pruning/evaluation jobs **276/277** were intentionally
cancelled and are retained in the resume plan.
The historical SparseGPT-70 endpoint used a different calibration/runtime
protocol and cannot support a controlled comparison with these geometry arms.
The new baseline changes only sparsity from the verified SparseGPT-55 recipe;
its actual config and weights still require verification when produced.

The [deployment-cast audit](../../results/tangent/gsm8k/eos-corrected-v1/audits/seventy-percent-deployment-casts.json)
verified all **192 local/downstream-70 tensors** after the BF16 cast: every
value remains finite, and every zero mask is unchanged. The saved float32
containers already contain BF16-rounded weights; both checkpoint writers
explicitly apply `candidate.to(torch.bfloat16).float()` before saving. Their
source hashes match the pruning configs. The
[reproduction record](../../results/tangent/gsm8k/eos-corrected-v1/audits/seventy-percent-deployment-casts-provenance.json)
links the exact checker and writer hashes. This excludes additional pruning or
nonfinite weights caused by deployment casting, not inference-precision effects
on task accuracy.


Corrected EOS re-evaluations of the frozen prompt-high, reasoning-high and
FP64 controls were **273/274/275**. They were intentionally cancelled before
evaluation and remain unfinished. Their complete pruning checkpoints are
reusable; only corrected EOS evaluation is needed on resume. Their historical
null results remain provisional for interpretation until these evaluations
finish under the corrected policy.

**Stopping artifact confirmed:** the numerical and calibration controls below
used the original EOS policy. That policy continued after `<|im_end|>` and
could reject a completed correct answer when the continuation reached the
token limit. The shared `gsm8k_pruning.behavior` path is also imported by
sequential local/downstream, whole-neuron, original, control, intervention,
and reevaluation drivers. Their old literal zeros require this caveat.
Corrected local-70 is 13/128 (271), and downstream-70 is 12/128 (272). The
historical comparisons below describe the old completion metric; they do not
exclude a stopping artifact or by themselves establish intrinsic math accuracy.

The observed zero is calibration dependent: matched local-70 reasoning
calibration scores **34/128 (26.56%)**, versus **0/128** for both original and
larger prompt-only calibration, with identical evaluation settings. Dense is
**123/128 (96.09%)**. This removes the literal sample zero but still leaves a
large quality loss. The FP64 solver control also scored
0/128 under the old EOS policy; corrected evaluation 275 is needed to assess
its effect on the revised score. The
[completed control comparison](../../results/tangent/gsm8k/pruning-artifact-audit-v1/controlled-comparison.json)
collects the evidence and scope limits.

| Condition | Task tokens | Correct / 128 | Truncated |
|---|---:|---:|---:|
| Dense | — | 123 | 1 |
| Local 70%, original prompts, FP32 solver | 4,096 | 0 | 128 |
| Local 70%, more prompts, FP32 solver | 32,768 | 0 | 128 |
| Local 70%, reasoning/endpoints, FP32 solver | 32,768 | 34 | 83 |
| Local 70%, original prompts, FP64 solver | 4,096 | 0 | 125 |

All pruned controls retain 4,096 frozen C4 tokens, FP32 geometry, BF16 model
weights, and identical batch-32 evaluation settings on the same 128 questions.

The [CPU audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/cpu-report.json)
verified all 96 stage hashes, upstream chains, tensor shapes, finite weights,
and dense-relative removal counts for each of four 70% candidates: task
SparseGPT, corrected local v2, downstream weight-diagonal, and representative
local. Every tensor has 69.9998856–69.9999491% zeros. No entire row or column
is zero; candidate/dense weight-norm ratios span 0.9011–1.0442. The fresh
representative local run used the requested 0.01 damping on its first
factorization attempt at every site and has geometric/ambient curvature trace
ratio 1.0. Older runs did not persist actual damping attempts. The 50%
representative local checkpoint was also checked as a positive control.

The fresh representative downstream 70% checkpoint has now also passed the
[96-tensor CPU audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/repcal-downstream-cpu-report.json):
valid hashes/chains, finite weights, 69.9998856–69.9999491% zeros, no entirely
zero rows/columns, and weight-norm ratios 0.8986–1.0391 relative to dense.
All 96 factorizations used 0.01 damping on the first attempt, and the
recorded geometry/ambient trace ratios span 1.0 to floating-point rounding.
The four-question engineering score independently rescored to 0/4 even when
truncation is ignored. Its separate matched 128-question screen subsequently
finished at 0/128, all truncated, versus dense 123/128. The
[independent screen rescore](../../results/tangent/gsm8k/pruning-artifact-audit-v1/repcal-downstream-screen-audit.json)
agrees; ignoring truncation recovers four strict correct markers. The screen
rejected the candidate, so there is no 512-question result.

Independent rescoring reproduced the saved scores. If truncation is ignored
while retaining the same strict final-answer parser, the fresh local 70%
screen contains 10/128 correct markers, versus its official 0/128. Raw outputs
also contain extensive repetition. Thus the zero includes a failure to finish;
it does not establish that every answer lacks correct arithmetic. Task
SparseGPT also collapses at 70%, so this is not specific evidence against
geometry. The corrected local, whole-neuron, input-side, and full-metric
methods have different scopes; canceled or unconverged runs are not measured
70% behavioral failures.

The focused CPU suite passed 90 tests (5 GPU skips, 1 known backend xfail),
and 16 added tests passed. The latter compare ordered OBS with explicit
trailing-system normal-equation solves at 50–99%, including rank-deficient
inputs, and check alpha-zero geometry parity and fixed-history cache alignment.
These checks do not rule out full-size GPU or calibration artifacts.

Gflow **232 completed** the full recorded-layer numerical check, using the
12,288-input block-15 down-projection without loading a language model. Its
[report](../../results/tangent/gsm8k/pruning-artifact-audit-v1/solver-precision/report.json)
compares production FP32 with a separate FP64 implementation of the same
ordered algorithm at 70%, holding inputs, initial FP32-scaled weights, mask
rule, block size, and requested relative ridge fixed. Ambient/local mask
disagreement is **5.883%/5.480% of all entries**; relative candidate differences
are **40.71%/40.80%**, also after BF16 casting. FP32/common-FP64 damped loss
ratios are **0.999784/0.999739**: the FP32 outputs are not worse on this
quadratic. Four-probe inverse-factor residuals reach 0.003879 in FP32 versus
1.09e-11 in FP64. Both used the initial 0.01 ridge rule; actual ridge values
differ by about four parts per million. This is substantial support/weight
sensitivity despite nearly equal surrogate losses, not proof that FP32 caused
the behavioral collapse. It does **not** close the model-wide numerical
artifact question. Five reference tests plus high-sparsity checks passed
(19 tests). This remains ordered OBS, not an exact coupled geometric refit.

Consequently, matched model-wide local-70 FP64 pruning **233 completed**;
its existing dense-matched 128-to-512 evaluation **234 completed**. The adapter
`scripts/run_gsm8k_fp64_control.py` retains the original 8,192-token calibration,
FP32 geometry/scaling, FP32 unscale boundary, BF16 model/checkpoint cast,
sequential chain, and scoring. It varies curvature accumulation,
factorization, and ordered compensation precision together. Concrete atlases
are refit on each candidate's own upstream state. It tags immutable configs
before hashing, with separate outputs and hashes for the adapter/reference.
Three new adapter tests plus the five reference tests passed (8 tests).
The [submission journal](../../results/tangent/gsm8k/pruning-artifact-audit-v1/fp64-control-submissions.json)
records dependencies, commands, and 64G GPU / 80G host caps; pruning followed
reasoning control 230. No active production pruning source was changed.

All 96 FP64 stages used the original 8,192-token calibration, with maximum
inverse-factor probe residual **9.63e-12** and first-attempt 0.01 damping.
The [full CPU checkpoint audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-fp64-cpu-report.json)
verified hashes/chains, finite weights, 69.9998856–69.9999491% zeros, no
all-zero rows/columns, norm ratios 0.8996–1.0367 relative to dense, and
geometry/ambient trace ratios equal to 1 within rounding. The engineering
check independently rescores to 0/4, all truncated (one correct strict marker
if truncation is ignored). The matched screen finished at **0/128**, with
**125 truncated**, and rejected the candidate; there is no 512-question result.
The [independent full-screen audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-fp64-screen-audit.json)
finds no scoring mismatches (nine strict correct markers if truncation is
ignored). Evaluation settings and IDs match all other local controls; the
shared pruning configuration is identical to the original FP32 baseline.

The [paired persisted-weight comparison](../../results/tangent/gsm8k/pruning-artifact-audit-v1/fp64-weight-comparison.json)
also verifies both runs' stage-file hashes. Their masks differ on **13.67%**
of all eligible MLP entries, and the FP64 chain differs by **61.22%** of the
original FP32 baseline's eligible-MLP Frobenius norm after BF16 casting. Later layers include
changed upstream activations and refitted atlases, so this is a chained
counterfactual difference, not a per-layer isolated numerical-error estimate.
Despite these large changes and improved factorization precision, the old-EOS
screen remained at zero. That score is confounded by the stopping bug and
does not settle precision's effect on corrected accuracy. Corrected evaluation
275 is pending. The control retains FP32 geometry and BF16 model execution;
it does not test every precision choice or an exact coupled geometric solver.

An additional [recorded-layer frontier audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/geometry-frontier/report.json)
compared ordered, congruence-exact, row-exact, and full-coupled-exact refits on
the same masks and induced ridge at 50%, 70%, and 90%. Across three fixed
8-by-16 crops and alpha in {0.3, 1, 3}, the ordered/full-exact loss ratio is
1.0313–1.4157 at 50% and 1.0322–1.3765 at 70%. The paired ratio changes by
0.906–1.093 times between 50% and 70%, with maximum free-coordinate exact
stationarity residual below 2.7e-18 over all 27 comparisons. This does not
show a special approximation-error cliff at 70% on these crops. It does not
validate the full-size metric, mask selection, or model-wide behavior.

All these entrywise runs share only 4,096 short C4 tokens plus 4,096 sampled
task-prompt tokens per site; no answer/reasoning continuations are calibrated.
The 12,288-input down-projection's undamped empirical curvature is necessarily
rank-deficient. Representative questions improve fitting-population coverage
but leave this budget and prompt-only mismatch in place. Rank deficiency is
not itself a solver defect; the controls below do not isolate a behavioral
effect of rank deficiency alone.

The [prepared reasoning control](../../results/tangent/gsm8k/pruning-artifact-audit-v1/reasoning-calibration.json)
contains the original training solutions for the same 640 fit/pruning parents,
with calculator annotations removed, the existing final-answer marker, and
an explicit end-of-answer EOS. It provides 61,757 response tokens; complete
prompt-plus-response contexts are at most 407 tokens. Preparation verifies
the training Arrow/source-row hashes, final answers, and exclusion of all
512 reserved evaluation parents and 64 independent engineering questions.
It neither loads a model nor supplies a behavioral result. Three additional
CPU tests cover source identity, split separation, EOS retention, and refusal
to silently truncate a solution; the focused follow-up suite passed 28 tests.

The [calibration-control plan](../../results/tangent/gsm8k/pruning-artifact-audit-v1/calibration-control-plan.json)
is now executable through `scripts/run_gsm8k_calibration_control.py`. It retains
the existing sequential model/solver/checkpoint loop and hashes the control
configuration before that loop creates any stage. Two model-wide
70% local runs each use 32,768 task tokens plus the same 4,096 frozen C4
tokens. One samples prompt tokens; the other samples reasoning tokens while
retaining every prompt readout and every pre-EOS answer endpoint. The atlas
fitting/readout rule and prescribed alpha remain fixed; actual geometry is
refit on each candidate's own upstream state as before. The prompt arm tests
coverage and mixture sensitivity against the existing 4,096-task-token baseline,
and the equal-budget reasoning arm tests token-population/endpoint sensitivity. Twelve focused
CPU tests passed, including causal prompt-readout equivalence through both
Qwen attention types, equal budgets, padding exclusion, endpoint retention,
and an unchanged legacy collector. Following completed diagnostic 226, these
GPU controls were submitted as **228/229** (larger prompt-only pruning /
evaluation) and **230/231** (equal-budget reasoning-inclusive pruning /
evaluation). Original experiment 227/225 completed before the new calibration
pruning jobs. The [submission journal](../../results/tangent/gsm8k/pruning-artifact-audit-v1/calibration-control-submissions.json)
records exact commands, resource caps, output namespaces, and dependencies.
The immutable prepared plan records its preparation-time `submitted=false`;
the submission journal and live scheduler are authoritative for execution.

Prompt control **228 completed** all 96 sites; its engineering check is 0/4,
all truncated (two correct strict markers if truncation is ignored). Its
[independent CPU audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-prompt-cpu-report.json)
verified all tensor hashes/chains and finite values, 69.9998856–69.9999491%
zeros, no all-zero rows/columns, weight-norm ratios 0.8851–0.9894 relative to
dense, first-attempt 0.01 damping at every site, and geometry/ambient trace
ratios equal to 1 within rounding. Matched evaluation **229 completed at
0/128, all truncated**, versus dense 123/128, and rejected the candidate.
The [independent screen audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-prompt-screen-audit.json)
finds no scoring mismatches; ignoring truncation recovers four strict correct
markers. Thus increased prompt coverage plus the changed task/C4 mixture did
not recover behavior under this recipe. There is no 512-question result.
Equal-budget reasoning-inclusive pruning **230 completed** all 96 sites with
the same production source hashes. Its config differs from prompt-high only
in the declared calibration-control `mode`. The engineering check is **1/4**,
with two truncated answers; ignoring truncation gives two strict correct
markers. This small check is not an accuracy estimate. The
[independent CPU audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-reasoning-cpu-report.json)
verified all hashes/chains, finite weights, 69.9998856–69.9999491% zeros, no
all-zero rows/columns, weight-norm ratios 0.8681–0.9971 relative to dense,
first-attempt 0.01 damping at every site, and geometry/ambient trace ratios
equal to 1 within rounding. Matched evaluation **231 completed at 34/128
(26.56%)**, with **83 truncated**, versus dense **123/128 (96.09%)** with one
truncated. It still failed the 15-point quality screen, so there is no
512-question result. The
[independent full-screen audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/control-reasoning-screen-audit.json)
finds no rescoring mismatches; ignoring truncation gives 42 strict correct
markers. All three local-70 candidates use exactly the same 128 question IDs
and evaluation configs except for source config/report hashes. The original
baseline's shared pruning fields are identical; prompt-high/reasoning-high
pruning configs differ only in calibration mode. Training calibration parents
remain disjoint from these evaluation questions.

Under the historical EOS policy, equal-budget reasoning calibration and
forced answer endpoints recovered nonzero finished-answer accuracy.
The literal zero cannot be treated as an intrinsic sparsity boundary. This
does not restore dense quality or establish a result for the other methods.
The completed FP64 control also scored zero under the historical EOS policy
with original calibration. The separate 55/60/65% bracket retains the original
prompt-only pruning calibration; its current evaluations use the corrected
EOS policy described above.

The [comparison-design annotation](../../results/tangent/gsm8k/pruning-artifact-audit-v1/calibration-comparison-design.json)
makes the attribution limit explicit: holding C4 at 4,096 while increasing
task tokens from 4,096 to 32,768 changes the task fraction from 1/2 to 8/9.
Prompt-high versus baseline therefore does not isolate sample count alone.
Reasoning-high versus prompt-high holds both counts fixed, but deliberately
changes token population and forces the pre-EOS endpoints. Geometry follows
the same fitting/readout/normalization rule; concrete atlases and absolute
gamma are recomputed on each candidate's current upstream activations.

Gflow **226** completed a bounded frozen-checkpoint diagnostic comparing
dense, representative local 50%, task SparseGPT 70%, and representative local
70% in 8 minutes 5 seconds. It scored identical dense reasoning continuations,
compared cached and uncached logits on identical token histories, and generated
short singleton, batch-two, and uncached continuations. Results are under
`results/tangent/gsm8k/pruning-artifact-audit-v1/gpu/`; the independently checked
[summary](../../results/tangent/gsm8k/pruning-artifact-audit-v1/gpu-summary.json)
records exact token equivalence and scope limits.

Every state agreed on all 65 tested fixed-history next-token choices for one
question; mean cached/uncached vocabulary KL was 8.85e-5 dense, 9.78e-5 local
50%, 4.04e-4 ambient 70%, and 3.12e-4 local 70%. Logits were not identical.
The local 70% singleton continuations were exactly identical across cache
modes on both questions. On question 5723, its early LaTeX loop repeated the
same token 8-gram 17 times with caching, 17 without caching, and 18 in batch
two; dense and local 50% had at most two repeats on the tested continuations.
Caching or batching alone therefore cannot explain that reproduced loop.
Other freely generated 50%/ambient-70 continuations can differ across modes;
this diagnostic does not establish general bitwise inference equivalence.

On the same two dense-generated reasoning targets, token-weighted mean NLL
was 0.06525 dense, 0.14512 local 50%, 0.54218 ambient 70%, and 0.49999 local
70%. This confirms deterioration on fixed targets without free decoding.
Two questions, 128 generated tokens, and 65 fixed-history positions are not a
model-wide accuracy estimate or a test of all later positions. No production
pruning or scoring code was changed. The completed calibration controls above
demonstrate protocol dependence of the observed zero; this bounded decoding
diagnostic alone cannot establish an intrinsic sparsity frontier.

Diagnostic 226 was expedited with a checkpoint-safe scheduler reorder:
the current code hashes and the latest completed stage of pruning job 224
were verified, its exact worker command was resubmitted as **227** after 226,
and evaluation **225** was retargeted to dependencies **221 and 227** before
224 was canceled through gflow. All 25 completed stages remain on disk. The
old systemd worker was confirmed inactive before 226 was released; no second
full model was loaded concurrently. Diagnostic 226 succeeded and 227 resumed,
advancing beyond the preserved stages. The [reorder journal](../../results/tangent/gsm8k/pruning-artifact-audit-v1/gflow-reorder.json)
records the original and replacement commands, dependencies, and recovery.

For context, the separate standard C4-calibrated family sweep records Qwen
9B WikiText2 perplexity 8.646 dense and 26.256 at 70%; PTB is 13.984 dense
and 42.564 at 70%. Those runs use a different pruning/evaluation protocol
and are not a matched GSM8K control. They also do not support interpreting
zero bounded-generation accuracy as complete loss of language modeling.

The [broader historical-sweep audit](../../results/tangent/gsm8k/pruning-artifact-audit-v1/broader-sweeps.json)
also checks all 11 archived cross-family tangent sweeps. At 70%, the legacy
score-only heuristic has lower WikiText2 perplexity than its paired ambient
baseline on five models and higher perplexity on six (changes span -3.21%
to +5.54%). Every 70% arm exited successfully; these are losses of quality,
not worker crashes. Saved paired commands differ only in tangent settings,
and all 139 recorded likelihood/perplexity/token triples satisfy the expected
arithmetic within 6.3e-8 in log perplexity. This does not verify archived
runtime/source identity, which those JSON reports do not record.

The separate Qwen 0.8B method-sweep results were committed with the Nov 6
rank-normalized implementation (`a97bc8b`), before the Nov 8 curvature-trace
correction (`c368013`). In particular, its `tangent_rescale` and
`tangent_heuristic` points must not be cited as corrected-v2 measurements.
The initial audit found that its resume path did not check implementation
versions. This is now fixed: the default is
`results/tangent/qwen08-method-sweep-curvature-trace-v2.json`, with separate
per-namespace logs. `--output` permits another fresh namespace, but a legacy
or mismatched file is rejected before pruning. Cached points bind source
hashes, the actual child interpreter/package versions, and the paired-input
bank; these are rechecked before and after each point. Twelve focused tests
passed, and the actual historical JSON was independently rejected without
changing its bytes. [Guard evidence](../../results/tangent/gsm8k/pruning-artifact-audit-v1/method-sweep-resume-guard.json).
This guard does not pin otherwise unversioned Hub references or establish
identical GPU dispatch. Raw measurements are retained and no corrected Qwen
0.8B measurements were invented. The current routing and method-family
CPU contracts passed 59 tests; that does not retroactively validate historical
execution or turn the legacy score-only heuristic into the full geometric
objective.

## Earlier local work

**Corrected local 50% · job 205:** completed the 512-question original validation at batch size 8: **436/512 (85.16%)**, with 7 truncated answers. It reused 96 cached pruning stages; the earlier batch-one answers were not mixed into this run. This is 11.33 percentage points below the earlier dense result, subject to the environment difference noted below.

**Representative-calibration single-block development:** the block 15 two-sided atlas smoke completed at 30% eligible sparsity with surrogate refit; behavior evaluation was skipped. The [full-metric trace](../../results/tangent/gsm8k/qwen-9b-full-metric-trace-v1/config.json) produced atlas and curvature artifacts but has no completed report. These single-block runs do not establish model-wide retention.

## Full-model whole-neuron pruning: stopped

The screen is restored to **128 questions** with a 15-point dense-relative cutoff. Passing candidates continue to 512 in the same evaluation job, reusing all screen answers. Batch size 8; the 128-question dense reference is **123/128**. Each cell lists pruning / evaluation job IDs and current outcome.

| Eligible MLP sparsity | Ambient | Downstream weight-diagonal |
|---:|---:|---:|
| 30% | **177 / 178: 42/128, failed screen** | **179 / 180: 69/128, failed screen** |
| 50% | **181 / 182: 0/128, failed screen** | **183 / 184: 0/128, failed screen** |
| 60% | **185 / 186: 0/128, failed screen** | **187 canceled: 30/32 blocks** / 188 canceled |
| 70% | 189 / 190 canceled | 191 / 192 canceled |
| 80% | 193 / 194 canceled | 195 / 196 canceled |
| 90% | 197 / 198 canceled | 199 / 200 canceled |
| 99% | 201 / 202 canceled | 203 / 204 canceled |

Pruning stages and evaluation answers are cached for resumption. Evaluation jobs depend on their corresponding pruning jobs; a rejected screen stops at 128. Each job has a 24-hour limit. At 30%, ambient scored **42/128 (32.81%)** with 49 truncated; downstream scored **69/128 (53.91%)** with 21 truncated. Both failed against dense **123/128** and stopped before 512. Both 50% candidates and the 60% ambient candidate scored 0/128; all answers were truncated. The 60% downstream pruning was canceled after 30 of 32 blocks; no evaluation was run. The remaining higher-sparsity jobs were canceled. The single-block pilot below is not evidence of full-model retention.

## Batch-eight: model-wide decision results

All candidates use the same 128 questions and dense reference. A 64-question screen allows a drop of up to 15 percentage points; passing candidates continue to 128 without repeating work. Sparsity is for eligible MLP weights.

| Candidate | First 64 | Full 128 | Outcome |
|---|---:|---:|---|
| Dense · 165 | 62/64 | **123/128 · 96.09%** | Complete |
| Downstream weight-diagonal 30% · 174 | 62/64 | **124/128 · 96.88%** | Complete; 1 truncated |
| Downstream weight-diagonal 50% · 175 | 54/64 | **107/128 · 83.59%** | Complete; 3 truncated |
| Downstream weight-diagonal 70% · 176 | 0/64 | — | Failed screen; all 64 truncated |

The 50% result is 12.50 percentage points below dense on 128 questions. The screen is a triage rule, not a statistical retention guarantee. Job 176 completed successfully but its candidate failed the screen.

## Batch-eight: single-block pilot

Block 15 only, at 30% eligible MLP sparsity; these are not model-wide pruning results. All arms passed the screen and finished 128 questions.

| Pilot method | First 64 | Full 128 | Truncated |
|---|---:|---:|---:|
| Dense reference · 165 | 62/64 | 123/128 | 0 |
| Entrywise ambient · 166/167 | 62/64 | 123/128 | 2 |
| Entrywise downstream weight-diagonal · 168/169 | 62/64 | 123/128 | 0 |
| Whole-neuron ambient · 170/171 | 62/64 | 124/128 | 1 |
| Whole-neuron downstream weight-diagonal · 172/173 | 61/64 | 122/128 | 0 |

These small differences on one block do not establish a method advantage. [Batching investigation](GPU-BATCHING-INVESTIGATION.md).

## Completed: original 512-problem validation

| Eligible MLP sparsity | Whole model removed | Task SparseGPT | Corrected local | Downstream |
|---:|---:|---:|---:|---:|
| **Dense** | 0% | **494/512 · 96.48%** | — | — |
| **30%** | 16.19% | [493/512 · 96.29%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s30-seed0-v1-original512/report.json) | [495/512 · 96.68%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s30-seed0-v2-original512/report.json) | [494/512 · 96.48%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-downstream-s30-seed0-v1-original512/report.json) |
| **50%** | 26.98% | [445/512 · 86.91%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s50-seed0-v1-original512/report.json) | [436/512 · 85.16%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s50-seed0-v2-original512-batch8-v1/report.json) | [416/512 · 81.25%](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-downstream-s50-seed0-v1-original512/report.json) |

[Dense report](../../results/tangent/gsm8k/qwen-9b-gsm8k-original-dense-512-v1/report.json). Corrected local and downstream used the GB10 environment; dense and task SparseGPT used the earlier default environment. Small differences cannot establish a method advantage. Earlier **local v1 results are invalid** after a normalization fix; they are excluded here. [Why?](LOCAL-PRUNING-VERIFICATION.md)

## Screening results: earlier 128-problem subset

| Method | Eligible MLP sparsity | Screen correct | Dense on same 128 | Outcome |
|---|---:|---:|---:|---|
| Task SparseGPT | 70%, 80%, 90%, 99% | 0/128 at each | 125/128 | Failed screen |
| Downstream | 70% | 8/128 | 125/128 | Failed screen |
| Downstream | 80% | 0/128 | 125/128 | Failed screen |
| Corrected local | 30% | 127/128 | 125/128 | Passed; full result above |

The screen stopped candidates more than 15 percentage points below dense; it is a triage rule, not a retention guarantee. Full or partial answers are retained. These 128 IDs differ from the new decision subset below.

## Earlier batch-one 64→128 decision results

Same fixed subset and GB10 environment; the 64-problem screen permits up to a 15-point drop versus matched dense. Passing candidates continue to 128 without repeating answers. These are exploratory validation, not retention claims.

| Candidate | First 64 | Full 128 | Status |
|---|---:|---:|---|
| Dense · 143 | 62/64 | **123/128 · 96.09%** | Complete |
| Corrected local 50% · 156 | 53/64 | **105/128 · 82.03%** | Passed screen; complete |
| Corrected local 70% · 157 | 0/64 | — | Failed screen; 63/64 truncated |
| Corrected local 80% · 158 | **0/10** | — | Cancelled; partial, all 10 truncated |
| Corrected local 90% / 99% · 159/160 | — | — | Cancelled at user request |
| Entrywise ambient pilot, block 15 · 145 | 63/64 | **124/128 · 96.88%** | Complete; one truncated |

The local 50% observed drop is **14.06 percentage points** versus dense on all 128; the screen is a computational triage rule. The entrywise pilot prunes one block and is not a model-wide comparison. These IDs differ from the earlier 512-problem results above.

## Other work

| Work | Status |
|---|---|
| Corrected local 30% · 118/119 | Complete: 495/512 on the earlier original-validation order |
| Corrected local 50%, 70%, 80%, 90%, 99% pruning · 120/122/124/126/128 | Complete; 70%–99% each scored 0/32 on controlled development |
| Downstream weight-diagonal smoke · 136 | **Complete:** 4/4 controlled examples; integration check |
| Weight-diagonal 30% · 137 | **Pruning complete:** 31/32 controlled development; decision evaluation complete: 124/128 |
| Weight-diagonal 50% · 139 | **Pruning complete:** 28/32 controlled development; decision evaluation complete: 107/128 |
| Weight-diagonal 70% · 141 | **Pruning complete:** 0/32 controlled development, all truncated; decision screen failed: 0/64 |
| Entrywise vs. whole-neuron pilot | Batch-eight jobs 166–173 complete; local arms remain canceled |

## Reference

- [Experiment results by technique](GSM8K-EXPERIMENT-SUMMARY.md) — completed results, screens, and pilot scope.

- [Detailed history and earlier diagnostics](GSM8K-EXPERIMENT-HISTORY.md) — preserved snapshot; historical queue states may be stale.
- [Experiment plan](GSM8K-EXPERIMENT-PLAN.md) · [Local correction](LOCAL-PRUNING-VERIFICATION.md) · [Method definitions](tangent-pruning-methods.md).
- Pruned weights and per-answer outputs are cached locally. GPU experiments are scheduled through gflow. The default time limit is 24 hours. The acceptable dense-relative accuracy drop for a final retention claim remains unset.
