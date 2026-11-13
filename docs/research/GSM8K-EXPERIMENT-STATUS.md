# GSM8K experiment status

## Queue update (2026-11-12)

Downstream 90% and 99% jobs 113–116 were cancelled at the user's request. Downstream 80% job 112 completed its screen with 0/128 correct versus dense 125/128, so it did not continue to all 512. Corrected local smoke job 117 was prioritized and started; local candidate and evaluation jobs 118–129 have priority 100 and remain gated by that smoke and their candidate dependencies. No corrected local outcome is available yet.

## Sum-then-square downstream comparisons queued

Jobs **136–142** add the versioned `downstream_weightdiag` arm. Job 136 is the two-block/six-projection smoke gate; 137/139/141 rerun completed 30/50/70% eligible-MLP candidates; 138/140/142 evaluate their candidates with dense reference 64. Historical 30/50% comparisons have 512-problem results; the 70% candidate has completed 32-problem development results. These comparison evaluations request the full 512-problem protocol, without the newer screen-128 early stop, to match completed comparisons. Existing jobs and historical artifacts are preserved.

The new accumulator sums `g.T @ X` before squaring, separately for every sequence/probe, then averages probes and sums independent examples/weighted anchors. Anchor sites, factor-local atlas, eight probes, 32 calibration problems, alpha 1 and ambient budgets are retained. Its own supported trace controls normalization. The geometric-arm guard runs after support filtering and diagnostics report raw/excluded/supported mass and achieved ratio. Row-congruence compensation remains approximate.

The plan and journal are `downstream-weightdiag-plan-v1.json` and `downstream-weightdiag-submissions-v1.json`; `scripts/submit_gsm8k_weightdiag.py` submits idempotently with GB10 Python, exclusive 64G GPU/80G host allocations, existing memory guards, 24-hour limits and smoke/candidate/dense dependencies. Original comparison config/report hashes are recorded in the plan.

GB10 CPU verification: **46 passed, 1 expected failure**, including cancellation, reinforcement, ranking reversal, sequence independence, nonlinear weight-Jacobian diagonals and a full-quadratic fixed-mask reference. The expected failure is the existing optional-backend dispatch defect. Backend helper policy is retained and its source hash/package versions are recorded; these are not verified reference-only PyTorch runs. GPU smoke and behavioral results are pending.

## 128-problem screening (2026-11-11 Central)

The user selected an exploratory promotion rule: evaluate the first 128 problems of the immutable original-validation manifest and continue to all 512 only if observed accuracy is at most 15 percentage points below dense on those same 128. Dense answered 125/128 correctly, so a candidate needs at least 106/128 to proceed. The 128 results are saved in the same `answers.jsonl` used by the full evaluation; a passing run continues in the same process and skips completed answers. A rejected screen writes `screen.json`, retains all answers, and does not claim a completed 512-problem report. This is a computational triage rule, not a confirmatory retention criterion; selected and completed 512-problem results remain validation results.

Existing saved answers already show failed screens for task SparseGPT 70%, 80%, and 90% (0/128 each) and downstream 70% (8/128). Their queued full-set retries 131–134 were cancelled after recording the screen decisions; all earlier partial answers remain on disk. The task-SparseGPT 99% continuation is job 135, replacing job 130 and reusing its saved answers. The queued downstream evaluations 112, 114, 116 and corrected local evaluations 119, 121, 123, 125, 127, 129 have screening enabled. Submission journals document original commands; inspect gflow for the updated commands. The existing dense reference used the default environment, while new candidates use GB10, so a cross-environment comparison limitation remains.

## Runtime update (2026-11-11 Central)

All currently queued GSM8K jobs now have 24-hour gflow limits. The guarded worker systemd cap and GSM8K submission defaults were also raised to 24 hours. The 99% task-SparseGPT evaluation was initially requeued as job 130 with 24-hour gflow and systemd limits; it has since been replaced by screened job 135. Timed-out validations were initially resubmitted as jobs 131–134, then cancelled after their saved answers failed the 128-problem screen above. Their original job IDs and submission journals remain historical records. Progress is saved per answer; retries reload the model, candidate chain, and language diagnostics.

Runtime audit: dense and task-SparseGPT 30% validation averaged 33 seconds per answer (about 364–369 generated tokens), while task-SparseGPT 70% and 80% averaged 183 seconds per answer and all saved outputs exhausted the 2,048-token budget. Pruned site weights are already checkpointed; generation is the major bottleneck at high sparsity. At roughly 183 seconds each, 512 such answers would take about 26 hours plus setup, so even a 24-hour allocation may need another resume. Batching and a predeclared screening subset are candidates for faster feedback; shorter generation budgets change the measurement protocol and require separate results.

Correction after this snapshot: the rank-normalized local measurements below
are invalid for assessing corrected local pruning. All nine remaining legacy
local jobs (80, 84, 85, 101, 102, 105, 106, 109, 110) were cancelled. Local
pruning now uses curvature-trace normalization unconditionally and requires a
fresh corrected smoke and candidates. See [verification](LOCAL-PRUNING-VERIFICATION.md)
and the [invalidation record](../../results/tangent/gsm8k/local-legacy-invalidated-v1.json).
The tables below preserve the earlier operational snapshot, not current local
results or queue state.

Corrected local jobs have now been submitted in fresh v2 directories at the
matching 30%, 50%, 70%, 80%, 90%, and 99% eligible MLP sparsities. Smoke job
117 gates pruning jobs 118, 120, 122, 124, 126, and 128; their respective
512-problem original-validation jobs are 119, 121, 123, 125, 127, and 129.
The runs use alpha 1, seed 0, the GB10 environment, all gate/up/down MLP
projections, and the same 2,048-token generation budget. Ten percent remains
skipped, matching the cancelled SparseGPT/downstream experiments. See the
[corrected plan](../../results/tangent/gsm8k/local-retention-sweep-plan-v2.json)
and [submission journal](../../results/tangent/gsm8k/local-retention-submissions-v2.json)
for exact commands, hashes, resource caps, and scheduler dependencies.

Snapshot: 2026-11-08, approximately 4:43 p.m. Central. Scheduler states are a snapshot, not live status; use `gqueue` and `gjob show <id>` to refresh. The [experiment plan](GSM8K-EXPERIMENT-PLAN.md) describes the intended design.

The practical question is how many weights can be removed while keeping accuracy near dense. The acceptable accuracy drop has **not** been chosen. The experiments below use Qwen3.5-9B and development/validation data; none is a final GSM8K test result. Eligible MLP sparsity and whole-model weights removed have different denominators.

## Completed: original GSM8K validation

These evaluations use the same 512 reserved original training problems and a 2,048-token generation budget. They are validation for selecting sparsity, not the official test set. Reports include per-problem answers and a small eight-window WikiText2 diagnostic.

| Candidate | Pruning job / evaluation job | Correct | Accuracy | Truncated | Whole-model weights removed | Report |
|---|---|---:|---:|---:|---:|---|
| Dense | — / 64 | 494/512 | 96.48% | 4 | 0% | [dense](../../results/tangent/gsm8k/qwen-9b-gsm8k-original-dense-512-v1/report.json) |
| Task-calibrated SparseGPT, 30% eligible MLP sparsity | 69 / 70 | 493/512 | 96.29% | 5 | 16.19% (1,449,549,824 weights) | [30% candidate](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s30-seed0-v1-original512/report.json) |
| Task-calibrated SparseGPT, 50% eligible MLP sparsity | 73 / 74 | 445/512 | 86.91% | 8 | 26.98% (2,415,919,104 weights) | [50% candidate](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-sparsegpt-task-s50-seed0-v1-original512/report.json) |

The observed accuracy difference at 30% is -0.195 percentage points; at 50% it is -9.57 points. These validation points do not establish retention at a specified tolerance, identify the maximum safe sparsity, or demonstrate a geometric-method advantage. The 30% candidate's eight-window language NLL is 0.0384 nats/token higher than dense; that small diagnostic is not a final language-quality assessment.

## Completed: controlled development and diagnostics

| Experiment and scope | What ran | Outcome and interpretation |
|---|---|---|
| Dense pilot, job 34 | 200 training-only development problems | 181/200 correct; 16 outputs truncated. [Report](../../results/tangent/gsm8k/qwen-9b-gsm8k-dense-pilot-v2/report.json). |
| Capture and geometry, jobs 35, 39–41 | Block 15 MLP output capture and three factor-atlas seeds | Five of eight blocks unused/unsupported per fit; active factors had supported charts. This alone does not establish semantic manifold recovery. [Seed 0 report](../../results/tangent/gsm8k/geometry-seed0-v2/report.json). |
| One-layer comparison, jobs 42–54 | 13 arms, one `mlp.down_proj` at 50% eligible sparsity, 32 controlled validation examples, 512-token budget | Dense 28/32; task SparseGPT 29/32; local charts 27/32; fixed factor spans 30/32. All failures were truncations. Small sample and budget confound ranking. [Paired summary](../../results/tangent/gsm8k/development-comparison.json). |
| Longer-budget re-evaluation, jobs 59–61 | Same 32 examples, dense/task SparseGPT/local, 2,048 tokens | All three 32/32, zero truncations. Each pruned candidate removed only 0.2811% of whole-model parameters. No evidence of a method advantage or model-wide compression limit. [Dense](../../results/tangent/gsm8k/qwen-9b-gsm8k-dense-a0-s0-v1-budget2048/report.json), [task](../../results/tangent/gsm8k/qwen-9b-gsm8k-sparsegpt-task-a0-s0-v1-budget2048/report.json), [local](../../results/tangent/gsm8k/qwen-9b-gsm8k-local-a1-s0-v1-budget2048/report.json). |
| Controls, jobs 57–58 | Whole-activation charts and empirical-Fisher control, 32 examples, 512 tokens | 28/32 and 29/32 respectively; not a ranking against the 2,048-token evaluations. [Whole-chart](../../results/tangent/gsm8k/qwen-9b-gsm8k-whole-chart-a1-s0-v1/report.json), [Fisher](../../results/tangent/gsm8k/qwen-9b-gsm8k-fisher-a1-s0-v1/report.json). |
| Stage 2, jobs 55–56 | Answer gradients and matched high/low geometry interventions | Completed 32 group conditions. High-minus-low answer-NLL contrast +0.00133 without compensation and -0.00161 with OBS; exploratory intervals include zero. Generated-accuracy differences were zero; this does not establish equivalence. [Matched analysis](../../results/tangent/gsm8k/stage2-v1/matched-analysis.json), [prediction analysis](../../results/tangent/gsm8k/stage2-v1/prediction-analysis.json). |
| Exactness audit | Nine small matrix crops, exact fixed-mask reference | Approximate-to-exact objective ratios 1.04–2.11; approximation error limits interpretation of negative geometry results. [Report](../../results/tangent/gsm8k/exactness-v1/report.json). |
| Sequential smoke, jobs 62–63 | Two blocks, six MLP projections, 50% eligible sparsity, four examples | Both task and local paths 4/4; 1.6864% whole-model weights removed. Integration checks only. [Task](../../results/tangent/gsm8k/qwen-9b-gsm8k-sequential-task-smoke-v1/report.json), [local](../../results/tangent/gsm8k/qwen-9b-gsm8k-sequential-local-smoke-v1/report.json). |

## Running now: local 50% validation

Job 97 is evaluating the completed local-chart 50%-eligible-sparsity candidate on the 512 original validation problems in the `gb10` environment. At this snapshot, **380 answers were saved, 322 correct, with 3 truncated**. On those same 380 problems, dense was correct on 367 and task-calibrated SparseGPT at 50% on 328. These are incomplete counts, not final scores; the local run uses a different software/kernel environment from the existing baseline runs, as noted below. The per-answer output is [answers.jsonl](../../results/tangent/gsm8k/qwen-9b-gsm8k-mlp-local-s50-seed0-v1-original512/answers.jsonl).

Job 98 (task SparseGPT 70% pruning) depends on successful completion of job 97; job 78 evaluates its candidate. The other 30%–99% candidates and evaluations remain queued. Downstream pruning is gated on smoke job 86. Job 97 is the sole running gflow job at this snapshot.

## Added high-sparsity sweep

On 2026-11-08, 80%, 90%, and 99% eligible MLP sparsity were submitted for task-calibrated SparseGPT, local charts, and downstream tangents. Each candidate has its own dependent 512-problem original validation. Jobs use the GB10 environment and exclusive guarded gflow allocations. These are queued experiments, not completed results; whole-model removal will be measured separately.

| Method | 80% prune / evaluation | 90% prune / evaluation | 99% prune / evaluation |
|---|---|---|---|
| Task SparseGPT | 99 / 100 | 103 / 104 | 107 / 108 |
| Local charts | 101 / 102 | 105 / 106 | 109 / 110 |
| Downstream | 111 / 112 | 113 / 114 | 115 / 116 |

Task and local pruning depend on their existing smoke jobs 62 and 63. Downstream pruning depends on smoke job 86; each evaluation depends on its candidate and dense reference 64. All workers check manifest provenance and candidate hashes; downstream workers also verify the smoke artifacts. The immutable [MLP plan](../../results/tangent/gsm8k/mlp-high-sparsity-plan-v1.json), [downstream plan](../../results/tangent/gsm8k/downstream-high-sparsity-plan-v1.json), and [submission journal](../../results/tangent/gsm8k/high-sparsity-submissions-v1.json) record the scope, hashes, and exact commands. The new tasks use the existing 512-problem protocol and cross-environment dense-reference limitation described below.

## Model-wide MLP sweep: status by candidate

All candidates prune gate/up/down MLP projections across all blocks. The development scores below are on 32 controlled examples; they must not be confused with the 512 original-problem evaluation above. Candidate completion and evaluation completion are separate phases. A completed pruning report includes a development evaluation, but is not a completed original-validation result.

| Method / eligible sparsity | Pruning status | 32-example development result | Original 512 validation status |
|---|---|---|---|
| Task SparseGPT 10% | Complete, job 65; 5.40% whole-model removed | 32/32, 0 truncated | Job 66 failed; retry 81 cancelled at user request (10% skipped) |
| Task SparseGPT 30% | Complete, job 69; 16.19% whole-model removed | 32/32, 0 truncated | Complete, job 70: 493/512 |
| Task SparseGPT 50% | Complete, job 73; 26.98% whole-model removed | 30/32, 2 truncated | Complete, job 74: 445/512 |
| Task SparseGPT 70% | Job 77 stopped; job 98 queued after 97, starts afresh in GB10 | Pending | Job 78 queued, depends on 98 |
| Local charts 10% | Job 67 failed; retry 82 cancelled at user request (10% skipped) | Pending | Job 68 cancelled after failure; retry 83 cancelled at user request |
| Local charts 30% | Job 71 failed, partial report only; retry 84 queued | Pending | Job 72 cancelled after failure; retry 85 queued, depends on 84 |
| Local charts 50% | Complete, job 75; 26.98% whole-model removed | 23/32, 2 truncated | Job 76 stopped for benchmark at user request; job 97 running in GB10 |
| Local charts 70% | Job 79 queued | Pending | Job 80 queued, depends on 79 |
| Downstream tangent 10% | Pruning job 87 cancelled at user request (10% skipped) | Pending | Evaluation job 88 cancelled at user request |
| Downstream tangent 30%, 50%, 70% | Smoke job 86 queued; pruning jobs 89, 91, 93 depend on it | Pending | Jobs 90, 92, 94 depend on respective pruning jobs |

Jobs 66 and 67 failed at the conservative CUDA headroom check; job 71 failed with CUDA out of memory during the solve. The 10% retries were cancelled at the user’s request; the 30% local retry remains queued and will rebuild with GB10. Scheduler success and a complete, provenance-validated report are both needed before treating an evaluation as finished. [Original sweep plan](../../results/tangent/gsm8k/mlp-retention-sweep-plan-v1.json) and [downstream plan](../../results/tangent/gsm8k/downstream-retention-sweep-plan-v1.json) record scope and commands; the downstream arm also uses extra backward geometry computation, so it is not compute matched to local or task SparseGPT.

## GB10 environment and convolution backend

Queue change on 2026-11-08: jobs 78, 79, 80, 84, 85, 86, 89–94 and 97 were updated in gflow to use `.pixi/envs/gb10/bin/python`; their resource requests and dependencies were retained. Task-SparseGPT 70% pruning job 77 was stopped at user request to run job 97; its three default-environment site checkpoints are preserved in `qwen-9b-gsm8k-mlp-sparsegpt-task-s70-seed0-v1-partial-default-before-gb10`. Replacement job 98 will start from scratch in GB10 after job 97 succeeds; evaluation job 78 now depends on job 98. The local 30% retry (84) starts afresh: the two site checkpoints from its failed default-environment attempt were preserved in `qwen-9b-gsm8k-mlp-local-s30-seed0-v1-partial-default-before-gb10`. Job 97 had written no answer records before job 76 was stopped. The original submission journals record the commands at submission, not these later gflow command updates; inspect `gjob show` for actual commands.

These GB10 evaluations are compared to a dense reference collected in the default environment (Transformers 5.20.0/FLA 0.5.2, PyTorch convolution). They use Transformers 5.22.2/FLA 0.6.0 and causal-conv1d; the full-model numerical effect of that change has not been measured, so interpret differences with that limitation. The kernel smoke and microbenchmark do not validate the full evaluation or downstream backpropagation.

Completed earlier GSM8K jobs used `.pixi/envs/default/bin/python`; that environment lacks `causal-conv1d`, and job 75 logs the PyTorch convolution fallback. The separate `gb10` environment installs the locally built CUDA 13 wheel at `/home/nlaanait/.pixi/wheels/cu130/causal_conv1d-1.7.0-cp312-cp312-linux_aarch64.whl`; use `pixi run -e gb10 ...` for new work. The wheel imports with torch 2.12.0+cu130 and contains SM 12.1 code. The absolute wheel path is machine-specific and must exist when installing this environment. Job 95 completed a small GPU forward/backward correctness check, as described below.

In the installed default environment (Transformers 5.20.0, FLA 0.5.2), inspection of the Qwen dispatcher after `force_pytorch_attention()` shows that chunk and recurrent gated-delta rules resolve to FLA, while both causal convolutions resolve to PyTorch. The helper prints `-> pytorch` for all four, but its unwrapping does not unwrap these `kernels` Func objects; the printout is misleading. This corrects the earlier assumption that all four use PyTorch. `use_kernels=False` disables the optional hub kernel path, not the installed-package fallback.

In `gb10` (Transformers 5.22.2, FLA 0.6.0), the same inspection resolves both gated-delta functions to FLA and both convolutions to causal-conv1d even after the helper runs. The dispatch inspections alone do not establish full-model correctness or performance. Scientific runs have been switched at the user’s request; a full-model cross-environment numerical comparison remains outstanding. Job 95 passed a small float32 convolution forward/backward correctness check. Job 96 measured warm forward-only kernels at batch 1, bfloat16 and Qwen3.5-9B dimensions on GB10: causal-conv1d was 3.42×/3.01× faster at 256/2048 tokens; FLA gated-delta was 6.25×/7.66× faster at 256/2048 and 3.52× at single-token decode. These synthetic kernel timings do not measure end-to-end speed, backward, or full-model numerical equivalence; maximum absolute convolution difference was 0.0625. Script: `scripts/benchmark_gb10_kernels.py`.

## Protocol and implementation notes

The controlled data registry has 23 source templates in 19 algebra/source groups; related templates stay in the same split. The reserved 512-problem original validation set is sampled from GSM8K training data, excludes earlier development/template source indices and exact normalized duplicates, and is kept separate from calibration; near-semantic duplicates are not guaranteed absent. The official test set remains untouched.

Sequential pruning refreshes activations and, for geometry arms, atlases after upstream changes. Candidates save an upstream-hashed per-site checkpoint chain. Evaluations verify candidate hashes and record per-problem answers, answer NLL, actual removal counts, and a small language diagnostic. Retention reporting requires an explicit allowed drop, separates observed retention from statistical support, and checks provenance and generation settings. The 32-example controlled sample cannot establish a two-point tolerance.

The downstream arm fits factor-local geometry at the next block's residual output (the final block uses its own post-MLP residual), then uses backpropagated tangent probes to estimate local sensitivities. It uses 32 family-balanced pruning-calibration problems, eight probes per prompt, alpha 1, and the same ambient token budget as the other arms; geometry and backward compute are additional. The two-block smoke job gates the full runs. See the [plan](GSM8K-EXPERIMENT-PLAN.md) and [submission journal](../../results/tangent/gsm8k/downstream-retention-submissions-v1.json) for details.

Jobs use exclusive gflow allocations, a guarded CUDA allocator cap up to 56 GiB, an 80 GiB systemd memory ceiling, and a 96 GiB host-pressure watchdog. The local retries request higher scheduler GPU-memory caps, but passing the scheduler does not guarantee that the process's adaptive allocator cap will suffice. Do not infer completion from a submission journal.

The sequential geometry encoder uses precomputed Gram products for sparse coding. A CPU microbenchmark reported a 35× encoding speedup, not an end-to-end speedup; changed floating-point association means near-tied assignments are not guaranteed bit-identical. The full design and limitations are in the experiment plan.

The user chose to skip all pending 10% runs on 2026-11-08. The original plans and submission journals remain immutable historical records and still list those jobs; consult gflow for their current states. The completed 10% task candidate and its 32-example development result remain recorded.

## Still unresolved

- Choose and freeze the acceptable dense-relative accuracy drop before confirmatory selection; 2 percentage points is only a provisional sensitivity value.
- Finish and compare the paired 512-problem evaluations, including uncertainty and the general-language diagnostic. Do not assume accuracy is monotonic in sparsity.
- Complete downstream smoke and candidate runs, and examine failed/retried local runs. No local or downstream model-wide retention result is available in this snapshot.
- Freeze the selected candidate and protocol before using the untouched official GSM8K test set. The current measurements are exploratory/validation results.
