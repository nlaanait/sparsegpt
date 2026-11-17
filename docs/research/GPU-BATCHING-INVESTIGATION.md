# GPU batching investigation

Inspected on 2026-11-13. There is substantial memory headroom and several
serial loops that can batch independent work. The current monitoring does
**not** establish SM saturation. Job 164 measured substantial generation
throughput gains, with smaller gains for prefill and batched backward probes.

## Observations

Completed gflow job **139**, `qwen-9b-gsm8k-mlp-downstream-s50-seed0-weightdiag-v1`,
ran for 88 minutes 37 seconds and completed all 96 pruning sites.

| Measurement | Observed value | Interpretation |
|---|---:|---|
| Peak CUDA reserved memory, saved report | 20.99 GiB | PyTorch allocator reservation, not total process memory |
| Peak system memory used, memory guard | 35.49 GiB | System-wide unified-memory pressure, including other host activity |
| System memory capacity | 121.69 GiB | CPU and GPU share physical memory on this GB10 |
| Serial behavioral evaluation | 1,253.5 seconds, 32 answers | 20.9 minutes, about 24% of this job's elapsed time |
| Generated tokens | 14,041 | About 11.2 tokens/s including prompt processing and answer-likelihood passes |
| Calibration prompts per site | 425 | Captured one prompt at a time |
| Downstream probes | 32 prompts × 8 VJPs per site | Separate backward calls for each probe |

The first live sample showed 94% GPU utilization during **answer generation**,
after pruning had finished. A later sample during pruning job **141** showed
77% GPU utilization, 39 W, and about 28.3 GiB system memory used in the recent
stage log. These are snapshots, not phase averages.

The `nvidia-smi` GPU-utilization counter measures the fraction of a sampling
period with one or more kernels executing. It does not measure achieved SM
occupancy, tensor-core utilization, or percentage of peak FLOPS.
[NVIDIA's counter definition](https://docs.nvidia.com/deploy/nvml-api/620/api/structnvmlUtilization__t.html).
Framebuffer memory counters were unavailable and memory-utilization samples
reported 0%; those readings cannot establish low memory-bandwidth demand.
CUDA allocator reservations and system memory use overlap and must not be added.

## Where batching fits

| Priority | Code | Opportunity | Required boundary |
|---|---|---|---|
| 1 | `gsm8k_sequential.collect` | Length-bucket independent calibration prompts; batch equal-length C4 windows | Preserve each example's readout, sampled token positions, seed, and original ordering; exclude padding |
| 1 | `gsm8k_pruning.behavior`, `gsm8k_decision.run_tiers` | Batch independent generations, initially 2/4/8 prompts | Verify padding/cache semantics and per-example EOS handling; finish the first 64 before generating the suffix |
| 2 | `downstream.tangent_pullback_diagonal` | Batch VJPs in small probe chunks on the same graph | Preserve the probe vectors, per-sequence contractions, averaging, and frozen geometry |
| 2 | Same metric loop | Cache detached CPU FP64 inputs once; reuse each unsquared contraction for diagonal and group metrics | Preserve precision and callback metadata; avoid duplicate contractions |
| 3 | `whole_neuron.refit_columns` | Increase the 256-row right-hand-side chunk when workspace permits | Keep the same factor, retained mask, ridge, and original reference weights |

The family runner already has `resolve_forward_batch` and a 4 GiB activation
budget in `models/qwen.py`. The dedicated GSM8K capture and generation loops
do not use that batching path. Raising the family runner's forward-batch flag
alone will not accelerate these experiment drivers.

For the downstream entrywise run, 425 math prompts plus 32 C4 windows per
site means approximately 43,872 serial capture forwards across 96 sites,
before the 3,072 additional backward-calibration forwards and 24,576 VJPs.
That makes calibration batching worth investigating even though generation
was the phase observed in the initial GPU sample. These counts describe
executed work, not a measured attribution of elapsed time.

### Preserve the metric when batching probes

For sequence \(s\) and probe \(p\), let

\[
 G_{s,p}=\frac{\partial\langle v_{s,p},a_s\rangle}{\partial Y_s},
 \qquad Z_{s,p}=G_{s,p}^{\mathsf T}X_s,
 \qquad K=\frac1m\sum_{p=1}^m\sum_s Z_{s,p}^{\odot2}.
\]

Compute several independent \(G_{s,p}\) together, but retain sequence and
probe axes until after each token-summed contraction is squared. Summing
independent sequences or probes before squaring introduces cross terms and
changes the method. The full neuron-group consumer also needs the unsquared
\(Z_{s,p}\), so the same contraction can serve both consumers.

Today, the metric code transfers/casts the same local inputs to CPU FP64 for
each probe. When a group consumer and weight diagonal are both requested,
it also computes \(G^{\mathsf T}X\) twice. Caching and reuse can remove this
work without changing the arithmetic precision. Moving contractions to GPU
FP32 is a separate numerical change and requires score/mask validation.

PyTorch exposes batched VJPs through `autograd.grad(is_grads_batched=True)`.
It uses vmap and may encounter unsupported operators or performance cliffs;
successful serial backward does not prove batched backward will work.
[PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html).

### Dependencies that should remain sequential

SparseGPT already processes all output rows together. Inside a column block,
the compensation at column \(j\) changes the weights used at \(j+1\).
The outer block update already uses a matrix multiplication. Batching these
columns as independent work changes the solve; increasing block size can also
change mask selection. Kernel fusion or compilation is a different optimization
from independent batching.

Different pruning stages use fresh activations from the partially pruned
network. Sharing captures across different candidate states changes that
protocol. Batch independent samples within one frozen stage first.

The repository documents an Atlas SM121 `gdn_prefill` bug that mixes batch
row zero into later rows (`models/qwen.py:_atlas_linear`). Identical-input
tests hide it. Production batching needs distinct-input equivalence checks
on the actual dispatched backend. An existing stream-based batch-one fallback
in the family runner is another option to benchmark where raw batching is
unsupported.

## Completed measurement: job 164

Gflow job **164**, `qwen-9b-gsm8k-batching-diagnostic-v1`, completed successfully
in 6 minutes 36 seconds. It ran
`scripts/benchmark_gsm8k_batching.py` with an exclusive allocation, 64G GPU
reservation, 80G host limit, and a 30-minute scheduler limit. It used priority
110 and ran after the preceding exclusive job. Existing experiments were
not modified.

The diagnostic uses the pinned dense Qwen3.5-9B checkpoint and the decision
runner's verified reference backend. It recorded:

- Prefill and fixed 32-token generation throughput for batches 1/2/4/8,
  using the same eight distinct prompts from the shared seeded manifest.
- Batched-versus-singleton logits, next-token agreement, and exact agreement
  of each 32-token generated sequence.
- Serial versus 2/4/8-probe VJPs on the actual block-15 down-output to
  block-16 residual graph, using the same fixed rank-two probes.
- Median wall time after warmup, individual timing samples, CUDA allocator
  peaks, system memory use, runtime versions, and explicit unsupported cases.

Results are in `results/performance/gsm8k-batching-v1/report.json` and
`measurements.jsonl`. Times are medians of three warmed repetitions, processing
the same eight prompts and 256 generated tokens in total at every batch size.

| Batch | Generation tokens/s | Generation speedup | Exact 32-token sequences matching serial | Prefill speedup | Peak CUDA allocated during generation |
|---|---:|---:|---:|---:|---:|
| 1 | 10.90 | 1.00× | 8/8 | 1.00× | 16.81 GiB |
| 2 | 23.24 | 2.13× | 8/8 | 1.44× | 16.92 GiB |
| 4 | 42.53 | 3.90× | 7/8 | 1.54× | 17.13 GiB |
| 8 | 65.55 | 6.01× | 7/8 | 1.22× | 17.54 GiB |

All batches agreed on the first generated token for all eight prompts, but
batched logits were not identical to singleton logits: maximum absolute
differences were 0.28–0.31. Batch two preserved every tested 32-token sequence;
batches four and eight each differed from serial on one sequence. This does
not establish a backend bug or an accuracy regression, but it rules out claiming
exact output equivalence for those larger batches. Batch two is the first
candidate for longer-generation and pruned-checkpoint validation.

Backward chunks of 1/2/4/8 took 31.59/26.10/23.34/24.75 ms for eight VJPs.
Chunk four was fastest, a **1.35×** backward-only speedup. Batched gradients
had maximum absolute difference 0.00024414 from serial, or 0.281% of the
reference gradient's maximum magnitude. That is not a bound on elementwise
relative error, and final tangent metrics and pruning masks were not compared.

The measured generation speedups confirm that the earlier utilization counter
did not imply throughput saturation for batch-one inference. They do not
predict a sixfold speedup of pruning or of a complete 2,048-token evaluation.
The diagnostic excludes atlas fitting, weight contractions, and complete
calibration. Prefill peaked at batch four, so one batch size should not be
assumed optimal for every phase. Padding-aware capture, full calibration,
longer generations, and pruned candidates still need validation before
enabling production batching. The scientific evaluation keeps its existing
2,048-token budget and shared 64/128 identities.

## Batch-eight rollout

The user selected batch eight for the experiments after reviewing job 164.
The shared GSM8K execution policy is now `gsm8k-batch8-v1` in
`sparsifier.eval.gsm8k_batching`. Calibration capture, generation, and
direct-answer likelihood forwards use contiguous groups of eight, with a
smaller final group when necessary. Per-example token positions and seeds
are unchanged, and padding is excluded from captured samples. Anchor hooks
collect every batch row. Backward-calibration samples and SparseGPT's
dependent column compensation retain their existing execution order.

Generation trims each example at its first EOS before counting tokens or
testing truncation. Partial batches are regenerated with their original
membership on resume; previously saved answers are not duplicated. Timing
per answer is batch wall time divided by group size, with the full batch time
also saved. The 64-question decision boundary is never crossed by a batch.

All new runs record batch size, padding, membership policy, source hash, and
verified reference backend. Legacy singleton answers cannot resume into a
batch-eight run, and candidate screens require a matching batch-eight dense
reference. Existing checkpoint weights remain usable as frozen candidates.

`scripts/propagate_gsm8k_batch8.py --submit` submitted jobs **165–176**:

| Jobs | Work |
|---|---|
| 165 | Fresh dense 128-question reference, batch eight |
| 166 / 167 | Entrywise ambient block-15 candidate / 64→128 evaluation |
| 168 / 169 | Entrywise downstream block-15 candidate / evaluation |
| 170 / 171 | Whole-neuron ambient block-15 candidate / evaluation |
| 172 / 173 | Whole-neuron downstream block-15 candidate / evaluation |
| 174 / 175 / 176 | Batch-eight evaluation of frozen downstream 30% / 50% / 70% candidates |

Queued predecessors 149/150/151/154/155/161/162/163 were canceled after
replacement submission, including dependent cancellations. Job 148 had
already started and was retained; it completed successfully. Previously
canceled local pilots remain excluded. The existing seeded manifest,
generation budget, screen cutoff, model revision, and calibration budgets
are unchanged. The rollout plan and submission journal are immutable/versioned
under `results/tangent/gsm8k/decision64-light128-batch8-rollout-*.json`.

Validation: 155 GSM8K, downstream, and whole-neuron CPU regression tests
passed, including actual tiny Qwen reference attention with eight distinct
padded sequences, EOS/truncation, per-example likelihood alignment, partial
resume, screen boundaries, and legacy-reference rejection. The full dense
reference is the first scheduled integration run for this rollout.

Prefer batching within a process before packing separate model jobs: it
shares model weights and avoids duplicate CPU/GPU memory traffic. Two shared
jobs may fit in the observed memory headroom, but their aggregate throughput
must be measured; memory capacity alone does not predict whether concurrent
jobs will contend for compute or unified-memory bandwidth. Both jobs would
need gflow shared allocations and realistic reservations checked against
the scheduler's device capacity; simply adding `--shared` is insufficient.
