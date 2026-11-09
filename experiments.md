# Experiments

WikiText2 perplexity for unstructured pruning of `Qwen/Qwen3.5-0.8B`.
Calibration is C4, 128 sequences, seed 0, sequence length 2048, PyTorch
attention, forward batch 1. The table rounds the saved JSON. Full values
are in the result files named below.

The first six columns are one run. The Tangent α=0.5 column is an earlier
run, recorded in a different commit, and its own SparseGPT baseline is not
the SparseGPT column here (35.718, 90.542, 360.736, and 2724.743 at
sparsities 0.5 through 0.8).

| Sparsity | SparseGPT | Wanda | Input-side (A) | Rescale (B) | Heuristic | Sparse+low-rank (D) | Tangent α=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 35.377 | 34.152 | 36.480 | 40.035 | 40.478 | 35.327 | 35.465 |
| 0.6 | 86.975 | 79.517 | 107.696 | 119.691 | 122.391 | 86.573 | 94.918 |
| 0.7 | 380.459 | 289.423 | 438.425 | 565.259 | 634.888 | 369.948 | 380.705 |
| 0.8 | 2350.268 | 2465.981 | 2370.105 | 3301.576 | 3134.470 | 2157.075 | 1999.488 |
| 0.9 | 9265.487 | 29274.812 | 11820.448 | 22428.852 | 17832.771 | 10447.258 | — |
| 0.99 | 478105.265 | 7.427e10 | 995705.586 | 4748198.600 | 78183.222 | 147012.936 | — |

## Method sweep

Commit `a97bc8b030fa6861b7f7effa5bc4da6ac78175ec` on
`feat/tangent-aware-pruning-revised`.

Result: `results/tangent/qwen35-0.8b-method-sweep.json`.

Sparsities are 0.5, 0.6, 0.7, 0.8, 0.9, and 0.99. SparseGPT is the
baseline. Wanda, input-side (A), rescale (B), the score-only heuristic,
and sparse-plus-low-rank (D) are the other arms. Fisher is omitted because
the layer loop does not collect loss gradients.

Input-side uses the nine concept-pair finite differences in
`results/tangent/qwen35-0.8b-concept-pairs`, with `--alpha-in 1`. Those
pairs are exported first:

```bash
PYTHONPATH=src:. python -m sparsifier.data.export_concept_pairs \
  --model Qwen/Qwen3.5-0.8B \
  --out-dir results/tangent/qwen35-0.8b-concept-pairs
```

The sweep driver is `drivers/qwen08_method_sweep.py`. It calls
`drivers/capped_sparsifier.py`, which caps the process at 4 GiB and stops
if host memory reaches 102 GiB. The run that finished the 36 points was:

```bash
PYTHONPATH=src:. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
SPARSIFIER_PYTHON=/home/nlaanait/sparsegpt/.pixi/envs/default/bin/python \
/home/nlaanait/sparsegpt/.pixi/envs/default/bin/python -u -m drivers.qwen08_method_sweep
```

Each point is:

```bash
python -u -m drivers.capped_sparsifier Qwen/Qwen3.5-0.8B c4 \
  --seed 0 --nsamples 128 --sparsity SPARSITY \
  --eval-datasets wikitext2 --pytorch-attention --forward-batch 1 \
  METHOD_FLAGS
```

`METHOD_FLAGS` by column:

| Column | Flags |
|---|---|
| SparseGPT | none |
| Wanda | `--pruner wanda` |
| Input-side (A) | `--pruner tangent_input_side --alpha-in 1 --pair-deltas results/tangent/qwen35-0.8b-concept-pairs` |
| Rescale (B) | `--pruner tangent_rescale --tangent-alpha 1` |
| Heuristic | `--pruner tangent_heuristic --tangent-alpha 1` |
| Sparse+low-rank (D) | `--pruner tangent_sparse_lowrank` |

Rescale and the heuristic use the default output charts: 8 charts, rank 2,
512 calibration samples. Sparse-plus-low-rank uses one rank-2 principal-component
frame of those calibration outputs. Neither reads the concept-pair bank.

## Tangent α=0.5

Commit `20b25cf6b85cd6e8d62f6e15c6db79f941044964`.

Result: `results/tangent/qwen35-0.8b-c4-a05-sparsity-sweep/sweep_report.json`
in that commit. The ladder is the default sparsity grid, 0.1 through 0.8.
Sparsities 0.9 and 0.99 were not run. The table above copies 0.5 through 0.8.

Driver: `experiments/tangent_sparsity_sweep.py`.

```bash
python -u -m experiments.tangent_sparsity_sweep \
  --model Qwen/Qwen3.5-0.8B \
  --calibration c4 \
  --nsamples 128 \
  --forward-batch 1 \
  --eval-dataset wikitext2 \
  --pytorch-attention \
  --tangent-alpha 0.5 \
  --tangent-charts 8 \
  --tangent-rank 2 \
  --tangent-samples 512 \
  --output-dir results/tangent/qwen35-0.8b-c4-a05-sparsity-sweep
```

Each tangent point recorded in the report is `python -m sparsifier` with
the same calibration settings and:

```bash
python -m sparsifier Qwen/Qwen3.5-0.8B c4 \
  --seed 0 --nsamples 128 --percdamp 0.01 --sparsity SPARSITY \
  --minlayer -1 --maxlayer 1000 --blocksize 128 --forward-batch 1 \
  --eval-datasets wikitext2 --pytorch-attention \
  --tangent-alpha 0.5 --tangent-charts 8 --tangent-rank 2 --tangent-samples 512
```

This is the output-tangent mask (`--tangent-alpha 0.5`) with ambient
SparseGPT compensation. It is not Variant B.

## Next experiments: revised tangent-pruning review

The CPU toy runs are harness checks, not evidence for model-level pruning
claims. Stage 1 now uses each local chart's stored anchor for reconstruction
and reports its storage cost. Variant D separately reports sparse-component
sparsity, effective dense sparsity, and representation-size reduction.

Run the follow-ups in this order:

1. **Stabilize the CPU harness.** Repeat the toy protocol over seeds 0–9;
   include planar-circle and multi-harmonic controls. Add a projector batch
   with varying ranks and gradient norms to verify that the revised
   rank-matched gradient-enrichment null is gradient-energy weighted.
2. **Run the real-layer diagnostic.** Evaluate Qwen3.5-0.8B block 0 on the
   existing concept and presence pair banks. The live harness reports Stages
   0–4 together; interpret Stage 0 as descriptive screening and inspect held-out
   score prediction and outcomes before making any geometry-weighting claim.
   Compare real directions with matched-rank random controls.
3. **Recheck Stage 1.** Use disjoint tangent-fit and held-out examples, with a
   planar circle as a negative control and multi-harmonic concepts where
   `b > r + 1`. Compare local and global representations at matched storage
   budget and report stability and held-out ambient error over multiple seeds.
4. **Validate Stage 2 predictions on the real layer.** Enumerate single-entry
   and selected structured knockouts with and without compensation. Compare
   geometric scores with magnitude, Wanda, ambient OBS, and Fisher against
   held-out tangent, presence, ambient, and downstream impact.
5. **Run Stage 3 only after score validation.** Sweep Variant A's
   `alpha_in` for unstructured and 2:4 masks at matched sparsity, calibration,
   and compensation settings. Compare SparseGPT, Wanda, Fisher, and the
   geometric arms; report the full alpha front and per-concept errors. Follow
   with Variant B and exact-refit checks only if Stage 2 supports them.
6. **Revisit Stage 4 after clarifying weighted-bias exactness.** Compare
   no-bias, centered/in-pass, and post-hoc mean arms on held-out examples,
   separating presence and normal residuals. Treat the ones-column result as
   an approximation for sample-varying output metrics unless a coupled solver
   is used.

### Immediate gflow submission

Job 25 currently requests 96 GiB and is marked exclusive. The real-layer
diagnostic is capped at 12 GiB (maximum combined allocation 108 GiB, below the
110 GiB limit). Submit it as a shared job; if gflow cannot co-locate it with
the exclusive job, it will remain queued until capacity is available.

```bash
gbatch \
  --gpus 1 \
  --gpu-memory 12G \
  --memory 24G \
  --name tangent-qwen35-0.8b-revised-stages \
  --project tangent-sparsegpt \
  --time 2:00:00 \
  --shared \
  --auto-close \
  -- pixi run python -u -m sparsifier.eval.live_stages \
    --max-gib 8 \
    --concept-pairs results/tangent/qwen35-0.8b-concept-pairs \
    --presence-pairs results/tangent/qwen35-0.8b-presence-pairs \
    --output results/tangent/qwen35-0.8b-revised-stages.json
```

### Run status (2026-11-06)

- CPU synthetic harness, seeds 0–9, initial version: output at
  `/tmp/tangent-stages-seeds0-9.json`. Tangent-Wanda score correlation was
  `0.9946 ± 0.0027`; magnitude and Wanda were `0.9400 ± 0.0121` and
  `0.9384 ± 0.0120`. Mean Stage-0 gradient enrichments were 7.79 for tangent
  and 0.029 for presence. These remain toy results.
- Corrected CPU rerun, seeds 0–9: `/tmp/tangent-stages-seeds0-9-revised.json`.
  Local-affine held-out errors averaged 0.0019 (planar circle) and 0.0030
  (multi-harmonic), versus approximately 0 and 0.450 globally. However, the
  local models store about 2,524 scalar units on average versus 24 for the
  global model. Stage 1 is therefore reported as
  `inconclusive_unmatched_storage`, not as an H1 win or loss. The previous
  local-error comparison omitted chart anchors and was not valid.
- Variant D in the corrected rerun uses 31/32 storage units: 15 sparse
  nonzeros, 16 low-rank factor values, and one unused unit. This is a 51.6%
  representation-size reduction and 76.6% sparsity in the sparse component,
  while the reconstructed dense matrix has 0% elementwise sparsity. The prior
  `actual_sparsity: 0.0` described only that dense reconstruction, not stored
  compression.
- Stage 4 arms were within tolerance in 10/10 runs for no-bias, centered, and
  post-hoc global-mean; the ones-column arm passed 5/10.
- Real Qwen3.5-0.8B Stages 0–4 completed as gflow job 26 with a 12 GiB cap
  and a measured peak of 3.84 GiB. Output:
  `results/tangent/qwen35-0.8b-revised-stages.json`. This is a one-layer smoke
  run on block 0 `linear_attn.in_proj_a` (16 × 1024 weights), using 128 random-token
  activation rows split equally for calibration and evaluation.

### Job 26 interpretation and next run

At approximately 50% unstructured sparsity, ambient SparseGPT has ambient,
tangent, and presence errors of 3.979, 0.1973, and 0.3186. The arm named
`local_tangents` reduces tangent error to 0.1825 (7.5%) but increases ambient
error to 4.039 (1.5%). Input-side alpha 3 reduces presence error to 0.2792
(12.4%), with ambient and tangent errors of 4.018 and 0.1992. Neither arm
improves every metric. The 2:4 alpha-3 arm reduces ambient error by 2.4% but
increases tangent and presence errors by 16.6% and 8.0%.

Stage 2 samples only eight knockouts: geometric and Wanda rank correlations
are both 0.857, versus 0.929 for magnitude and ambient OBS. The test does not
show an advantage for the geometric score. Stage 4 centering lowers ambient
error by 9.1%; all four bias arms meet the current local 10% tolerance.

Limitations that must be corrected before a larger sweep:

- Calibration uses random token ids. Recorded gradients come from different
  forward passes and are repeated to match the activation count, invalidating
  interpretation of the joint activation-gradient Fisher scores.
- Held-out tangent/presence directions are averages of the same saved bank,
  repeated across evaluation rows. The `local_tangents` arm uses a fixed
  metric; it is not a sample-conditioned local-chart experiment.
- `kl_to_dense` is the KL between softmaxes of layer outputs, not vocabulary
  logits. The reported probes and KL do not measure language-model task loss.
- Static-union and global-BSF arms use identical scales. D4 splits different
  knob directions rather than independent estimates of each same concept.
  Its `stop_unstable` flag is descriptive evidence requiring a better control.

Next: collect aligned activations and gradients from natural text, fit
directions on separate concept examples, and measure actual held-out token
NLL/KL after inserting each pruned layer. Start with a middle-block down/output
projection, 256–1024 unique Stage-2 knockouts, matched-rank random controls,
and confidence intervals. Then compare SparseGPT, Wanda, corrected Fisher,
input-side alpha 0/0.3/1/3, and fixed tangent weighting at 50% sparsity over
three calibration seeds. Require a repeatable concept-error improvement with
a predeclared bound on held-out NLL degradation before expanding the grid.

### Natural-text Stage 2/3 sequence

Runner: `python -m sparsifier.eval.natural_layer`. The previous
`live_stages` entry point is retained as a legacy random-token smoke harness.
The new runner captures aligned activations and output gradients from the
same C4 token forward and computes held-out NLL and dense-to-pruned KL from
final vocabulary logits after inserting the candidate weight into the model.

The first sequence uses Qwen3.5-0.8B block 12 `mlp.down_proj`, 16 C4 windows
of 128 tokens per calibration seed, and exactly 50% elementwise sparsity.
Concept fitting and evaluation use disjoint prompt prefixes for each of the
nine knobs; the knob values are shared, so this tests context generalization.
Paired input-difference errors describe the local linear layer output. The
geometric weighting is a fixed span with the standard post-scaling damping
surrogate; no local-chart or coupled-solver exactness claim is made.

| gflow job | Experiment | Dependency | Output |
|---|---|---|---|
| 28 | Stage 2, seed 0: 256 unique entries, with/without OBS compensation | none | `results/tangent/natural-layer-v1/stage2-seed0.json` |
| 29 | Stage 3, seed 0 | 28 | `results/tangent/natural-layer-v1/stage3-seed0.json` |
| 30 | Stage 3, seed 1 | 29 | `results/tangent/natural-layer-v1/stage3-seed1.json` |
| 31 | Stage 3, seed 2 | 30 | `results/tangent/natural-layer-v1/stage3-seed2.json` |

Stage 2 uses two WikiText2 test windows and reports rank correlations with
bootstrap intervals for held-out pair errors, projected layer errors, actual
token-NLL changes, and vocabulary KL. Scores include magnitude, Wanda,
ambient OBS, aligned Fisher, geometric Wanda/OBS, and three matched-rank
random-span controls. Stage 3 uses eight different WikiText2 test windows
and compares SparseGPT, Wanda, input-side alpha 0.3/1/3 (SparseGPT is alpha 0),
fixed tangent weighting, aligned Fisher, and three random-span controls.
Each arm reports per-knob errors and window-level language metrics. Before
expanding the grid, require repeatable concept-error improvement and an
upper confidence bound on NLL increase versus SparseGPT of at most 0.02
nats/token. These short-window diagnostics do not replace the full 2048-token
WikiText2/PTB sweep.

Each job requests 12 GiB GPU and 24 GiB host RAM, with an 8 GiB CUDA allocator
cap. Dependencies serialize the new jobs, keeping their combined allocation
with a 96 GiB shared GPU job below 110 GiB. Job 27 initially blocked these
jobs by reserving 120 GiB of host RAM; it was cancelled at the user's request.
Job 28 then started, and jobs 29–31 remain dependent on the preceding phase.
Progress is saved after each knockout or arm, with source/config provenance
checked on resume. The CPU checks cover gradient alignment, true vocabulary
metrics, restoration after temporary weight changes, exact sparsity, and the
rank-one knockout calculation.

### GSM8K downstream tangent pullbacks (2026-11-08)

The `downstream` arm joins the Qwen3.5-9B MLP retention study via
`sparsifier.eval.gsm8k_downstream`. gflow 86 is a two-block/six-projection
smoke test; jobs 87/89/91/93 prune all MLP projections at 10/30/50/70%
eligible sparsity; jobs 88/90/92/94 run the matching original 512-problem
validation. Full runs depend on the smoke artifacts; validations depend on
their candidates and dense reference 64. All are exclusive guarded jobs.

Next-block residual tangents are pulled back with eight probes on 32
pruning-calibration problems. Ambient data remains the same 4096 C4 + 4096
math tokens; backward prompts do not enlarge that budget. Alpha is 1,
with diagonal output/token metrics and matched rescaled compensation.
Existing jobs and immutable plans are preserved. See
[the status note](docs/research/GSM8K-EXPERIMENT-STATUS.md) for limitations,
the additive plan, journal, provenance checks, and CPU verification.
