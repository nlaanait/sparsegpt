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
