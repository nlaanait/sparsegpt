# SparseGPT

This repository contains code to reproduce the key results of the paper [SparseGPT: Massive Language Models Can be Accurately Pruned in One-shot](https://arxiv.org/abs/2301.00774).

Specifically, it provides scripts and implementations to:

* Evaluate baseline and pruned models on raw-WikiText2, PTB and C4-subset. (`sparsegpt/datautils.py`, `sparsegpt/opt.py`, `sparsegpt/bloom.py`)
* Perform unstructured, n:m and sparse + quantized SparseGPT compression on OPT and BLOOM models. (`sparsegpt/sparsegpt.py`, `sparsegpt/opt.py`, `sparsegpt/bloom.py`)

We note that this SparseGPT implementation is based on our open-source [GPTQ code](https://github.com/IST-DASLab/gptq). 

## Dependencies

* `torch`: tested on v1.10.1+cu111
* `transformers`: tested on v4.21.2
* `datasets`: tested on v1.17.0

## Usage

Here are some sample commands to run baselines and sparsification on OPT models, followed by perplexity evaluations on raw-WikiText2, PTB and C4.
See also the CMD-argument documentation.

```
# Run dense baseline
python -m sparsegpt.opt facebook/opt-125m c4

# Run magnitude baseline
python -m sparsegpt.opt facebook/opt-125m c4 --sparsity .5 --gmp

# Prune to 50\% uniform sparsity with SparseGPT
python -m sparsegpt.opt facebook/opt-125m c4 --sparsity .5

# Prune to full 2:4 sparsity with SparseGPT
python -m sparsegpt.opt facebook/opt-125m c4 --prunen 2 --prunem 4

# Prune to 50\% + 4-bit with SparseGPT
python -m sparsegpt.opt facebook/opt-125m c4 --sparsity .5 --wbits 4
```

To run on other OPT models, replace "facebook/opt-125m" by the HuggingFace name of the corresponding model.
For the 175B model, access must first be requested from Meta and the checkpoint converted to HuggingFace format, then its location can simply be passed as a name to this script.

The BLOOM module `sparsegpt.bloom` has a very similar interface, however some features are currently only available for OPT, e.g.:

```
# Sparsify BLOOM-176B with SparseGPT
python -m sparsegpt.bloom bigscience/bloom c4 --sparsity .5
```

We also provide LLaMA pruning script with the very same interface:

```
# Sparsify LLaMa with SparseGPT
python -m sparsegpt.llama LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity 0.5
```

In case one would like to save the sparsified model specify path to saved checkpoint via  `--save` flag.

One can optionally log evalution results to W&B with `--log_wandb`. 

## Geometry-aware calibration

This fork keeps the upstream SparseGPT solver (`fasterprune`) and adds an optional
tangent-aware calibration objective:

```
min ||(S - W) H||_F² / N  +  alpha ||(S - W) T||_F² / P
```

Activation-only SparseGPT is unchanged when `--geometry-alpha 0` (the default).
Positive alpha augments the Hessian with ambient tangent probes after activation
hooks finish, then runs the same prune/compensate path on CPU or CUDA.

### Layer demo (no model download)

```bash
# CPU
python -m sparsegpt.geometry_demo --output geometry_results --device cpu

# CUDA (requires a GPU build of torch)
python -m sparsegpt.geometry_demo --output geometry_results_cuda --device cuda

python -m unittest -v tests.test_geometry
```

Compare `activation_only` vs `geometry_aware` in `geometry_results/report.json`.
Supply a real layer with `--input layer.npz` containing `W`, `H_train`, `H_test`
and optional `T_train` / `T_test` (columns are samples / probes).

### Full-model OPT / BLOOM / LLaMA

1. Export per-layer probes into `<probes>/<block>/<module>.npy`:

```bash
python -m sparsegpt.geometry_export \
  --activations layer0_fc1_acts.npz --key H \
  --output probes --block-index 0 --module-name fc1
```

Module names must match SparseGPT keys (`fc1`, `self_attn.q_proj`, ...).

2. Prune with the same solver and geometry calibration:

```bash
python -m sparsegpt.opt facebook/opt-125m c4 --sparsity .5 \
  --geometry-alpha 1.0 --geometry-probes probes

python -m sparsegpt.bloom bigscience/bloom-560m c4 --sparsity .5 \
  --geometry-alpha 1.0 --geometry-probes probes

python -m sparsegpt.llama LLAMA_HF_WEIGHTS_LOCATION c4 --sparsity .5 \
  --geometry-alpha 1.0 --geometry-probes probes
```

`--geometry-alpha 0` is bit-compatible with upstream activation calibration.
Geometry mode requires SparseGPT (not `--gmp`). Layers and Hessians follow the
device chosen by the existing scripts (CUDA when available, otherwise CPU).

## Output-tangent pruning experiments

`--tangent-alpha` enables a separate, output-space candidate. During the normal
layer calibration forward pass, it keeps a bounded uniform reservoir of paired
input/output token samples, clusters outputs into local charts, and fits a
rank-limited PCA tangent basis in each chart. The mask multiplier for output
row `p` and input column `q` is

```text
G[p,q] = sum_c diag(U_c U_c.T)[p] * sum_{n in c} x[n,q]^2
         / sum_n x[n,q]^2
M[p,q] = (1 - alpha) + alpha * G[p,q]
```

The candidate multiplies the current ambient SparseGPT mask score by `M`.
SparseGPT still performs its usual ambient Cholesky compensation, so this is a
mask-ranking heuristic, not an exact solver for the sample-dependent projected
quadratic. Each sample is assigned to one chart; overlapping concept factors,
chart uncertainty, and cross-output curvature are not represented. A zero
calibration-energy input feature receives zero measured tangent coupling and is
reported as inactive in the tangent calibration diagnostics.

```bash
python -m sparsifier Qwen/Qwen3.5-2B c4 --sparsity 0.5 \
  --tangent-alpha 0.5 --tangent-charts 8 --tangent-rank 2 \
  --tangent-samples 512 --eval-datasets wikitext2
```

The same mode is wired into Qwen, OLMo, Llama, and Gemma layer loops, including
Gemma's packed experts. It cannot be combined with the existing input-side
`--geometry-alpha` mode. Tangent estimator settings enter the checkpoint cache
identity only when tangent pruning is enabled; baseline SparseGPT cache keys
stay unchanged.

### Experiments

The small CPU reference study compares held-out local and global tangent
approximations, direct and OBS score rankings, matched-sparsity masks, projected
bias fitting, and a control where off-tangent error remains visible to a
downstream readout. It uses five synthetic seeds by default and writes a JSON
report without modifying repository results:

```bash
pixi run tangent-reference
# or
python examples/tangent_pruning_experiment.py --seeds 0 1 2 3 4 \
  --output /tmp/tangent-reference.json
```

For a full-model matched comparison, `experiments.tangent_pruning_pair` runs
ambient SparseGPT and the tangent mask sequentially with the same model,
calibration, seed, layer range, sparsity, damping, and evaluation datasets. It
writes a log per arm and a `report.json` with settings and measured perplexities.
Run it inside the repository's Pixi GPU environment, through the configured
GPU scheduler on this machine:

```bash
gbatch --gpus 1 --gpu-memory 16G --memory 32G \
  --name qwen-2b-tangent-c4-sp50 --project tangent-sparsegpt \
  --time 8:00:00 --auto-close -- \
  pixi run python -m experiments.tangent_pruning_pair \
    --model Qwen/Qwen3.5-2B --calibration c4 --nsamples 128 \
    --sparsity 0.5 --eval-dataset wikitext2 \
    --tangent-alpha 0.5 --tangent-charts 8 --tangent-rank 2 \
    --tangent-samples 512 --output-dir results/tangent/qwen3.5-2b-c4-sp50
```

Interpret the synthetic and model runs as hypothesis tests. They can support
or disconfirm score prediction, post-compensation reconstruction, and task
quality claims; neither a lower projected error nor a low tangent score by
itself establishes safety or preservation of downstream behavior.

## Demo

One can try SparseGPT via the colab demo - `demo.ipynb`. 

## Cite

If you found this work useful, please consider citing:

```
@article{frantar-sparsegpt,
  title={{SparseGPT}: Massive Language Models Can Be Accurately Pruned in One-Shot}, 
  author={Elias Frantar and Dan Alistarh},
  year={2023},
  journal={arXiv preprint arXiv:2301.00774}
}
```
