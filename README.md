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
