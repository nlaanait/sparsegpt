# SparseGPT fork

This fork measures unstructured SparseGPT on Qwen and Gemma 4 and fits one
sparsity law per family. Upstream OPT / BLOOM / LLaMA entry points are still
in `sparsegpt/`. The measurements that matter now are the Qwen and Gemma
sweeps.

## Perplexity

A dense WikiText2 perplexity for these models is on the order of the Qwen
dense scores, about 10–20. PTB is the same kind of number, a bit higher.
Qwen’s saved dense scores live in `figures/qwen/`. A dense score in the
hundreds or thousands is not a language-model perplexity. Do not fit it, do
not quote it, and do not treat it as a point on the law.

`experiments/gemma_gate.py` only checks that the layer-wise loop matches
`model()` on the same windows. Those two paths once agreed near 12,000 on
WikiText2. Agreement is not a valid score.

When a Gemma dense score is implausible, trace `sparsegpt/gemma.py`:
`get_gemma`, `gemma_eval`, `gemma_reference_nlls`, and the inputs from
`sparsegpt/datautils.py`. Logs are `figures/gemma/logs/<model>_<dataset>_sdpa_dense.log`.
The line `Perplexity:` is the number that was stored. Leave `analysis/sweep.py`,
`experiments/run_gemma.sh`, and any running `python -m analysis.sweep` or
`python -m sparsegpt.gemma` process alone. Do not load another full model
onto the GPU while one of those is running.

## Plots

Three pages, per family and per eval (WikiText2 unsuffixed, PTB suffixed
`_ptb`):

- `*_sparsity_perplexity.html` — raw perplexity versus sparsity.
- `*_sparsity_loglog.html` — `ln(perplexity / dense)` versus sparsity, both
  axes logarithmic. Reference: `plot_loglog` in `qwen_figure1_family.py`.
- `*_sparsity_normalized.html` — perplexity divided by that model’s own dense
  perplexity, versus sparsity. The y-axis is logarithmic. A dashed line sits
  at 1. Each model is markers plus its curve from the shared law. Reference:
  `plot_normalized` in `qwen_figure1_family.py`. The Gemma page is
  `figures/gemma/gemma_sparsity_normalized.html`, written by `_normalized` in
  `analysis/plots.py`.

The normalized page is not `ln(perplexity / dense)` against
`s ^ (b1 log10(N / N*))`. That collapse is not a figure we publish.

## The law

```
ln(perplexity / dense) = a · (N / N*) ^ (b1 log10 s)
```

Fit separately on WikiText2 and on PTB. `s` is unstructured sparsity. `N`
and `N*` are stored parameters in billions, counted as safetensors bytes / 2.
Google’s effective parameter counts are not `N`. Points at or below a model’s
own dense perplexity stay out of the fit. The fit is `analysis/fit.py`.

## Gemma sweep

Base repos only: `google/gemma-4-E2B`, `E4B`, `12B`, `26B-A4B`, `31B`. The
12B checkpoint is the unified text tower (`Gemma4UnifiedForCausalLM`).
`26B-A4B` is the MoE text tower (128 experts, top-8); the fit still uses
total stored N (safetensors bytes / 2), not active parameters. The
instruction-tuned measurements are in `figures/gemma-it/`.
Calibration is C4, 128 sequences of 2048, seed 0. Eval is raw WikiText2 and
PTB at sequence length 2048, SDPA, text tower only. No magnitude pruning.
Results are `figures/gemma/<short-name>_sparsity.json`.
