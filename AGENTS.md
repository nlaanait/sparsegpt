# SparseGPT fork

This fork measures unstructured SparseGPT on Qwen, Gemma 4, OLMo 2
Instruct, and Llama 3, and fits one sparsity law per family. Upstream OPT /
BLOOM / LLaMA entry points are still in `sparsegpt/`. The measurements that
matter now are the Qwen, Gemma, OLMo 2, and Llama 3 sweeps.

## Perplexity

A dense WikiText2 perplexity for these models is on the order of the Qwen
dense scores, about 10–20. Larger OLMo 2 Instruct can land a bit lower (~4–9).
PTB is the same kind of number, a bit higher. Qwen’s saved dense scores live
in `figures/qwen/`. A dense score in the hundreds or thousands is not a
language-model perplexity. Do not fit it, do not quote it, and do not treat
it as a point on the law.

`experiments/gemma_gate.py` only checks that the layer-wise loop matches
`model()` on the same windows. Those two paths once agreed near 12,000 on
WikiText2. Agreement is not a valid score.

When a Gemma dense score is implausible, trace `sparsegpt/gemma.py`:
`get_gemma`, `gemma_eval`, `gemma_reference_nlls`, and the inputs from
`sparsegpt/datautils.py`. Logs are `figures/gemma/logs/<model>_<dataset>_sdpa_dense.log`.
The line `Perplexity:` is the number that was stored. Leave `analysis/sweep.py`,
`experiments/run_gemma.sh`, and any running `python -m analysis.sweep` or
`python -m sparsegpt.gemma` process alone. Do not load another full model
onto the GPU while one of those is running — submit through gflow instead
(see Scheduling below).

## Plots

Three pages, per family and per eval (WikiText2 unsuffixed, PTB suffixed
`_ptb`):

- `*_sparsity_perplexity.html` — raw perplexity versus sparsity.
- `*_sparsity_loglog.html` — scale-free collapse,
  `N^(−β) · ln(perplexity / dense)` versus sparsity, both axes logarithmic,
  with one curve `α · s^b`. Reference: `figures/universal_scaling_law.html`.
- `*_sparsity_normalized.html` — perplexity divided by that model’s own dense
  perplexity, versus sparsity. The y-axis is logarithmic. A dashed line sits
  at 1. Each model is markers plus its curve from the scale-free law.
  Reference: `figures/universal_scaling_law.html` and `plot_normalized` in
  `qwen_figure1_family.py`.

The normalized page is not `ln(perplexity / dense)` against
`s ^ (b1 log10(N / N*))`. That reference-law collapse is not a figure we publish.

## The law

```
ln(perplexity / dense) = α · N^β · s^b
```

This is the scale-free form on `figures/sparsity_principle.html`. Size and
sparsity factorize; changing the unit of N only rescales α. Fit separately
on each eval. `s` is unstructured sparsity. `N` is stored parameters in
billions, counted as safetensors bytes / 2. Google’s effective parameter
counts are not `N`. Points at or below a model’s own dense perplexity stay
out of the fit. The fit is `solve_scale_free_law` in `analysis/fit.py`.
The older `a · (N / N*)^(b1 log10 s)` form is a comparison row on the Qwen
section of `figures/sparsity_principle.html`. Family pages and the universal
figure use the scale-free law.

## Canonical pages

Two HTML files are the cross-family record. Per-family files under
`figures/<family>/` do not replace them.

- `figures/sparsity_principle.html` — the scale-free law on the Qwen family,
  plus a table of the same fit for every other family that has enough C4
  points. Written by `python -m sparsity_principle`.
- `figures/universal_scaling_law.html` — one row per family. Left panel is
  the collapse `N^(−β) · ln(perplexity / dense)` versus sparsity, both axes
  logarithmic, with the family curve `α · s^b`. Right panel is
  perplexity/dense versus sparsity, y logarithmic, dashed line at 1.
  Each panel has its own menu and shows one pair at a time: `c4→wt`,
  `c4→ptb`, `wt→c4`, `wt→ptb`. The default is `c4→wt`. Written by
  `python -m universal_scaling`.

When a family gains C4 WikiText2 or PTB points above its own dense
perplexity, regenerate both pages:

```
pixi run --frozen -e gh200 python -m sparsity_principle
pixi run --frozen -e gh200 python -m universal_scaling
```

Each family keeps its own α, β, b, and each calibration → eval pair is fit
on its own. Gemma-4 on the universal page is carried forward from the
previous figure until `figures/gemma/` has sparsity JSON again, so its
`wt→` menu entries are empty. WikiText2-calibrated sweeps also keep their
own family pages under `figures/*_wikitext2_cal/`.

## Gemma sweep

Base repos: `google/gemma-4-E2B`, `E4B`, `12B`, `26B-A4B`, `31B`. The 12B
checkpoint is the unified text tower (`Gemma4UnifiedForCausalLM`). `26B-A4B`
is the MoE text tower (128 experts, top-8); the fit still uses total stored
N (safetensors bytes / 2), not active parameters. Instruction-tuned
measurements live in `figures/gemma-it/`. Calibration is C4, 128 sequences of
2048, seed 0. Eval is raw WikiText2 and PTB at sequence length 2048, SDPA,
text tower only. No magnitude pruning. Results are
`figures/gemma/<short-name>_sparsity.json`. The WikiText2-calibrated
counterpart for the instruction-tuned models on the universal Gemma row
(`E4B-it`, `12B-it`, `31B-it`; eval C4 and PTB) writes
`figures/gemma_it_wikitext2_cal/`. Driver: `gemma_it_wikitext2_cal_sparsity.py`.

## OLMo 2 sweep

Instruction-tuned repos only: `allenai/OLMo-2-0425-1B-Instruct`,
`OLMo-2-1124-7B-Instruct`, `OLMo-2-1124-13B-Instruct`,
`OLMo-2-0325-32B-Instruct`. Calibration is C4, 128 × 2048, seed 0. Eval is
raw WikiText2 and PTB (no chat template). Dense WikiText2 must clear
`SMOKE_PPL_RANGE` in `experiments/olmo.py` before any prune. Results land
under `figures/olmo2/`. Driver: `olmo2_sparsity.py`.

## Llama 3 sweep

One family for Llama 3.1 and Llama 3.2. Repos, in size order:
`meta-llama/Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.1-8B`,
`Llama-3.2-11B-Vision`, `Llama-3.1-70B`, `Llama-3.2-90B-Vision`. Plots, the
fit, and `figures/llama3/` are shared. Each JSON file records the full repo
id. Stored N is safetensors bytes / 2 of that full checkpoint. The 11B and
90B Vision runs load the text tower; text-only forward skips cross-attention,
and so does the prune. Calibration is C4, 128 × 2048, seed 0. Eval is raw
WikiText2 and PTB. Dense WikiText2 must clear `SMOKE_PPL_RANGE` in
`experiments/llama.py` before any prune. Driver: `llama3_sparsity.py`.
The WikiText2-calibrated counterpart (eval C4 and PTB) writes
`figures/llama3_wikitext2_cal/`. Driver: `llama3_wikitext2_cal_sparsity.py`.

## Scheduling (gflow)

All GPU experiments on this machine go through **gflow**. Do not start
`python -m sparsegpt.*`, `olmo2_sparsity.py`, or `analysis.sweep` in a bare
tmux/shell while another GPU job may run. Ad-hoc `pgrep` wait loops are not
a queue.

### Daemon

```bash
gflowd status          # must be Running
gflowd up              # if down; systemd user unit gflowd.service preferred
```

MCP for Cursor is `gflow mcp serve` via `~/.cursor/mcp.json`. The daemon must
be up for MCP tools to work.

### Submit

```bash
gbatch \
  --gpus 1 \
  --gpu-memory <CAP> \
  --memory <HOST_CAP> \
  --name <experiment-id> \
  --project <family>-sparsegpt \
  --time <HH:MM:SS> \
  --auto-close \
  --depends-on <prev-job-id> \   # when serializing a family
  -- pixi run python -u <driver> ...
```

- `--name` / experiment id: lowercase, stable, unique. Pattern:
  `<family>-<size>-<variant>-<cal>-sparsegpt-<phase>`
  Example: `olmo2-7b-instruct-c4-sparsegpt-sweep`.
- `--project`: family bucket, e.g. `olmo2-sparsegpt`, `gemma-sparsegpt`.
- `--gpu-memory`: expected peak device memory for SparseGPT (layer-wise still
  spikes on activations). Cap to what the model needs so shared packing stays
  honest; on this single-GPU host exclusive jobs (`--gpus 1` without
  `--shared`) serialize. Co-locate small jobs with `--shared --gpu-memory`.
- `--memory`: host RAM cap for the full checkpoint on CPU plus activations.
- One model (or one clearly scoped phase) per job. Prefer
  `--depends-on` / `--depends-on-all` over launching competing waiters.
- Resumable drivers (`--skip-smoke`, existing JSON points, prune cache) are
  required so a requeue does not redo finished work.

### OLMo 2 VRAM / RAM caps (expected)

| Model | Experiment id | `--gpu-memory` | `--memory` | `--time` |
|---|---|---|---|---|
| 1B-Instruct | `olmo2-1b-instruct-c4-sparsegpt-sweep` | `16G` | `32G` | `8:00:00` |
| 7B-Instruct | `olmo2-7b-instruct-c4-sparsegpt-sweep` | `32G` | `48G` | `16:00:00` |
| 13B-Instruct | `olmo2-13b-instruct-c4-sparsegpt-sweep` | `48G` | `64G` | `24:00:00` |
| 32B-Instruct | `olmo2-32b-instruct-c4-sparsegpt-sweep` | `96G` | `96G` | `48:00:00` |

Submit helper: `experiments/submit_olmo2_gflow.sh`.

### Llama 3 VRAM / RAM caps (expected)

| Model | Experiment id | `--gpu-memory` | `--memory` | `--time` |
|---|---|---|---|---|
| 1B | `llama3-1b-base-c4-sparsegpt-sweep` | `16G` | `32G` | `8:00:00` |
| 3B | `llama3-3b-base-c4-sparsegpt-sweep` | `24G` | `48G` | `16:00:00` |
| 8B | `llama3-8b-base-c4-sparsegpt-sweep` | `32G` | `64G` | `24:00:00` |
| 11B-Vision | `llama3-11b-vision-c4-sparsegpt-sweep` | `48G` | `64G` | `24:00:00` |
| 70B | `llama3-70b-base-c4-sparsegpt-sweep` | `64G` | `224G` | `72:00:00` |
| 90B-Vision | `llama3-90b-vision-c4-sparsegpt-sweep` | `64G` | `320G` | `96:00:00` |

Submit helper: `experiments/submit_llama3_gflow.sh`. These jobs are `--shared` and are not chained. The WikiText2-calibrated jobs use the same caps with `wikitext2` in the experiment id. Helper: `experiments/submit_llama3_wikitext2_cal_gflow.sh`.

### Inspect

```bash
gqueue
gjob show <id>
gjob log <id>
ginfo
```

Cancel with `gcancel <id>`. Do not kill random `python -m sparsegpt.*`
processes that gflow owns; cancel the job instead.
