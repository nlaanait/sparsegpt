# SparseGPT fork

This fork measures unstructured SparseGPT on Qwen, Gemma 4, OLMo 2 Instruct,
and Llama 3, and fits one sparsity law per family. Those four families run
as `python -m sparsifier`. OPT and BLOOM stay in `sparsegpt/`
(`python -m sparsegpt.opt`, `python -m sparsegpt.bloom`).

## Layout

- `src/sparsifier/` — the pixi project. One run: a pruner, a metric, and a registered model.
  Perplexity is the metric recorded today. Bits per byte is the other
  metric, once an eval counts bytes. `sparsifier.prune` owns the SparseGPT
  and geometry solvers plus the pruner adapters that bind them to a layer.
  `sparsifier.models` holds the catalog (Hugging Face id, family, runner,
  stored parameter count), the family protocols, and the Qwen, OLMo, Llama,
  and Gemma layer loops.
  It does not import `experiments`. The Click command
  is `sparsifier.cli` (`python -m sparsifier`). N:M flags are
  `--zeros-per-group` and `--group-size`. `sparsifier.data` loads the
  calibration windows (`loaders`) and finds a pruned checkpoint (`cache`).
  `sparsifier.sweep` is the run
  config, `SweepSpec`, `SweepRecord`, `validate`, and
  `sparsifier.sweep.commands` (the argv and the tenth-step ladder). Family
  sweep records stay in `experiments` and are `SweepRecord` values.
- `sparsegpt/` — the legacy package. OPT and BLOOM still run from here.
  The copied Qwen, OLMo, Llama, and Gemma modules stay on disk for parity
  tests. The Click command does not import them.
- `experiments/` — one sweep record per family and calibration, plus the
  submit scripts. The default prune namespace is
  `sparsifier.sweep.default_prune_args`. `experiments.sweeps` lists the
  records.
  `python -m experiments.reproduce plan <sweep>` prints the commands still
  missing. `python -m experiments.reproduce figures <sweep>` rewrites the
  three pages and the family principle page from saved JSON.
- `analysis/` — load perplexity rows, fit the law, write the three pages,
  run a sweep.
- `drivers/` — command-line entry points for a sweep or a Qwen figure.
- `results/<family>/` — that sweep's JSON, likelihood dumps, and HTML.
  Cross-family pages are `results/sparsity_principle.html` and
  `results/universal_scaling_law.html`.
- Qwen 3.5 family bundles live in `results/qwen3.5/`. Qwen 3.6 and 3.8
  per-model JSON lives in `results/qwen/`. Run logs under `results/**/logs/`
  are gitignored.

## Perplexity

A dense WikiText2 perplexity for these models is on the order of the Qwen
dense scores, about 10–20. Larger OLMo 2 Instruct can land a bit lower (~4–9).
PTB is the same kind of number, a bit higher. Qwen 3.5 dense scores live
in `results/qwen3.5/`. Qwen 3.6 and 3.8 live in `results/qwen/`. A dense
score in the hundreds or thousands is not a language-model perplexity. Do
not fit it, do not quote it, and do not treat it as a point on the law.

`experiments/gemma_gate.py` checks that the layer-wise loop matches
`model()` on the same windows for `google/gemma-4-E2B-it`. Those two paths
once agreed near 12,000 on WikiText2. The gap is not stored. A dense score
in the base sweep is the `Perplexity:` line from `python -m sparsifier`.

When a Gemma dense score is implausible, trace `sparsifier/models/gemma.py`:
`get_gemma`, `gemma_eval`, `gemma_reference_nlls`, and the inputs from
`sparsifier/data/loaders.py`. Logs are `results/gemma/logs/<model>_<dataset>_sdpa_dense.log`.
The line `Perplexity:` is the number that was stored. Leave `analysis/sweep.py`,
`experiments/run_gemma.sh`, and any running `python -m analysis.sweep` or
`python -m sparsifier` process alone. Do not load another full model
onto the GPU while one of those is running — submit through gflow instead
(see Scheduling below).

## Plots

Three pages, per family and per eval (WikiText2 unsuffixed, PTB suffixed
`_ptb`):

- `*_sparsity_perplexity.html` — raw perplexity versus sparsity.
- `*_sparsity_loglog.html` — `ln(perplexity / dense)` versus sparsity, both
  axes logarithmic. Reference: `plot_loglog` in `drivers.qwen_figure1_family.py`.
- `*_sparsity_normalized.html` — perplexity divided by that model’s own dense
  perplexity, versus sparsity. The y-axis is logarithmic. A dashed line sits
  at 1. Each model is markers plus its curve from the shared law. Reference:
  `plot_normalized` in `drivers.qwen_figure1_family.py`. The Gemma page is
  `results/gemma/gemma_sparsity_normalized.html`, written by `_normalized` in
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

Base repos: `google/gemma-4-E2B`, `E4B`, `12B`, `26B-A4B`, `31B`. The 12B
checkpoint is the unified text tower (`Gemma4UnifiedForCausalLM`). `26B-A4B`
is the MoE text tower (128 experts, top-8); the fit still uses total stored
N (safetensors bytes / 2), not active parameters. Instruction-tuned
measurements live in `results/gemma-it/`. Calibration is C4, 128 sequences of
2048, seed 0. Eval is raw WikiText2 and PTB at sequence length 2048, SDPA,
text tower only. No magnitude pruning. Results are
`results/gemma/<short-name>_sparsity.json`.

## OLMo 2 sweep

Instruction-tuned repos only: `allenai/OLMo-2-0425-1B-Instruct`,
`OLMo-2-1124-7B-Instruct`, `OLMo-2-1124-13B-Instruct`,
`OLMo-2-0325-32B-Instruct`. Calibration is C4, 128 × 2048, seed 0. Eval is
raw WikiText2 and PTB (no chat template). Dense WikiText2 must clear
`SMOKE_PPL_RANGE` in `experiments/olmo.py` before any prune. Results land
under `results/olmo2/`. Driver: `drivers.olmo2_sparsity`.

## Scheduling (gflow)

All GPU experiments on this machine go through **gflow**. Do not start
`python -m sparsifier`, `python -m sparsegpt.*`, `python -m drivers.olmo2_sparsity`, or `analysis.sweep`
in a bare tmux/shell while another GPU job may run. Ad-hoc `pgrep` wait loops
are not a queue.

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

### Inspect

```bash
gqueue
gjob show <id>
gjob log <id>
ginfo
```

Cancel with `gcancel <id>`. Do not kill random `python -m sparsifier` or `python -m sparsegpt.*`
processes that gflow owns; cancel the job instead.
