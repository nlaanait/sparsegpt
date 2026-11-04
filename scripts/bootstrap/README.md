# GH200 bare-metal bootstrap

Scripts to bring up an Ubuntu **GH200** (Grace Hopper, `linux-aarch64`, CUDA 13)
node for the `tangent-aware-pruning` branch: system packages, NVIDIA check,
Pixi, gflow (`runqd`), repo env, and verify.

## One shot

On the node (as a user with passwordless sudo for apt):

```bash
git clone --branch tangent-aware-pruning --single-branch \
  https://github.com/nlaanait/sparsegpt.git
cd sparsegpt
./scripts/bootstrap/bootstrap.sh
```

Or copy only `scripts/bootstrap/` first; phase 5 clones the repo into
`BOOTSTRAP_WORKDIR` (default: repo root inferred from the script path).

## After bootstrap

```bash
gflowd status
ginfo
pixi run cuda-check

# CPU synthetic tangent study (already run by verify)
pixi run tangent-reference

# Fork kernel build + schedule jobs without waiting an hour
./scripts/schedule_tangent_gh200.sh
```

### causal-conv1d without blocking

There is no aarch64 cu130 wheel on PyPI for our torch pin; a from-source build
is ~an hour. Bootstrap **strips** that path dep so `pixi install` is fast, then
phase 7 submits `scripts/submit_causal_conv1d_build_gflow.sh` (shared 2G GPU).

| Job | When | Kernels |
|---|---|---|
| `causal-conv1d-…-build` | immediately (background) | builds wheel → `~/.pixi/wheels/cu130/` |
| tangent smoke | immediately, no `--depends-on` | `--pytorch-attention` fallback |
| full tangent pair | `--depends-on` build id | `ensure_causal_conv1d.sh` then fast path |

Manual pieces:

```bash
./scripts/submit_causal_conv1d_build_gflow.sh   # prints BUILD_JOB_ID=…
# …schedule any non-kernel jobs now…
gbatch --depends-on "$BUILD_JOB_ID" --gpus 1 … -- \
  bash -lc './scripts/ensure_causal_conv1d.sh && pixi run python -m …'
```

Needs `nvcc` (CUDA 13 toolkit). Build sets `TORCH_CUDA_ARCH_LIST` from
`nvidia-smi` compute capability (GH200 → `9.0+PTX`).

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `BOOTSTRAP_BRANCH` | current HEAD, else `tangent-aware-pruning` | Git branch for phase 5 |
| `BOOTSTRAP_WORKDIR` | this repo root | Checkout / `pixi install` directory |
| `BOOTSTRAP_REPO_URL` | `https://github.com/nlaanait/sparsegpt.git` | Clone URL if no `.git` |
| `BOOTSTRAP_REQUIRE_GPU` | `1` | Fail phase 2 without `nvidia-smi` |
| `BOOTSTRAP_ALLOW_STRIP_CONV1D` | `1` | Drop local `causal-conv1d` path pin if wheel missing |
| `BOOTSTRAP_SUBMIT_CONV1D_BUILD` | `1` | Phase 7: gbatch the wheel build in the background |
| `CAUSAL_CONV1D_WHEEL` | unset | Path to `causal_conv1d-*-linux_aarch64.whl` to stage |
| `BOOTSTRAP_ONLY` | unset | Comma list of phases to run (`1`…`7`) |
| `BOOTSTRAP_SKIP` | unset | Comma list of phases to skip |
| `HF_TOKEN` | unset | Hugging Face token for gated models |

## Phases

1. `01-system-packages.sh` — build tools, tmux, git-lfs, …
2. `02-nvidia-check.sh` — `nvidia-smi` + NVML (drivers are assumed preinstalled)
3. `03-pixi.sh` — official Pixi installer → `~/.pixi/bin`
4. `04-gflow.sh` — `uv tool install runqd`, `gflowd up`, optional systemd user unit + Cursor MCP
5. `05-repo-env.sh` — checkout branch, stage/strip `causal-conv1d`, `pixi install`
6. `06-verify.sh` — cuda-check, geometry/SparseGPT tests, short CPU tangent reference
7. submit `causal-conv1d` wheel build via gbatch (non-blocking)

## Notes for GH200

- Workspace `pixi.toml` already targets `{ platform = "linux-aarch64", cuda = "13.0" }`, which matches GH200 + recent drivers.
- Missing local `causal-conv1d` wheel → strip pin, install, build in background.
- All GPU experiments go through **gflow** (`gbatch`), not a bare shell.
- Atlas-Inference/gdn hub kernels in the Qwen path are SM121 (GB10). On GH200
  prefer `--pytorch-attention` for GDN, or validate hub kernel support separately;
  the local `causal-conv1d` wheel still helps the conv op when not forced to PyTorch.
