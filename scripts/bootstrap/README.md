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

# GPU smokes through gflow — do not launch bare python on the GPU
./scripts/submit_tangent_smoke_gflow.sh
./scripts/submit_tangent_pair_gflow.sh
```

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `BOOTSTRAP_BRANCH` | `tangent-aware-pruning` | Git branch for phase 5 |
| `BOOTSTRAP_WORKDIR` | this repo root | Checkout / `pixi install` directory |
| `BOOTSTRAP_REPO_URL` | `https://github.com/nlaanait/sparsegpt.git` | Clone URL if no `.git` |
| `BOOTSTRAP_REQUIRE_GPU` | `1` | Fail phase 2 without `nvidia-smi` |
| `BOOTSTRAP_ALLOW_STRIP_CONV1D` | `1` | Drop local `causal-conv1d` path pin if wheel missing |
| `CAUSAL_CONV1D_WHEEL` | unset | Path to `causal_conv1d-*-linux_aarch64.whl` to stage |
| `BOOTSTRAP_ONLY` | unset | Comma list of phases to run (`1`…`6`) |
| `BOOTSTRAP_SKIP` | unset | Comma list of phases to skip |
| `HF_TOKEN` | unset | Hugging Face token for gated models |

## Phases

1. `01-system-packages.sh` — build tools, tmux, git-lfs, …
2. `02-nvidia-check.sh` — `nvidia-smi` + NVML (drivers are assumed preinstalled)
3. `03-pixi.sh` — official Pixi installer → `~/.pixi/bin`
4. `04-gflow.sh` — `uv tool install runqd`, `gflowd up`, optional systemd user unit + Cursor MCP
5. `05-repo-env.sh` — checkout branch, stage/strip `causal-conv1d`, `pixi install`
6. `06-verify.sh` — cuda-check, geometry/SparseGPT tests, short CPU tangent reference

## Notes for GH200

- Workspace `pixi.toml` already targets `{ platform = "linux-aarch64", cuda = "13.0" }`, which matches GH200 + recent drivers.
- The absolute `causal-conv1d` wheel path in `pixi.toml` is rewritten to
  `~/.pixi/wheels/cu130/...` when `CAUSAL_CONV1D_WHEEL` is provided; otherwise
  the pin is stripped so install still works (Qwen3.5 linear-attention CUDA
  kernels may fall back).
- All GPU experiments go through **gflow** (`gbatch`), not a bare shell.
