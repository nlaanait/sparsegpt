"""Find or record a pruned checkpoint under sparse_opt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("sparse_opt")
PRUNE_CONFIG_NAME = "prune_config.json"
_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def _current_keys(config: dict) -> dict:
    """N:M fields are ``zeros_per_group`` and ``group_size``.

    A checkpoint written before that rename stored ``prunen`` and ``prunem``.
    Lookup rewrites those so the directory still matches.
    """
    config = dict(config)
    if "zeros_per_group" not in config and "prunen" in config:
        config["zeros_per_group"] = config["prunen"]
    if "group_size" not in config and "prunem" in config:
        config["group_size"] = config["prunem"]
    config.pop("prunen", None)
    config.pop("prunem", None)
    return config


def prune_config_from_args(args) -> dict:
    """Fields that change the pruned weights. Eval-only flags are omitted."""
    probes = args.geometry_probes
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "percdamp": args.percdamp,
        "sparsity": args.sparsity,
        "zeros_per_group": args.zeros_per_group,
        "group_size": args.group_size,
        "blocksize": args.blocksize,
        "gmp": bool(args.gmp),
        "wbits": args.wbits,
        "minlayer": args.minlayer,
        "maxlayer": args.maxlayer,
        "prune_only": args.prune_only,
        "invert": bool(args.invert),
        "true_sequential": bool(args.true_sequential),
        "geometry_alpha": args.geometry_alpha,
        "geometry_probes": None if probes is None else str(probes),
    }
    # Preserve cache identities for all pre-existing pruning modes. Tangent
    # estimator settings change the mask only when the mode is enabled.
    tangent_alpha = float(getattr(args, "tangent_alpha", 0.0))
    if tangent_alpha > 0.0:
        config.update(
            tangent_alpha=tangent_alpha,
            tangent_charts=int(args.tangent_charts),
            tangent_rank=int(args.tangent_rank),
            tangent_samples=int(args.tangent_samples),
        )
    return config


def canonical_config(config: dict) -> dict:
    return json.loads(json.dumps(_current_keys(config), sort_keys=True))


def write_prune_config(path: Path, config: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(canonical_config(config), indent=2, sort_keys=True) + "\n"
    (path / PRUNE_CONFIG_NAME).write_text(payload)


def _has_weights(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_NAMES)


def _read_config(path: Path) -> dict | None:
    cfg_path = path / PRUNE_CONFIG_NAME
    if not cfg_path.is_file() or not _has_weights(path):
        return None
    return canonical_config(json.loads(cfg_path.read_text()))


def cache_roots(save: str = "", cache_dir: Path = CACHE_DIR) -> list[Path]:
    roots = [Path(cache_dir)]
    if save:
        requested = Path(save)
        roots.append(requested)
        if requested.parent != Path("."):
            roots.append(requested.parent)
    unique: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_pruned_model(config: dict, save: str = "", cache_dir: Path = CACHE_DIR) -> Path | None:
    """Return a checkpoint directory whose prune_config.json matches `config`."""
    want = canonical_config(config)
    seen = set()
    for root in cache_roots(save, cache_dir):
        if not root.exists():
            continue
        candidates = [root] if root.is_dir() else []
        if root.is_dir():
            candidates.extend(path for path in root.iterdir() if path.is_dir())
        for path in candidates:
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            found = _read_config(path)
            if found == want:
                return path
    return None


def config_slug(config: dict) -> str:
    config = canonical_config(config)
    model = str(config["model"]).rstrip("/").split("/")[-1].lower()
    parts = [model, config["dataset"]]
    if config["zeros_per_group"] and config["group_size"]:
        parts.append(f"{config['zeros_per_group']}-{config['group_size']}")
    else:
        parts.append(f"sp{config['sparsity']}")
    if config["gmp"]:
        parts.append("gmp")
    if config["wbits"] < 16:
        parts.append(f"w{config['wbits']}")
    if config["geometry_alpha"]:
        parts.append(f"geo{config['geometry_alpha']}")
    if config.get("tangent_alpha", 0.0):
        parts.append(
            f"tan{config['tangent_alpha']}-c{config['tangent_charts']}"
            f"-r{config['tangent_rank']}"
        )
    if config["nsamples"] != 128:
        parts.append(f"n{config['nsamples']}")
    if config["maxlayer"] != 1000:
        parts.append(f"maxlayer{config['maxlayer']}")
    if config["minlayer"] != -1:
        parts.append(f"minlayer{config['minlayer']}")
    if config["seed"] != 0:
        parts.append(f"seed{config['seed']}")
    if config["true_sequential"]:
        parts.append("seq")
    if config["prune_only"]:
        parts.append(str(config["prune_only"]).replace("/", "-"))
    if config["invert"]:
        parts.append("invert")
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:8]
    parts.append(digest)
    return "-".join(str(part) for part in parts)


def resolve_save_dir(save: str, config: dict, cache_dir: Path = CACHE_DIR) -> Path:
    """Use `save` unless that directory already holds a different pruned model."""
    requested = Path(save)
    if not _has_weights(requested):
        return requested
    found = _read_config(requested)
    if found == canonical_config(config):
        return requested
    return Path(cache_dir) / config_slug(config)
