"""Symbolic regression of the Qwen sparsity law with PySR.

Target: ln(perplexity / dense) for SparseGPT points above dense.
Features: sparsity and architecture (N, L, H, I, heads).

The second search regresses ln(target) on logarithmic transforms of sparsity
and parameter count.  Products in that space represent power-law exponents,
including exponents that vary with model size.

Dataset-size scaling is deliberately not fitted: every point used the same
C4 calibration sample (128 sequences of 2048 tokens).  WikiText2 and PTB are
evaluation corpora, not two controlled calibration/training-set sizes.

Run with the isolated venv that has pysr:
  /tmp/pysr-venv/bin/python -u analysis/pysr_qwen_forms.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "results" / "qwen3.5"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
OUT = ROOT / "results" / "qwen" / "pysr_forms"

ORDER = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]

# Avoid sympy-reserved names (N, I, ...).
FEATURE_COLS = [
    "s",
    "Nb",
    "n_layers",
    "hidden",
    "ffn",
    "n_head",
    "log10_s",
    "log10_Nb",
    "log10_layers",
    "log10_hidden",
]

POWER_FEATURE_COLS = [
    "ln_s",
    "ln_Nb",
    "ln_odds_s",
    "ln_damage_s",
]

CALIBRATION_SEQUENCES = 128
CALIBRATION_SEQUENCE_LENGTH = 2048


def arch_table() -> dict[str, dict[str, float]]:
    sizes = json.loads((FAMILY / "qwen3.5_family_wikitext2.json").read_text())["params_b"]
    out = {}
    for model in ORDER:
        name = model.split("/")[-1]
        root = HUB / f"models--Qwen--{name}"
        cfg_path = next(c for c in root.rglob("config.json") if ".no_exist" not in c.parts)
        cfg = json.loads(cfg_path.read_text())
        t = cfg.get("text_config", cfg)
        out[model] = {
            "Nb": float(sizes[model]),
            "n_layers": float(t["num_hidden_layers"]),
            "hidden": float(t["hidden_size"]),
            "ffn": float(t["intermediate_size"]),
            "n_head": float(t["num_attention_heads"]),
        }
    return out


def load_rows(path: Path, arch: dict[str, dict[str, float]]) -> pd.DataFrame:
    data = json.loads(path.read_text())
    dense = {
        point["model"]: float(point["perplexity"])
        for point in data["points"]
        if str(point.get("method", "")).lower() == "dense"
    }
    rows = []
    for point in data["points"]:
        if str(point.get("method", "")).lower() != "sparsegpt":
            continue
        model = point["model"]
        base = dense.get(model)
        ppl = float(point["perplexity"])
        sparsity = float(point["sparsity"])
        if base is None or base <= 0 or ppl <= base or sparsity <= 0:
            continue
        features = arch[model]
        rows.append(
            {
                "model": model,
                "s": sparsity,
                "y": math.log(ppl / base),
                "Nb": features["Nb"],
                "n_layers": features["n_layers"],
                "hidden": features["hidden"],
                "ffn": features["ffn"],
                "n_head": features["n_head"],
                "log10_s": math.log10(sparsity),
                "log10_Nb": math.log10(features["Nb"]),
                "log10_layers": math.log10(features["n_layers"]),
                "log10_hidden": math.log10(features["hidden"]),
                "ln_s": math.log(sparsity),
                "ln_Nb": math.log(features["Nb"]),
                "ln_odds_s": math.log(sparsity / (1.0 - sparsity)),
                "ln_damage_s": math.log(-math.log1p(-sparsity)),
            }
        )
    return pd.DataFrame(rows)


def make_model(seed: int) -> PySRRegressor:
    return PySRRegressor(
        niterations=80,
        populations=24,
        population_size=40,
        maxsize=20,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["log10", "exp", "neg"],
        constraints={"pow": (-1, 1)},
        nested_constraints={
            "pow": {"pow": 0, "exp": 0, "log10": 0},
            "exp": {"exp": 0, "pow": 0, "log10": 0},
            "log10": {"log10": 0, "exp": 0},
        },
        elementwise_loss="L2DistLoss()",
        model_selection="best",
        parsimony=1e-3,
        precision=64,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        temp_equation_file=True,
        progress=False,
        verbosity=1,
    )


def make_power_model(seed: int) -> PySRRegressor:
    """Search power laws by making their logarithms symbolic polynomials."""
    return PySRRegressor(
        niterations=160,
        populations=24,
        population_size=40,
        maxsize=14,
        binary_operators=["+", "-", "*", "/"],
        elementwise_loss="L2DistLoss()",
        model_selection="best",
        parsimony=1e-3,
        precision=64,
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        temp_equation_file=True,
        progress=False,
        verbosity=1,
    )


def ref_rmse(df: pd.DataFrame) -> float:
    """RMSE of the published 3-param reference form on these rows."""
    from scipy.optimize import least_squares

    s = df["s"].to_numpy()
    y = df["y"].to_numpy()
    n = df["Nb"].to_numpy()

    def residual(theta):
        a, b0, b1 = theta
        return a * np.power(s, b0 + b1 * np.log10(n)) - y

    best = None
    for guess in ([12.5, 4.1, 2.7], [11.4, 3.9, 2.6], [10.0, 4.0, 1.0]):
        fit = least_squares(residual, guess, bounds=([1e-6, -5, -10], [1e4, 30, 10]), max_nfev=4000)
        err = float(np.sqrt(np.mean(np.square(residual(fit.x)))))
        if best is None or err < best:
            best = err
    return best


def run_eval(name: str, path: Path, arch: dict[str, dict[str, float]], seed: int) -> None:
    df = load_rows(path, arch)
    OUT.mkdir(parents=True, exist_ok=True)
    X = df[FEATURE_COLS]
    y = df["y"]
    baseline = ref_rmse(df)
    print(f"\n=== {name} ===", flush=True)
    print(f"points={len(df)} models={sorted({m.split('/')[-1] for m in df['model']})}", flush=True)
    print(f"reference 3-param RMSE={baseline:.4f}", flush=True)

    model = make_model(seed)
    model.fit(X, y)
    equations = model.equations_
    equations.to_csv(OUT / f"{name}_equations.csv", index=False)

    print(f"{'complexity':>10} {'loss':>12} {'score':>10}  equation", flush=True)
    for _, row in equations.sort_values("complexity").iterrows():
        # loss is MSE under L2DistLoss; report RMSE to match principle page
        rmse = math.sqrt(float(row["loss"])) if float(row["loss"]) >= 0 else float("nan")
        eq = row["equation"]
        print(
            f"{int(row['complexity']):10d} {rmse:12.4f} {float(row['score']):10.4f}  {eq}",
            flush=True,
        )

    best = model.get_best()
    pred = model.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y.to_numpy()) ** 2)))
    print(f"selected: {best}", flush=True)
    print(f"selected RMSE={rmse:.4f} (ref {baseline:.4f})", flush=True)
    (OUT / f"{name}_best.txt").write_text(
        f"equation={best['equation']}\n"
        f"complexity={int(best['complexity'])}\n"
        f"RMSE={rmse:.6f}\n"
        f"reference_RMSE={baseline:.6f}\n"
        f"n={len(df)}\n"
    )


def run_power_eval(name: str, path: Path, arch: dict[str, dict[str, float]], seed: int) -> None:
    """Search alternatives of the form y = exp(f(log(s), log(N), ...))."""
    df = load_rows(path, arch)
    X = df[POWER_FEATURE_COLS]
    log_y = np.log(df["y"].to_numpy())

    print(f"\n=== {name}: power-law search ===", flush=True)
    print(
        "target=ln(ln(perplexity/dense)); "
        f"features={','.join(POWER_FEATURE_COLS)}",
        flush=True,
    )

    model = make_power_model(seed)
    model.fit(X, log_y)
    equations = model.equations_
    equations.to_csv(OUT / f"{name}_power_equations.csv", index=False)

    print(f"{'complexity':>10} {'log RMSE':>12} {'score':>10}  ln(y) equation", flush=True)
    for _, row in equations.sort_values("complexity").iterrows():
        rmse = math.sqrt(float(row["loss"])) if float(row["loss"]) >= 0 else float("nan")
        print(
            f"{int(row['complexity']):10d} {rmse:12.4f} "
            f"{float(row['score']):10.4f}  {row['equation']}",
            flush=True,
        )

    best = model.get_best()
    pred_log_y = model.predict(X)
    log_rmse = float(np.sqrt(np.mean(np.square(pred_log_y - log_y))))
    y_rmse = float(
        np.sqrt(np.mean(np.square(np.exp(pred_log_y) - df["y"].to_numpy())))
    )
    print(f"selected ln(y): {best}", flush=True)
    print(f"selected log-RMSE={log_rmse:.4f}; y-RMSE={y_rmse:.4f}", flush=True)
    (OUT / f"{name}_power_best.txt").write_text(
        "The equation predicts ln(y), where y = ln(perplexity / dense).\n"
        f"ln_y_equation={best['equation']}\n"
        f"y_equation=exp({best['equation']})\n"
        f"complexity={int(best['complexity'])}\n"
        f"log_RMSE={log_rmse:.6f}\n"
        f"y_RMSE={y_rmse:.6f}\n"
        f"n={len(df)}\n"
    )


def write_dataset_size_note() -> None:
    calibration_tokens = CALIBRATION_SEQUENCES * CALIBRATION_SEQUENCE_LENGTH
    note = {
        "dataset_size_power_law_identifiable": False,
        "reason": (
            "All Qwen SparseGPT points use one fixed C4 calibration size. "
            "WikiText2 and PTB are different evaluation distributions, not "
            "controlled values of calibration or training dataset size."
        ),
        "calibration_sequences": CALIBRATION_SEQUENCES,
        "calibration_sequence_length": CALIBRATION_SEQUENCE_LENGTH,
        "calibration_tokens_per_point": calibration_tokens,
        "unique_calibration_sizes": [calibration_tokens],
    }
    (OUT / "dataset_size_identifiability.json").write_text(
        json.dumps(note, indent=2) + "\n"
    )


def main() -> None:
    arch = arch_table()
    OUT.mkdir(parents=True, exist_ok=True)
    write_dataset_size_note()
    run_eval("wikitext2", FAMILY / "qwen3.5_family_sparsity_wikitext2.json", arch, seed=0)
    run_eval("ptb", FAMILY / "qwen3.5_family_sparsity_ptb.json", arch, seed=1)
    run_power_eval(
        "wikitext2",
        FAMILY / "qwen3.5_family_sparsity_wikitext2.json",
        arch,
        seed=2,
    )
    run_power_eval(
        "ptb",
        FAMILY / "qwen3.5_family_sparsity_ptb.json",
        arch,
        seed=3,
    )


if __name__ == "__main__":
    main()
