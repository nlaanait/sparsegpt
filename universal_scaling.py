"""Cross-family scale-free pages.

`figures/sparsity_principle.html` states the law on Qwen and tables the same
fit for every other family that has enough C4 points.
`figures/universal_scaling_law.html` draws one row per family. Each panel
shows one calibration → eval pair at a time. The menu on the panel switches
among c4→wt, c4→ptb, wt→c4, and wt→ptb.

Gemma-4 has no result JSON in this tree. Its C4 curves are carried forward
from the previous universal figure. The wt→ pairs are empty for Gemma until
that JSON exists.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import numpy as np

from analysis.fit import solve_scale_free_law
from experiments.llama import LLAMA3, stored_params as llama_stored_params
from experiments.olmo import OLMO2, stored_params as olmo_stored_params
from experiments.qwen import ROOT

PAGE = ROOT / "figures" / "universal_scaling_law.html"
COLORS = ("#1d4ed8", "#0f766e", "#b45309", "#6d28d9", "#be123c", "#0369a1")
CURVE_S = np.logspace(np.log10(0.08), np.log10(0.99), 80)
GEMMA_CACHE = ROOT / "figures" / "gemma" / "universal_carried.json"
# One calibration → eval pair. A panel shows exactly one of these.
PROTOCOLS = (
    ("c4-wt", "c4→wt", "c4", "wikitext2"),
    ("c4-ptb", "c4→ptb", "c4", "ptb"),
    ("wt-c4", "wt→c4", "wikitext2", "c4"),
    ("wt-ptb", "wt→ptb", "wikitext2", "ptb"),
)


def _label(model: str) -> str:
    return model.rstrip("/").split("/")[-1]


def _usable_dense(dense: float) -> bool:
    return 0.5 < dense < 80.0


def _load_family(spec, sizes: dict[str, float]) -> dict:
    """Points above each model's own dense perplexity, plus every positive ratio."""
    above = []
    ratios = []
    seen: list[str] = []
    for model in spec.models:
        path = spec.result_dir / f"{_label(model).lower()}_sparsity.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get("calibration") != spec.calibration:
            continue
        billions = float(sizes[model])
        evals = set(spec.eval_datasets)
        dense: dict[str, float] = {}
        for point in payload["points"]:
            if point["method"] == "dense" and point["dataset"] in evals:
                dense[point["dataset"]] = float(point["perplexity"])
        entered = False
        for point in payload["points"]:
            dataset = point["dataset"]
            if point["method"] != "sparsegpt" or dataset not in evals or dataset not in dense:
                continue
            base = dense[dataset]
            perplexity = float(point["perplexity"])
            if not _usable_dense(base) or perplexity <= 0 or not np.isfinite(perplexity):
                continue
            ratio = perplexity / base
            sparsity = float(point["sparsity"])
            if sparsity <= 0 or sparsity >= 1:
                continue
            row = {
                "model": model,
                "label": _label(model),
                "dataset": dataset,
                "s": sparsity,
                "ratio": ratio,
                "log_ratio": float(np.log(ratio)),
                "n": billions,
            }
            ratios.append(row)
            entered = True
            if ratio > 1.0:
                above.append(row)
        if entered:
            seen.append(model)
    return {"spec": spec, "above": above, "ratios": ratios, "models": seen, "sizes": sizes}


def _fit(rows: list[dict]) -> tuple[float, float, float, float] | None:
    models = {row["model"] for row in rows}
    if len(models) < 2 or len(rows) < 4:
        return None
    alpha, beta, exponent, rmse = solve_scale_free_law(
        np.asarray([row["s"] for row in rows], dtype=float),
        np.asarray([row["log_ratio"] for row in rows], dtype=float),
        np.asarray([row["n"] for row in rows], dtype=float),
    )
    return alpha, beta, exponent, rmse


def fit_c4_families() -> list[dict]:
    """OLMo 2 and Llama 3, C4 calibration, WikiText2 and PTB."""
    loaded = [
        ("OLMo 2 Instruct", _load_family(OLMO2, olmo_stored_params())),
        ("Llama 3", _load_family(LLAMA3, llama_stored_params())),
    ]
    families = []
    for title, payload in loaded:
        laws = {}
        for dataset in ("wikitext2", "ptb"):
            fitted = _fit([row for row in payload["above"] if row["dataset"] == dataset])
            if fitted is not None:
                laws[dataset] = fitted
        pooled = _fit(payload["above"])
        if pooled is not None:
            laws["pooled"] = pooled
        payload["title"] = title
        payload["laws"] = laws
        families.append(payload)
    return families


def _fmt(alpha: float, beta: float, exponent: float) -> str:
    return f"{alpha:.2f} · N<sup>{beta:.3f}</sup> · s<sup>{exponent:.2f}</sup>"


def other_family_section() -> str:
    """HTML table for the Qwen principle page. No extra <div> elements."""
    families = fit_c4_families()
    rows = []
    for family in families:
        labels = ", ".join(_label(model) for model in family["models"]) or "—"
        for dataset, name in (("wikitext2", "WikiText2"), ("ptb", "PTB")):
            fitted = family["laws"].get(dataset)
            if fitted is None:
                rows.append(
                    "<tr>"
                    f"<td>{family['title']}</td><td>{name}</td>"
                    "<td class=\"num\" colspan=\"5\">waiting for two models above dense</td>"
                    f"<td>{labels}</td></tr>"
                )
                continue
            alpha, beta, exponent, rmse = fitted
            count = sum(1 for row in family["above"] if row["dataset"] == dataset)
            rows.append(
                "<tr>"
                f"<td>{family['title']}</td><td>{name}</td>"
                f"<td class=\"num\">{alpha:.2f}</td>"
                f"<td class=\"num\">{beta:.3f}</td>"
                f"<td class=\"num\">{exponent:.2f}</td>"
                f"<td class=\"num\">{rmse:.3f}</td>"
                f"<td class=\"num\">{count}</td>"
                f"<td>{labels}</td></tr>"
            )
    body = "".join(rows)
    return f"""
  <h2>Same law, other families</h2>
  <p class="small muted">C4 calibration, shown here as its own table. On <code>figures/universal_scaling_law.html</code> each panel shows one calibration → eval pair, switched from the menu on that panel: c4→wt, c4→ptb, wt→c4, wt→ptb. A model enters once it has points above its own dense perplexity. Gemma-4 stays on that figure at its published WikiText2 law, 17.15 · N<sup>−0.309</sup> · s<sup>4.48</sup> (RMSE 0.137).</p>
  <p class="small muted">Regenerate both pages after a family gains C4 points: <code>python -m sparsity_principle</code> and <code>python -m universal_scaling</code>.</p>
  <table>
    <thead>
      <tr><th>Family</th><th>Eval</th><th class="num">α</th><th class="num">β</th><th class="num">b</th><th class="num">RMSE</th><th class="num">n</th><th>Models in the figure</th></tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
"""


def _decode(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "bdata" in value:
        raw = base64.b64decode(value["bdata"])
        code = {"f8": "d", "f4": "f", "i4": "i"}[value["dtype"]]
        size = struct.calcsize(code)
        count = len(raw) // size
        return list(struct.unpack("<" + code * count, raw))
    raise TypeError(f"cannot decode plot array of type {type(value).__name__}")


def _existing_traces() -> list[dict]:
    text = PAGE.read_text()
    start = text.find("[", text.rfind("Plotly.newPlot"))
    return json.loads(_grab(text, start, "[", "]"))


def _grab(text: str, start: int, open_c: str, close_c: str) -> str:
    depth = 0
    in_str = False
    esc = False
    for index in range(start, len(text)):
        char = text[index]
        if in_str:
            if esc:
                esc = False
            elif char == "\\":
                esc = True
            elif char == '"':
                in_str = False
            continue
        if char == '"':
            in_str = True
            continue
        if char == open_c:
            depth += 1
        elif char == close_c:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise RuntimeError("unbalanced plotly payload")


def _pack(ratios: list[dict], above: list[dict], order: list[str]) -> dict:
    present = []
    seen = {row["model"] for row in ratios}
    for model in order:
        if model in seen:
            present.append(model)
    for model in seen:
        if model not in present:
            present.append(model)
    return {"ratios": ratios, "above": above, "models": present, "law": _fit(above)}


def _law_text(law) -> str:
    if law is None:
        return "no points"
    alpha, beta, exponent, rmse = law
    return f"{alpha:.2f}·N^({beta:.3f})·s^{exponent:.2f}  RMSE {rmse:.3f}"


def _protocols_from_spec(spec, sizes: dict[str, float], order: list[str]) -> dict[str, dict]:
    loaded = _load_family(spec, sizes)
    found = {}
    for key, _label, calibration, dataset in PROTOCOLS:
        if spec.calibration != calibration or dataset not in spec.eval_datasets:
            continue
        ratios = [row for row in loaded["ratios"] if row["dataset"] == dataset]
        above = [row for row in loaded["above"] if row["dataset"] == dataset]
        found[key] = _pack(ratios, above, order)
    return found


def _empty_protocols() -> dict[str, dict]:
    return {key: _pack([], [], []) for key, *_rest in PROTOCOLS}


def _merge_protocols(*parts: dict[str, dict]) -> dict[str, dict]:
    merged = _empty_protocols()
    for part in parts:
        for key, payload in part.items():
            merged[key] = payload
    return merged


def _qwen_protocols() -> tuple[dict[str, dict], list[str]]:
    from experiments.qwen import QWEN_35_WIKITEXT2_CAL, QWEN_WIKITEXT2_CAL
    from sparsity_principle import PTB_PATH, WIKI_PATH, _eval_points, load_sizes, model_order

    order = model_order()
    sizes = load_sizes(order)
    protocols = _empty_protocols()
    for key, path, dataset in (
        ("c4-wt", WIKI_PATH, "wikitext2"),
        ("c4-ptb", PTB_PATH, "ptb"),
    ):
        points = []
        for point in _eval_points(path, order):
            item = dict(point)
            item["dataset"] = dataset
            points.append(item)
        ratios, above = _rows_from_points(points, sizes)
        protocols[key] = _pack(ratios, above, order)
    base = _protocols_from_spec(QWEN_35_WIKITEXT2_CAL, sizes, order)
    later = _protocols_from_spec(QWEN_WIKITEXT2_CAL, sizes, order)
    for key in ("wt-c4", "wt-ptb"):
        protocols[key] = _pack(
            base[key]["ratios"] + later[key]["ratios"],
            base[key]["above"] + later[key]["above"],
            order,
        )
    return protocols, order


def _rows_from_points(points: list[dict], sizes: dict[str, float]) -> tuple[list[dict], list[dict]]:
    dense: dict[tuple[str, str], float] = {}
    for point in points:
        if point.get("method") == "dense":
            dense[(point["model"], point["dataset"])] = float(point["perplexity"])
    ratios, above = [], []
    for point in points:
        if point.get("method") != "sparsegpt":
            continue
        model = point["model"]
        dataset = point["dataset"]
        base = dense.get((model, dataset))
        if model not in sizes or base is None:
            continue
        perplexity = float(point["perplexity"])
        if not _usable_dense(base) or perplexity <= 0 or not np.isfinite(perplexity):
            continue
        sparsity = float(point["sparsity"])
        if sparsity <= 0 or sparsity >= 1:
            continue
        row = {
            "model": model,
            "label": _label(model),
            "dataset": dataset,
            "s": sparsity,
            "ratio": perplexity / base,
            "log_ratio": float(np.log(perplexity / base)),
            "n": float(sizes[model]),
        }
        ratios.append(row)
        if row["ratio"] > 1.0:
            above.append(row)
    return ratios, above


def _xy(trace: dict) -> tuple[list[float], list[float]]:
    return [float(value) for value in _decode(trace["x"])], [float(value) for value in _decode(trace["y"])]


def _gemma_carried() -> dict:
    """C4 Gemma series from the previous universal figure.

    The wt→ pairs stay empty. Result JSON for Gemma is not in this tree.
    """
    if GEMMA_CACHE.is_file():
        return json.loads(GEMMA_CACHE.read_text())
    traces = [trace for trace in _existing_traces() if "gemma" in trace.get("name", "").lower()]
    if not any(trace.get("name", "").startswith("Gemma WT2") for trace in traces):
        raise RuntimeError("carried Gemma traces are missing from the universal page")
    collapse = {"c4-wt": {"markers": [], "lines": []}, "c4-ptb": {"markers": [], "lines": []}}
    normalized: list[dict] = []
    for trace in traces:
        name = trace.get("name", "")
        xs, ys = _xy(trace)
        if name.startswith("Gemma WT2"):
            collapse["c4-wt"]["lines"].append({"name": name, "x": xs, "y": ys})
        elif name.startswith("Gemma PTB"):
            collapse["c4-ptb"]["lines"].append({"name": name, "x": xs, "y": ys})
        elif "· WT2" in name:
            collapse["c4-wt"]["markers"].append(
                {"name": name.split("·")[0].strip(), "x": xs, "y": ys, "color": (trace.get("marker") or {}).get("color", "#1d4ed8")}
            )
        elif "· PTB" in name:
            collapse["c4-ptb"]["markers"].append(
                {"name": name.split("·")[0].strip(), "x": xs, "y": ys, "color": (trace.get("marker") or {}).get("color", "#c2410c")}
            )
        elif "·" not in name and not name.startswith("Gemma "):
            normalized.append({"name": name, "x": xs, "y": ys, "mode": trace.get("mode", "markers"), "color": (trace.get("marker") or {}).get("color") or (trace.get("line") or {}).get("color", "#1d4ed8")})
    # Original order is WT2 marker, PTB marker, then that model's law line.
    buckets = {"c4-wt": [], "c4-ptb": []}
    lines = {"c4-wt": [], "c4-ptb": []}
    pending = "c4-wt"
    for item in normalized:
        if item["mode"] == "lines" or item["name"].endswith(" law"):
            lines["c4-wt"].append(item)
            pending = "c4-wt"
            continue
        buckets[pending].append(item)
        pending = "c4-ptb" if pending == "c4-wt" else "c4-wt"
    payload = {}
    for key in ("c4-wt", "c4-ptb"):
        payload[key] = {
            "collapse": collapse[key],
            "normalized": {"markers": buckets[key], "lines": lines[key]},
        }
    for key, *_rest in PROTOCOLS:
        payload.setdefault(key, {"collapse": {"markers": [], "lines": []}, "normalized": {"markers": [], "lines": []}})
    GEMMA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    GEMMA_CACHE.write_text(json.dumps(payload))
    return payload


def _family_rows() -> list[dict]:
    from experiments.llama import LLAMA3_WIKITEXT2_CAL
    from experiments.olmo import OLMO2_WIKITEXT2_CAL

    from experiments.gemma import GEMMA_IT_WIKITEXT2_CAL, stored_params as gemma_stored_params

    qwen, qwen_order = _qwen_protocols()
    olmo_sizes = olmo_stored_params()
    llama_sizes = llama_stored_params()
    gemma_order = list(GEMMA_IT_WIKITEXT2_CAL.models)
    rows = [
        {"title": "Qwen", "kind": "fit", "protocols": qwen, "order": qwen_order},
        {
            "title": "Gemma-4",
            "kind": "carried",
            "protocols": _gemma_carried(),
            "fit_protocols": _protocols_from_spec(
                GEMMA_IT_WIKITEXT2_CAL, gemma_stored_params(), gemma_order
            ),
            "fit_keys": ("wt-c4", "wt-ptb"),
            "order": gemma_order,
        },
        {
            "title": "OLMo 2 Instruct",
            "kind": "fit",
            "protocols": _merge_protocols(
                _protocols_from_spec(OLMO2, olmo_sizes, list(OLMO2.models)),
                _protocols_from_spec(OLMO2_WIKITEXT2_CAL, olmo_sizes, list(OLMO2.models)),
            ),
            "order": list(OLMO2.models),
        },
        {
            "title": "Llama 3",
            "kind": "fit",
            "protocols": _merge_protocols(
                _protocols_from_spec(LLAMA3, llama_sizes, list(LLAMA3.models)),
                _protocols_from_spec(LLAMA3_WIKITEXT2_CAL, llama_sizes, list(LLAMA3.models)),
            ),
            "order": list(LLAMA3.models),
        },
    ]
    return rows


def _colors(order: list[str]) -> dict[str, str]:
    from sparsity_principle import MODEL_COLORS

    palette = {}
    for index, model in enumerate(order):
        palette[model] = MODEL_COLORS.get(model, COLORS[index % len(COLORS)])
    return palette


def _add(fig, row: int, col: int, visible: bool, **kwargs) -> int:
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(visible=visible, **kwargs), row=row, col=col)
    return len(fig.data) - 1


def _protocol_pack(family: dict, protocol: str) -> dict:
    if protocol in family.get("fit_keys", ()):
        return family["fit_protocols"][protocol]
    return family["protocols"][protocol]


def _draw_fit(fig, family: dict, protocol: str, kind: str, row: int, visible: bool) -> list[int]:
    pack = _protocol_pack(family, protocol)
    colors = _colors(family["order"])
    indices = []
    law = pack["law"]
    column = 1 if kind == "collapse" else 2
    if kind == "collapse" and law is not None:
        alpha, beta, exponent, _rmse = law
        indices.append(
            _add(
                fig, row, column, visible,
                x=CURVE_S,
                y=alpha * np.power(CURVE_S, exponent),
                mode="lines",
                name=f"{family['title']} law",
                legendgroup=family["title"],
                line=dict(color="#111827", width=2.4),
                hovertemplate="s=%{x:.2f}<br>collapsed=%{y:.3f}<extra></extra>",
            )
        )
        for model in pack["models"]:
            points = [point for point in pack["above"] if point["model"] == model]
            if not points:
                continue
            indices.append(
                _add(
                    fig, row, column, visible,
                    x=[point["s"] for point in points],
                    y=[point["log_ratio"] * point["n"] ** (-beta) for point in points],
                    mode="markers",
                    name=points[0]["label"],
                    legendgroup=family["title"],
                    marker=dict(color=colors[model], size=8, symbol="circle", line=dict(color=colors[model], width=1.2)),
                    hovertemplate=f"{points[0]['label']} ({points[0]['n']:.2f}B)<br>s=%{{x:.2f}}<br>collapsed=%{{y:.3f}}<extra></extra>",
                )
            )
    if kind == "normalized":
        if law is not None:
            alpha, beta, exponent, _rmse = law
            for model in pack["models"]:
                sample = next((point for point in pack["ratios"] if point["model"] == model), None)
                if sample is None:
                    continue
                indices.append(
                    _add(
                        fig, row, column, visible,
                        x=CURVE_S,
                        y=np.exp(alpha * sample["n"] ** beta * np.power(CURVE_S, exponent)),
                        mode="lines",
                        name=f"{sample['label']} law",
                        legendgroup=family["title"],
                        showlegend=False,
                        line=dict(color=colors[model], width=1.5),
                        hovertemplate=f"{sample['label']} law<br>s=%{{x:.2f}}<br>ppl/dense=%{{y:.3f}}<extra></extra>",
                    )
                )
        for model in pack["models"]:
            points = [point for point in pack["ratios"] if point["model"] == model]
            if not points:
                continue
            indices.append(
                _add(
                    fig, row, column, visible,
                    x=[point["s"] for point in points],
                    y=[point["ratio"] for point in points],
                    mode="markers",
                    name=points[0]["label"],
                    legendgroup=family["title"],
                    marker=dict(color=colors[model], size=8, symbol="circle", line=dict(color=colors[model], width=1.2)),
                    hovertemplate=f"{points[0]['label']}<br>s=%{{x:.2f}}<br>ppl/dense=%{{y:.3f}}<extra></extra>",
                )
            )
    return indices


def _draw_carried(fig, family: dict, protocol: str, kind: str, row: int, visible: bool) -> list[int]:
    block = family["protocols"].get(protocol) or {}
    side = block.get("collapse" if kind == "collapse" else "normalized") or {"markers": [], "lines": []}
    column = 1 if kind == "collapse" else 2
    indices = []
    for item in side.get("lines", []):
        indices.append(
            _add(
                fig, row, column, visible,
                x=item["x"], y=item["y"], mode="lines",
                name=item["name"] if kind == "collapse" else f"{family['title']} law",
                legendgroup=family["title"],
                showlegend=kind == "collapse",
                line=dict(color=item.get("color") or "#111827", width=2.2),
                hovertemplate="s=%{x:.2f}<br>%{y:.3f}<extra></extra>",
            )
        )
    for item in side.get("markers", []):
        indices.append(
            _add(
                fig, row, column, visible,
                x=item["x"], y=item["y"], mode="markers",
                name=item["name"],
                legendgroup=family["title"],
                marker=dict(color=item.get("color") or "#1d4ed8", size=8, symbol="circle"),
                hovertemplate=f"{item['name']}<br>s=%{{x:.2f}}<br>%{{y:.3f}}<extra></extra>",
            )
        )
    return indices


def _button_label(family: dict, protocol: str, caption: str) -> str:
    if protocol in family.get("fit_keys", ()):
        return f"{caption} · {_law_text(family['fit_protocols'][protocol]['law'])}"
    if family["kind"] == "carried":
        lines = (family["protocols"].get(protocol) or {}).get("collapse", {}).get("lines", [])
        if lines:
            return f"{caption} · {lines[0]['name'].split(':', 1)[-1].strip()}"
        markers = (family["protocols"].get(protocol) or {}).get("collapse", {}).get("markers", [])
        return f"{caption} · {'no points' if not markers else 'carried'}"
    return f"{caption} · {_law_text(family['protocols'][protocol]['law'])}"


def _panel_menu(family: dict, groups: dict[str, list[int]], default: str, x_right: float, y_top: float) -> dict:
    order = [key for key, *_rest in PROTOCOLS]
    all_idx = [index for key in order for index in groups[key]]
    buttons = []
    for key, caption, _cal, _dataset in PROTOCOLS:
        chosen = set(groups[key])
        buttons.append(
            dict(
                label=_button_label(family, key, caption),
                method="restyle",
                args=[{"visible": [index in chosen for index in all_idx]}, all_idx],
            )
        )
    return dict(
        type="dropdown",
        direction="down",
        x=x_right,
        xanchor="right",
        y=y_top,
        yanchor="top",
        buttons=buttons,
        active=order.index(default),
        bgcolor="#f8fafc",
        bordercolor="#cbd5e1",
        font=dict(size=11),
        pad=dict(b=6, r=0),
        showactive=True,
    )


def write_page() -> None:
    from plotly.subplots import make_subplots

    families = _family_rows()
    titles = []
    for family in families:
        titles.append(f"{family['title']} · collapse")
        titles.append(f"{family['title']} · ppl/dense")
    fig = make_subplots(
        rows=len(families),
        cols=2,
        subplot_titles=tuple(titles),
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )
    menus = []
    placements = []
    for row, family in enumerate(families, start=1):
        for column, kind in ((1, "collapse"), (2, "normalized")):
            if kind == "normalized":
                _add(
                    fig, row, column, True,
                    x=[0.0, 0.99], y=[1.0, 1.0], mode="lines",
                    name="dense", showlegend=False, hoverinfo="skip",
                    line=dict(color="#94a3b8", width=1.2, dash="dash"),
                )
            groups = {key: [] for key, *_rest in PROTOCOLS}
            has = {}
            for key, _caption, _cal, _dataset in PROTOCOLS:
                if family["kind"] == "fit" or key in family.get("fit_keys", ()):
                    groups[key] = _draw_fit(fig, family, key, kind, row, visible=False)
                    has[key] = bool(_protocol_pack(family, key)["ratios"])
                else:
                    groups[key] = _draw_carried(fig, family, key, kind, row, visible=False)
                    side = "collapse" if kind == "collapse" else "normalized"
                    block = family["protocols"].get(key, {}).get(side, {})
                    has[key] = bool(block.get("markers") or block.get("lines"))
            default = "c4-wt" if has.get("c4-wt") else next((key for key, *_r in PROTOCOLS if has.get(key)), "c4-wt")
            for index in groups[default]:
                fig.data[index].visible = True
            placements.append((family, groups, default, row, column))
            law = family["protocols"].get(default, {})
            if family["kind"] == "fit":
                fitted = law.get("law") if isinstance(law, dict) else None
                print(f"{family['title']} {default} {kind}: {_law_text(fitted)}", flush=True)
    for row in range(1, len(families) + 1):
        fig.update_xaxes(type="log", title_text="Unstructured sparsity s (log)", row=row, col=1)
        fig.update_yaxes(type="log", title_text="N^(−β) · ln(ppl/dense) (log)", row=row, col=1)
        fig.update_xaxes(title_text="Unstructured sparsity s", range=[0, 1], row=row, col=2)
        fig.update_yaxes(type="log", title_text="perplexity / dense (log)", row=row, col=2)
    for family, groups, default, row, column in placements:
        x_key = "xaxis" if (row, column) == (1, 1) else f"xaxis{(row - 1) * 2 + column}"
        y_key = "yaxis" if (row, column) == (1, 1) else f"yaxis{(row - 1) * 2 + column}"
        menus.append(
            _panel_menu(
                family,
                groups,
                default,
                float(fig.layout[x_key].domain[1]),
                float(fig.layout[y_key].domain[1]),
            )
        )
    fig.update_layout(
        template="plotly_white",
        width=1480,
        height=560 * len(families),
        margin=dict(t=120, b=80, l=70, r=30),
        title=dict(
            text=(
                "<b>Scale-Free Sparsity Law by Model Family</b><br><sup>"
                "Each panel shows one calibration → eval pair. "
                "Use the menu on the panel: c4→wt · c4→ptb · wt→c4 · wt→ptb. "
                "Default is c4→wt.</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5, font=dict(size=10)),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color="#0f172a"),
        updatemenus=menus,
    )
    fig.write_html(PAGE, include_plotlyjs=True, full_html=True, config={"displaylogo": False, "responsive": True})
    print(f"Wrote {PAGE}", flush=True)


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()

