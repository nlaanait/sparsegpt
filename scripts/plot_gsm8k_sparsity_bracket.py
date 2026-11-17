"""Plot audited GSM8K sparsity measurements, leaving pending levels blank."""
from __future__ import annotations

import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path

import plotly.graph_objects as go


METHODS = {
    "sparsegpt-task": ("SparseGPT", "#2166ac"),
    "local": ("Local", "#b35806"),
    "downstream-weightdiag": ("Downstream weight diagonal", "#238443"),
}
LEVELS = (55, 60, 65, 70)


def load(path: Path):
    return json.loads(path.read_text())


def checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verified_audit(root: Path, label: str, ids_sha256: str | None = None):
    path = root / "audits" / f"{label}.json"
    audit = load(path)
    folder = Path(audit["evaluation"])
    common = audit["common128"]
    assert audit["common128_complete"] and common["count"] == 128
    assert audit["raw_token_rescore_mismatches"] == 0
    assert audit["stopping_policy"]["version"] == "gsm8k-model-and-tokenizer-eos-v1"
    assert audit["config_sha256"] == digest(load(folder / "config.json"))
    assert audit["report_sha256"] == checksum(folder / "report.json")
    assert audit["answers_sha256"] == checksum(folder / "answers.jsonl")
    if ids_sha256 is not None:
        assert common["ids_sha256"] == ids_sha256
    for filename, expected in audit["raw_files"].items():
        assert checksum(folder / "raw-generations" / filename) == expected
    inputs_path = root / "audits" / f"{label}-generation-inputs.json"
    inputs = load(inputs_path)
    assert inputs["common128_complete"] and inputs["count"] >= 128
    assert inputs["packed_input_and_prompt_mismatches"] == 0
    assert inputs["config_sha256"] == audit["config_sha256"]
    assert inputs["audited_records_sha256"] == audit["audited_records_sha256"]
    assert inputs["answers_snapshot_sha256"] == audit["answers_sha256"]
    assert inputs["raw_files_sha256"] == audit["raw_files"]
    return audit, {"raw_audit_sha256": checksum(path),
                   "generation_inputs_audit_sha256": checksum(inputs_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/tangent/gsm8k/eos-corrected-v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    collector = load(root / "collector" / "progress.json")
    collected = {row["label"]: row for row in collector["verified"]}
    assert "dense" in collected
    dense, dense_hashes = verified_audit(root, "dense")
    ids = dense["common128"]["ids_sha256"]
    points, pending = [], []
    for method in METHODS:
        for level in LEVELS:
            label = f"{method}-s{level}"
            if label not in collected:
                pending.append(label)
                continue
            audit, hashes = verified_audit(root, label, ids)
            common = audit["common128"]
            assert collected[label]["common128"] == common
            link_path = root / "audits" / f"{label}-checkpoint-score-link.json"
            link = load(link_path)
            assert link["raw_audit_sha256"] == hashes["raw_audit_sha256"]
            assert link["verified_stage_count"] == 96
            assert link["common128_ids_sha256"] == ids
            assert link["evaluation_config_sha256"] == audit["config_sha256"]
            source = Path(link["source"])
            assert link["source_config_sha256"] == digest(load(source / "config.json"))
            assert link["source_report_sha256"] == checksum(source / "report.json")
            assert link["source_cpu_audit_sha256"] == checksum(Path(link["source_cpu_audit"]))
            points.append({"label": label, "method": method, "sparsity_percent": level,
                           **common, "accuracy_percent": 100 * common["correct"] / 128,
                           "evaluation_job": collected[label]["job_id"], **hashes,
                           "checkpoint_score_link_sha256": checksum(link_path)})

    figure = go.Figure()
    for method, (name, color) in METHODS.items():
        rows = {p["sparsity_percent"]: p for p in points if p["method"] == method}
        figure.add_trace(go.Scatter(
            name=name, x=list(LEVELS),
            y=[rows[s]["accuracy_percent"] if s in rows else None for s in LEVELS],
            mode="lines+markers", connectgaps=False,
            line=dict(color=color, width=2), marker=dict(size=10),
            customdata=[[rows[s]["correct"], rows[s]["truncated"], rows[s]["evaluation_job"]]
                        if s in rows else [None, None, None] for s in LEVELS],
            hovertemplate=("Sparsity: %{x}%<br>Correct: %{customdata[0]}/128 (%{y:.2f}%)"
                           "<br>Truncated: %{customdata[1]}/128<br>Evaluation job: %{customdata[2]}"
                           "<extra>%{fullData.name}</extra>")))
    dense_correct = dense["common128"]["correct"]
    figure.add_hline(y=100 * dense_correct / 128, line_dash="dash", line_color="#777",
                     annotation_text=f"Dense {dense_correct}/128", annotation_position="top left")
    grid_done = sum(p["sparsity_percent"] in (55, 60, 65) for p in points)
    zeros = [p["label"] for p in points if p["correct"] == 0]
    conclusion = ("No observed zero among the measured points; zero-accuracy threshold unestablished."
                  if not zeros else "Observed sample zeros: " + ", ".join(zeros) + ".")
    figure.update_layout(
        template="plotly_white", width=1100, height=680,
        title=dict(x=.025, text=("GSM8K accuracy across MLP sparsity"
            "<br><sup>Qwen3.5-9B | corrected EOS | seed 0 | batch 32 | 2,048 new tokens</sup>")),
        xaxis=dict(title="Eligible MLP weight sparsity (%)", tickvals=list(LEVELS), range=[54, 71]),
        yaxis=dict(title="Correct answers (% of common 128)", range=[-2, 103]),
        legend=dict(orientation="h", x=0, y=1.1), margin=dict(l=75, r=35, t=130, b=135),
        annotations=list(figure.layout.annotations) + [dict(
            x=0, y=-.2, xref="paper", yref="paper", showarrow=False, xanchor="left",
            font=dict(size=11, color="#555"),
            text=(f"{grid_done}/9 requested 55–65% points verified. Blank levels are pending; lines do not bridge them."
                  "<br>Same held-out training-validation questions; exploratory, single seed. No fitted threshold."
                  f"<br>{conclusion}"))])
    output = root / "figures"
    output.mkdir(exist_ok=True)
    html_path = output / "sparsity-bracket.html"
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)

    class Scripts(HTMLParser):
        remote = None

        def __init__(self):
            super().__init__()
            self.remote = []

        def handle_starttag(self, tag, attrs):
            if tag == "script" and dict(attrs).get("src"):
                self.remote.append(dict(attrs)["src"])

    parsed = Scripts()
    parsed.feed(html_path.read_text())
    assert not parsed.remote
    data = {"scope": "common128_sparsity_measurements_not_population_zero_threshold",
            "model": "Qwen/Qwen3.5-9B", "common128_ids_sha256": ids,
            "dense_correct": dense_correct, "dense_audit_hashes": dense_hashes,
            "completed_requested_grid_points": grid_done, "expected_requested_grid_points": 9,
            "points": points, "pending_labels": pending, "observed_sample_zeros": zeros,
            "generator_sha256": checksum(Path(__file__)), "html_sha256": checksum(html_path),
            "external_script_dependencies": parsed.remote}
    (output / "sparsity-bracket.data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"html": str(html_path), "verified_points": len(points),
                      "requested_grid_complete": grid_done == 9, "observed_sample_zeros": zeros}))


if __name__ == "__main__":
    main()
