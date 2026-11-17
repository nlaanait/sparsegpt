# Representative GSM8K calibration

The default fitting population is now a seeded sample of original GSM8K
training questions. The controlled arithmetic variants remain a separate
diagnostic for known parameter interventions and tangent-direction checks.
Earlier template-calibrated results remain identifiable as transfer experiments
from a narrow fitting distribution; they are not relabeled as representative
calibration results.

## Frozen data split

`results/tangent/gsm8k/data/representative-calibration-seed0-v1.json` contains:

| Use | Original questions | Manifest split |
|---|---:|---|
| Fit local output, input, and downstream tangent atlases | 512 | `geometry_fit` |
| Calibrate pruning / estimate downstream and two-sided moments | 128 | `pruning_calibration` |
| Independent engineering diagnostics and finite-output checks | 64 | `validation` |

These are **704 distinct parent questions**, not generated numerical variants.
The 64 engineering questions are separate from the 64-question evaluation
screen. All three calibration/diagnostic splits exclude the entire reserved
512-question evaluation pool, the previous controlled-template parents, and
the development-pilot parents. Preparation also excludes exact casefolded,
whitespace-normalized question duplicates across these boundaries.

Selection uses a dedicated `random.Random(0)` on the original training indices,
uniformly without replacement. This samples the training distribution; it
does not assert balanced semantic categories or eliminate all possible
semantic near-duplicates. `family` identifies the original parent question,
not an inferred problem-type label. Original text, exact answer, source index,
source-row digest, exclusions, seed, source-file hash, and manifest hash are
recorded. The official test split is never opened.

Output/downstream atlas fitting and two-sided input atlas fitting use only
the 512 fit readouts. The 128 pruning questions receive frozen chart
assignments and default to **8 backward probes each: 1,024 VJPs per site**.
Original-question selection is seeded rather than taking a sorted prefix of
unique family IDs. Controlled diagnostic selection retains its family-balanced
rule and historical 32-question backward default.

Batch size remains 8. Ambient exposure remains 4,096 math + 4,096 C4 tokens;
the math budget includes all 640 fit/pruning readouts. The cached C4 token
windows can be reused, but every activation is refreshed from the current
partially pruned model. Original-question runs do not reuse controlled math
activations, fitted atlases, or pruned candidates.

## Evaluation identity

Two new manifests bind the existing evaluation questions to this calibration:

- `data/decision64-light128-repcal-seed0-v1.json`: the same ordered first
  64/128 evaluation IDs as before.
- `data/decision128-full512-repcal-seed0-batch8-v1.json`: the same ordered
  128/512 evaluation IDs as before.

Their IDs and shuffle order are unchanged; hashes change because the
calibration binding changes. The 2,048-token generation cap and 15-point
screen cutoff are unchanged. Representative submissions use new `repcal-v1`
output names and a matching dense-reference identity. Parent-source separation
is checked again before candidate evaluation loads a model.

## Preparation and execution

Preparation is CPU-only and uses the cached training Arrow source:

```bash
PYTHONPATH=src:. .pixi/envs/gb10/bin/python -m sparsifier.data.gsm8k_calibration \
  --train-arrow /path/to/gsm8k-train.arrow \
  --evaluation-manifest results/tangent/gsm8k/data/original-validation-v1.json \
  --diagnostic-manifest results/tangent/gsm8k/data/controlled-v1.json \
  --diagnostic-manifest results/tangent/gsm8k/data/development.json \
  --prepare-evaluation
```

The capture, sequential, downstream, and whole-neuron pruning CLIs default to
the new calibration manifest. An explicit `--manifest` still permits historical
controlled diagnostics. Two-sided and full-metric options use the same default
source, and fitting code continues to enforce geometry-fit-only atlas training.
The older paired-difference controls still require their explicit controlled
pairs; unrelated original questions are not silently treated as intervention
pairs.

`scripts/submit_gsm8k_decision_pilot.py` now defaults to the representative
single-block pilot. `scripts/submit_gsm8k_whole_neuron_full.py` uses the same
calibration for full-model candidates and the reserved 128-to-512 evaluation
protocol. Both print a plan unless `--submit` is supplied. Full submissions
create or reuse the matching representative dense-reference job before
dependent candidates. Historical controlled manifests, plans, journals, and
result directories remain unchanged; legacy migration helpers retain their
explicit diagnostic protocol.

The manifests and dry-run plans are prepared. No new representative-calibration
GPU experiment was submitted as part of this protocol change.
