# GSM8K experiment plan: geometry-guided preservation of mathematical behavior

**Status:** implementation and development runs in progress; see GSM8K-EXPERIMENT-STATUS.md.<br>
**Primary model:** `Qwen/Qwen3.5-9B`, post-trained checkpoint, text tower, BF16.<br>
**Primary question:** how many model weights can be removed while keeping GSM8K accuracy within an allowed drop from the dense model, and does factor-local geometry extend that sparsity limit? Dense performance is a retention reference, not a score to beat.

The experiment connects Stage 2 (predict intervention effects) to Stage 3 (preserve behavior under pruning). Controlled math word problems provide the mechanism test; untouched GSM8K test problems provide the transfer test. A positive result on reconstruction of calibration directions alone is insufficient.

## 1. Relationship to the existing stages

The research specification is [tangent-aware-pruning-revised.md](tangent-aware-pruning-revised.md), especially Section 6. The [review](tangent-aware-pruning-revised-luna6-review.md) qualifies gradient-enrichment screening, geometric nulls, and claims of exactness.

| Existing code | Reuse | Limitation to address |
|---|---|---|
| `src/sparsifier/eval/revised_protocol.py` | Synthetic Stages 0–4; solver and geometry checks | Constructed geometry does not establish language-model utility |
| `src/sparsifier/eval/live_stages.py` | Memory checks and small shared helpers | Legacy random-token harness; its activation-space probe is not a vocabulary-level behavioral metric |
| `src/sparsifier/eval/natural_layer.py` | Aligned activations/gradients, reversible weight insertion, compensated knockouts, real vocabulary NLL/KL, resumable reports | Existing pairs hold out prefixes only; pruning is fixed at 50%; no generated-answer evaluation |
| `src/sparsifier/prune/input_side.py` | Variant A difference-covariance augmentation | Protects local linear-output differences; downstream usefulness remains empirical |
| `src/sparsifier/data/concept_pairs.py` | Small regression fixture | Nine token contrasts do not constitute the scientific dataset |
| `src/sparsifier/models/loop.py` and `cli.py` | Existing `tangent_input_side` / saved pair-delta integration | Sequential experiments need explicit provenance and a policy for refreshing pairs after upstream pruning |

Keep the nine-pair bank for engineering regressions. Do not extend it as the utility benchmark. Do not interpret completion of a software stage as confirmation of its scientific hypothesis.

### Geometric premise: factor identity and location within a factor

The [BSF paper, Definition 1 and Section 2.2](https://arxiv.org/html/2606.25234v1) motivates a **sparse additive composition** of manifold-valued factors. Its formal model is not a single cluster assignment for an entire activation. Each active factor contributes a point; multiple contributions coexist. Its linear block is a container for a factor's geometry, not an assertion that the occupied geometry fills that linear subspace.

For this experiment, distinguish:

```text
h(x) ≈ sum_{i in S(x)} m_i(x) + residual(x)
m_i(x) belongs to M_i
V_i = span(M_i)                       # factor's global linear container
U_i,c(x) spans a local tangent        # chart within that factor
Q(x) = orth(span of active U_i,c(x))   # local span, not the global data support
```

Across active sets, the model's support is a union of sums of factor manifolds. Within a factor, allow multiple components/branches and multiple charts when supported by data; do not attribute a requirement of multiple disconnected manifolds per factor to the paper. A chart covers a local region and is not necessarily a separate component. At intersections or support changes, a unique smooth tangent need not exist.

This distinction was underrepresented in the initial plan. A pooled difference covariance retains directions from multiple regions but discards which region/factor configuration belongs to which input. Reassigning identical difference rows to different charts leaves Variant A unchanged if their weights remain the same. Therefore Variant A is a useful contrast baseline, but cannot on its own test the value of input-dependent manifold structure.

The existing implementations cover only parts of this hierarchy: `tangent.py` assigns a chart to the whole sample; `factor_union.py` accepts multiple active factors but one fixed basis per factor; `geometry_features.py` explicitly implements a small block dictionary rather than the paper's BSF. None of these facts establishes a factor-specific local atlas. Audit and extend the representation before labeling a run as that experiment.

## 2. Hypotheses and claims

| Hypothesis | Supporting observation | Null or limiting outcome |
|---|---|---|
| G: repeated task changes define stable local directions | Directions generalize across independent instances; local bases improve held-out reconstruction over a global span at matched budget | Unstable directions or no local advantage; retain the finite-difference interpretation |
| L: input-dependent factor geometry matters | Correctly assigned factor-local metrics improve behavioral preservation over fixed factor spans, pooled differences, and shuffled assignments | A global or pooled representation performs equally well; locality has no demonstrated added value |
| P: geometry adds predictive information about weight importance | Geometric scores predict held-out behavioral damage beyond ambient OBS and other baselines, including after compensation | Only calibration/geometric error is predicted; no demonstrated behavioral importance |
| U: meaningful differences improve pruning utility | Variant A permits higher whole-model sparsity at the same allowed accuracy drop from dense than task-calibrated SparseGPT and matched controls | No improvement, or improvement explained by additional task data or arbitrary regularization |
| T: the improvement transfers | Frozen model-wide configurations retain near-dense accuracy on untouched GSM8K at their selected sparsity | Benefit is restricted to controlled problem families |

P and U are related but separate: aggregate pruning can help even if the particular single-weight score is poor. Report each outcome independently. A positive U without G supports task-contrast-guided pruning, not a local-tangent or manifold claim. A one-layer result does not establish whole-model compression utility.

G and L are required for the central manifold-aware claim. If G fails, report the failure and any remaining contrast-pruning result separately; do not silently replace the main hypothesis with U.

## 3. Data construction and splits

### Controlled task families

Construct parameterized problems from GSM8K **training** problems, with executable solution functions. Begin with short two- and three-operation families, such as:

> Maya has a boxes with b pencils in each. She gives away c pencils. How many pencils remain?

The answer is `a*b - c`. Store the expression/program, valid parameter ranges, units, constraints, and exact answer. Validate arithmetic and manually review each source template and its paraphrases. Use integer or rational arithmetic, not floating-point equality, for answer checking.

For each anchor instance, vary one relevant quantity at a time over several neighboring values, holding other factors fixed. Reject invalid substitutions such as negative physical counts or disallowed fractional objects. Record the factor, signed step size, source family, and all nuisance variables.

Also generate crossed grids of two or more independently controlled quantities, covering multiple regions and their combinations. Hold out combinations of familiar factors/regions as a compositional-generalization test, separately from entirely unseen templates. Include problems with different subsets of required operations. Mathematical knobs and operations are intervention labels, not assumed one-to-one neural factors: verify selectivity and co-activation in the fitted representation. Carry boundaries and numeral-length changes are candidate regime changes to investigate, not predeclared proof of distinct manifolds.

Use separate variation classes:

- **Quantity changes:** meaningful finite differences with a known answer change.
- **Names and wording:** nuisance changes preserving the answer; controls for lexical sensitivity.
- **Operation changes:** a secondary contrast experiment with independently validated semantics and solutions. Do not treat switching addition to subtraction as a small continuous tangent step.

Inspect tokenization. The primary controlled analysis should match token lengths within each pair where feasible and report the resulting exclusions. Include a secondary variable-length analysis aligned by semantic readout position. Do not silently confuse token-position changes with mathematical variation.

### Split unit and separation

Assign source problems and their template/paraphrase families to splits **before generating variants**. All near-duplicate source families stay together. Disjoint numeric samples of the same template are an interpolation test, not unseen-template generalization.

| Split | Permitted use |
|---|---|
| Development pilot | Debugging, dense competence, precision/noise estimates, power planning, finalizing prompts and budgets |
| Geometry fit | Estimate differences, local bases, factor normalization, and random controls |
| Pruning calibration | Estimate ambient curvature and task-gradient/Fisher baselines on independent examples |
| Validation | Select alpha, layer/site, sparsity, and any tuned baseline settings |
| Stage 2 diagnostic holdout | Evaluate intervention predictions; never use as the Stage 3 final test |
| Controlled final test | Confirm preservation on unseen families, with a separately labeled interpolation subset if desired |
| GSM8K official test | Final transfer evaluation only, after settings are frozen |

Reserve separate general-text validation and final-evaluation windows as well. Never tune alpha or the language-loss tolerance on the final WikiText2/PTB windows.

Target a pilot of roughly 20 reviewed source families with 20–40 instances each. This is an engineering and variance-estimation budget, not a sufficient final sample size. Choose the final number of independent families from pilot variability and the preregistered minimum effect; generating more variants does not replace more families.

### GSM8K and GSM-Symbolic

[GSM8K](https://github.com/openai/grade-school-math) supplies training problems and an official held-out benchmark. [GSM-Symbolic](https://github.com/apple-aiml-research/ml-gsm-symbolic) supplies useful template and generated-instance examples. Use its construction approach as a reference; do not assume its templates are independent of GSM8K test problems. Audit provenance before using external templates for fitting.

An untouched benchmark here means untouched by this experiment's fitting and selection. It does not establish absence from the checkpoint's pretraining data.

## 4. Model, prompting, and competence gate

Use **`Qwen/Qwen3.5-9B`** as the primary model. It is already in the repository catalog and provides a middle-size dense model for repeated interventions without starting at the 27B scale. Use the post-trained checkpoint's text tower in BF16, with no quantization in the primary experiment. This is a budget/capability choice, not a claim of measured competence on our tasks; retain the dense pilot gate.

The [official configuration](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json) specifies 32 decoder layers, hidden width 4096, and MLP intermediate width 12288. Start at zero-based block **15**, `mlp.down_proj`; its weight has shape `[4096, 12288]`. Pin model/tokenizer revisions and record chat template, attention implementation, generation settings, and reasoning mode. The 0.8B model is permitted only for engineering smoke checks, not as the primary scientific model.

Profile one scheduled 9B forward/backward pilot before setting job caps. A 12288-square FP32 input covariance alone is about 0.56 GiB; solver workspaces, activations, gradients, and model weights add substantially more. Do not inherit the 0.8B harness's 8 GiB cap or assume layer-wise pruning bounds all memory. Freeze any model change as a protocol amendment before confirmation.

Use one frozen generation protocol per evaluation track. The primary behavioral track permits a fixed reasoning budget and requests an explicit final numeric answer. Keep temperature, token budget, stopping rules, and answer extraction identical across pruning arms. Record truncations and malformed answers as failures, with separate counts.

On the development split, require at least 100 dense-correct examples spanning at least 10 source families before conducting a retention study. This is a feasibility floor, not a statistical power guarantee. If the model fails, move to a larger supported checkpoint or simplify the controlled task, documenting the changed scope before freezing the experiment. Do not select final-test examples to manufacture competence.

Report overall accuracy on every final problem. Dense-correct retention is a secondary metric with the same dense reference set for every arm; also report dense-wrong to pruned-correct transitions.

## 5. Stages 0–1: measurement and geometry

Start at block 15 `mlp.down_proj` of Qwen3.5-9B. This measures a linear write into the residual stream. Select any alternative site using development/validation data and freeze it before the final test.

Capture input activations at a shared readout marker after the complete question and before generated reasoning or answer tokens. Confirm that all operands are visible through the causal attention mask. Never use the edited operator token as the sole measurement site. Do not average unrelated token positions to repair alignment.

For a quantity parameter t, record raw differences and step-normalized differences:

```text
delta_phi = phi(problem(t + step)) - phi(problem(t))
direction_estimate = delta_phi / step
```

Use raw differences for the primary Variant A objective, with the existing per-factor trace normalization. Analyze step-normalized estimates for stability and preregister any use in a separate pruning arm. Integer and tokenized changes are finite differences; small numeric steps do not guarantee smooth hidden-state changes.

Measure:

1. Direction/subspace agreement across disjoint instances of the same factor and comparable contexts.
2. Sensitivity to step size, numeral length, carry boundaries, and paraphrases.
3. Held-out local versus global reconstruction at matched rank/storage, including chart metadata and assignment cost.
4. Gradient enrichment using aligned answer-conditioned gradients, alongside matched-rank nulls.
5. Coordinate leverage and mask sensitivity as descriptive diagnostics.

Chart assignment for held-out examples must use available inputs and a fitted assignment rule; it cannot select the best chart using the held-out target difference. Report both within-family stability and transfer across families.

Fit sparse factor activity and then local charts **within each active factor's coordinates**. Use controlled interventions to test the correspondence between recovered factors and mathematical variation. Full-activation k-means alone is a baseline, not factor discovery. If using the repository block dictionary, label it as an approximation and validate factor reconstruction/selectivity before interpreting its blocks. Compare learned assignments with task-label-assisted assignments only as a clearly labeled diagnostic upper bound.

Record factor IDs, active-set probabilities, component/region IDs where justified, chart centers, local ranks, bases, occupancy, and assignment uncertainty. Compare stability at corresponding regions rather than demanding identical tangents across a curved factor. Balance evaluation across supported regions; report rare-region failures and abstain from fitting unsupported charts. Count factor dictionaries, all charts, and assignment overhead in budget comparisons.

During intervention evaluation, measure both (a) error projected on the dense reference's frozen local frames and (b) changes in factor activity/chart assignments under a frozen encoder applied to pruned activations. The second metric detects drift that would be hidden by selecting a new favorable frame. Tangent preservation alone does not guarantee preservation of factor presence, chart identity, or branch transitions.

Do not infer utility from a leverage distribution, mask-flip rate, or gradient-energy threshold. If local geometry fails, stop the local-chart claim but retain the predeclared Variant A contrast experiment. Do not invent a concept-absent reference for arithmetic; presence-direction and Stage 4 claims are outside the initial experiment.

## 6. Stage 2: predict causal intervention effects

Reuse reversible weight replacement and compensated knockout machinery from `natural_layer.py`. Restore the dense weight after every intervention. Evaluate the actual stored precision and measure no-op/repeated-forward numerical noise first.

### Candidate scores and interventions

Compare magnitude, Wanda, ambient OBS, task-gradient/Fisher, existing output-geometric scores, and an input-difference score aligned with Variant A. For a proposed compensated weight change A, the latter is its predicted mean squared local difference error:

```text
mean_k || A delta_phi_k ||^2
```

Include random-orientation versions of the geometric scores. Keep ground-truth-label gradient scores labeled as empirical-Fisher-style baselines; do not call them model Fisher unless labels are sampled from the model.

Use two candidate sets:

- A representative sample of entries, stratified by layer/ambient importance where applicable.
- Disjoint discordant groups: high geometry/low ambient and low geometry/high ambient, matched in ambient OBS cost and number of removed parameters as closely as possible.

Predeclare quantile rules, matching tolerances, group sizes, and how unmatched candidates are handled using pilot data. Report representative and enriched samples separately. Evaluate both uncompensated and OBS-compensated removal. For groups, use a consistent joint/sequential compensation procedure and compare predicted effects of the resulting update; do not sum isolated removal costs as if interactions were absent.

Start with approximately 128 sampled entries and 16 matched group comparisons for feasibility. Choose group sizes and final counts from pilot effect/noise estimates, then freeze them. Single-entry effects below the precision floor are inconclusive; use grouped interventions for behavioral accuracy.

### Outcomes

- Held-out correct-answer conditional NLL, normalized consistently across answer lengths.
- Generated final-answer accuracy and dense-correct retention for grouped interventions.
- Vocabulary KL to the dense model under identical prefixes.
- Local held-out difference error and ambient reconstruction error.

The conditional-NLL diagnostic uses a frozen direct-answer prompt. If a reference rationale is supplied in any secondary diagnostic, label it teacher-forced and keep it identical across arms; do not substitute that result for free-generation accuracy.

Test incremental predictive value beyond ambient OBS using a baseline predictor and an augmented predictor fitted on development/validation interventions and assessed on held-out interventions. Also report rank correlations, calibration plots, and matched-group outcome differences. Repeated use of the same problems/intervention groups creates dependence; retain identifiers for crossed or clustered uncertainty estimates.

Support P only when prediction improves for behavioral outcomes, including the compensated condition. Predicting the geometric objective by construction is not evidence for P.

## 7. Stage 3: compensated pruning

### Required arms

| Arm | Purpose |
|---|---|
| Dense model | Behavioral reference |
| SparseGPT, general-text calibration | Existing operational baseline |
| SparseGPT, general text plus math calibration | Critical control for exposure to relevant task data |
| Variant A, meaningful differences | Proposed intervention |
| Variant A, randomly re-paired examples | Test whether controlled pairing matters |
| Variant A, randomly rotated differences | Test whether meaningful orientation matters |
| Task-gradient/Fisher-weighted pruning | Direct downstream-sensitivity comparator |

### Required locality comparison

The table above tests U. To test the central L hypothesis, add this matched ladder on the same factor fit and samples:

1. Fixed global span across all factors/regions.
2. Input-dependent active factors, each using its fixed global span `V_i`.
3. Input-dependent active factors, each using its assigned local chart `U_i,c(x)`.
4. The same local charts with assignments shuffled within factor/occupancy strata, preserving the marginal chart frequencies and ranks.
5. Whole-activation charts without factor decomposition, to test whether generic clustering explains any gain.

Compare local versus fixed-factor methods at matched total fitting/storage budgets and report their protected ranks. Include a rank-matched control so that reduced rank alone cannot explain the result. Keep regularization normalization and solver quality comparable. The assignment shuffle must change sample-to-projector correspondence; a relabeling that leaves the objective unchanged is invalid.

For a direct-write layer, a proposed output-side objective is:

```text
A = W_dense - W_pruned
L = E_x ||A phi(x)||^2 + gamma E_x ||Q(x)^T A phi(x)||^2
```

Here Q(x) is formed from the tangents of co-active factors at their respective locations. Estimate geometry at the output site where this error lands and pair each projector with that sample's layer input. Factor presence and assignment drift remain separate measurements. If using soft chart weights, define a positive-semidefinite weighted metric explicitly; it is generally not an orthogonal projector.

This sample-dependent output metric is not implemented exactly by adding one shared input covariance. Use an explicitly labeled approximation and validate it against a feasible small-layer/fixed-mask reference. Reuse the existing output-metric machinery only after checking its factor/chart semantics and damping. A negative result from an inaccurate solver cannot isolate failure of the geometric hypothesis.

All math-aware arms receive access to the same fitting examples and use the same ambient calibration token budget and mixture. The task-calibrated SparseGPT control must include the pair endpoints as ordinary calibration examples, with token counts matched; it cannot be disadvantaged by withholding the extra task data. Keep separate records for data access, activation collection, and gradient compute.

For re-pairing, actually change pair membership within compatible family/length strata, using a seeded derangement where possible. Reordering rows, changing signs, or permuting labels with no covariance change is not a valid null.

For the orientation control, apply a seeded orthogonal transformation in input space to the difference vectors, preserving the Gram matrix, covariance spectrum, rank, and factor energies while changing alignment with the model. A consistent signed coordinate permutation is an inexpensive orthogonal construction; identify its restricted null explicitly. Use at least three null seeds and report their distribution, not the worst control. These controls must enter the same **input-side Hessian augmentation**, unlike the existing output-side random controls.

### Initial configuration

- One selected `mlp.down_proj`, 50% unstructured sparsity.
- Alpha sweep: `0, 0.3, 1, 3`; alpha zero must reproduce the corresponding task-calibrated SparseGPT path.
- Same damping, block size, precision, mask constraints, and compensation settings across applicable arms.
- Three calibration seeds, with paired evaluation problems and generation settings.
- Measure actual zero counts after casting to model precision.

The implemented alpha is the added-to-ambient trace ratio: `gamma = alpha * tr(H) / tr(H_delta)`. Its fraction of the augmented trace is `alpha / (1 + alpha)`. Report the implementation's actual normalization and damping placement.

The one-layer 50% run is an engineering/measurement pilot, not the compression target. For the model-wide study, evaluate a predeclared coarse sparsity ladder of 0, 0.1, 0.3, 0.5, and 0.7 on validation, then refine the retention boundary in 0.05 steps. Each candidate starts from the same dense checkpoint; sequential layer pruning within that candidate uses refreshed activations. Report all evaluated points rather than assuming accuracy is monotone. Treat 2:4 as a separate extension. Freeze alpha, eligible layers, generation protocol, and the validation-selected sparsity before final evaluation.

### Multi-layer extension

After the one-layer study, prune a predefined set of MLP output projections sequentially, followed by a broader model experiment only if justified. Report eligible-layer sparsity and whole-model sparsity separately. Maintain identical layer sets and budgets across arms.

Refresh paired activations from each arm's current partially pruned model before pruning the next layer. Saved dense-model deltas may be evaluated as an explicitly labeled approximation, not silently reused as current activations. Cache keys must identify the upstream checkpoint/masks, dataset, site, and model revision.

## 8. Metrics, uncertainty, and decision rules

Primary practical outcome: the largest **measured whole-model sparsity** whose generated-answer accuracy remains within delta percentage points of dense under an identical generation protocol. The acceptable delta is configurable and awaiting user selection; 2 percentage points is a provisional sensitivity-analysis value, not an agreed threshold. Publish accuracy-versus-sparsity and accuracy-drop-versus-sparsity curves, with uncertainty, for each method. Count removed weights over both eligible matrices and the full text model; a 50% sparse single matrix is not a 50% sparse model. State whether embeddings, attention, and other projections are excluded.

Primary utility outcome U: improvement in attainable sparsity at the same dense-relative accuracy tolerance versus task-calibrated SparseGPT. Accuracy comparisons at fixed sparsity are supporting diagnostics. Primary locality outcome L: whether correctly assigned factor-local geometry extends this frontier beyond fixed-factor, pooled-difference, and shuffled-assignment controls. No method is required to outperform dense accuracy.

Secondary outcomes: dense-correct retention, dense-wrong recovery, answer NLL, pair-both-correct rate, vocabulary KL, geometric and ambient errors, general-text NLL/perplexity, runtime, peak device/host memory, and geometry collection overhead. Pair-both-correct rate requires both counterfactual problems to be answered correctly; preserving an incorrect relationship is not success.

Use a hierarchical paired bootstrap over source families and instances for controlled tasks, and paired problem-level resampling for GSM8K. Report variation across calibration/null seeds without counting those runs as new independent problems. For Stage 2, account for both shared test problems and intervention groups. Predeclare simultaneous/multiple-comparison handling for confirmatory control comparisons; all unplanned analyses are exploratory.

Before confirmation, freeze:

1. The allowed dense-relative accuracy drop delta, supplied by the user and frozen before confirmation, plus the minimum meaningful sparsity advantage for method comparisons.
2. A general-language noninferiority margin, initially proposed as **0.02 nats/token relative to task-calibrated SparseGPT**, checked on separate validation data for feasibility and then frozen.
3. Final sample sizes, selected configuration, comparisons, uncertainty procedure, and exclusions.

For a candidate to establish retention, the upper one-sided 95% confidence bound on its paired accuracy drop from dense must be no larger than delta. Distinguish point-estimate retention from statistically established retention. Select the largest qualifying sparsity on validation and evaluate that frozen candidate on the untouched test; do not scan test outcomes to choose a sparsity. A small sample that cannot establish retention is inconclusive, even if its observed accuracy equals dense.

Support comparative U when the selected method permits more sparsity under the same retention rule than the matched baseline. Require matched controls to attribute that advantage to meaningful geometry. If geometry is preserved but task accuracy is not, report geometric preservation without practical utility. General-language degradation is a secondary guardrail with the frozen tolerance above. Correct-answer retention and dense-wrong recovery must be reported separately so compensating errors are visible.

Before freezing generation, measure truncation on development data and use a sufficient common reasoning budget across arms. Do not selectively give failing methods or examples extra tokens. Report budget sensitivity separately from task retention.

Run the frozen transfer comparison on the full GSM8K test set after model-wide settings are selected. A successful controlled experiment with failed GSM8K transfer supports a restricted-domain result, not general reasoning preservation.

## 9. Implementation work

Proposed modules and artifacts below are work items, not existing commands.

1. **Dataset and manifest:** implement reviewed template loading, exact solutions, constrained generation, nuisance controls, family-level splitting, tokenization checks, and provenance hashes under `src/sparsifier/data/`.
2. **Readout capture and factor atlas:** implement semantic readout-position selection, crossed factor sweeps, factor activity estimation, within-factor charts, and frozen assignment rules, with explicit input/output sites and split-aware caches.
3. **Behavior evaluator:** add frozen prompt rendering, deterministic generation, exact numeric parsing, per-problem outcomes, and answer-conditioned NLL under `src/sparsifier/eval/`.
4. **Stage 2 integration:** extend the natural-layer intervention path with input-difference predictions, matched groups, noise checks, and held-out predictive comparisons.
5. **Stage 3 integration:** parameterize sparsity/calibration, add input-side nulls and equal-data controls, implement and validate the factor-local output metric and its fixed-span/shuffled-assignment controls, persist masks and validation selections, and integrate sequential pair/atlas refresh with explicit reference provenance.
6. **Reporting:** write per-example JSONL plus aggregate reports under `results/tangent/gsm8k/`; keep configuration, source hashes, checkpoint provenance, and split manifests with every run.
7. **Scheduling:** add resumable, one-phase-per-job gflow submission helpers only after CPU validation and pilot sizing.

Meaningful CPU tests should cover generator arithmetic/constraints; source-family leakage; readout alignment; parser handling of fractions, signs, malformed/truncated outputs; covariance preservation and nontriviality of null controls; alpha-zero parity; exact weight restoration; and cache/config mismatch rejection. Small scheduled GPU checks should validate full-forward readout capture and end-to-end metrics before the scientific runs.

Do not modify the existing family sweeps or running processes as part of this experiment. All GPU work, including the dense pilot, goes through gflow. Inspect daemon/queue state, set measured VRAM/RAM caps, and serialize phases with dependencies. Never start an additional full-model load in an ad-hoc shell while another experiment may run.

## 10. Execution order and deliverables

| Phase | Deliverable | Decision |
|---|---|---|
| CPU preparation | Reviewed templates, immutable splits, tested evaluator and nulls | Proceed when arithmetic, provenance, and alignment checks pass |
| Dense pilot | Competence, precision/noise, runtime and variance report | Freeze feasible model, prompts, budgets, and confirmation criteria |
| Stages 0–1 | Stability and local/global geometry report | Scope tangent claims; retain contrast experiment if geometry is unsupported |
| Stage 2 | Compensated/uncompensated behavioral prediction report | Assess P independently; refine only on validation before freezing Stage 3 |
| One-layer Stage 3 | Equal-data, matched-control accuracy/language-loss comparison | Assess U at one layer; proceed to multi-layer study if warranted |
| Multi-layer Stage 3 | Frozen pruning configuration and held-out controlled-task report | Establish model-level utility before benchmark transfer |
| GSM8K transfer | Full per-problem outputs and paired comparisons | Assess T without retuning |

The first scientific deliverables are the factor/local-geometry validation and a Stage 2 report connecting predicted weight importance to held-out mathematical behavior. Stage 3 must distinguish benefits from task calibration, pooled contrasts, factor selection, and location within factors. The factor-local output comparison is part of the main experiment; Stage 4 bias folding, invented presence references, and unrelated output-metric variants remain deferred.

## 11. Downstream tangent pullback extension (2026-11-08)

Add a `downstream` arm to the same all-MLP eligible-layer scope and sparsity ladder (0.1, 0.3, 0.5, 0.7), followed by the existing 512-problem original-GSM8K validation at a common 2,048-token budget. A separate immutable plan and submission journal preserve the already running experiment definitions. A two-block/six-projection smoke test gates all full candidates; candidate provenance gates their evaluations.

For target linear output \(y_\ell=W_\ell\phi_\ell\), fit factor-local tangent projectors \(P_{k,n}\) on geometry-fit activations at the next block's residual output. The final block uses its own residual output after the MLP. Seed detached random tangent probes at the final prompt position and estimate the diagonal of

\[ B_{\ell,n}=J_{k\leftarrow\ell,n}^{\top}P_{k,n}J_{k\leftarrow\ell,n} \]

using backpropagation. The intervening Jacobian includes token mixing; the pruning approximation subsequently drops cross-token/output correlations. Pair each target-position sensitivity with its actual local input to accumulate additional row-curvature diagonals. Apply the Variant B rescale/solve/unscale method with matched compensation. This tests propagation of activation geometry rather than a task-loss/Fisher metric.

Initial settings are alpha 1, seed 0, eight probes per prompt, and 32 seeded family-balanced pruning-calibration problems, held separate from the geometry-fit examples. The optional local input-tangent regularizer is disabled for this first arm. Refresh anchor geometry and all local activations from the current partially pruned model before each projection. Keep the same 4,096 C4 + 4,096 math-token ambient budget as task SparseGPT and the local arm; backward prompts contribute only to geometric statistics. Record all fitting/probe IDs, anchors, atlas and candidate hashes, raw traces, and precision settings. Extra gradient compute and geometry sample counts differ from the local-only arm and must be reported explicitly.

Use exclusive guarded gflow allocations and stop the model forward at the selected anchor to bound the backward graph. Save per-site checkpoints in the existing upstream-hashed format so original validation and retention reporting can consume this arm. A successful smoke test validates integration only. Compare actual whole-model removal counts, held-out generated answers, answer NLL, language diagnostics, runtime and memory; preserving downstream tangent coordinates alone does not establish behavioral retention. The acceptable accuracy drop and final-test selection remain unset.
