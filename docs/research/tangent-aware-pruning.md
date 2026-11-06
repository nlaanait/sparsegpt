# Research note: Tangent-aware pruning via local activation geometry

**Created:** 2026-11-01  
**Status:** exploratory — weight-level pruning / compression guided by local concept tangents  
**Prerequisite:** validate local data tangents and their relationship to weight perturbations as described in [bsf-def1-beyond-global-subspaces.md](bsf-def1-beyond-global-subspaces.md).  
**Related notes:** [preserving-isometry-under-truncation.md](preserving-isometry-under-truncation.md), [corpus-quantization-ideas-oct2026.md](corpus-quantization-ideas-oct2026.md), and [neural-operator-steering-on-activations.md](neural-operator-steering-on-activations.md).  
**Related corpus papers:** ID 86 (BSF); ID 51 (Manifold Steering); ID 228 (DBW); ID 230 (SliceGPT); ID 85 (Bonsai 2).

---

## Research question and scope

Standard activation-aware pruning methods rank weights using activation statistics or a layer-reconstruction objective. This note asks whether weighting reconstruction error by the **input-conditioned local tangent directions** of concept activations can improve weight selection.

This is a hypothesis about a useful pruning metric, not an established property of neural networks. The proposed projected loss measures local activation reconstruction at a chosen site. By itself it does **not** establish preservation of manifold distances, downstream behavior, task accuracy, or chart assignments. Those require separate measurements.

The local tangent model is motivated by the local-chart reading of BSF Def. 1 in [bsf-def1-beyond-global-subspaces.md](bsf-def1-beyond-global-subspaces.md):

1. An input may activate a sparse set of factors \(S(x)\subset[M]\).
2. Each active factor may vary locally along a tangent \(T_{m_i(x)}\mathcal M_i\), with basis \(U_{i,n}\in\mathrm{St}(r_i,d)\).
3. The rank of the active union is
   \[
   r_{\mathrm{union}}(x)
   =\dim\operatorname{span}\{T_{m_i(x)}\mathcal M_i:i\in S(x)\},
   \]
   which is at most \(\sum_{i\in S(x)}r_i\). Equality requires the factor tangents to be linearly independent. It can be much less than the ambient dimension \(d\), but this must be measured.

Chart anchors are 0th-order offsets; tangent vectors describe local variation. A fixed layer bias can absorb only a suitable global offset, not arbitrary chart-specific or input-dependent residuals.

### Hypotheses

- **H1 — Local geometry helps at fixed representation budget.** For a concept with curved activation support and stable local tangent estimates, a local-tangent representation will yield lower held-out tangent-projected reconstruction error than a fixed global span at matched retained dimension or storage budget. The hypothesis is disconfirmed if local charts are unstable, do not improve held-out error, or their estimation cost outweighs the gain.
- **H2 — Tangent scores predict tangent knockout impact.** A score derived from the projected objective will rank candidate weight removals more accurately than ambient magnitude, Wanda, or ambient OBS scores for measured held-out tangent-projected layer error. The hypothesis is disconfirmed if score ranking or calibration is no better than those baselines.
- **H3 — Tangent-aware masks improve the quality/budget tradeoff.** At matched sparsity and with the same compensation budget, a tangent-aware candidate will reduce held-out tangent-projected layer error, and may improve downstream task or concept metrics, relative to ambient pruning baselines. Failure to improve the geometric metric disconfirms the first part; no task improvement would limit the practical claim without disproving the local reconstruction result.
- **H4 — Some normal residuals can be removed or folded safely.** If the pruned residual lies mainly outside the active tangent union and is approximately a constant offset, removing it or applying the derived bias correction will preserve held-out task and concept metrics within a predeclared tolerance. Input-dependent normal residuals, chart changes, or task degradation disconfirm the relevant case.
- **H5 — Scores reveal distinct parameter roles.** Low tangent knockout impact with high ambient impact may identify parameters whose local effect is mostly normal to the measured tangent union; high editability with low current knockout impact may identify candidate control directions. Per-concept participation may predict selective versus broad impact. These are diagnostic hypotheses only; score patterns do not establish that weights own concepts, are safe to prune, or will be useful for steering.

---

## 1. Editability and knockout impact

Let \(h(x)\in\mathbb R^d\) be the activation at the selected site. For a scalar parameter \(W_\alpha\), local editability along concept \(i\) is

\[
e_i^{\mathrm{edit}}(\alpha;x)
=
\frac{1}{\sqrt{r_i}}
\left\|
U_{i,n}^{\top}\frac{\partial h(x)}{\partial W_\alpha}
\right\|_F.
\]

For a parameter block, interpret the Jacobian as a linear map from the vectorized parameter perturbation to activation space. Contract it with the specified perturbation; do not treat a matrix-valued parameter as a scalar without defining that contraction.

Editability is a local sensitivity, not current importance. Removing a scalar parameter is a finite change \(\delta W_\alpha=-W_\alpha\), so the first-order activation change is

\[
\delta h^{(-\alpha)}(x)
\approx
-\frac{\partial h(x)}{\partial W_\alpha}W_\alpha.
\]

Its per-factor tangent impact is

\[
\tau_i(\alpha;x)
=
\frac{1}{\sqrt{r_i}}
\left\|
U_{i,n}^{\top}
\frac{\partial h(x)}{\partial W_\alpha}W_\alpha
\right\|_F.
\]

This is a first-order estimate; for large knockouts, measure the actual forward-pass change as well.

There are two valid but different aggregation choices:

1. **Union-subspace metric.** Let \(Q_n\) be an orthonormal basis for the actual span of all active factor tangents, \(P_{\mathrm{union},n}=Q_nQ_n^\top\). Then
   \[
   \tau_{\mathrm{union}}(\alpha)^2
   =
   \mathbb E_x\left[
   \left\|
   Q_n^\top
   \frac{\partial h(x)}{\partial W_\alpha}W_\alpha
   \right\|_F^2
   \right].
   \]
   This counts a shared direction once. With the ambient score
   \[
   \tau_{\mathrm{ambient}}(\alpha)^2
   =
   \mathbb E_x\left[
   \left\|
   \frac{\partial h(x)}{\partial W_\alpha}W_\alpha
   \right\|_F^2
   \right],
   \]
   the ratio \(\rho_{\mathrm{union}}=\tau_{\mathrm{union}}^2/\tau_{\mathrm{ambient}}^2\) lies in \([0,1]\), assuming the same perturbation and activation site in both terms.
2. **Per-concept metric.** Report each \(\tau_i\), or define a nonnegative weighted sum of per-concept losses. This intentionally may count overlapping directions more than once. It is not generally equal to the union-subspace score, and its ratio to ambient energy need not be at most one unless the weighting operator is bounded by the identity.

Do not use \(\sum_i r_i\) as the union rank unless independence has been checked. The per-concept sum and union projector answer different questions.

---

## 2. Candidate parameter regimes

The following are score-defined categories to test, not semantic labels or pruning guarantees.

| Candidate regime | Measured pattern | Hypothesis to test |
|---|---|---|
| Tangent-sensitive | High tangent knockout score, possibly modest weight magnitude | Removing these parameters may increase held-out tangent error; magnitude ranking may miss some |
| Ambient-heavy, tangent-light | High ambient but low union-tangent impact | Some of the measured output change may be normal to the estimated union; safety depends on downstream/task evaluation |
| High-editability, low-current-impact | High local editability but small current knockout effect | May indicate capacity for a chosen perturbation direction; does not establish a useful dormant steerer |
| Low measured impact | Low ambient and tangent knockout impact, low editability | Candidate low-priority parameters within the tested data distribution |

For concept participation, define
\[
\tau_i(\alpha)
=
\left(
\mathbb E_x[
\mathbf 1_{i\in S(x)}\,\tau_i(\alpha;x)^2]
\right)^{1/2},
\qquad
\mathrm{PR}(\alpha)
=
\frac{\left(\sum_i\tau_i(\alpha)^2\right)^2}
{\sum_i\tau_i(\alpha)^4}.
\]
The participation ratio ranges from 1 to \(M\) when the denominator is nonzero. It measures concentration of this score across the chosen concept set. The interpretation “specialist” versus “broadly shared” is a hypothesis to validate with selective ablations, not a conclusion from PR alone.

---

## 3. Direct-layer projected reconstruction objective

This section assumes the target weights write directly into the activation site:
\[
h_n=h_{\mathrm{prev},n}+W\phi_n+b,
\qquad
W\in\mathbb R^{d\times d_{\mathrm{in}}}.
\]
For \(A=W-\hat W\), the residual before bias correction is \(A\phi_n\). Let \(M_n\succeq0\) be the output-space metric. The union-tangent choice is \(M_n=P_{\mathrm{union},n}\). An optional anchor term can be added, but its interpretation is discussed below.

The sample-projected reconstruction objective is
\[
\min_{\hat W\in\mathcal C,\,\Delta b\in\mathbb R^d}
\frac1N\sum_{n=1}^N
(A\phi_n-\Delta b)^\top M_n(A\phi_n-\Delta b).
\]
This is a local activation-reconstruction loss. It is not, by itself, a manifold-geodesic, isometry, or task loss.

For fixed \(A\) and \(M_n\), the least-squares bias correction satisfies
\[
\left(\sum_n M_n\right)\Delta b
=
\sum_n M_n A\phi_n.
\]
A minimum-norm solution is
\[
\Delta b=
\left(\sum_n M_n\right)^\dagger
\left(\sum_n M_n A\phi_n\right),
\]
with arbitrary components in the null space of \(\sum_n M_n\). If every \(M_n=I\), this reduces to \(\Delta b=A\,\mathbb E[\phi]\). If \(M_n\) varies, the unweighted global-mean formula is not generally optimal. Even an optimal global bias cannot absorb input-dependent residuals.

An anchor penalty could use
\[
M_n=P_{\mathrm{union},n}+\eta P_{\mathrm{anchor},n}.
\]
However, projecting onto the span of absolute chart-anchor vectors is origin-dependent and does not by itself guarantee correct discrete chart selection. Until an explicit chart-selection metric is defined, treat anchor preservation as an auxiliary hypothesis and report it separately.

---

## 4. SparseGPT and Wanda: derivations and candidate variants

### Ambient reconstruction and OBS convention

For calibration activations collected as columns of \(X\), the unregularized layer reconstruction loss is
\[
\mathcal L_{\mathrm{amb}}(A)
=
\frac1N\|AX\|_F^2.
\]
Its input covariance is \(H_0=XX^\top/N\). A ridge term \(\lambda I\) may stabilize the inverse used by an algorithm; distinguish \(H_0\) from \(H=H_0+\lambda I\).

For a single row and the quadratic OBS problem with curvature \(H\), constraining coordinate \(q\) to zero and compensating over the other coordinates gives a loss increase proportional to
\[
\frac{W_{pq}^2}{[H^{-1}]_{qq}},
\]
and the corresponding free-coordinate update is
\[
\Delta W_{p,j\ne q}
=
-\frac{W_{pq}}{[H^{-1}]_{qq}}[H^{-1}]_{qj}.
\]
A factor of \(1/2\) depends on the loss convention. SparseGPT implementations may use a Cholesky factor of the inverse Hessian in their iterative update and score; do not substitute the square of a literal inverse-Hessian diagonal for a factor-diagonal expression without deriving the mapping for that implementation. In the ambient objective, the same input curvature is shared by every output row. Thus, within one input column, the OBS curvature factor is shared across rows, so its ranking is by the current weight magnitude. This statement is about the ambient layer-reconstruction surrogate, not downstream task importance.

Wanda ranks by \(|W_{pq}|\|\phi_q\|_2\), compared per output row. Its squared score corresponds to a diagonal input-covariance approximation to the ambient reconstruction objective.

### Direct tangent single-weight score

For a direct-layer single-entry knockout,
\[
\delta h_n=-W_{pq}\phi_{q,n}e_p.
\]
Under the union-tangent projector, the exact projected squared activation error for that un-compensated one-entry perturbation is
\[
I_{pq}^{\mathrm{tan,diag}}
=
\frac{W_{pq}^2}{N}
\sum_n
\|Q_n^\top e_p\|_2^2\phi_{q,n}^2.
\]
This is the direct diagonal tangent analogue of squared Wanda. It is not the same as a second-order OBS score after compensation.

Define the tangent coupling
\[
G_{pq}
=
\frac{\sum_n (P_{\mathrm{union},n})_{pp}\phi_{q,n}^2}
{\sum_n\phi_{q,n}^2},
\]
when the denominator is positive. Then
\[
I_{pq}^{\mathrm{tan,diag}}
=
W_{pq}^2\,G_{pq}\,
\frac1N\sum_n\phi_{q,n}^2.
\]
If the denominator is zero, the feature is inactive on calibration data and this score is undefined by the ratio; assign zero direct calibration impact and flag that the feature has no observed support. The formula uses sample second moments, not a ridge-inflated covariance diagonal.

### Option 1 — Reweighted SparseGPT mask (heuristic)

A low-overhead candidate is to multiply the ambient mask score by \(G_{pq}\). This can test whether output-coordinate tangent occupancy improves mask selection. It is **not generally an exact Tangent-SparseGPT derivation**:

\[
\mathcal L_{\mathrm{tan}}(A)
=
\frac1N\sum_n
\operatorname{tr}\!\left(
A^\top P_{\mathrm{union},n}A\,\phi_n\phi_n^\top
\right).
\]
In vectorized form, its curvature is proportional to
\[
\sum_n(\phi_n\phi_n^\top)\otimes P_{\mathrm{union},n}.
\]
When \(P_{\mathrm{union},n}\) has off-diagonal entries, output rows are coupled. Even if each projector is diagonal, sample-varying diagonal weights generally give each output row a different input covariance. \(G\) retains only diagonal output occupancy and featurewise squared activations; it omits these couplings and cross-feature covariance.

Only in restricted cases—such as a fixed diagonal output metric that scales each row’s ambient curvature—does multiplying row scores by a fixed output weight reproduce the corresponding rowwise quadratic weighting. For the general input-conditioned case, label the reweighted score a heuristic, retain it as an ablation, and measure its actual projected loss after compensation. Exact optimization of the full projected quadratic would require handling its coupled curvature or a justified structured approximation.

### Option 2 — Per-chart OBS (candidate with restricted exactness)

For a fixed chart \(c\), basis \(U_c\), and calibration subset \(I_c\), define \(B_c=U_c^\top W\) and conditional input curvature
\[
H_c=\frac1{|I_c|}\sum_{n\in I_c}\phi_n\phi_n^\top+\lambda I.
\]
The projected chart loss is an ordinary quadratic reconstruction loss in \(B_c\). If pruning an ambient entry \(W_{pq}\) and restricting compensation to the same ambient output row, its projected perturbation is \(U_c^\top e_p W_{pq}\); the sum of tangent-coordinate OBS costs contains the factor \(\|U_c^\top e_p\|_2^2\) multiplied by the scalar OBS cost based on \([H_c^{-1}]_{qq}\).

This is exact only for that specified fixed-chart quadratic and compensation restriction, with the corresponding inverse-Hessian convention. If compensation can use other output rows, if charts use different masks, or if independently optimized chart updates must be combined into one global sparse matrix, the displayed per-chart costs are not automatically the global optimum. Validate the candidate against direct quadratic solves on a small layer before using “exact” for any broader claim.

### Option 3 — Fixed-union subspace rotation

If one fixed orthonormal frame \(U\) spans all tangent directions to be preserved, write
\[
W_\parallel=U^\top W,\qquad W_\perp=(I-UU^\top)W.
\]
For the fixed union-projected objective, the loss depends only on \(W_\parallel\), so ordinary ambient reconstruction in the rotated tangent coordinates uses the same input covariance. This gives a coherent candidate for preserving the fixed union projection.

It is not equivalent to the input-conditioned objective when different examples activate different tangent subspaces: the fixed union may retain directions that are inactive on each individual example. \(W_\perp\) has zero loss under the fixed union-projected surrogate, but may still affect downstream behavior. Folding its mean into bias removes only the constant component, not input-dependent residuals. Test those effects separately.

### Option 4 — Input-space probe covariance on the ambient Hessian

A separate, already-implemented candidate augments the **ambient input Hessian** rather than projecting the output residual. Collect calibration inputs \(\phi_n\) as usual so
\[
H_0
\propto
\sum_n\phi_n\phi_n^\top,
\]
and add a bank of ambient input probes \(t_k\) (columns of a probe matrix \(T\)) with relative weight \(\alpha\ge 0\):
\[
H
\propto
\sum_n\phi_n\phi_n^\top
+\alpha\cdot\frac{N_{\mathrm{act}}}{K}\sum_{k=1}^{K}t_kt_k^\top,
\]
up to the same SparseGPT scaling convention used for \(H_0\) (this repository’s `GeometrySparseGPT` matches the upstream \(2\sum\phi\phi^\top/n_{\mathrm{seq}}\) scale on both terms). Ordinary ambient SparseGPT then runs on \(H\).

This preserves a surrogate of the form
\[
\mathbb E\|\!(S-W)\phi\|_2^2
+\alpha\,\mathbb E\|\!(S-W)t\|_2^2
\]
in **input** space. It is not equivalent to Options 1–3: those weight or rotate **output**-space residuals by local tangent projectors \(P_{\mathrm{union},n}\) or a fixed \(U\). Input probes may come from a block-dictionary / local-PCA pipeline on activations, but that does not by itself implement the projected loss \(\mathcal L_{\mathrm{tan}}\). Treat Option 4 as its own ablation arm; do not describe it as exact tangent-projected OBS.

---

## 5. Relation to other compression methods

| Method | Geometry or objective used | Open question for this proposal |
|---|---|---|
| Magnitude, Wanda, SparseGPT | Weight magnitude, activation statistics, or ambient layer reconstruction | Does tangent weighting improve held-out tangent or task metrics at matched sparsity? |
| SliceGPT | Global activation subspace and dimension reduction | Do local charts help when curvature and changing active support are measured? |
| Global BSF truncation | Fixed factor spans \(V_g=\operatorname{span}(\mathcal M_g)\) | Does local tangent rank provide a useful budget advantage over the factor span? |
| DBW | Weight spectral components and bit-width allocation | Do input-conditioned activation tangents add value beyond weight-spectrum selection? |
| Option 4 input-probe Hessian | Ambient reconstruction plus input-probe covariance on \(H\) | Does input-space probe energy improve task or tangent metrics versus output-projected Options 1–3 at matched sparsity? |
| Tangent-aware candidates | Input-conditioned local tangent projections | Requires reliable tangent estimates and validation beyond the calibration set |

Connection to QTEA/Bonsai remains speculative: a tangent-derived residual allocation score could be tested after ternary quantization, but it should not be described as allocating bits only to concept-distorting weights until measured.

---

## 6. Experiments that can support or disconfirm the hypotheses

Use separate data for tangent fitting, pruning calibration, and final evaluation wherever feasible. Report tangent stability across held-out examples and chart boundaries. Compare at identical sparsity/pattern and with identical compensation settings.

### Stage 1 — Validate local tangent estimates (H1 prerequisite)

1. Use a controlled concept sweep or other dataset with known variation; estimate local bases on one split and evaluate on held-out points.
2. Compare local tangent rank and stability with the global span. Do not assume \(r_i<b_i\); measure it with a stated rank threshold.
3. At matched rank/storage, compare local-chart and fixed-global-subspace reconstruction on held-out activations. Report local projected error, ambient error, and tangent-estimation cost.
4. Disconfirm H1 if local bases are unstable or provide no held-out gain at the same budget.

### Stage 2 — Validate score predictions before pruning (H2, H5)

1. On a small layer, enumerate single-entry knockouts and selected structured knockouts. Record actual un-compensated tangent and ambient errors on held-out calibration-like inputs.
2. Compare predicted tangent scores against measured tangent errors using rank correlation and calibration plots; include magnitude, Wanda, and ambient OBS scores.
3. Separately measure editability with small signed perturbations. Test whether it predicts the effect of those perturbations, not whether it implies current knockout importance.
4. For PR/regime claims, perform concept-selective and broad ablations and compare observed impacts to the predicted pattern.
5. Disconfirm H2/H5 if the proposed scores do not outperform baselines or do not predict the stated selective/broad effects.

### Stage 3 — Compare compensated pruning methods (H3)

At matched sparsity, compare magnitude, Wanda, standard SparseGPT, Option 1, and—where feasible—Options 2 and 3. Use the same calibration examples, pruning patterns, and compensation budget. Report:

- held-out union-tangent and per-concept projected reconstruction errors;
- ambient layer reconstruction error;
- downstream task and concept-probe metrics;
- runtime, memory, and tangent-estimation overhead.

For discordant groups, define disjoint sets by explicit score quantiles, equalize the number of removed parameters, and report both pre- and post-compensation results. A small ambient error also bounds the same-example orthogonal tangent-projected error from above; any task degradation despite small local errors is a separate downstream-sensitivity result.

### Stage 4 — Test bias folding and normal-residual safety (H4)

Compare no bias adjustment, the derived weighted least-squares bias, and the simple global-mean bias. Evaluate held-out projected residuals, chart assignments if available, and downstream behavior. Include cases with input-dependent residuals. Disconfirm safety for any condition where the relevant held-out metric exceeds its predeclared tolerance.

---

## Reporting standard

Present positive, null, and negative outcomes as evidence about the stated hypotheses. A failed hypothesis narrows when tangent-aware pruning helps; it does not by itself invalidate the broader research question. Do not describe a score as “safe,” a parameter as a concept owner, or a method as geometry-preserving unless the corresponding held-out experiment supports that claim.

---

## Implementation ablation registry (this repository)

Each candidate is a distinct `--pruner` name so Stage-3 comparisons are mechanical. Optional `--bias-fold` applies the §3 projected least-squares bias update after the chosen prune when the layer has a bias.

| `--pruner` | Note mapping | Notes |
|---|---|---|
| `sparsegpt` | Ambient OBS / SparseGPT | Baseline |
| `wanda` | Ambient Wanda | `|W|·RMS(φ)` per row; no OBS compensation |
| `geometry` | Option 4 input-probe Hessian | Requires `--geometry-alpha` and `--geometry-probes` |
| `tangent` | Option 1 (k-means charts) | Score blend `(1-α)+αG` retained; ambient compensation |
| `tangent_factor_union` | Option 1 with concept-factor unions | QR/SVD span of active factor tangents; not `∑ r_i` |
| `tangent_chart_obs` | Option 2 | Per-chart OBS mask scores; chart-local compensation |
| `tangent_fixed_union` | Option 3 | Fixed PCA union frame; sparsify `W_∥=U^⊤W` |
| `tangent_knockout` | §1 `I^{tan,diag}` / `τ_union` ranking | Ambient compensation |
| `tangent_editability` | §1 editability ranking | No `|W|` factor; ambient compensation |
| `tangent_exact_obs` | Exact `L_tan` OBS | `--tangent-exact-mode={diag,full,auto}`; diag-P row Hessians or small Kronecker |

Optional `--tangent-anchor-eta η` builds `M_n=P_union+η P_anchor` (report separately). Disjoint tangent-fit vs prune-cal windows: `--tangent-fit-nsamples` / `--tangent-fit-seed`. Stage 1–4 held-out harness: `python -m sparsifier.eval.tangent_stages`.

§1 editability, per-concept knockout, union/ambient ratio, and participation ratio live in `sparsifier.prune.scores` for diagnostics and score-ranking ablations.
