# Research note: Tangent-aware pruning via local activation geometry (revised)

**Created:** 2026-11-01; **revised:** 2026-11-05  
**Status:** exploratory. Weight-level pruning and compression guided by local concept geometry measured in activations.  
**Revision of:** [tangent-aware-pruning.md](tangent-aware-pruning.md), kept unchanged for history. The changes follow [tangent-aware-pruning-review.md](tangent-aware-pruning-review.md), which has the derivations and numerical checks behind them.  
**Prerequisite:** validate local data tangents and their relationship to weight perturbations as described in [bsf-def1-beyond-global-subspaces.md](bsf-def1-beyond-global-subspaces.md).  
**Related notes:** [preserving-isometry-under-truncation.md](preserving-isometry-under-truncation.md), [corpus-quantization-ideas-oct2026.md](corpus-quantization-ideas-oct2026.md), [neural-operator-steering-on-activations.md](neural-operator-steering-on-activations.md), [bonsai2-ternary-conversion-spec.md](bonsai2-ternary-conversion-spec.md).  
**Related corpus papers:** ID 86 (BSF); ID 51 (Manifold Steering); ID 88 (SparseGPT); ID 230 (SliceGPT); ID 228 (DBW); ID 85 (Bonsai 2); ID 210 (Debias-SparseGPT); ID 254 (diagonal Fisher and update sparsity); ID 220 (GA-SVD); ID 142 (Fisher-Rao geometry of LLMs); ID 107 (curved but unused number geometry); ID 223 (QTEA).

---

## What changed in this revision

1. The geometric loss is now a regularizer on top of ambient reconstruction, never a replacement, because the pure tangent projection is degenerate (Sec. 3.1).
2. Each active concept gets a presence direction in addition to its tangent, because tangent-only metrics cannot see a concept being attenuated (Sec. 3.2).
3. A new input-side objective penalizes the layer's error on paired concept differences. It runs exactly in unmodified SparseGPT and is now the first pruning experiment (Sec. 3.3 and 4.4).
4. SparseGPT's exact score, its mask comparison groups, and the basis dependence of entrywise scores are spelled out, because together they decide when an output-side reweighting can change a mask at all (Sec. 4.1 to 4.3).
5. Option 1 (scores times \(G\)) is replaced by a rescaling trick with matching compensation; Option 2 is restated as row Hessians with an exact refit; Option 3 is replaced by a sparse-plus-tangent construction (Sec. 4.5 to 4.7).
6. A Fisher baseline and an output-metric ladder share one implementation (Sec. 4.8).
7. Scores use raw energies in losses and per-dimension or enrichment-normalized values in comparisons; participation is reported conditionally and unconditionally; editability of direct-write entries is redirected to upstream or neuron-level perturbations (Sec. 1 and 2).
8. The hypotheses are restated to remove circularity, and a premise hypothesis H0 is added.
9. Bias correction is built into the pruning pass (centered covariance, or a constant input column processed last) instead of post-hoc folding (Sec. 3.4).
10. Stage-0 diagnostics come before any pruning, and the experiment order is revised (Sec. 6).

---

## Research question and scope

Standard activation-aware pruning ranks weights by activation statistics or by a layer-reconstruction objective. This note asks whether adding concept-geometry terms to that objective improves weight selection at matched sparsity. The terms emphasize either the input-conditioned local tangent and presence directions of concept activations at the layer output, or the concept variation entering the layer at its input.

This is a hypothesis about a useful pruning metric, not an established property of neural networks. The geometric terms measure local activation reconstruction at a chosen site. They do not by themselves establish preservation of manifold distances, downstream behavior, task accuracy, or chart assignments; those require separate measurements.

Two constraints frame everything below. First, the geometric terms are regularizers on top of ambient reconstruction: on their own they ignore everything the layer writes outside the concept directions (Sec. 3.1). Second, whether emphasizing concept directions improves average downstream loss is an empirical premise (H0). If the premise fails, the method remains a concept-preservation tool and is judged on concept metrics only.

### Local model

The local model follows the local-chart reading of BSF Def. 1 in [bsf-def1-beyond-global-subspaces.md](bsf-def1-beyond-global-subspaces.md):

1. An input activates a sparse set of factors \(S(x) \subset [M]\).
2. Each active factor varies locally along a tangent \(T_{m_i(x)}\mathcal{M}_i\). Its basis \(U_{i,n} := U_{i,c(i,x_n)} \in \mathrm{St}(r_i, d)\) is that of the chart \(c\) assigned to factor \(i\) at input \(x_n\) (bsf-def1 Step 1).
3. Each active factor also has a presence direction: its chart anchor measured from an explicit concept-absent reference, \(m_{i,c} = \bar h(s_0) - h_0\) (bsf-def1 Step 1A), orthogonalized against the active tangents. On a curved factor this direction is generally normal to the tangent, and it carries concept strength (bsf-def1 Sec. 2).
4. \(Q_n\) is an orthonormal basis, computed with a stated rank threshold, for the span of the active tangents. Then \(P_n = Q_n Q_n^\top\) and \(r_n = \operatorname{rank} P_n \le \sum_{i \in S(x_n)} r_i\), with equality only if the factor tangents are linearly independent. \(P^{\mathrm{pres}}_n\) projects onto the presence directions, and \(P^{\mathrm{geo}}_n = P_n + P^{\mathrm{pres}}_n\), with rank \(r^{\mathrm{geo}}_n\).

Chart anchors are zeroth-order offsets; tangent vectors describe local variation. A fixed layer bias can absorb only a suitable global offset, not chart-specific or input-dependent residuals.

**Site.** Fix the activation site \(h\) and state whether it is the residual stream after the layer's write or a normalized block input. Estimate tangents and presence directions at the site where the pruning error lands. In pre-norm blocks the next reader sees \(\mathrm{RMSNorm}(h)\), whose Jacobian removes the direction of \(h_n\) itself to first order, so errors along \(h_n\) are partly invisible downstream (not fully, since the residual stream carries them forward).

### Notation

| Symbol | Meaning |
|---|---|
| \(h_n = h_{\mathrm{prev},n} + W\phi_n + b\) | direct-write layer at the site, with \(W \in \mathbb{R}^{d \times d_{\mathrm{in}}}\) |
| \(A = W - \hat W\) | pruning residual in weight space |
| \(H_0 = XX^\top / N\) and \(H = H_0 + \lambda I\) | input second moment and its dampened version |
| \(P_n\), \(P^{\mathrm{pres}}_n\), \(P^{\mathrm{geo}}_n\) | projectors onto the active tangents, the presence directions, and both |
| \(\bar r\), \(\bar r^{\mathrm{geo}}\) | means of \(r_n\) and \(r^{\mathrm{geo}}_n\) over the calibration set, zeros included |
| \(\gamma\), \(\alpha\) | geometric weight and its normalized form, \(\gamma = \alpha d / \bar r^{\mathrm{geo}}\) |
| \(\nu\) | weight density after pruning |
| \(C\) | upper Cholesky factor with \(H^{-1} = C^\top C\) |
| \(g_n = \partial \ell_n / \partial h_n\) | loss gradient at the site |

### Hypotheses

- **H0: downstream sensitivity is enriched along concept directions.** Loss gradients at the site carry more energy in the active tangent and presence directions than a random subspace of the same rank does, i.e. the ratio \(\kappa\) of Sec. 4.8 is well above 1. If \(\kappa \approx 1\), geometric emphasis is not expected to improve average downstream loss over ambient SparseGPT, and the method is judged on concept metrics only.
- **H1: local charts beat a global span at fixed budget, where they can.** For a concept with curved support and stable local tangents, an atlas of local charts reconstructs held-out concept-attributable variation better than a fixed global span at matched budget. The comparison must be metric-neutral (ambient reconstruction or description length) and must count chart storage and per-input chart assignment. For pruning metrics a gain is only possible when \(b_i > r_i + 1\) (Sec. 3.2), so choose test concepts accordingly. Disconfirmed if charts are unstable, give no held-out gain, or cost more than they save.
- **H2: geometric scores predict compensated and downstream impact.** Tangent and presence scores rank candidate removals better than magnitude, Wanda, ambient OBS, and Fisher scores when the target is the compensated layer error and the downstream change (loss, KL to the dense model, concept probes). This includes upstream layers, where the scores are only first-order. Predicting the uncompensated single-entry tangent error of a direct-write layer does not count, because that score is the calibration tangent error by construction. Disconfirmed if the geometric scores do no better than the baselines.
- **H3: geometry-aware pruning improves the tradeoff front.** At matched sparsity, pattern, and compensation budget, a geometry-aware method traces a better front in (ambient error, held-out tangent and presence error, downstream metric) than ambient methods across the \(\alpha\) sweep. Lower tangent error alone is expected by construction and does not count as support. Failure to improve downstream metrics limits the practical claim without disproving the geometric result.
- **H4: residual normal to all concept spans can be removed or folded safely.** Residual normal to the tangent but inside the concept span (radial, presence) is not expected to be safe to drop. Residual normal to all concept spans that is approximately constant can be removed or folded into the bias while preserving held-out task and concept metrics within a predeclared tolerance. Input-dependent normal residuals, chart changes, or task degradation disconfirm the relevant case.
- **H5: scores reveal distinct parameter roles.** Low tangent enrichment with high ambient impact may identify parameters whose effect is mostly normal to the concept directions. High editability with low current impact, measured on upstream weights or neuron-level perturbations, may identify candidate control directions. Conditional participation may predict selective versus broad impact. These are diagnostic hypotheses only: score patterns do not establish that weights own concepts, are safe to prune, or will be useful for steering.

---

## 1. Scores: editability, knockout impact, and enrichment

Let \(h(x) \in \mathbb{R}^d\) be the activation at the site and \(j_\alpha(x) = \partial h(x) / \partial W_\alpha\) the Jacobian column of a scalar parameter \(W_\alpha\). For a parameter block, interpret the Jacobian as a linear map from the vectorized perturbation to activation space, and contract it with a specified perturbation rather than treating a matrix-valued parameter as a scalar.

**Editability** is local sensitivity, not current importance. Keep bsf-def1's rank-normalized definition and its enrichment over a random \(r_i\)-plane:

\[ e_i^{\mathrm{edit}}(\alpha; x) = \frac{1}{\sqrt{r_i}} \lVert U_{i,n}^\top j_\alpha(x) \rVert_2, \qquad \tilde e_i^{\mathrm{edit}}(\alpha; x) = \frac{\lVert U_{i,n}^\top j_\alpha(x) \rVert_2 / \sqrt{r_i}}{\lVert j_\alpha(x) \rVert_2 / \sqrt{d}}. \]

**Knockout impact.** Removing a scalar parameter is the finite change \(\delta W_\alpha = -W_\alpha\), so the first-order activation change, exact for a direct-write layer, is

\[ \delta h^{(-\alpha)}(x) \approx -j_\alpha(x)\, W_\alpha. \]

Use raw energies wherever a score feeds a loss, and normalized values wherever concepts of different rank are compared:

- per-factor tangent energy \(\tau_i(\alpha; x)^2 = \lVert U_{i,n}^\top \delta h^{(-\alpha)}(x) \rVert_2^2\);
- per-dimension energy \(\tau_i(\alpha; x)^2 / r_i\), which is the original note's rank-normalized score;
- enrichment \(\tilde\tau_i(\alpha; x) = (d / r_i)\, \tau_i(\alpha; x)^2 / \lVert \delta h^{(-\alpha)}(x) \rVert_2^2\), equal to 1 on average for a random \(r_i\)-plane; it is the square of \(\tilde e^{\mathrm{edit}}\) applied to the single vector \(\delta h\);
- presence energy \(\tau_{\mathrm{pres}}(\alpha; x)^2 = \lVert P^{\mathrm{pres}}_n \delta h^{(-\alpha)}(x) \rVert_2^2\).

These are first-order estimates; for large knockouts, also measure the actual forward-pass change.

**Union metric.** With the union projector,

\[ \tau_{\mathrm{union}}(\alpha)^2 = \mathbb{E}_x \lVert Q_n^\top \delta h^{(-\alpha)}(x) \rVert_2^2, \qquad \tau_{\mathrm{amb}}(\alpha)^2 = \mathbb{E}_x \lVert \delta h^{(-\alpha)}(x) \rVert_2^2. \]

The ratio \(\rho_{\mathrm{union}} = \tau_{\mathrm{union}}^2 / \tau_{\mathrm{amb}}^2\) lies in \([0, 1]\), assuming the same perturbation and site in both terms. Compare it with its random-subspace null, \(\mathbb{E}_x[r_n \lVert \delta h \rVert^2] / (d\, \mathbb{E}_x \lVert \delta h \rVert^2)\), which is about \(\bar r / d\), rather than with 0 or 1. For a direct-write entry at a single sample, \(\rho_{\mathrm{union}}\) is exactly the coordinate leverage of Sec. 4.3.

**Per-concept metric.** Report each \(\tau_i\), or a nonnegative weighted sum of per-concept losses. This may count overlapping directions more than once, so it is not generally equal to the union score, and its ratio to ambient energy need not be at most one unless the weighting operator is bounded by the identity. Do not use \(\sum_i r_i\) as the union rank unless independence has been checked.

**Direct-write caveat.** For an entry \(W_{pq}\) of a direct-write layer, \(j = \phi_q e_p\). Editability and knockout both reduce to \(\lvert \phi_q \rvert\, \lVert U_{i,n}^\top e_p \rVert_2\) up to constant factors (\(1/\sqrt{r_i}\) for editability, \(\lvert W_{pq} \rvert\) for knockout), so editability adds no information there. Following bsf-def1 Sec. 0, measure editability on upstream weights, or on column- and neuron-level perturbations of direct-write layers.

---

## 2. Candidate parameter regimes

The following are score-defined categories to test, not semantic labels or pruning guarantees.

| Candidate regime | Measured pattern | Hypothesis to test |
|---|---|---|
| Tangent-sensitive | High tangent enrichment and energy, possibly modest weight magnitude | Removal may increase held-out tangent error; magnitude ranking may miss some |
| Presence-sensitive | High presence energy, low tangent energy | Removal may attenuate the concept without moving it along the manifold; tangent-only scores miss these |
| Ambient-heavy, concept-light | High ambient impact, tangent and presence enrichment near 1 or below | Much of the output change may be normal to the concept directions; safety depends on downstream and task evaluation |
| High-editability, low-current-impact | High editability (upstream or neuron-level) with a small current knockout effect | May indicate capacity for a chosen perturbation direction; does not establish a useful dormant steerer |
| Low measured impact | Low ambient and geometric impact, low editability | Candidate low-priority parameters within the tested data distribution |

For concept participation, separate impact from frequency:

\[ \tau_i^{\mathrm{cond}}(\alpha)^2 = \mathbb{E}_x\bigl[\tau_i(\alpha; x)^2 \mid i \in S(x)\bigr], \qquad \tau_i^{\mathrm{uncond}}(\alpha)^2 = P(i \in S)\, \tau_i^{\mathrm{cond}}(\alpha)^2. \]

The original note used the unconditional form, which makes rare concepts look negligible. Compute participation on per-dimension energies \(q_i = \tau_i^{\mathrm{cond}}(\alpha)^2 / r_i\):

\[ \mathrm{PR}(\alpha) = \frac{\left(\sum_i q_i\right)^2}{\sum_i q_i^2}. \]

It ranges from 1 to the number of concepts with nonzero \(q_i\). Use the conditional version for specialist-versus-shared questions and the unconditional one for expected loss, and report both. The specialist-versus-shared reading remains a hypothesis to validate with selective ablations, not a conclusion from \(\mathrm{PR}\) alone.

---

## 3. Layer objectives

This section assumes the target weights write directly into the site, \(h_n = h_{\mathrm{prev},n} + W\phi_n + b\), with \(A = W - \hat W\) and residual \(A\phi_n - \Delta b\) after a bias correction \(\Delta b\). For an output metric \(M_n \succeq 0\), the sample objective is

\[ \min_{\hat W \in \mathcal{C},\, \Delta b \in \mathbb{R}^d} \; \frac{1}{N} \sum_{n=1}^N (A\phi_n - \Delta b)^\top M_n (A\phi_n - \Delta b). \]

This is a local activation-reconstruction loss, not a manifold-geodesic, isometry, or task loss. Upstream layers are handled through their pullback in Sec. 4.3.

### 3.1 Why the pure tangent projection fails

Take \(M_n = P_n\), as in the original note. Let \(\mathcal{T} = \sum_n \operatorname{range}(Q_n)\) be the span of all tangent unions seen in calibration, with \(R = \dim \mathcal{T}\) and orthogonal projector \(\Pi\). Since \(Q_n^\top = Q_n^\top \Pi\), the objective depends on \(A\) and \(\Delta b\) only through \(\Pi A\) and \(\Pi \Delta b\). Consequences:

- \((I - \Pi)A\) is unconstrained: the objective is blind to everything the layer writes outside \(\mathcal{T}\). The null space of \(\sum_n M_n\) in the original bias solution is the same fact at the bias level.
- Zero calibration loss is reached whenever \(\Pi \hat W = \Pi W\) on the span of the inputs. That is \(R\) linear equations per input column, generically solvable at column density \(R/d\), and a 2:4 pattern admits zero tangent loss whenever \(R < d/2\). In a random toy (\(d = 64\), \(d_{\mathrm{in}} = 20\), \(R = 6\)), keeping \(R\) entries per column (9.4% density) gives zero tangent loss and an ambient error 157 times the layer's output energy.
- The original Option 3 ("\(W_\perp\) has zero loss") is the extreme case. The original Option 1 (scores times raw \(G\)) inherits the problem in unstructured mode: rows with near-zero leverage get near-zero scores and are pruned first, whatever their ambient importance.

Geometric terms are therefore used only as regularizers on top of the ambient loss.

### 3.2 Regularized output-side objective with presence directions

Use

\[ M_n = I + \gamma\, P^{\mathrm{geo}}_n, \qquad \gamma = \alpha\, d / \bar r^{\mathrm{geo}}, \]

so that \(\mathcal{L}_\gamma = \mathcal{L}_{\mathrm{amb}} + \gamma\, \mathcal{L}_{\mathrm{geo}}\). For isotropic errors, \(\alpha\) is the ratio of geometric to ambient loss, which makes it comparable across layers and concept sets. Sweep \(\alpha\) (for example 0, 0.1, 0.3, 1, 3, 10, and the limit below) to trace the front required by H3. Tangent-only (\(P^{\mathrm{geo}}_n = P_n\)) is kept as an ablation.

**Why presence directions.** Attenuating a concept, a common pruning failure, moves it along its presence direction, which a tangent-only metric cannot see. Position along the manifold (which chart is active) is tangential to first order, so the tangent term protects chart identity while the presence term protects concept strength. The presence direction replaces the original anchor term: it is measured from an explicit concept-absent reference \(h_0\) rather than from the origin, and it is orthogonalized against the tangent. It remains reference-dependent, so report results for the chosen \(h_0\).

**Bias.** The bias equation becomes positive definite, so the pseudo-inverse is no longer needed:

\[ \Bigl(N I + \gamma \textstyle\sum_n P^{\mathrm{geo}}_n\Bigr) \Delta b = \textstyle\sum_n \bigl(I + \gamma P^{\mathrm{geo}}_n\bigr) A\phi_n. \]

If every \(M_n = I\), this reduces to \(\Delta b = A\, \mathbb{E}[\phi]\); if \(M_n\) varies, the unweighted global-mean formula is not generally optimal. Even an optimal global bias cannot absorb input-dependent residuals. Sec. 3.4 builds the bias into the pruning pass itself.

**Lexicographic limit.** As \(\gamma \to \infty\), the problem becomes: minimize ambient error subject to zero calibration geometric error. With \(R\) now the dimension of the span of all geometric directions, this spends \(R\, d_{\mathrm{in}}\) constraints out of \(\nu\, d\, d_{\mathrm{in}}\) surviving parameters, a fraction \(R/(\nu d)\) (3% for \(R = 64\), \(d = 4096\), \(\nu = 1/2\)). Exact preservation on calibration data is therefore cheap; the open questions are held-out generalization and whether downstream loss cares (H0).

**Consequence for H1.** With presence included, each active concept contributes \(r_i + 1\) protected directions locally, against \(b_i\) for its global span, so local charts reduce the protected rank only when \(b_i > r_i + 1\). A planar circle (\(b = 2\), \(r = 1\)) gives no reduction. ID 86 gives calibration points. DINOv3 block codes have mean stable rank near 3 (Fig. 8), which leaves little room unless \(r_i = 1\). The InceptionV1 curve block puts 89% of the variance of a one-dimensional orientation manifold into its first three Fourier planes (59/18/12%). Choose H1 test concepts from that favorable end, and use planar circles as a negative control.

**Calibration and chart assignment.** On generic calibration text most tokens have no active concept, so \(P^{\mathrm{geo}}_n = 0\) for most \(n\) and the geometric term is diluted. Mix concept data into calibration and compute \(\bar r^{\mathrm{geo}}\) over the mixed set. With a dense unlabeled atlas (bsf-def1 Step 1B), hard chart assignments make \(P^{\mathrm{geo}}_n\) noisy; use soft assignments, \(M_n = I + \gamma \sum_c w_{nc} P^{\mathrm{geo}}_c\), and report sensitivity to them.

### 3.3 Input-side (pushforward) objective

Instead of measuring errors along the output concept directions, measure the layer's errors on the concept variation entering it. Take paired inputs \((x_k, x_k')\) that differ along a concept knob (bsf-def1 Step 1A), or by presence versus absence of the concept, and let \(\delta\phi_k = \phi(x_k) - \phi(x_k')\) be the difference at the layer input. For a linear layer, \(W \delta\phi_k\) is exactly this layer's contribution to the output difference, so \(A\, \delta\phi_k\) is the pruning error on that pushforward:

\[ \mathcal{L}(A) = \operatorname{tr}(A H A^\top) + \gamma_{\mathrm{in}} \frac{1}{K} \sum_{k=1}^{K} \lVert A\, \delta\phi_k \rVert_2^2 = \operatorname{tr}\bigl(A (H + \gamma_{\mathrm{in}} H_\Delta) A^\top\bigr), \qquad H_\Delta = \frac{1}{K} \sum_{k=1}^{K} \delta\phi_k\, \delta\phi_k^\top. \]

- Rows stay decoupled with one shared Hessian \(H' = H + \gamma_{\mathrm{in}} H_\Delta\), so SparseGPT applies exactly (Sec. 4.4).
- It never uses coordinate leverages, and the ambient term stays in, so it avoids the problems of Sec. 3.1 and 4.3.
- It changes N:M masks, because it changes the per-column curvature and the compensation (Sec. 4.2).
- Knob pairs protect how the layer transports movement along the manifold; presence pairs protect concept strength.
- \(H_\Delta\) is a difference covariance, so it removes the context shared within each pair; upweighting concept tokens in \(H\) would mostly upweight that shared context.
- It needs only paired inputs, not tangent estimates.

It does not see errors from non-concept inputs that land along concept directions (interference); only the output-side term sees those. The two combine directly: use \(H'\) as the shared Hessian in the rescaling trick of Sec. 4.5. The closest in-corpus precedent is ID 210 (Debias-SparseGPT), which adds a second-order term over contrasting inputs to SparseGPT, possibly with the opposite goal of suppressing the difference; its PDF is not parsed yet.

### 3.4 Bias handling inside the pruning pass

If a bias correction is planned, mask selection and compensation should already account for it. Either use the centered covariance in place of \(H_0\), or append a constant-one input column that is excluded from mask selection (and from N:M groups) and processed last, so that every compensation step can update it. For \(M_n = I\) the two are equivalent, since the Schur complement of the augmented second moment is the centered covariance; for \(M_n \ne I\), the ones column handles the weighted bias automatically. The position matters because SparseGPT only updates columns not yet processed. If the same pass also quantizes, keep this column unquantized: ID 223 documents error accumulating in the late columns of GPTQ-order solvers, which here is exactly what the bias column should absorb. Mixing uncentered SparseGPT with post-hoc bias folding is internally inconsistent.

---

## 4. SparseGPT and Wanda: mechanics and variants

### 4.1 Ambient reconstruction and SparseGPT's score

For calibration activations collected as columns of \(X\), the ambient loss is \(\mathcal{L}_{\mathrm{amb}}(A) = \frac{1}{N} \lVert AX \rVert_F^2\). For one row with curvature \(H\), pruning coordinate \(q\) and compensating with the others costs \(W_{pq}^2 / [H^{-1}]_{qq}\), with update \(\Delta W_{pj} = -(W_{pq} / [H^{-1}]_{qq})\, [H^{-1}]_{qj}\) for \(j \ne q\). Factors such as \(1/2\) and \(1/N\) do not affect ranking.

SparseGPT (ID 88, Algorithm 1) works with the upper Cholesky factor \(C\) of \(H^{-1}\). By the GPTQ identity,

\[ C_{qq}^2 = \bigl[(H_{F_q F_q})^{-1}\bigr]_{qq}, \qquad F_q = \{q, q+1, \dots, d_{\mathrm{in}}\}, \]

so its score \(W_{pq}^2 / C_{qq}^2\) is exactly the OBS cost of pruning \(W_{pq}\) when compensation is restricted to the columns not yet processed, and the update \(W_{pj} \leftarrow W_{pj} - (W_{pq} / C_{qq})\, C_{qj}\) for \(j \ge q\) is the matching restricted update. Practical consequences:

- Scores are evaluated on current weights, which already carry compensation from earlier columns. Apply any reweighting at selection time inside the column loop.
- Within one column the criterion ranks by magnitude, because the Hessian diagonal is shared across rows (ID 88, Sec. 3.2).
- Dampening is \(\lambda\) times the mean diagonal of \(H\); recompute it whenever \(H\) is modified.

### 4.2 Wanda and mask comparison groups

Wanda (outside the corpus) ranks by \(\lvert W_{pq} \rvert\, \lVert X_{q,:} \rVert_2\) within each output row. Its square, \(N\, W_{pq}^2 (H_0)_{qq}\), is exactly the uncompensated single-entry ambient cost; it is also OBS with \(H\) replaced by its diagonal.

What a multiplicative score factor \(f_{pq}\) can change depends on the comparison group (ID 88, Sec. 3.2 and 3.3):

| Factor type | Wanda (per row) | SparseGPT N:M (per row, groups of \(m\)) | SparseGPT unstructured (all rows, blocks of 128 columns) |
|---|---|---|---|
| Row-only, e.g. a static leverage \(P_{pp}\) | cancels | cancels | moves sparsity between rows |
| Varies within a row, e.g. input-conditioned \(G_{pq}\) | changes the mask | changes the mask | changes the mask |

For input-conditioned scores, only the within-row variation \(G_{pq} - \bar G_p\) survives under N:M or Wanda; it comes from covariation between feature \(q\)'s activity and which concept directions are active when \(q\) fires. With a static output metric, the row-diagonal treatment leaves 2:4 masks and compensation exactly as in vanilla SparseGPT; only cross-row compensation (Sec. 4.5) can change that result. The input-side objective (Sec. 3.3) acts under every pattern.

### 4.3 Entrywise output-side scores and basis dependence

For a direct-write single-entry knockout, \(\delta h_n = -W_{pq}\, \phi_{q,n}\, e_p\), and its exact uncompensated geometric error on calibration data is

\[ I_{pq}^{\mathrm{geo,diag}} = \frac{W_{pq}^2}{N} \sum_{n=1}^N (P^{\mathrm{geo}}_n)_{pp}\, \phi_{q,n}^2 = W_{pq}^2\, G_{pq}\, \frac{1}{N} \sum_{n=1}^N \phi_{q,n}^2, \qquad G_{pq} = \frac{\sum_n (P^{\mathrm{geo}}_n)_{pp}\, \phi_{q,n}^2}{\sum_n \phi_{q,n}^2}. \]

This is the geometric analogue of squared Wanda and inherits its exactness; it is not a compensated OBS score. If a feature is inactive on calibration data (zero denominator), assign zero direct calibration impact and flag that it has no observed support.

Every output-side quantity for a direct-write entry thus reduces to the coordinate leverage \(\ell_{pn} = (P^{\mathrm{geo}}_n)_{pp}\), which depends on the basis:

- Leverages sum to the rank, so their mean is \(r^{\mathrm{geo}}_n / d\). For a subspace in general position relative to the axes, \(\ell_{pn} \sim \mathrm{Beta}(r/2, (d - r)/2)\), with relative spread about \(\sqrt{2/r}\): order one for \(r\) between 1 and 4, and for \(r = 1\) the density diverges at zero. Random tangents already produce large, meaningless variation in \(G\).
- Leverages are invariant to permutations and sign flips of coordinates, but not to rotations. Residual-stream rotations (QuaRot-style Hadamard, or SliceGPT's PCA basis from ID 230) change them, and incoherence rotations flatten them by design. An input-side rotation such as the fold in [bonsai2-ternary-conversion-spec.md](bonsai2-ternary-conversion-spec.md) flattens column structure but leaves a layer's output rows unchanged.
- BSF's synthetic benchmark (ID 86) embeds factors with random orthonormal maps, so its leverages follow the Beta null exactly. It is a ready-made negative control.
- Real residual streams are partly privileged (outlier dimensions, RMSNorm gains). If real tangents concentrate leverage on outlier dimensions that SparseGPT already protects through \(H\), gains should not be attributed to geometry.

Test leverage concentration against the Beta null before relying on entrywise output-side scores (D1 in Sec. 6). If the test fails, use the input-side objective, or quantities that live in a privileged basis, as follows.

**Neuron write alignment.** For \(W_{\mathrm{down}}\), column \(q\) is neuron \(q\)'s write vector, and its alignment with the active concept directions is

\[ \chi_{qn} = \lVert P^{\mathrm{geo}}_n w^{\mathrm{down}}_q \rVert_2^2 \,/\, \lVert w^{\mathrm{down}}_q \rVert_2^2, \]

with null \(r^{\mathrm{geo}}_n / d\). It uses the whole write vector, so it is rotation-invariant in the residual space, while the neuron index is privileged by the elementwise nonlinearity. It is the natural score for structured (whole-neuron) pruning.

**Upstream layers.** Take a SwiGLU block with \(x = \mathrm{RMSNorm}(h_{\mathrm{in}})\), \(z = W_{\mathrm{gate}} x\), \(u = W_{\mathrm{up}} x\), and \(h = h_{\mathrm{in}} + W_{\mathrm{down}}(\mathrm{silu}(z) \odot u)\). Removing \(W^{\mathrm{up}}_{qk}\) changes the block output by exactly \(-W^{\mathrm{up}}_{qk}\, \mathrm{silu}(z_q)\, x_k\, w^{\mathrm{down}}_q\), a dense direction whose geometric energy at sample \(n\) is

\[ (W^{\mathrm{up}}_{qk})^2\, \mathrm{silu}(z_{qn})^2\, x_{kn}^2\, \lVert P^{\mathrm{geo}}_n w^{\mathrm{down}}_q \rVert_2^2. \]

For upstream entries the geometric factor is therefore the neuron's write alignment, not a coordinate leverage. Pulling the regularized metric back through \(W_{\mathrm{down}}\) and the gate, and dropping cross-neuron coupling, gives row \(q\) of \(W_{\mathrm{up}}\) its own sample weights:

\[ \omega_{qn} = \mathrm{silu}(z_{qn})^2 \bigl(\lVert w^{\mathrm{down}}_q \rVert_2^2 + \gamma\, \lVert P^{\mathrm{geo}}_n w^{\mathrm{down}}_q \rVert_2^2\bigr). \]

The rescaling trick of Sec. 4.5 handles these weights, and the \(\gamma = 0\) case is itself a non-trivial downstream-aware baseline. For \(W_{\mathrm{gate}}\), the same holds to first order with \(\mathrm{silu}'(z_q)\, u_q\) in place of \(\mathrm{silu}(z_q)\). This is how the isotropy trap of bsf-def1 Sec. 0 shows up for single entries: dense perturbations of a direct-write block are isotropic, and single entries trade that for basis dependence.

### 4.4 Variant A: input-side augmentation (recommended first)

The objective of Sec. 3.3 needs no change to SparseGPT, only to its Hessian:

```python
# H: accumulated input Hessian for this layer, before dampening
# D: (K, d_in) paired input differences phi(x_k) - phi(x_k') at this layer, from the current partially pruned model
H_delta = D.T @ D / D.shape[0]
gamma_in = alpha_in * torch.trace(H) / torch.trace(H_delta)
H = H + gamma_in * H_delta  # then the usual dampening, Cholesky, mask selection, and updates
```

Here \(\alpha_{\mathrm{in}}\) is the share of total curvature contributed by the difference term. Pairs should be token-aligned, or restricted to the concept positions. If pairs come from several knobs or step sizes, normalize per knob so that one does not dominate. Constants cancel in differences, so the bias column of Sec. 3.4 is unaffected.

### 4.5 Variant B: the rescaling trick (replaces Option 1)

Replacing \(P^{\mathrm{geo}}_n\) by its diagonal gives each output row its own Hessian, which breaks SparseGPT's shared inverse:

\[ H^{(p)} = H' + \gamma\, \frac{1}{N} \sum_n (P^{\mathrm{geo}}_n)_{pp}\, \phi_n \phi_n^\top, \]

where \(H'\) is the shared Hessian: \(H_0\), or \(H_0 + \gamma_{\mathrm{in}} H_\Delta\) when combined with Variant A. Approximate each row Hessian by a congruence of \(H'\) that matches its diagonal:

\[ H^{(p)} \approx \Lambda_p H' \Lambda_p, \qquad \Lambda_p = \operatorname{diag}(\lambda_{p1}, \dots, \lambda_{p d_{\mathrm{in}}}), \qquad \lambda_{pq}^2 = H^{(p)}_{qq} / H'_{qq}. \]

With \(H' = H_0\), \(\lambda_{pq}^2 = 1 + \gamma G_{pq}\); with Variant A, replace \(G_{pq}\) by \(G_{pq} (H_0)_{qq} / H'_{qq}\). Compute \(\lambda\) from undamped moments, and set \(\lambda_{pq} = 1\) for features with no calibration support. Substituting \(\tilde a_p = \Lambda_p a_p\) turns row \(p\)'s loss into \(\tilde a_p^\top H' \tilde a_p\). A positive diagonal rescaling preserves sparsity patterns, so the constrained problem is vanilla SparseGPT on \(W \odot \Lambda\):

```python
scale = torch.sqrt(1 + gamma * G)        # lambda; gamma = alpha * d / r_bar_geo; G as in Sec. 4.3, shape (d_out, d_in)
W_tilde_hat = sparsegpt(W * scale, H)    # unmodified: same H, Cholesky, mask rule, updates
W_hat = W_tilde_hat / scale              # same support
```

- The mask score becomes \((1 + \gamma G_{pq})\, W_{pq}^2 / C_{qq}^2\), a bounded reweighting without the degeneracy of Sec. 3.1.
- The compensation is rescaled to match, \(\Delta W_{pj} = -W_{pq}\, (\lambda_{pq} / \lambda_{pj})\, C_{qj} / C_{qq}\). The original score-only Option 1 lacked this.
- \(G\) costs one pass over the calibration set: a single \(d \times N\) by \(N \times d_{\mathrm{in}}\) product of leverages and squared activations, or one product per chart if \(P^{\mathrm{geo}}_n\) is constant within charts.
- The approximation is close when the per-sample weights are uncorrelated with the feature products \(\phi_{qn}\phi_{jn}\). Measure its error on a few rows (D5).
- Do not multiply scores by raw \(G\) as the original Option 1 did. If a score-only ablation is wanted, use \(1 + \gamma G\) without rescaled compensation and label it heuristic.

**Static sanity case.** If \(P^{\mathrm{geo}}_n = P\) is fixed, then \(H^{(p)} = (1 + \gamma P_{pp}) H_0\) exactly, so the trick is exact: a row weight multiplies the scores and the compensation is unchanged, which makes it a no-op under N:M. Exact OBS over the full Kronecker curvature \(H \otimes M\), with \(M = I + \gamma P\) and compensation allowed across rows, gives

\[ s_{pq}^{\mathrm{Kron}} = \frac{W_{pq}^2}{[H^{-1}]_{qq}\, [M^{-1}]_{pp}}, \qquad [M^{-1}]_{pp} = 1 - \frac{\gamma}{1 + \gamma} P_{pp}. \]

Its update spreads the error across rows along \(M^{-1} e_p = e_p - \frac{\gamma}{1+\gamma} P e_p\), i.e. along the concept directions. The row-diagonal score agrees with it at \(P_{pp} \in \{0, 1\}\) and over-protects rows in between, because it cannot use other rows to absorb geometric error. For low-rank \(P = QQ^\top\), the Kronecker quantities need only \(R \times R\) solves (Woodbury), so a column-sequential SparseGPT variant with cross-row compensation along \(P e_p\) looks tractable. It is the only route by which a static output metric can change a 2:4 result; prototype it on a small layer against the exact refit of Sec. 4.6.

### 4.6 Variant C: per-chart OBS and the exact refit (restates Option 2)

For a fixed chart \(c\) with basis \(U_c\), calibration subset \(I_c\), and conditional covariance \(H_c = \frac{1}{\lvert I_c \rvert} \sum_{n \in I_c} \phi_n \phi_n^\top + \lambda I\), pruning \(W_{pq}\) with compensation restricted to row \(p\) costs \(\lVert U_c^\top e_p \rVert_2^2\, W_{pq}^2 / [H_c^{-1}]_{qq}\) under the projected chart loss. This is exact for that fixed-chart quadratic and compensation restriction only. Summing chart losses with chart frequencies \(\pi_c\) and adding the ambient term gives row Hessians \(H^{(p)} = H + \gamma \sum_c \pi_c\, \lVert U_c^\top e_p \rVert_2^2\, H_c\), so Variant B's trick applies directly.

For the full coupled objective with a fixed mask, the gradient is matrix-free. If \(M_n = M_c\) is constant within chart \(c\),

\[ \nabla_A \mathcal{L} \propto \sum_c \pi_c\, M_c\, A\, H_c. \]

Projected conjugate gradient on the surviving entries gives the exact refit for a given mask. On a small layer, it measures what the congruence approximation and the missing cross-row compensation cost, and it is the reference against which any "exact" claim for Variants B and C should be checked.

### 4.7 Variant D: fixed global union (replaces Option 3)

If one fixed orthonormal frame \(Q\) (projector \(\Pi\), rank \(R\)) spans all concept directions to be preserved, prune with the ambient objective to get a sparse \(W_{\mathrm{sp}}\), then restore the concept block exactly with a rank-\(R\) dense correction:

\[ \hat W = W_{\mathrm{sp}} + \Pi\, (W - W_{\mathrm{sp}}). \]

The residual becomes \((I - \Pi)(W - W_{\mathrm{sp}})\): zero geometric error, and an ambient error no larger than that of \(W_{\mathrm{sp}}\). Storage is \(W_{\mathrm{sp}}\) plus \(Q\) (\(d \times R\)) and \(Q^\top (W - W_{\mathrm{sp}})\) (\(R \times d_{\mathrm{in}}\)); compare with other methods at matched total storage. Sparse-plus-low-rank compression (outside the corpus) provides related baselines.

An alternative is to rotate the residual stream into concept and complementary coordinates using SliceGPT's computational invariance (ID 230). In that basis the static regularized metric is diagonal, so SparseGPT with row weights is exact. The rotation is model-wide, however, and conflicts with the incoherence rotations used for quantization.

Both alternatives share the original Option 3's limitation: a fixed union is not equivalent to the input-conditioned objective when different examples activate different concept directions, since it also protects directions that are inactive on each example.

### 4.8 One implementation, a ladder of output metrics, and the premise test

The trick of Sec. 4.5 is generic. Any per-sample, per-row weight \(\omega_{pn}\) defines \(H^{(p)} = \frac{1}{N} \sum_n \omega_{pn} \phi_n \phi_n^\top\) and \(\lambda_{pq}^2 = \sum_n \omega_{pn} \phi_{qn}^2 / \sum_n \phi_{qn}^2\), so every rung below runs through identical code:

| Rung | Per-sample row weight | Role |
|---|---|---|
| Ambient | \(1\) | vanilla SparseGPT |
| Static concept union | \(1 + \gamma P_{pp}\) for one fixed union | sanity case; a no-op under N:M |
| Global BSF blocks | \(1 + \gamma (P^{V}_n)_{pp}\), projector onto the active global spans \(V_g\) | the H1 control at the pruning level |
| Local tangents | \(1 + \gamma (P_n)_{pp}\) | the original proposal, regularized |
| Tangents and presence | \(1 + \gamma (P^{\mathrm{geo}}_n)_{pp}\) | the default in this revision |
| Fisher | \(1 + \alpha\, g_{np}^2 / \overline{g^2}\), with \(\overline{g^2}\) the mean of \(g_{np}^2\) over rows and samples | downstream-aware baseline |

Using \(\alpha\) directly in the Fisher rung matches the normalization of \(\gamma\): for isotropic errors, both extra terms weigh \(\alpha\) times the ambient loss. The global-BSF rung is where bsf-def1's argument for local charts over \(V_g\) becomes testable for pruning.

**Fisher and Taylor scores.** For a direct-write entry, \(\partial \ell_n / \partial W_{pq} = g_{np}\, \phi_{qn}\), so the squared first-order Taylor (empirical Fisher) score is \(W_{pq}^2 \sum_n g_{np}^2 \phi_{qn}^2\). The geometric score is the same expression with \(g_{np}^2\) replaced by the leverage \((P^{\mathrm{geo}}_n)_{pp}\): a geometric stand-in for downstream sensitivity. It can beat the Fisher rung only if gradients are too noisy at the calibration size, or if the goal is concept preservation rather than average loss. Prefer labels sampled from the model (model Fisher) to ground-truth labels. In the corpus, ID 254 selects sparse masks with the diagonal model Fisher, and ID 220 builds compression whitening from on-policy distillation gradients.

**Premise test (H0).** Over tokens where the concept is active, compute

\[ \kappa = \frac{\mathbb{E}_n \lVert P^{\mathrm{geo}}_n g_n \rVert_2^2 \,/\, \mathbb{E}_n \lVert g_n \rVert_2^2}{\bar r^{\mathrm{geo}} / d}. \]

If \(\kappa \approx 1\), geometric emphasis is unlikely to improve average downstream loss over ambient SparseGPT. If \(\kappa \gg 1\), H0 holds. The corpus gives reasons to expect either outcome: ID 107 reports that Qwen2.5-7B compares numbers through linear structure despite curved number geometry, and ID 142 argues that output (Fisher-Rao) geometry is more consistent across models than coordinate-dependent activation geometry.

---

## 5. Relation to other compression methods

| Method | Geometry or objective used | Open question for this proposal |
|---|---|---|
| Magnitude, Wanda, SparseGPT (ID 88) | Weight magnitude, activation statistics, or ambient layer reconstruction | Do geometric terms improve held-out concept or task metrics at matched sparsity, beyond what ambient error already bounds? |
| Fisher and Taylor scores (ID 254); gradient-informed whitening (ID 220) | Downstream loss sensitivity | Does geometry add anything beyond the Fisher rung, or only regularize it at small calibration sizes? |
| Debias-SparseGPT (ID 210) | Second-order term over contrasting input pairs | Same machinery as the input-side objective; preserving versus suppressing the difference |
| SliceGPT (ID 230) | Global activation subspace, rotation invariance, dimension deletion | Do local charts help when curvature and changing active support are measured? |
| Global BSF truncation (ID 86) | Fixed factor spans \(V_g = \operatorname{span}(\mathcal{M}_g)\) | Does local tangent rank give a useful budget advantage over the span, which requires \(b_i > r_i + 1\)? |
| DBW (ID 228) | Weight spectral components and bit-width allocation | Do input-conditioned activation directions add value beyond weight-spectrum selection? |
| Sparse plus low-rank (outside corpus) | Sparse matrix with a dense low-rank correction | Baseline for Variant D at matched storage |
| Geometry-aware variants (this note) | Regularized output-side metrics; input-side difference curvature | Require reliable tangent estimates and validation beyond the calibration set |

Connection to QTEA (ID 223) and Bonsai (ID 85) remains speculative. A geometry-derived residual allocation score could be tested after ternary quantization, but it should not be described as allocating bits only to concept-distorting weights until measured. Before combining with those pipelines, check which rotations they apply: residual-stream incoherence rotations flatten the leverages that entrywise output-side scores rely on (Sec. 4.3), while the input-side objective does not rely on coordinate structure.

---

## 6. Experiments that can support or disconfirm the hypotheses

Use separate data for tangent fitting, pruning calibration, and final evaluation wherever feasible. Report tangent stability across held-out examples and chart boundaries. Compare at identical sparsity and pattern, with identical compensation settings, and state the pattern (2:4 or unstructured), because it determines which variants can act at all.

Order of work: Stage 0 (using first-pass tangents from bsf-def1 Step 1), then Variant A, which needs only paired inputs, then Stages 1 and 2, then the rest of Stage 3, then Stage 4.

### Stage 0: diagnostics before any pruning

Run these on one layer and one concept family. Each diagnostic has a null and a decision rule.

| ID | Measure | Null | Decision |
|---|---|---|---|
| D1 | Coordinate leverages \((P^{\mathrm{geo}}_n)_{pp}\) of the real concept directions | \(\mathrm{Beta}(r/2, (d-r)/2)\); BSF synthetic data | Indistinguishable from the null: skip entrywise output-side scores in this basis; use Variant A, write alignment, or a rotated basis |
| D2 | Mask flip rate between vanilla SparseGPT and Variant B at \(\alpha = 1\), for the pattern to be deployed | random tangents of matched rank | Near the null rate: output-side metrics cannot matter for that pattern |
| D3 | Gradient enrichment \(\kappa\) (H0), for tangents and presence directions separately | \(\kappa = 1\) | \(\kappa \approx 1\): judge geometric methods on concept metrics only |
| D4 | Tangent stability (principal angles across disjoint splits) and the spectrum of \(\frac{1}{N}\sum_n P^{\mathrm{geo}}_n\), which gives \(R\) | random subspaces of matched rank | Unstable tangents: stop, since the H1 prerequisite fails; \(R\) sets the budget \(R/(\nu d)\) |
| D5 | Congruence error of \(\Lambda_p H' \Lambda_p\) against \(H^{(p)}\) on sampled rows, and rank correlation of the resulting OBS costs | none | Large error: use the exact refit, or Hessians shared by groups of rows |
| D6 | Radial share: fraction of knob and presence variance normal to the local tangent | none | Large share: presence directions are required |

### Stage 1: validate local tangents and presence directions (H1 prerequisite)

1. Use a controlled concept sweep or another dataset with known variation; estimate local bases and presence directions on one split and evaluate on held-out points.
2. Compare local tangent rank and stability with the global span. Measure rather than assume \(r_i < b_i\), with a stated rank threshold.
3. At matched budget, including chart storage and per-input assignment, compare local-chart and global-span reconstruction of held-out concept-attributable variation in a metric-neutral way (ambient error or description length). Report tangent-estimation cost.
4. Prefer concepts with \(b_i > r_i + 1\), such as multi-harmonic curves, and include a planar circle as a negative control.
5. Disconfirm H1 if local bases are unstable or provide no held-out gain at the same budget.

### Stage 2: validate score predictions before pruning (H2, H5)

1. On a small layer, enumerate single-entry and selected structured knockouts, with and without compensation. Record actual tangent, presence, and ambient errors on held-out inputs, together with downstream changes (loss, KL to the dense model, concept probes).
2. Compare geometric scores against the measured effects with rank correlation and calibration plots, including magnitude, Wanda, ambient OBS, and Fisher scores. Include upstream layers, where the scores are first-order.
3. Measure editability with small signed perturbations of upstream weights or neuron-level directions, and test whether it predicts the effect of those perturbations, not whether it implies current knockout importance.
4. For participation and regime claims, perform concept-selective and broad ablations and compare the observed impacts with the predicted pattern.
5. Disconfirm H2 or H5 if the geometric scores do not outperform the baselines or do not predict the stated selective or broad effects.

### Stage 3: compensated pruning (H3)

Run in this order, at matched sparsity, with the same calibration examples, patterns, and compensation budget:

1. Variant A (input-side augmentation), sweeping \(\alpha_{\mathrm{in}}\), for both 2:4 and unstructured patterns. This is the cheapest experiment and the only one guaranteed to act on N:M masks.
2. The output-metric ladder of Sec. 4.8 through Variant B, mainly for unstructured sparsity, including the global-BSF rung (H1 control) and the Fisher rung (downstream baseline).
3. Exactness checks on a small layer: the projected-CG refit (Sec. 4.6) and cross-row compensation for the static case (Sec. 4.5).
4. Upstream and structured pruning: \(W_{\mathrm{up}}\) and \(W_{\mathrm{gate}}\) with write-alignment weights, and whole-neuron pruning ranked by \(\chi_{qn}\) (Sec. 4.3).
5. Variant D at matched total storage.

Report:

- held-out tangent, presence, and per-concept projected errors;
- ambient layer reconstruction error;
- downstream task and concept-probe metrics;
- the \(\alpha\) sweep as a front, not a single point;
- runtime, memory, and tangent-estimation overhead.

For discordant groups, define disjoint sets by explicit score quantiles, equalize the number of removed parameters, and report both pre- and post-compensation results. A small ambient error also bounds the same-example geometric error from above, so geometric methods can only win where the ambient error is non-negligible and its direction matters downstream (H0). Any task degradation despite small local errors is a separate downstream-sensitivity result.

### Stage 4: bias folding and normal-residual safety (H4)

Compare no bias adjustment, the in-pass bias of Sec. 3.4 (ones column or centered covariance), and post-hoc folding with the global mean. Separate residual inside the concept spans (presence) from residual normal to all concept spans. Evaluate held-out projected residuals, chart assignments where available, and downstream behavior, including cases with input-dependent residuals. Disconfirm safety for any condition whose held-out metric exceeds its predeclared tolerance.

---

## 7. Speculative: transfer operators between factor tangents

For a block with Jacobian \(J_n = \partial h_{\mathrm{out}} / \partial h_{\mathrm{in}}\), define

\[ T_{ij,n} = U_{j,\mathrm{out}}^\top\, J_n\, U_{i,\mathrm{in}} \in \mathbb{R}^{r_j \times r_i}, \]

the block's action from concept \(i\)'s input tangent to concept \(j\)'s output tangent. The input-side term (Sec. 3.3) and the output-side term (Sec. 3.2) are its two marginals; the bilinear objective \(\sum_k \lVert Q_{\mathrm{out},k}^\top A\, t_k \rVert_2^2\), with input tangents \(t_k\), preserves \(T\) itself.

For the project's main objective, the chart atlas \(\{(m_{i,c}, U_{i,c})\}\) together with the small matrices \(T_{ij}\) is a candidate for the compact runnable object: the layer's action restricted to the concept atlas, stored as \(r_j \times r_i\) matrices instead of \(d \times d_{\mathrm{in}}\) weights. Pruning that preserves \(T\) on held-out inputs would be direct evidence that this object captures what the weights do on those concepts. This links to [neural-operator-steering-on-activations.md](neural-operator-steering-on-activations.md) and stays a hypothesis until the earlier stages pass.

---

## Reporting standard

Present positive, null, and negative outcomes as evidence about the stated hypotheses. A failed hypothesis narrows when geometry-aware pruning helps; it does not by itself invalidate the broader research question. Do not describe a score as "safe," a parameter as a concept owner, or a method as geometry-preserving unless the corresponding held-out experiment supports that claim. Report the Stage-0 outcomes alongside every pruning result, state the sparsity pattern, and give the \(\alpha\) sweep as a front.

---

## References

**In the corpus:** ID 86 (BSF: Def. 1, block stable rank in Fig. 8, InceptionV1 curve harmonics, synthetic benchmark); ID 51 (Manifold Steering); ID 88 (SparseGPT: Algorithm 1, mask rules in Sec. 3.2 and 3.3); ID 230 (SliceGPT: rotation invariance); ID 228 (DBW); ID 85 (Bonsai 2); ID 210 (Debias-SparseGPT; abstract only, PDF not parsed); ID 254 (diagonal Fisher and sparse masks); ID 220 (GA-SVD); ID 142 (Fisher-Rao geometry of LLMs; abstract only, PDF not parsed); ID 107 (number comparison in Qwen2.5-7B); ID 223 (QTEA: late-column error accumulation).

**Outside the corpus (candidates to ingest):** Frantar et al., GPTQ (2022), for the Cholesky identity; Frantar et al., Optimal Brain Compression (2022); Hassibi and Stork, Optimal Brain Surgeon (1993); Sun et al., Wanda (2023); Ashkboos et al., QuaRot (2024); Molchanov et al. (2019), Taylor and Fisher importance; Engels et al., "Not all language model features are linear" (2024), circular features as the unfavorable case for H1; Kantamneni and Tegmark (2025), multi-frequency number helices as a favorable case; sparse-plus-low-rank pruning such as OATS (2024).
