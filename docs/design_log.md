# Design log

Decisions, paper/code deviations and interpretations, newest last. Each entry is dated.

## 2026-10-10: DyG-MF source audit (Week 1, step 1)

**Source:** https://github.com/Clearloveyuan/DyG-MF at commit `cb25db878b9435b836ae13fda2527af712fabbee` (15 Oct 2024), cloned into `external/DyG-MF` (gitignored).
**Paper:** Li et al., *Revisiting Dynamic Graph Clustering via Matrix Factorization*, WWW 2025, DOI 10.1145/3696410.3714646.

### Repository contents
- `Algorithm/esP.m` (3,642 lines): first snapshot. The node set minus landmarks is split into 50 hard-coded blocks, with each block's code written out by hand.
- `Algorithm/dyesP.m` (112 lines): later snapshots, including the selective update.
- `Algorithm/select1.m`: landmark selection, combining a KMM multiple-means initialisation with a degree-ordered greedy pick of mutually non-adjacent nodes.
- Helpers: `KMM.m`, `CSBG.m`, `ConstructA_NP.m`, `svd2uv.m`, `eig1.m`, `EProjSimplex_new.m`, `gen_nn_distanceA.m`, `meanInd.m`, `sqdist.m`, `struG2la.m`, `ClusteringMeasure.m` (ACC and NMI via Hungarian matching) and `NMI.m`.
- `Initialization/Initialize.p`, `Initialize2.p`, `Initialize3.p`: **MATLAB P-code**, which is encrypted and cannot be read.
- `Dirth-Death/`: one generated benchmark (birth/death), 10 snapshots, about 100k nodes, about 1.9M edge lines per snapshot, 499 communities of 200–400 nodes, plus a Linux x86-64 generator binary (`bench_birthdeath`).
- `Datasets/` holds only a placeholder file. There is no driver script, no parameter file and no PMI routine.

### Findings
| # | Finding | Evidence | Consequence |
|---|---|---|---|
| F1 | All three initialisation routines are P-code. | `file Initialization/*.p` → data; called at `esP.m:450`, `esP.m:489…` and `dyesP.m:73` | GNU Octave cannot execute P-code. MATLAB can run them only if the P-code version is compatible (repo says MATLAB 2023). We cannot read them to port them. |
| F2 | **The initialisers receive ground-truth labels.** | `Initialize(landmark, real1, h1, KK, label1)`, `Initialize2(..., real1)`, `Initialize3(landmark, real2, KK)` | We cannot inspect how the labels are used. The released pipeline therefore cannot be shown to be label-free, which must be stated in E1. |
| F3 | **Selection uses ground-truth changed nodes.** | `dyesP.m:12–19`: `b = find(realde~=0)`; truly changed nodes ranked below position 30 are appended to `ldmark` | The released selective update is not purely score-based. A faithful port must expose this, and our variants must not inherit it. |
| F4 | The change score uses the next snapshot A3. | `dyesP.m:4`: `CA = (0.64*A2 + 0.36*(A1+A3))./3`; `delta(i) = ‖A2(i,:) − CA(i,:)‖₂` | This confirms the lookahead that RQ1 removes. The score is computed on raw adjacency A, not the PMI matrix. The weights (0.64, 0.36, /3) must be checked against paper §4.3. |
| F5 | `PMI` is called but not defined. | `dyesP.m:3`: `M2 = PMI(A2,2)`; there is no PMI definition in the repo | `dyesP.m` errors as released. The PMI window and normalisation must come from the paper. |
| F6 | `n3` is undefined in `dyesP.m`. | `dyesP.m:106`: `Whole(ldmark,:) = WW1(n3+1:n3+KK,:)` | `dyesP.m` errors as released. In `esP.m`, `n3` is an output of `Initialize`, which is P-code. |
| F7 | `dyesP` never assigns its outputs `n2` and `aa1`. | `dyesP.m:109–110`: `kmeans` and `NMI` are commented out | The final clustering step and its settings (k, replicates) are unspecified for later snapshots. |
| F8 | `esP` drops the remainder nodes. | `hn = hnn − rem(hnn,50)`; blocks cover rows 1..hn only | Up to 49 non-landmark nodes get zero rows in `Whole`, and the row normalisation `Whole./sum(Whole,2)` then gives NaN. |
| F9 | The dynamic budget is a count, not a fraction. | `dyesP.m:16`: `ldmark = [bb(1:KK,:); b(add)]` | The paper's 16% setting must be mapped to KK. The actual budget is KK plus the ground-truth additions (F3). |
| F10 | `select1` adds the missing paths `KMM/` and `KMM/funs`. | `select1.m:10–11` | This is harmless because the KMM files are in `Algorithm/`, but it shows the repo is not a runnable snapshot. |

### Verdict on reproduction feasibility
- **The released code cannot be run as an end-to-end reference** in Octave (F1) or in MATLAB without fixes (F5, F6, F7). It is also not label-free (F2, F3).
- I did not install Octave, because it cannot execute the P-code initialisers.
- **Implication for Week 2:** E1 cannot be a "public-code reproduction". Following report §6, it becomes a **paper-based reimplementation**, using the public `.m` code as a structural guide, with these rules:
  - Every gap (PMI, initialisation, final clustering, n3 bookkeeping) is filled with one transparent, label-free replacement, documented here and used by every variant.
  - The ground-truth paths (F2, F3) are left out of the faithful port. The E1 deviation record says so explicitly.
  - The only published numbers to compare against are the paper's reported NMI and NF1.
- **Open question:** do we have MATLAB access? MATLAB could show whether the P-code runs at all. Even so, F2, F3, F5 and F6 mean a MATLAB run would not be a clean label-free reference. MATLAB is therefore useful but not decisive.

### Benchmark note
The birth/death benchmark (about 100k nodes and 2M edges per snapshot) is too large for early parity and debugging. Week 2 will use our own small planted sequences for parity, and use the birth/death benchmark (possibly subsampled, documented here) only for the published-number comparison.

## 2026-10-10: Week 1 implementation decisions

- **NF1 definition.** Ported from CDLIB's `cdlib/evaluation/internal/NF1.py` (Rossetti, Pappalardo & Rinzivillo, 2016), checked against the source on 2026-10-10. Quirks kept on purpose:
  - Each per-match F1 is rounded to 2 decimals before averaging.
  - Ties produce multiple matches.
  - A node counts only the first time it is seen.
  - redundancy = predicted communities / matched reference communities.
  - With no matches, CDLIB raises an error; we return 0.0 instead.
  - **Still to confirm:** that the DyG-MF paper's NF1 is this definition, by checking its citation.
- **NMI.** scikit-learn with arithmetic-mean normalisation. DyG-MF's `NMI.m` may normalise differently (to check in Week 2 against `ClusteringMeasure.m`).
- **Ground truth separated from method input.** `GraphSnapshot` carries no labels; `EvalAnnotations` goes only to evaluation. `tests/test_online.py` scrambles evaluation labels and asserts predictions are unchanged.
- **Method interface.** A single `step(GraphSnapshot)` call; the first call initialises the method. A separate `init` was not needed for Louvain (YAGNI). DyG-MF's first-snapshot path (esP) can sit behind the same call in Week 2.
- **Tracker.**
  - Jaccard threshold 0.3 on a many-to-many overlap graph. Each connected component is classified by (#old, #new) communities. Many-to-many components become `complex`, reported separately from the five tracked classes.
  - Continuations keep their identity; merges, splits and births get fresh identities.
  - A two-way split of equal halves gives a Jaccard of 0.5 per link, so the 0.3 threshold leaves a margin.
- **Event matching.**
  - A match needs the same kind, a delay between 0 and the tolerance, and a Jaccard overlap of involved nodes (before ∪ after) of at least `min_overlap`.
  - Matching is one-to-one and greedy, by smallest delay, then largest overlap.
  - Precision, recall and F1 are None when undefined.
- **Determinism.** Louvain's seed for snapshot t is derived from (algo_seed, t), so output at t is independent of the rest of the sequence.
- **First smoke observation (e0_smoke, seeds 1/1).** Louvain kept a planted 28-node community whole for two snapshots after it was split (t=5 → detected at t=7). With tolerance 1 this counts as a missed split. This is a genuine detection delay of the baseline, not a tracker error, and it is visible as an NF1 dip at t=5–6.

## 2026-10-11: DyG-MF paper audit against the released code (Week 2, step 1)

**Source:** arXiv:2502.06117v1 (10 Feb 2025), the WWW '25 version, supplied by the user.

### Metric definitions (resolves the Week 1 open items)
- **NMI:** the paper cites Danon et al. (2005) [8]. The code's `NMI.m` computes −2·Σ n_ij log(n_ij N / n_i n_j) / (Σ n_i log(n_i/N) + Σ n_j log(n_j/N)) = 2I / (H₁ + H₂), which is **arithmetic normalisation**. This matches our scikit-learn setting.
  - `ClusteringMeasure.m` instead uses I / max(H₁, H₂) after Hungarian matching. It is not the one wired into `dyesP`.
  - `NMI.m` assumes the reference labels are 1..k (`rows = max(r_labels)`), so reference labels must be remapped before comparing.
- **NF1:** the paper cites Rossetti, RDyn (2017) [49], which is the same reference CDLIB's NF1 cites. This confirms our Week 1 port.

### Paper vs code vs our plan
| # | Component | Paper | Released code | Consequence |
|---|---|---|---|---|
| P1 | PMI | §3: m_ij = log(w_ij Σ_k d_k / (d_i d_j)) | `PMI(A2,2)` is called but not defined (F5); the second argument suggests a window of 2 | The paper's formula is −∞ wherever w_ij = 0, yet NMF needs a non-negative target. Following Qiu et al. [47] (NetMF), use shifted positive PMI: max(m_ij, 0). |
| P2 | Landmark centres, Eq. 7 | θ = Σ (1+λ)(m_t + m_{t−1}) / \|Θ\| | Not in code: `select1` uses KMM plus greedy non-adjacent nodes | **Eq. 7 is not the minimiser of Eq. 6.** Setting the gradient to zero gives θ = Σ (m_t + λ m_{t−1}) / ((1+λ)\|Θ\|). Use the correct minimiser and record the erratum. |
| P3 | Landmark count | \|U_t\| = 0.5 ("percentage of landmarks"), tuned in [0.48, 0.52] | `num` is passed in | Half the nodes as landmarks contradicts "select a few nodes" and makes O(\|U\|³) no better than O(n³). Follow 0.5 for E1 fidelity and note the contradiction. |
| P4 | Embedding size r and cluster read-out | Eq. 2: each row of C is on the simplex and its argmax is the cluster; §5 sets r = 1000 | The final `kmeans(Whole, 4, 'Replicates', 100)` is commented out (F7) | *Corrected 2026-10-11:* this is consistent. r is a fixed embedding width (Eq. 15 needs constant r so static rows of Φ can be kept), and k_t enters through landmark centres and the regulariser's eigenvector count. See D2. |
| P5 | Number of clusters ϱ_t | Elbow method (Fig. 7A) | Hard-coded 4 in the commented `kmeans` | The elbow is label-free but unstable to automate. An oracle k (the true count) uses ground truth. Decision needed (D3). |
| P6 | Dynamic-node score, Eq. 14 | Δε = ‖w_t − w̄_t‖² + ‖(w_t − θ_t) − (w̄_t − θ̄_t)‖², w̄ = (w_{t−1} + w_t + w_{t+1})/3; at t = 1 or t = τ, missing terms are ignored | `CA = (0.64·A2 + 0.36·(A1+A3))/3`; unsquared Euclidean distance; topology term only; ground-truth changed nodes injected (F3) | The paper confirms the lookahead (§4.4: "takes only G_{t−1}, G_t and G_{t+1} as input"). The code's weights differ from the paper and drop the second term. Use the paper's version. |
| P7 | Dynamic budget | μ = 0.16 of nodes | KK (a count) plus ground-truth additions (F9) | Use μ·n, rounded. |
| P8 | Landmark factorisation, Eqs. 8 and 15 | M⁰⁰ ≈ ΦΨ with Φ ≥ 0, Ψ ≥ 0, Φ1 = 1; static landmark blocks kept fixed (Φ_y = Φ_{y,t−1}, Ψ_y = Ψ_{y,t−1}) | In P-code / `select1` (BiGraph and A from KMM) | Use multiplicative NMF updates. The text swaps the x/y (static/dynamic) subscripts; read them as x = dynamic, y = static, matching "update Φ_x" in Algorithm 1. |
| P9 | Separated factorisation, Eqs. 10–11 | C^i = P^i Φ, H^i = Ψ Q^i; intra and inter terms | Hand-written loop over 50 blocks with multiplicative updates on H and B | Implement for general s with vectorised updates. The paper uses s = 50 for large graphs and s = 1 for small ones (§5.5). Eq. 11 indexes Q^i in both terms, likely a typo for Q^j; we use Q^j. |
| P10 | Bi-clustering regulariser, Eq. 13 | β·Tr(Fᵀ L_S F), L_S the **normalised** Laplacian of S = [[0, C], [Cᵀ, 0]], FᵀF = I; β = 20 | **Unnormalised** L = D − S built on H (not C); F = the first KK eigenvectors; distances added as `alpha*X` in the H update | Follow the paper (normalised Laplacian on C, weight β), using the KMM/CSBG alternating scheme [44]: F is the bottom eigenvectors, then a distance penalty in the factor update. |
| P11 | Temporal smoothing | Eq. 13 has α‖C_t − C_{t−1}‖²; Eq. 16 removes it once selective updating applies | Absent from `dyesP` | Consistent with Eq. 16; no smoothing term in the selective variant. |
| P12 | Repetitions | §5: five runs; README: ten runs | n/a | Minor inconsistency; report against the paper. |

### Findings that matter for evaluation (and for the thesis)
- **E1 (tuning on the test set):** §5 says Birth-Death-30K was the validation set for hyperparameter tuning, and Table 2 reports test results on Birth-30K. The published numbers therefore come from tuning on the evaluation dataset. Our protocol (report §5) separates the two.
- **E2 (real-data pseudo-labels):** Appendix C says the real-world "gold" labels come from running **DYNMOGA**, a competing method, on the graph aggregated over **all** timestamps. This confirms the proposal's pseudo-label concern, and these labels also use future information. Real-data NMI and NF1 are agreement with DYNMOGA's partition, not accuracy.
- **E3 (benchmark mismatch):** Appendix C describes Birth-Death with 40–60 communities and mixing parameter 0.2. The repo's `Dirth-Death/` files have 499 communities of 200–400 nodes, so they do not match. Treat them as "a Birth-Death instance", not the paper's exact data.
- **Hyperparameters (§5):** s = 50, r = 1000, |U| = 0.5, μ = 0.16, λ = 0.2, β = 20. The number of factorisation iterations (`ll` in the code) and α are not reported.

### Reimplementation spec (decisions confirmed by the user, 2026-10-11)
- **D1. PMI:** positive PMI, m_ij = max(log(w_ij · vol / (d_i d_j)), 0) on edges and 0 elsewhere (NetMF convention [47]).
- **D2. Embedding and read-out:**
  - Fixed width r = min(r_cfg, |U_t|) across the sequence; the paper uses r_cfg = 1000.
  - Labels are the argmax of each row of C, with C = Φ for landmarks and C = PΦ otherwise; rows lie on the simplex because P and Φ are row-normalised after each update.
  - k_t sets the number of landmark centres (Eq. 6) and the number of bottom eigenvectors F in the regulariser (Theorem 2).
- **D3. Choosing k_t:**
  - Main setting: an automated elbow, computing k-means inertia over the rows of M_t for k in [k_min, k_max] and picking the point of maximum distance from the chord on the normalised curve.
  - Diagnostic: **oracle k**, the reference community count at t, passed in by the runner as `k_schedule`. This is an explicit, labelled exception to the ground-truth separation and is never used for headline results.
- **D4. Landmarks:**
  - |U_t| = round(0.5·n), following §5.
  - Centres come from Eq. 6 using the **corrected** minimiser θ = Σ (m_t + λ_a m_{t−1}) / Σ (1 + λ_a), where λ_a = λ for nodes present at t−1 and 0 otherwise.
  - Each centre takes its nearest unassigned |U|/k nodes in turn.
- **D5. Optimisation:** multiplicative updates, since the paper gives only pseudocode, using Eq. 10's first-line form (C, H = ΨQ, Φ, Ψ).
  - Bi-clustering: in each subset, F holds the bottom k eigenvectors of the normalised Laplacian of [[0, C_i], [C_iᵀ, 0]]. With degrees frozen, the term becomes β·Σ C_al X_al, where X_al = ‖u_a/√d_a − v_l/√d_l‖², and it enters the P update's denominator as β·XΦᵀ.
  - F is refreshed every `f_every` iterations.
  - Iterations (`ll`) and α are not reported in the paper, so both are configurable.
- **D6. Initialisation (label-free replacement for the P-code `Initialize*`):**
  - Φ and Ψ start random; P starts from the row-normalised M_S0 plus noise; Q starts random.
  - The paper says embeddings start random (Fig. 8A).
- **D7. Selective update (Eqs. 14–16), original lookahead version:**
  - Δε follows Eq. 14 exactly: equal 1/3 weights; at t = 1 or t = τ the missing terms are dropped; θ is the nearest centre at each time. The paper subtracts centres in M-row space from rows in W-space, and this is implemented literally.
  - The top ⌈μn⌉ nodes are dynamic. New and reappearing nodes are always updated, count as mandatory work, and are reported separately.
  - Static nodes keep their C row. Landmark Φ rows and Ψ columns are fixed for static nodes that were landmarks at t−1.
  - Q (shared context) is re-estimated for all non-landmark nodes every step. This is shared work, not a membership update, and is reported as such.
- **D8. Lookahead adapter:**
  - The output for t is produced once t+1 arrives, and rows record `available_at`.
  - The final snapshot is processed without t+1, as the paper specifies.
  - Lookahead runs do not support checkpoint resume yet (E1 runs are small).
- **D9. Subsets:** s is configurable. With Eq. 11 included, the split objective equals the unsplit one in (P, Q), so s affects only the scheduling and the per-subset regulariser. Small graphs use s = 1, as in §5.5.

### 2026-10-11: read-out revised after a measured failure (supersedes D2's argmax)
- **Measurement** (planted graph, 5 communities × 40 nodes, oracle k = 5, scratch script): argmax of C gives NMI 1.00 at r = 5, 0.85 at r = 10 (10 clusters), 0.69 at r = 25 (24 clusters) and 0.55 at r = 100 (62 clusters). K-means with k on the same C gives 1.00 for r ≤ 25. With the paper's r = 1000 on a small graph, r = min(r, |U|) = |U|, so the landmark factorisation has full rank, each landmark gets its own column, and argmax fragments the clusters.
- **Decision (user, 2026-10-11):**
  - r is fixed at k_max (default 20), an upper bound on k_t, so static embeddings can be kept (Eq. 15).
  - Labels come from k-means with k_t on the rows of C (n_init = 10), matching the code's commented-out `kmeans(Whole, k, 'Replicates', 100)`.
  - Deviations from the paper: r ≠ 1000, and k-means replaces Eq. 2's argmax.
  - The k-means read-out runs over all nodes every snapshot. This is shared work that is timed (`time_readout`) but not counted as node updates; a static node's label can change even though its embedding is frozen.
- **Finding, β scale:** with our formulation, the penalty entries X average about 0.04, so β·X ≈ 0.8 at β = 20, against P-update denominators in the hundreds. Labels were identical at β = 20 and β = 0. The paper's β is not transferable across graph sizes or formulations, so the bi-clustering contribution must be checked by ablation (w/o BR), not assumed.

### 2026-10-11: PMI window revised after a measured failure (supersedes D1)
- **Measurement** (Birth-Death subsample, 6 communities, about 1,200 nodes, oracle k = 6, r = 20, first snapshot, graph seeds 1–3):

  | PMI | NMI at t0 | Empty landmark rows (of 600) | Density of M |
  |---|---|---|---|
  | T = 1 (the paper's printed formula) | 0.511 / 0.420 / 0.524 | 256 / 225 / 217 | 3.2% |
  | T = 2 | 1.000 / 1.000 / 1.000 | 0 / 0 / 0 | 16.6% |

  With T = 1, M is non-zero only on edges. The landmark block M⁰⁰ keeps only landmark–landmark edges, so many landmark rows are empty and Φ carries no structure (k-means on Φ: NMI 0.07). The paper's landmark scheme needs multi-hop co-occurrence.
- **Decision (user, 2026-10-11):** positive NetMF PMI with window T = 2 (`pmi_window`), max(log(vol/T · Σ_{r≤T} (D⁻¹A)^r D⁻¹), 0). This matches the code's `PMI(A2, 2)` and the paper's reference [47]. Deviation: the paper's printed formula is the T = 1 case, which fails with its own landmark scheme.
- **Diagnostic note:** with T = 2, rescaling Φ's rows after an unconstrained multiplicative step remains a heuristic. In isolation, a constrained (sum-to-one) multiplicative update did not beat it (scratch test, k-means on Φ: 0.83 vs 0.86), so it is kept.

## 2026-10-11: E1 pilot results (Birth-Death subsample)

**Setup:** configs `configs/e1_birthdeath_*.yaml`, graph and algorithm seeds 1–3 (each seed draws a different 6-community subsample of about 1,200 nodes), 10 snapshots. The summary is in `reports/e1_pilot_birthdeath.csv`; raw runs are in `results/` (not committed).

| Implementation | Mean NMI | Mean NF1 | Nodes updated (t ≥ 1) | Total time |
|---|---|---|---|---|
| Published (paper Table 2, Birth-30K) | 99.9 | 90.2 | 16% (nominal) | not comparable |
| MATLAB local run | — | — | — | not possible (P-code, F1–F7) |
| Python DyG-MF, oracle k | 0.999 ± 0.001 | 1.000 ± 0.000 | 32.1% ± 0.4 | 10.1 s ± 0.7 |
| Python DyG-MF, elbow k | 0.988 ± 0.017 | 0.992 ± 0.014 | 31.6% ± 1.1 | 19.6 s ± 0.8 |
| Python DyG-MF, μ = 1 (like "w/o SEU") | 0.999 ± 0.001 | 1.000 ± 0.001 | 100% | 15.7 s ± 0.6 |
| Per-snapshot Louvain | 1.000 | 1.000 | 100% | 0.8 s |

**Interpretation and limits:**
- **Ceiling effect.** The subsample drops every edge to the 493 communities that were not drawn, so it is far easier than the full benchmark. Louvain is perfect, so this pilot shows the reimplementation *works*, but it cannot tell DyG-MF apart from simple baselines. A harder E1 setting is needed: more communities, inter-community edges kept, or the planted generator with mixing.
- **Realised budget ≈ 2× nominal μ.** Landmarks are re-selected every snapshot, and newly chosen or dynamic landmarks are re-estimated, so about 32% of nodes are recomputed at μ = 0.16. The budget accounting in E4/E5 must use realised counts.
- **Selective updating saved time:** 10.1 s against 15.7 s for full updating, at equal NMI here.
- **Contrast with the planted pilot (scratch, 2026-10-11).** On a small planted sequence with a merge and a birth, μ = 0.16 drifted (NMI from 1.00 to 0.60 over 6 snapshots), while μ = 1 tracked from-scratch runs. The Birth-Death subsample draws contain no events. This is preliminary evidence that frozen embeddings drift under structural events, which bears directly on RQ3 and E6. It is a 3-seed pilot, not a result.
- **Checkpoint verdict (25 Oct gate, preliminary):** the reimplementation is label-free, fully specified (D1–D9 plus two measured revisions) and recovers easy structure. It is acceptable to build on, provided E1 is repeated on a harder setting before the final tables.
