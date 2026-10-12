# Related work notes

## Temporal baseline shortlist (Week 1, step 10)

The report (E8) requires one reproducible temporal baseline. Its information access, compute needs and interface must be checked before choosing. The choice is made in Week 2.

Status key: **checked** means confirmed from the paper or code; **verify** means recalled from memory and still to be confirmed against the source before relying on it.

| Candidate | Information access | Compute | Interface / code | Fit | Status |
|---|---|---|---|---|---|
| **Warm-start Louvain**: Louvain at t initialised from the partition at t−1 (python-louvain `best_partition(partition=...)`) | Causal: current snapshot plus previous partition | Same order as static Louvain; usually faster from a warm start | Python; fits `step(GraphSnapshot)` directly | Strong. A cheap, transparent incremental baseline that separates "temporal smoothing" from "selective updating" | **checked 2026-10-11:** python-louvain 0.16 `best_partition(graph, partition=None, weight, resolution, randomize, random_state)`; the starting partition must cover every node, so new nodes need a singleton community first |
| **Evolutionary spectral clustering**, PCQ/PCM (Chi et al., KDD 2007) | Causal: blends snapshot cost with the previous partition (PCQ) or affinity (PCM) | One eigendecomposition per snapshot; fine for small and medium graphs | No official code; about 50 lines on numpy/scipy | Good. A classic temporal-smoothness baseline, close in spirit to factorisation methods | verify the formulation in the paper |
| **DF Louvain / dynamic frontier** (Sahu, 2024, arXiv:2404.19634) | Causal: previous partition plus the batch of edge changes; re-processes only affected vertices | Very fast; designed for large graphs | C++/OpenMP code (puzzlef); would need a Python port or a subprocess wrapper | Good conceptual match (selective updating via a change frontier), but a heavier integration | verify the code licence and build |
| **Δ-screening** (Zarayeneh & Kalyanaraman, ASONAM 2019, arXiv:1904.08553) | Causal: screens vertices affected by the edge batch, then runs incremental Louvain | Reported about 3× speedup over static | Code availability unclear | Very close to our selective-update question. Strong as related work; as a baseline only if code exists or a port is small | verify code availability |
| **PisCES** (Liu et al., PNAS 2018) | **Not causal**: global smoothing across past *and future* snapshots | Iterative spectral; whole sequence at once | MATLAB (github.com/letitiaLiu/PisCES; listed among DyG-MF's baselines) | Only as a labelled offline reference, not a fair online competitor | verify |

**Chosen (Week 2, 2026-10-11):** warm-start Louvain. It will be implemented and run in Week 7 (E8), since nothing needs it before then (YAGNI).

**Original recommendation:** warm-start Louvain as the primary temporal baseline, because it is causal, fits our interface and is cheap to reproduce. Evolutionary spectral clustering would be a second choice if Kelly wants a smoothness-based method. Δ-screening and DF Louvain are the closest related work for the literature comparison (report §15, Chapter 2) whether or not they are run.


## Contribution review (end of Week 3, 2026-10-12)

**Limitation demonstrated so far (E1–E3 pilots):**
- DyG-MF's selective updating depends on next-snapshot information (Eq. 14 uses t+1, and §4.4 states that the method takes G_{t−1}, G_t and G_{t+1}).
- Its change score selects changed nodes barely better than random under gradual change (recall 0.19 vs 0.17).
- With a fixed budget, frozen embeddings drift after structural events (Week 2 planted pilot).

**Intervention so far:**
- Causal operation, which costs no measured quality and cuts operational delay from about 1.25 to about 0.1 snapshots.
- The corrected community-profile score, which recovers all changed nodes under gradual change and nearly matches full updating at about a third of the updates.

**How this differs from the closest work (to verify against sources when writing Chapter 2):**
- jLMDC (Li et al., 2023) also splits nodes into dynamic and static groups by topology. Our profile score instead measures change *relative to the previous partition*, not raw topology.
- Δ-screening and DF Louvain select affected vertices from the edge batch, within modularity methods. Our setting is an incremental factorisation method, with explicit accounting and an evaluation of causal delay.

**Not yet shown, and the remaining contribution:**
- Budgets that respond to how much the graph changed (E4).
- Cost-aware allocation (E5).
- Drift control by refresh (E6).

The current fixed μ = 0.16 is a weakness of both DyG-MF and our causal variants, which motivates Weeks 4–6.

**Framing risk:** per-snapshot Louvain beats DyG-MF on the planted graphs. Contribution claims must compare selection policies inside the incremental method (with full updating as the reference), and E8 must check real data.
