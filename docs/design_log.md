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
- **Open question for Kelly:** do we have MATLAB access? MATLAB could show whether the P-code runs at all. Even so, F2, F3, F5 and F6 mean a MATLAB run would not be a clean label-free reference. MATLAB is therefore useful but not decisive.

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
