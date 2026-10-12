"""DyG-MF reimplementation (Li et al., WWW 2025), following docs/design_log.md D1-D9.

Per snapshot: positive NetMF PMI matrix M (window T); temporal landmark selection (Eq. 6 with the
corrected centre update); landmark factorisation M00 ~ Phi Psi (Eqs. 8/15); then the
separated factorisation (Eq. 10, first-line form) over non-landmark nodes with
C = P Phi, H = Psi Q and the bi-clustering regulariser (Eq. 13). Labels come from
k-means with k_t on the rows of C (design log D2, revised: argmax fragments clusters
whenever r > k). After the first snapshot only dynamic nodes (Eq. 14) are
re-estimated; static nodes keep their embedding row.

Dense numpy throughout: intended for graphs of up to a few thousand nodes.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from dgc.data.types import GraphSnapshot
from dgc.methods import scores as sc
from dgc.methods.base import StepResult

EPS = 1e-10
SCORERS = ("eq14", "one_step", "ewma", "profile", "combined", "random")


@dataclass(frozen=True)
class DyGMFConfig:
    """Hyperparameters; defaults are the paper's section 5 settings where reported.

    `r` is a fixed embedding width and should bound every k_t (default k_max); the
    paper's r = 1000 is not used because the read-out needs r close to k.
    `scorer` picks the dynamic-node score: "eq14" is the paper's (it uses t+1 when the
    lookahead adapter supplies it); the others are causal estimators from `scores`.
    """

    r: int = 20
    pmi_window: int = 2
    landmark_fraction: float = 0.5
    mu: float = 0.16
    lam: float = 0.2
    beta: float = 20.0
    s: int = 1
    n_iter: int = 100
    f_every: int = 5
    k_min: int = 2
    k_max: int = 20
    kmeans_iter: int = 100
    scorer: str = "eq14"
    ewma_gamma: float = 0.5

    def __post_init__(self) -> None:
        if self.scorer not in SCORERS:
            raise ValueError(f"unknown scorer {self.scorer!r}; known: {list(SCORERS)}")


@dataclass
class _Frame:
    """One snapshot in matrix form, indexed by its sorted node list."""

    t: int
    nodes: np.ndarray
    adj: np.ndarray
    pmi: np.ndarray
    k: int = 0
    landmarks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    centres: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


# ---- pure helpers ---------------------------------------------------------------


def kmeans(k: int, n_init: int, seed: int) -> KMeans:
    """KMeans with an integer n_init (sklearn has no stubs; pyright infers str)."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed)  # pyright: ignore[reportArgumentType]


def adjacency(snapshot: GraphSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Sorted node array and dense 0/1 adjacency matrix of a snapshot."""
    nodes = np.array(sorted(snapshot.nodes), dtype=int)
    index = {int(n): i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)))
    for u, v in snapshot.edges:
        adj[index[u], index[v]] = adj[index[v], index[u]] = 1.0
    return nodes, adj


def ppmi(adj: np.ndarray, window: int = 2) -> np.ndarray:
    """Positive NetMF PMI (Qiu et al. 2018, one negative sample):
    max(log(vol / T * sum_{r=1..T} (D^-1 A)^r D^-1), 0). T = 1 is the paper's
    printed formula restricted to edges; design log D1 (revised) uses T = 2."""
    deg = adj.sum(axis=1)
    vol = deg.sum()
    if vol == 0:
        return np.zeros_like(adj)
    inv_deg = np.where(deg > 0, 1 / np.maximum(deg, EPS), 0.0)
    walk = inv_deg[:, None] * adj
    power, total = np.eye(len(adj)), np.zeros_like(adj)
    for _ in range(window):
        power = power @ walk
        total += power
    scaled = vol / window * total * inv_deg[None, :]
    out = np.zeros_like(adj)
    positive = scaled > 1
    out[positive] = np.log(scaled[positive])
    return out


def sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between the rows of a and b."""
    d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.maximum(d, 0.0)


def reindex(matrix: np.ndarray, old: np.ndarray, new: np.ndarray, rows: bool = True) -> np.ndarray:
    """Map a matrix indexed by `old` nodes onto `new` nodes; missing entries become 0.

    Columns are always re-indexed; rows too when `rows` is True.
    """
    pos = {int(n): i for i, n in enumerate(old)}
    keep_new = [i for i, n in enumerate(new) if int(n) in pos]
    keep_old = [pos[int(new[i])] for i in keep_new]
    if rows:
        out = np.zeros((len(new), len(new)))
        out[np.ix_(keep_new, keep_new)] = matrix[np.ix_(keep_old, keep_old)]
    else:
        out = np.zeros((matrix.shape[0], len(new)))
        out[:, keep_new] = matrix[:, keep_old]
    return out


def elbow_k(rows: np.ndarray, k_min: int, k_max: int, seed: int) -> int:
    """Elbow of the k-means inertia curve: max distance below the normalised chord."""
    k_max = min(k_max, len(rows) - 1)
    if k_max <= k_min:
        return max(1, min(k_min, len(rows)))
    ks = np.arange(k_min, k_max + 1)
    inertia = np.array([kmeans(int(k), 3, seed).fit(rows).inertia_ for k in ks])
    span = inertia[0] - inertia[-1]
    if span <= 0:
        return int(k_min)
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia[-1]) / span
    return int(ks[np.argmax(1 - x - y)])


def temporal_centres(
    m_t: np.ndarray,
    m_prev: np.ndarray,
    lam_per_node: np.ndarray,
    k: int,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 6 by alternating assignment and the corrected closed-form centre update.

    Returns the centres (k x n) and each node's combined distance to every centre.
    """
    centres, _ = kmeans_plusplus(m_t, n_clusters=k, random_state=seed)
    assign = None
    dist = np.empty((len(m_t), k))
    for _ in range(max_iter):
        dist = sqdist(m_t, centres) + lam_per_node[:, None] * sqdist(m_prev, centres)
        new_assign = dist.argmin(axis=1)
        if assign is not None and np.array_equal(assign, new_assign):
            break
        assign = new_assign
        for c in range(k):
            members = assign == c
            if members.any():
                weight = 1 + lam_per_node[members]
                num = m_t[members] + lam_per_node[members, None] * m_prev[members]
                centres[c] = num.sum(axis=0) / weight.sum()
    return centres, dist


def nearest_landmarks(dist: np.ndarray, n_landmarks: int) -> np.ndarray:
    """Each centre in turn takes its nearest unassigned nodes until its quota is met."""
    k = dist.shape[1]
    quotas = [n_landmarks // k + (1 if c < n_landmarks % k else 0) for c in range(k)]
    taken = np.zeros(len(dist), dtype=bool)
    chosen: list[int] = []
    for c in range(k):
        order = np.argsort(dist[:, c], kind="stable")
        picked = [int(i) for i in order if not taken[i]][: quotas[c]]
        taken[picked] = True
        chosen.extend(picked)
    return np.array(sorted(chosen), dtype=int)


def biclustering_penalty(c_block: np.ndarray, k: int) -> np.ndarray:
    """X with X_al = ||u_a/sqrt(d_a) - v_l/sqrt(d_l)||^2 from the bottom-k eigenvectors
    of the normalised Laplacian of the bipartite graph [[0, C], [C^T, 0]]."""
    n, r = c_block.shape
    s = np.zeros((n + r, n + r))
    s[:n, n:] = c_block
    s[n:, :n] = c_block.T
    deg = s.sum(axis=1)
    inv_sqrt = np.where(deg > 0, 1 / np.sqrt(np.maximum(deg, EPS)), 0.0)
    lap = np.eye(n + r) - inv_sqrt[:, None] * s * inv_sqrt[None, :]
    _, vecs = np.linalg.eigh(lap)
    f = vecs[:, : min(k, n + r)] * inv_sqrt[:, None]
    return sqdist(f[:n], f[n:])


def eq14_scores(w_rows: list[np.ndarray], centres: list[np.ndarray], current: int) -> np.ndarray:
    """Eq. 14: topology change plus shift relative to the nearest centres.

    `w_rows` and `centres` hold the available snapshots among t-1, t, t+1, all
    re-indexed to the nodes at t; `current` is the position of t in those lists.
    """
    w_t = w_rows[current]
    w_bar = sum(w_rows) / len(w_rows)

    def nearest(cent: np.ndarray) -> np.ndarray:
        return cent[sqdist(w_t, cent).argmin(axis=1)]

    theta_t = nearest(centres[current])
    theta_bar = sum(nearest(c) for c in centres) / len(centres)
    topo = ((w_t - w_bar) ** 2).sum(axis=1)
    shift = (((w_t - theta_t) - (w_bar - theta_bar)) ** 2).sum(axis=1)
    return topo + shift


def _row_normalise(x: np.ndarray) -> np.ndarray:
    sums = x.sum(axis=1, keepdims=True)
    uniform = np.full_like(x, 1.0 / x.shape[1])
    return np.where(sums > EPS, x / np.maximum(sums, EPS), uniform)


# ---- method ------------------------------------------------------------------------


class DyGMFCore:
    """DyG-MF state machine. `process(cur, nxt)` returns the output for `cur`.

    `nxt` is snapshot t+1 (the published lookahead) or None at the final snapshot.
    `k_schedule`, if given, is an oracle number of clusters per t (diagnostic only).
    """

    def __init__(self, config: DyGMFConfig, seed: int, k_schedule: dict[int, int] | None = None):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.k_schedule = k_schedule
        self.r: int | None = None
        self.prev: _Frame | None = None
        self.ahead: _Frame | None = None
        self.emb: dict[int, np.ndarray] = {}
        self.phi: dict[int, np.ndarray] = {}
        self.psi: dict[int, np.ndarray] = {}
        self.labels: dict[int, int] = {}
        self.history: np.ndarray | None = None

    def _seed(self) -> int:
        return int(self.rng.integers(2**31 - 1))

    def _frame(self, snapshot: GraphSnapshot, prev: _Frame | None) -> _Frame:
        """Matrices, k_t, centres and landmarks for a snapshot (cached for lookahead)."""
        if self.ahead is not None and self.ahead.t == snapshot.t:
            return self.ahead
        nodes, adj = adjacency(snapshot)
        frame = _Frame(t=snapshot.t, nodes=nodes, adj=adj, pmi=ppmi(adj, self.cfg.pmi_window))
        n = len(nodes)
        if self.k_schedule is not None:
            frame.k = min(self.k_schedule[snapshot.t], n)
        else:
            frame.k = elbow_k(frame.pmi, self.cfg.k_min, self.cfg.k_max, self._seed())
        if prev is None:
            m_prev = np.zeros_like(frame.pmi)
            lam = np.zeros(n)
        else:
            m_prev = reindex(prev.pmi, prev.nodes, nodes)
            present = np.isin(nodes, prev.nodes)
            lam = np.where(present, self.cfg.lam, 0.0)
        frame.centres, dist = temporal_centres(
            frame.pmi, m_prev, lam, frame.k, self._seed(), self.cfg.kmeans_iter
        )
        n_landmarks = max(frame.k, round(self.cfg.landmark_fraction * n))
        frame.landmarks = nearest_landmarks(dist, min(n_landmarks, n))
        return frame

    def process(self, cur: GraphSnapshot, nxt: GraphSnapshot | None) -> StepResult:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        frame = self._frame(cur, self.prev)
        n = len(frame.nodes)
        ids = [int(x) for x in frame.nodes]
        if self.r is None:
            self.r = max(1, min(self.cfg.r, len(frame.landmarks)))
        timings["landmarks"] = time.perf_counter() - start

        tick = time.perf_counter()
        if self.prev is None:
            dynamic, selected = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
        else:
            dynamic, selected = self._dynamic_mask(frame, nxt)
        timings["score"] = time.perf_counter() - tick

        tick = time.perf_counter()
        emb, phi_rows, psi_cols, updated = self._factorise(frame, dynamic)
        timings["cluster"] = time.perf_counter() - tick

        tick = time.perf_counter()
        km = kmeans(min(frame.k, n), 10, self._seed())
        labels = {ids[i]: int(c) for i, c in enumerate(km.fit_predict(emb))}
        timings["readout"] = time.perf_counter() - tick
        # History is updated only after the output for t exists.
        self.emb = {ids[i]: emb[i] for i in range(n)}
        self.phi, self.psi = phi_rows, psi_cols
        if self.history is None or self.prev is None:
            self.history = frame.adj.copy()
        else:
            aligned = reindex(self.history, self.prev.nodes, frame.nodes)
            self.history = sc.ewma_update(aligned, frame.adj, self.cfg.ewma_gamma)
        self.labels = labels
        self.prev = frame
        return StepResult(
            labels=labels,
            updated_nodes=frozenset(ids[i] for i in np.flatnonzero(updated)),
            timings=timings,
            selected=frozenset(ids[i] for i in np.flatnonzero(selected)),
        )

    def _dynamic_mask(
        self, frame: _Frame, nxt: GraphSnapshot | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dynamic nodes: every node without a stored embedding (mandatory) plus the
        top ceil(mu*n) others by the configured score (returned as `selected`)."""
        scores = self._scores(frame, nxt)
        nodes = frame.nodes
        mandatory = np.array([int(x) not in self.emb for x in nodes])
        selected = np.zeros(len(nodes), dtype=bool)
        budget = int(np.ceil(self.cfg.mu * len(nodes)))
        order = np.lexsort((nodes, -scores))
        selected[[i for i in order if not mandatory[i]][:budget]] = True
        return mandatory | selected, selected

    def _scores(self, frame: _Frame, nxt: GraphSnapshot | None) -> np.ndarray:
        prev = self.prev
        assert prev is not None and self.history is not None
        nodes, w_t = frame.nodes, frame.adj
        w_prev = reindex(prev.adj, prev.nodes, nodes)
        name = self.cfg.scorer
        if name == "eq14":
            w_rows = [w_prev, w_t]
            centres = [reindex(prev.centres, prev.nodes, nodes, rows=False), frame.centres]
            if nxt is not None:
                self.ahead = self._frame(nxt, frame)
                w_rows.append(reindex(self.ahead.adj, self.ahead.nodes, nodes))
                centres.append(reindex(self.ahead.centres, self.ahead.nodes, nodes, rows=False))
            return eq14_scores(w_rows, centres, current=1)
        if name == "random":
            return self.rng.random(len(nodes))
        one = sc.one_step(w_t, w_prev)
        ewma = sc.ewma_score(w_t, reindex(self.history, prev.nodes, nodes))
        if name == "one_step":
            return one
        if name == "ewma":
            return ewma
        prev_labels = np.array([self.labels.get(int(x), sc.UNKNOWN) for x in nodes])
        profile = sc.community_profile(w_t, w_prev, prev_labels, prev.k)
        if name == "profile":
            return profile
        return sc.combined([one, ewma, profile])

    def _factorise(self, frame: _Frame, dynamic: np.ndarray):
        """Landmark factorisation, then the separated factorisation of free rows."""
        cfg, r = self.cfg, self.r
        assert r is not None
        m, lm = frame.pmi, frame.landmarks
        ids = [int(x) for x in frame.nodes]
        non = np.setdiff1d(np.arange(len(ids)), lm)
        n_u = len(lm)

        # Landmark rows: fixed when the node is static and was a landmark before.
        fixed = np.array([not dynamic[i] and ids[i] in self.phi for i in lm], dtype=bool)
        phi = _row_normalise(self.rng.random((n_u, r)))
        psi = self.rng.random((r, n_u))
        for row, i in enumerate(lm):
            if fixed[row]:
                phi[row], psi[:, row] = self.phi[ids[i]], self.psi[ids[i]]
        free = ~fixed
        a = m[np.ix_(lm, lm)]
        for _ in range(cfg.n_iter):
            if not free.any():
                break
            new_phi = phi * (a @ psi.T) / (phi @ psi @ psi.T + EPS)
            phi[free] = _row_normalise(new_phi[free])
            new_psi = psi * (phi.T @ a) / (phi.T @ phi @ psi + EPS)
            psi[:, free] = new_psi[:, free]

        # Non-landmark rows: dynamic rows are optimised via P, static rows keep C.
        emb = np.zeros((len(ids), r))
        emb[lm] = phi
        dyn = non[dynamic[non]]
        static = non[~dynamic[non]]
        for i in static:
            emb[i] = self.emb[ids[i]]
        if len(non):
            self._separated(m, lm, non, dyn, emb, phi, psi, frame.k)

        updated = dynamic.copy()
        updated[lm[free]] = True
        phi_rows = {ids[i]: phi[row] for row, i in enumerate(lm)}
        psi_cols = {ids[i]: psi[:, row] for row, i in enumerate(lm)}
        return emb, phi_rows, psi_cols, updated

    def _separated(self, m, lm, non, dyn, emb, phi, psi, k) -> None:
        """Eq. 10 (first-line form) and Eq. 13 by multiplicative updates; fills emb[dyn]."""
        cfg = self.cfg
        n_u = len(lm)
        m_ln, m_nn = m[np.ix_(lm, non)], m[np.ix_(non, non)]
        m_dn, m_dl = m[np.ix_(dyn, non)], m[np.ix_(dyn, lm)]
        q = self.rng.random((n_u, len(non)))
        p = _row_normalise(m_dl + 1e-3 * self.rng.random((len(dyn), n_u)))
        subsets = np.array_split(self.rng.permutation(len(dyn)), max(1, min(cfg.s, len(dyn))))
        phipsi = phi @ psi
        x = np.zeros((len(dyn), phi.shape[1]))

        for it in range(cfg.n_iter):
            if len(dyn) and it % cfg.f_every == 0:
                c_dyn = p @ phi
                for sub in subsets:
                    if len(sub):
                        x[sub] = biclustering_penalty(c_dyn[sub], k)
            if len(dyn):
                emb[dyn] = p @ phi
            c_non = emb[non]
            g1 = c_non @ psi
            q *= (g1.T @ m_nn + phipsi.T @ m_ln) / ((g1.T @ g1 + phipsi.T @ phipsi) @ q + EPS)
            if len(dyn):
                b1 = phipsi @ q
                numer = m_dn @ b1.T + m_dl @ phipsi.T
                denom = p @ (b1 @ b1.T + phipsi @ phipsi.T) + cfg.beta * x @ phi.T
                p = _row_normalise(p * numer / (denom + EPS))
        if len(dyn):
            emb[dyn] = p @ phi

    def get_state(self) -> dict[str, Any]:
        return {
            "rng": self.rng.bit_generator.state,
            "r": self.r,
            "prev": self.prev,
            "ahead": self.ahead,
            "emb": dict(self.emb),
            "phi": dict(self.phi),
            "psi": dict(self.psi),
            "labels": dict(self.labels),
            "history": self.history,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.r, self.prev, self.ahead = state["r"], state["prev"], state["ahead"]
        self.emb, self.phi, self.psi = dict(state["emb"]), dict(state["phi"]), dict(state["psi"])
        self.labels, self.history = dict(state["labels"]), state["history"]


class CausalDyGMF:
    """DyG-MF that never sees t+1: an `OnlineClusterer` with a causal scorer.

    With scorer "eq14" this is the paper's score with the t+1 terms dropped, as the
    paper itself does at the final snapshot.
    """

    def __init__(self, config: DyGMFConfig, seed: int, k_schedule: dict[int, int] | None = None):
        self.core = DyGMFCore(config, seed, k_schedule)

    def step(self, snapshot: GraphSnapshot) -> StepResult:
        return self.core.process(snapshot, None)

    def get_state(self) -> dict[str, Any]:
        return self.core.get_state()

    def set_state(self, state: dict[str, Any]) -> None:
        self.core.set_state(state)
