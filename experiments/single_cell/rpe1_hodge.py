"""Cell cycle (RPE1-FUCCI, labelling velocity): is the velocity's
direction-carrying part a gradient, a local curl, or HARMONIC (a flow around
the hole of the cycle)? Scored on the circle against the FUCCI position.

Three-way graph Hodge decomposition (HodgeRank, Jiang, Lim, Yao & Ye 2011) of
the edge flow f_ij = (v_i + v_j)/2 . (x_j - x_i) on the symmetrised kNN graph
in 30-PC space:
    f = B phi + C^T u + h
  gradient  B phi    least squares on the incidence matrix B (as in
                     gastrulation_hodge.py)
  curl      C^T u    least squares on the triangle-edge matrix C of the graph's
                     triangles ((C f)_ijk = f_ij + f_jk - f_ik: circulation)
  harmonic  h        the rest: divergence-free AND circulation-free on every
                     triangle -- only possible around holes of the graph
The three parts are orthogonal; their shares of ||f||^2 are reported, with a
null (centred velocities permuted among cells).

Ground truth: theta = 2 pi * Cell_cycle_relativePos (FUCCI). Per part:
  phase corr   correlation over edges between the part's edge values and the
               wrapped phase difference along the edge (does it point forward
               around the cycle?)
  coherence    as in gastrulation_hodge.py (1 = smooth field, ~0.2 noise)
  walk         cosine walk in 30-PC space (k = 30, weights exp(10 cos)) with
               the part's per-cell vectors: mean phase progress after s steps
               (unwrapped, fraction of a full cycle; see phase_progress) minus
               the direction-blind walk's
And the gradient potential's fit to the phase: R^2 of phi ~ cos(theta) +
sin(theta) (a potential can follow the cycle only up to the point where it
has to come back down).

Run:    python experiments/single_cell/rpe1_hodge.py
Needs:  data/rpe1.npz (rpe1_data.py, dynamo env)
Writes: experiments/single_cell/rpe1_results/<tag>/{hodge.json, HODGE.md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gastrulation_hodge import cell_vectors, coherence, edge_flow, edges_from_knn, hodge, incidence  # noqa: E402
from gastrulation_repr import chain, knn  # noqa: E402

TWO_PI = 2 * np.pi


def wrap(a):
    """Angle difference wrapped to [-pi, pi)."""
    return (a + np.pi) % TWO_PI - np.pi


def triangles(E, n):
    """All triangles (i < j < k) of the undirected graph with edges E (i < j)."""
    nb = [[] for _ in range(n)]
    for i, j in E:
        nb[i].append(j)
    nb = [np.array(sorted(x), dtype=np.int64) for x in nb]
    T = []
    for i in range(n):
        for j in nb[i]:
            ks = np.intersect1d(nb[i][nb[i] > j], nb[j], assume_unique=True)
            if len(ks):
                T.append(np.column_stack([np.full(len(ks), i), np.full(len(ks), j), ks]))
    return np.concatenate(T) if T else np.zeros((0, 3), dtype=np.int64)


def curl_matrix(T, E, n):
    """C (t x m): (C f)_ijk = f_ij + f_jk - f_ik for edges oriented i < j."""
    key = E[:, 0] * n + E[:, 1]
    order = np.argsort(key)

    def eid(a, b):
        return order[np.searchsorted(key[order], a * n + b)]
    t = len(T)
    rows = np.repeat(np.arange(t), 3)
    cols = np.column_stack([eid(T[:, 0], T[:, 1]), eid(T[:, 1], T[:, 2]), eid(T[:, 0], T[:, 2])]).ravel()
    vals = np.tile([1.0, 1.0, -1.0], t)
    return sp.csr_matrix((vals, (rows, cols)), shape=(t, len(E)))


def hodge3(B, C, f):
    """f = grad + curl + harmonic (orthogonal)."""
    _, g, r = hodge(B, f)
    u = lsqr(C.T, r, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    c = C.T @ u
    return g, c, r - c


def phase_progress(M, theta, steps):
    """Per cell: expected phase travelled after s steps of the walk M, in
    cycles, UNWRAPPED -- the sum over steps of the expected wrapped increment
    of each step, sum_{t < s} M^t g with g_i = sum_j M_ij wrap(theta_j - theta_i).
    Exact as long as a single step stays below half a cycle; unlike wrapping
    the start-to-end difference, a walk that travels far is not counted as
    going backward. Returns {s: (n,) array}."""
    g = np.asarray(M.multiply(wrap(theta[None, :] - theta[:, None])).sum(1)).ravel() / TWO_PI
    out, acc, h = {}, np.zeros_like(g), g.copy()
    for s in range(1, max(steps) + 1):
        acc = acc + h
        h = M @ h
        if s in steps:
            out[s] = acc.copy()
    return out


def phase_advance(M, theta, steps):
    """Mean over cells of ``phase_progress``."""
    return {s: float(v.mean()) for s, v in phase_progress(M, theta, steps).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--tag", default="hodge")
    a = ap.parse_args()
    res = os.path.join(HERE, "rpe1_results", a.tag)
    os.makedirs(res, exist_ok=True)

    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X = d["X_pca"]
    theta = TWO_PI * d["cycle_pos"]
    n = len(X)
    idx = knn(X, a.k)
    E = edges_from_knn(idx)
    B = incidence(E, n)
    t0 = time.time()
    T = triangles(E, n)
    C = curl_matrix(T, E, n)
    dtheta = wrap(theta[E[:, 1]] - theta[E[:, 0]])
    # how well the 30-PC neighbourhoods respect the cycle at all
    nb_spread = float(np.median(np.abs(wrap(theta[idx] - theta[:, None])).mean(1)) / TWO_PI)
    print(f"rpe1: {n} cells, {len(E)} edges, {len(T)} triangles ({time.time() - t0:.0f}s); "
          f"median |phase difference| to 30-PC neighbours: {nb_spread:.3f} cycle", flush=True)

    def walk(F):
        adv = phase_advance(chain(X, F, a.k), theta, a.steps)
        adv0 = phase_advance(chain(X, F, a.k, directed=False), theta, a.steps)
        return {f"s{s}": adv[s] - adv0[s] for s in a.steps}

    out = dict(args=vars(a), n=n, m=len(E), t=len(T), neighbour_phase_spread=nb_spread,
               velocity_share_in_pca=float(np.median(d["vel_share"])), velocities={})
    rng = np.random.default_rng(0)
    for name, V in (("direct", d["V_pca_direct"]), ("dynamo_kernel", d["V_pca_kernel"])):
        V = V.astype(np.float64)
        f = edge_flow(X, V, E)
        g, c, h = hodge3(B, C, f)
        tot = (f ** 2).sum()
        null = []
        for _ in range(a.n_null):
            fn = edge_flow(X, (V - V.mean(0))[rng.permutation(n)], E)
            gn, cn, hn = hodge3(B, C, fn)
            null.append([float((x ** 2).sum() / (fn ** 2).sum()) for x in (gn, cn, hn)])
        phi, _, _ = hodge(B, f)
        A = np.column_stack([np.cos(theta), np.sin(theta), np.ones(n)])
        fit = A @ np.linalg.lstsq(A, phi, rcond=None)[0]
        parts = {"full": f, "gradient": g, "curl": c, "harmonic": h}
        U = {p: cell_vectors(X, E, v) for p, v in parts.items()}
        o = dict(share={p: float((parts[p] ** 2).sum() / tot) for p in ("gradient", "curl", "harmonic")},
                 share_null=dict(zip(("gradient", "curl", "harmonic"), np.mean(null, 0).tolist())),
                 phase_corr={p: float(np.corrcoef(v, dtheta)[0, 1]) for p, v in parts.items()},
                 coherence={p: coherence(u, idx) for p, u in U.items()},
                 coherence_raw=coherence(V, idx),
                 phi_phase_R2=float(1 - ((phi - fit) ** 2).sum() / ((phi - phi.mean()) ** 2).sum()),
                 walk={p: walk(u) for p, u in U.items()}, walk_raw=walk(V),
                 mean_share=float((V.mean(0) ** 2).sum() / (V ** 2).sum(1).mean()))
        out["velocities"][name] = o
        print(f"  {name}: shares " + ", ".join(f"{p} {v:.3f} (null {o['share_null'][p]:.3f})"
                                              for p, v in o["share"].items())
              + "; phase corr " + ", ".join(f"{p} {v:+.3f}" for p, v in o["phase_corr"].items())
              + f"; walk s{a.steps[-1]} " + ", ".join(f"{p} {w[f's{a.steps[-1]}']:+.4f}"
                                                      for p, w in o["walk"].items()), flush=True)
        json.dump(out, open(os.path.join(res, "hodge.json"), "w"), indent=1)
    write_report(out, res)


def write_report(out, res):
    steps = out["args"]["steps"]
    L = [f"# RPE1-FUCCI cell cycle: Hodge decomposition of the velocity (n = {out['n']}, "
         f"{out['m']} edges, {out['t']} triangles, k = {out['args']['k']})", "",
         f"Velocity: dynamo's labelling-based velocity of total RNA, projected linearly on dynamo's "
         f"30-PC PCA (median share of its norm kept: {out['velocity_share_in_pca']:.2f}); "
         f"`dynamo_kernel` = dynamo's own kernel-based projection. Median phase difference between a "
         f"cell and its 30-PC neighbours: {out['neighbour_phase_spread']:.3f} cycle.", "",
         "| velocity | common mean | gradient | curl | harmonic | φ ~ phase R² | "
         "phase corr: full / grad / curl / harm | coherence: raw / grad / curl / harm |",
         "|---|---|---|---|---|---|---|---|"]
    for nm, o in out["velocities"].items():
        s, sn, pc, co = o["share"], o["share_null"], o["phase_corr"], o["coherence"]
        L.append(f"| {nm} | {o['mean_share']:.2f} | " + " | ".join(
            f"{s[p]:.3f} (null {sn[p]:.3f})" for p in ("gradient", "curl", "harmonic"))
            + f" | {o['phi_phase_R2']:.2f} | {pc['full']:+.3f} / {pc['gradient']:+.3f} / "
            f"{pc['curl']:+.3f} / {pc['harmonic']:+.3f} | {o['coherence_raw']:.2f} / "
            f"{co['gradient']:.2f} / {co['curl']:.2f} / {co['harmonic']:.2f} |")
    L += ["", "Shares of ||f||²; null = centred velocities permuted among cells. Phase corr = "
          "correlation over edges between the part's edge values and the wrapped FUCCI phase "
          "difference.", "",
          "## Walk: phase advance over the direction-blind walk (fraction of a cycle)", "",
          "| velocity | part | " + " | ".join(f"s={s}" for s in steps) + " |",
          "|---|---|" + "---|" * len(steps)]
    for nm, o in out["velocities"].items():
        for p, w in {"raw velocity": o["walk_raw"], **o["walk"]}.items():
            L.append(f"| {nm} | {p} | " + " | ".join(f"{w[f's{s}']:+.4f}" for s in steps) + " |")
    open(os.path.join(res, "HODGE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "HODGE.md"))


if __name__ == "__main__":
    main()
