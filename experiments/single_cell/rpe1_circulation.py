"""Does the RPE1 velocity CIRCULATE around the cell cycle? A gradient
(potential) flow has exactly zero circulation around any closed loop, so a
consistent non-zero circulation is what no potential -- no landscape -- can
represent. (The Hodge shares of rpe1_hodge.py cannot show it: a cycle is
locally a gradient, and even the ideal phase flow comes out ~80% gradient.)

Loops: --n-loops closed loops around the cycle, each through one random cell
per FUCCI phase bin (--bins bins), consecutive cells joined by shortest paths
on the kNN graph (edge length = 30-PC distance). Along a loop, with the edge
flow f_ij = (v_i + v_j)/2 . (x_j - x_i):
  net/abs = sum of f along the loop / sum of |f| (+1: always forward around
            the cycle, -1: always backward, 0: no net circulation)
Statistic: its mean over the loops. Null: the same loops, velocities
(centred) permuted among cells, --n-null times. Reference: the ideal phase
flow (wrapped FUCCI phase difference along each edge).

Where the circulation comes from: the mean flow per segment (bin b -> b+1)
over the loops, relative to the segment's mean |flow| -- is it spread around
the cycle or concentrated at the M -> G1 seam?

Run:    python experiments/single_cell/rpe1_circulation.py
Writes: experiments/single_cell/rpe1_results/circulation/{circulation.json, CIRCULATION.md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gastrulation_hodge import edge_flow, edges_from_knn  # noqa: E402
from gastrulation_repr import knn  # noqa: E402
from rpe1_hodge import TWO_PI, wrap  # noqa: E402


def build_loops(X, pos, E, n_loops, bins, rng):
    """Each loop: list of (edge id, sign) going once around the phase bins."""
    n, m = len(X), len(E)
    L = np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1)
    G = sp.csr_matrix((np.r_[L, L], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    eid = sp.csr_matrix((np.r_[np.arange(m), np.arange(m)] + 1, (np.r_[E[:, 0], E[:, 1]],
                                                                 np.r_[E[:, 1], E[:, 0]])), shape=(n, n))
    members = [np.flatnonzero((pos * bins).astype(int) % bins == b) for b in range(bins)]
    loops = []
    for _ in range(n_loops):
        nodes = [rng.choice(mb) for mb in members]
        _, pred = shortest_path(G, indices=nodes, return_predecessors=True, directed=False)
        segs = []
        for b in range(bins):
            a, t = nodes[b], nodes[(b + 1) % bins]
            path = [t]
            while path[-1] != a:
                path.append(pred[b, path[-1]])
            path = path[::-1]
            segs.append([(int(eid[i, j]) - 1, 1.0 if i < j else -1.0) for i, j in zip(path[:-1], path[1:])])
        loops.append(segs)
    return loops


def stats(loops, f, bins):
    ratios, seg_net, seg_abs = [], np.zeros(bins), np.zeros(bins)
    for segs in loops:
        net = tot = 0.0
        for b, seg in enumerate(segs):
            v = np.array([s * f[e] for e, s in seg]) if seg else np.zeros(0)
            net += v.sum()
            tot += np.abs(v).sum()
            seg_net[b] += v.sum()
            seg_abs[b] += np.abs(v).sum()
        ratios.append(net / max(tot, 1e-300))
    return float(np.mean(ratios)), float(np.mean(np.array(ratios) > 0)), seg_net / np.maximum(seg_abs, 1e-300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[10, 30])
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--n-loops", type=int, default=300)
    ap.add_argument("--n-null", type=int, default=200)
    a = ap.parse_args()
    res = os.path.join(HERE, "rpe1_results", "circulation")
    os.makedirs(res, exist_ok=True)
    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X, pos = d["X_pca"], d["cycle_pos"]
    theta = TWO_PI * pos
    n = len(X)
    rng = np.random.default_rng(0)
    out = dict(args=vars(a), n=n, results={})
    for k in a.k:
        E = edges_from_knn(knn(X, k))
        loops = build_loops(X, pos, E, a.n_loops, a.bins, rng)
        r = {"ideal_phase_flow": stats(loops, wrap(theta[E[:, 1]] - theta[E[:, 0]]), a.bins)}
        r = {"ideal_phase_flow": dict(mean=r["ideal_phase_flow"][0], frac_pos=r["ideal_phase_flow"][1],
                                      per_segment=r["ideal_phase_flow"][2].tolist())}
        for name in ("V_pca_direct", "V_pca_kernel"):
            V = d[name].astype(np.float64)
            mean, frac, seg = stats(loops, edge_flow(X, V, E), a.bins)
            Vc = V - V.mean(0)
            null = np.array([stats(loops, edge_flow(X, Vc[rng.permutation(n)], E), a.bins)[0]
                             for _ in range(a.n_null)])
            r[name] = dict(mean=mean, frac_pos=frac, per_segment=seg.tolist(), null_mean=float(null.mean()),
                           null_sd=float(null.std()), z=float((mean - null.mean()) / null.std()),
                           p_one_sided=float((np.sum(null >= mean) + 1) / (len(null) + 1)))
            print(f"k={k} {name}: mean net/abs over {a.n_loops} loops {mean:+.3f} (positive in "
                  f"{frac:.0%}); null {null.mean():+.3f} ± {null.std():.3f}; z {r[name]['z']:+.1f}, "
                  f"p {r[name]['p_one_sided']:.3f}", flush=True)
        out["results"][f"k{k}"] = r
        json.dump(out, open(os.path.join(res, "circulation.json"), "w"), indent=1)
    write_report(out, res)


def write_report(out, res):
    bins = out["args"]["bins"]
    L = ["# RPE1-FUCCI: does the velocity circulate around the cycle?", "",
         f"{out['args']['n_loops']} closed loops per graph, each through one random cell per FUCCI phase "
         f"bin ({bins} bins), joined by shortest kNN-graph paths. net/abs = sum of the edge flow "
         "along the loop / sum of its absolute values; a gradient gives 0 on every loop. Null: centred "
         f"velocities permuted among cells ({out['args']['n_null']} times, same loops).", "",
         "| graph | flow | mean net/abs | loops with net > 0 | null | z | p (one-sided) |",
         "|---|---|---|---|---|---|---|"]
    for k, r in out["results"].items():
        for nm, o in r.items():
            if "null_mean" in o:
                L.append(f"| {k} | {nm} | {o['mean']:+.3f} | {o['frac_pos']:.0%} | "
                         f"{o['null_mean']:+.3f} ± {o['null_sd']:.3f} | {o['z']:+.1f} | {o['p_one_sided']:.3f} |")
            else:
                L.append(f"| {k} | {nm} (reference) | {o['mean']:+.3f} | {o['frac_pos']:.0%} | – | – | – |")
    L += ["", "## Per segment: net / abs flow from phase bin b to b+1 (summed over loops)", "",
          "Bin b covers FUCCI positions [b/12, (b+1)/12); the last segment closes the loop across the "
          "M → G1 seam.", "",
          "| graph | flow | " + " | ".join(f"{b}→{(b + 1) % bins}" for b in range(bins)) + " |",
          "|---|---|" + "---|" * bins]
    for k, r in out["results"].items():
        for nm, o in r.items():
            L.append(f"| {k} | {nm} | " + " | ".join(f"{v:+.2f}" for v in o["per_segment"]) + " |")
    open(os.path.join(res, "CIRCULATION.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "CIRCULATION.md"))


if __name__ == "__main__":
    main()
