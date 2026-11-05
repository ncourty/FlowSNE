"""Real-data test of the Helmholtz theory (article/ICML, Section 6).

Prediction: a Sinkhorn-balanced Randers affinity keeps (mostly) the
non-gradient part of a velocity field and drops its gradient part, while a
row-stochastic one keeps both. So, relative to each normalisation's own
direction-blind control,
  * on a developmental trajectory (pancreas, velocity mostly gradient), the
    balanced affinity should lose most of the forward signal;
  * on the cell cycle (RPE1-FUCCI, velocity circulating), it should keep it.

For each dataset / velocity, cost power q in {1, 2} and normalisation
(balanced "DS", row-stochastic "RS"), with flowsne.EntropicRandersAffinity
(dense, perplexity sqrt(n), per-row bandwidths):
  shift     mean outgoing displacement of the drifted affinity minus that of
            the symmetric one (same q, same normalisation), per cell;
            reported: cosine with the velocity, and with its graph-Hodge
            gradient part and remainder (30-NN graph, gastrulation_hodge.py)
  progress  walk on the affinity truncated to its 30 largest entries per row:
            ground-truth progress after s steps minus the symmetric
            affinity's (pancreas: pseudotime; RPE1: unwrapped FUCCI phase)

Run:    conda run -n dev python experiments/theory/balanced_vs_rs.py
Writes: experiments/theory/results/balanced_vs_rs.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "..", "single_cell")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SC)

from flowsne import EntropicRandersAffinity  # noqa: E402
from gastrulation_hodge import cell_vectors, edge_flow, edges_from_knn, hodge, incidence  # noqa: E402
from gastrulation_repr import knn  # noqa: E402
from rpe1_hodge import TWO_PI, phase_progress  # noqa: E402

OUT = os.path.join(HERE, "results")


def datasets(subsample=None):
    d = np.load(os.path.join(SC, "data", "pancreas.npz"), allow_pickle=True)
    pan = dict(X=d["X_pca"].astype(np.float64), V=d["V_pca"].astype(np.float64),
               truth=d["pseudotime"].astype(np.float64), kind="linear")
    r = np.load(os.path.join(SC, "data", "rpe1.npz"))
    base = dict(X=r["X_pca"].astype(np.float64), truth=TWO_PI * r["cycle_pos"], kind="circle")
    out = {"pancreas": pan,
           "rpe1_direct": dict(base, V=r["V_pca_direct"].astype(np.float64)),
           "rpe1_kernel": dict(base, V=r["V_pca_kernel"].astype(np.float64))}
    if subsample:
        rng = np.random.default_rng(0)
        for k, v in out.items():
            idx = np.sort(rng.choice(len(v["X"]), min(subsample, len(v["X"])), replace=False))
            out[k] = {kk: (vv[idx] if isinstance(vv, np.ndarray) else vv) for kk, vv in v.items()}
    return out


def affinity(X, V, ds, q):
    aff = EntropicRandersAffinity(perplexity="auto", omega=None if V is None else torch.tensor(V),
                                  doubly_stochastic=ds, cost_power=q)
    P = aff(torch.tensor(X)).detach().double().numpy()
    err = float(getattr(aff, "marginal_error_", float("nan")))
    return P / P.sum(1, keepdims=True), err


def shift(P, P0, X):
    """(P - P0) mean displacement per cell."""
    return (P - P0) @ X - ((P - P0).sum(1))[:, None] * X


def truncate(P, k):
    idx = np.argpartition(-P, k, axis=1)[:, :k]
    vals = np.take_along_axis(P, idx, 1)
    n = len(P)
    M = sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    rs = np.asarray(M.sum(1)).ravel()
    return (sp.diags(1 / rs) @ M).tocsr()


def progress(M, truth, kind, steps):
    if kind == "circle":
        return {s: float(v.mean()) for s, v in phase_progress(M, truth, steps).items()}
    x, out = truth.copy(), {}
    for s in range(1, max(steps) + 1):
        x = M @ x
        if s in steps:
            out[s] = float((x - truth).mean())
    return out


def cos_global(A, B):
    """Mean over cells of cos(A_i, B_i), cells with non-negligible B."""
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    ok = (nb > np.quantile(nb, 0.2)) & (na > 0)
    return float(((A * B).sum(1)[ok] / (na[ok] * nb[ok])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--k-chain", type=int, default=30)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--powers", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = {"args": vars(a), "data": {}}
    path = os.path.join(OUT, f"balanced_vs_rs{a.tag}.json")
    sym_cache = {}
    for name, d in datasets(a.subsample).items():
        t0 = time.time()
        X, V, truth = d["X"], d["V"], d["truth"]
        n = len(X)
        E = edges_from_knn(knn(X, 30))
        f = edge_flow(X, V, E)
        _, g, rest = hodge(incidence(E, n), f)
        Vg, Vr = cell_vectors(X, E, g), cell_vectors(X, E, rest)
        r = {"n": n, "gradient_share": float((g ** 2).sum() / (f ** 2).sum()), "fits": {}}
        print(f"[{name}] n={n} gradient share {r['gradient_share']:.3f}", flush=True)
        for q in a.powers:
            for ds in (True, False):
                key = (d["kind"], n, q, ds, name.split("_")[0])
                if key not in sym_cache:
                    sym_cache[key] = affinity(X, None, ds, q)
                P0, e0 = sym_cache[key]
                P, e = affinity(X, V, ds, q)
                S = shift(P, P0, X)
                Mt, M0t = truncate(P, a.k_chain), truncate(P0, a.k_chain)
                p, p0 = (progress(M, truth, d["kind"], a.steps) for M in (Mt, M0t))
                pd, pd0 = (progress(sp.csr_matrix(M), truth, d["kind"], a.steps) for M in (P, P0))
                St = (Mt - M0t) @ X - np.asarray((Mt - M0t).sum(1)) * X
                lab = f"{'DS' if ds else 'RS'}_q{q:g}"
                r["fits"][lab] = dict(
                    marginal_error=e, cos_velocity=cos_global(S, V), cos_gradient=cos_global(S, Vg),
                    cos_rest=cos_global(S, Vr), shift_norm=float(np.linalg.norm(S, axis=1).mean()),
                    progress={s: p[s] - p0[s] for s in a.steps},
                    progress_dense={s: pd[s] - pd0[s] for s in a.steps},
                    cos_velocity_trunc=cos_global(St, V), cos_gradient_trunc=cos_global(St, Vg),
                    cos_rest_trunc=cos_global(St, Vr))
                print(f"  {lab:8s} {r['fits'][lab]}", flush=True)
        r["seconds"] = time.time() - t0
        res["data"][name] = r
        json.dump(res, open(path, "w"), indent=1)
    write_report(res, path.replace(".json", ".md"))


def write_report(res, out):
    steps = res["args"]["steps"]
    L = ["# Balanced vs row-stochastic Randers affinities on real velocity fields", "",
         "See `balanced_vs_rs.py`. `cos` = mean cosine between each affinity's shift in mean "
         "outgoing displacement (drifted minus symmetric, same normalisation and cost power) and "
         "the velocity / its graph-Hodge gradient part / its remainder. `progress` = ground-truth "
         "progress of the 30-truncated walk minus the symmetric walk's (pancreas: pseudotime; "
         "RPE1: fraction of a cycle).", ""]
    for name, r in res["data"].items():
        L += [f"## {name} (n = {r['n']}, gradient share of the edge flow {r['gradient_share']:.3f})", "",
              "| affinity | cos velocity | cos gradient | cos rest | |shift| | " +
              " | ".join(f"progress s={s}" for s in steps) + " | marginal err |",
              "|---" * (6 + len(steps)) + "|"]
        for lab, v in r["fits"].items():
            L.append(f"| {lab} | {v['cos_velocity']:+.3f} | {v['cos_gradient']:+.3f} | {v['cos_rest']:+.3f} | "
                     f"{v['shift_norm']:.3g} | " + " | ".join(f"{v['progress'][s]:+.4f}" for s in steps) +
                     f" | {v['marginal_error']:.1e} |")
        L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
