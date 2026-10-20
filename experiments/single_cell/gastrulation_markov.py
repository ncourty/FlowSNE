"""Does a FlowSNE embedding carry a useful directed Markov chain? Scored
against REAL developmental time (mouse gastrulation, embryonic stage
E6.5-E8.5; see gastrulation_data.py).

Each kernel below becomes a row-stochastic Markov chain on the same cells:
  scvelo            scVelo's velocity transition matrix (the CellRank-style
                    velocity kernel; restricted to the kept cells)
  input_randers     FlowSNE's input affinity alone (row-stochastic Randers,
                    sparse), no embedding
  flowsne_d<d>_nu<v>  FlowSNE's output kernel in d dimensions with v degrees
                    of freedom: each cell's k nearest cells under the FITTED
                    Randers distance in the embedding, weighted by the Student
                    kernel (1 + d_F^2 / v)^(-(v+1)/2), row-normalised
  symmetric_d2      the same with a symmetric output kernel (no flow): a
                    direction-blind control

Scores: run the chain k steps from every cell (k in --steps) and measure the
embryonic days gained, mean(M^k stage - stage), and the fraction of cells
moving forward; plus a per-stage breakdown at k = 5. A correct directed chain
moves cells forward at every stage but the last. Even a direction-blind chain
drifts (toward dense regions, which are not spread evenly across stages), so
kernels are read RELATIVE to the symmetric control (reported too).

Ceiling: the input velocity's own forward fraction per cell type (do its
neighbours in the velocity direction come from later stages?). Where
scVelo's velocity points backwards (known for the erythroid lineage), no
velocity-based kernel can be right there.

Run (GPU, full data):  python experiments/single_cell/gastrulation_markov.py --subset full --n-cells 30000
Quick test (CPU):      python experiments/single_cell/gastrulation_markov.py --subset erythroid \\
                           --n-cells 1500 --iters 100 --configs 2:1 10:1 --tag test
Writes: experiments/single_cell/gastrulation_results/<tag>/{markov.json, RESULTS.md}
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
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, randers_knn  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_data import load  # noqa: E402


def velocity_ceiling(X, V, stage, celltype, k=30):
    """Per cell: stage gained toward neighbours in the velocity direction minus
    against it; per cell type, the fraction of cells pointing forward."""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    D = X[idx] - X[:, None]
    cos = (D * V[:, None]).sum(-1) / (np.linalg.norm(D, axis=-1) * np.linalg.norm(V, axis=1)[:, None]
                                      + 1e-12)
    dst = stage[idx] - stage[:, None]
    w, wb = np.clip(cos, 0, None), np.clip(-cos, 0, None)
    gain = (w * dst).sum(1) / np.maximum(w.sum(1), 1e-12) - (wb * dst).sum(1) / np.maximum(
        wb.sum(1), 1e-12)
    per_type = {c: float((gain[celltype == c] > 0).mean()) for c in np.unique(celltype)}
    return float((gain > 0).mean()), per_type


def chain_from_rows(values, indices, n):
    rows = np.repeat(np.arange(n), indices.shape[1])
    M = sp.csr_matrix((values.ravel(), (rows, indices.ravel())), shape=(n, n))
    rs = np.asarray(M.sum(1)).ravel()
    return (sp.diags(1.0 / np.clip(rs, 1e-300, None)) @ M).tocsr()


def chain_scores(M, stage, steps):
    x = stage.astype(np.float64).copy()
    out, k_done = {}, 0
    for k in sorted(steps):
        while k_done < k:
            x = M @ x
            k_done += 1
        adv = x - stage
        out[f"days_k{k}"] = float(adv.mean())
        out[f"forward_k{k}"] = float((adv > 0).mean())
        if k == 5:
            out["by_stage_k5"] = {f"E{s:g}": float(adv[stage == s].mean()) for s in np.unique(stage)}
    return out


def output_chain(aff, Z, k, dof, backend):
    """k nearest cells under the fitted Randers distance in the embedding,
    Student-weighted, row-normalised."""
    with torch.no_grad():
        omega = aff.omega_at(Z)
        d, idx = randers_knn(Z, omega, k, mode=aff.drift_mode, backend=backend)
        w = (1 + d ** 2 / dof) ** (-(dof + 1) / 2)
    n = Z.shape[0]
    return chain_from_rows(w.cpu().double().numpy(), idx.cpu().numpy(), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--configs", nargs="+", default=["2:1", "10:1", "10:9"],
                    help="embedding dimension:degrees of freedom")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--k-chain", type=int, default=30, help="neighbours per row of each chain")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    res = os.path.join(HERE, "gastrulation_results", a.tag)
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "markov.json")

    d = load(a.subset, n_cells=a.n_cells, seed=a.seed)
    ok = ~np.isnan(d["stage"])                          # drop cells with no parsable stage
    X_np, V_np, stage = d["X_pca"][ok].astype(np.float64), d["V_pca"][ok].astype(np.float64), d["stage"][ok]
    celltype = d["celltype"][ok]
    T = d["T_scvelo"][ok][:, ok]
    T = (sp.diags(1.0 / np.clip(np.asarray(T.sum(1)).ravel(), 1e-12, None)) @ T).tocsr()
    n = len(X_np)
    backend = "keops" if (cuda or n > 6000) else None
    print(f"{a.subset}: {n} cells, stages {sorted(set(np.unique(stage)))}, backend {backend}, "
          f"{dtype}", flush=True)

    ceiling, ceiling_by_type = velocity_ceiling(X_np, V_np, stage, celltype)
    print(f"input velocity points forward in time for {ceiling:.1%} of cells", flush=True)
    out = dict(args=vars(a), n=n, velocity_forward_fraction=ceiling,
               velocity_forward_by_celltype=ceiling_by_type, kernels={})

    def record(name, M, extra=None):
        s = chain_scores(M, stage, a.steps)
        out["kernels"][name] = dict(s, **(extra or {}))
        print(f"  {name:22s} " + "  ".join(f"{k} {v:+.3f}" for k, v in s.items()
                                           if not isinstance(v, dict)), flush=True)
        json.dump(out, open(path, "w"), indent=1)

    record("scvelo", T)
    X = torch.tensor(X_np, dtype=dtype, device=device)
    V = torch.tensor(V_np, dtype=dtype, device=device)
    perp = a.perplexity if a.perplexity == "auto" else float(a.perplexity)
    t0 = time.time()
    P, idx = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True,
                                     perplexity=perp, backend=backend)(X)
    print(f"  input affinity: k = {idx.shape[1]}  ({time.time() - t0:.0f}s)", flush=True)
    record("input_randers", chain_from_rows(P.cpu().double().numpy(), idx.cpu().numpy(), n))

    Xc = X_np - X_np.mean(0)
    pcs = np.linalg.svd(Xc, full_matrices=False)[2]
    runs = [("symmetric_d2", 2, 1.0, False)] + [
        (f"flowsne_d{c.split(':')[0]}_nu{c.split(':')[1]}", int(c.split(":")[0]),
         float(c.split(":")[1]), True) for c in a.configs]
    for name, dim, dof, train_flow in runs:
        torch.manual_seed(a.seed)
        flow = input_std_norm_flow(dim, hidden_dim=32) if train_flow else mlp_flow(dim)
        aff = RandersAffinity(n_components=dim, flow=flow, degrees_of_freedom=dof,
                              backend=backend).to(device=device, dtype=dtype)
        init = torch.tensor(Xc @ pcs[:dim].T, dtype=dtype, device=device)
        t0 = time.time()
        Z = fit_embedding(aff, P, init, indices=idx, max_iter=a.iters, lr=a.lr, seed=a.seed,
                          train_flow=train_flow)
        with torch.no_grad():
            loss = aff.cross_entropy(Z, P, idx).item()
        record(name, output_chain(aff, Z, a.k_chain, dof, backend),
               dict(loss=loss, fit_secs=round(time.time() - t0, 1)))
        np.save(os.path.join(res, f"Z_{name}.npy"), Z.cpu().float().numpy())

    write_report(out, res)


def write_report(out, res):
    ks = sorted(int(k[6:]) for k in next(iter(out["kernels"].values())) if k.startswith("days_k"))
    L = [f"# Gastrulation Markov chains vs real stage ({out['args']['subset']}, n = {out['n']})", "",
         f"Input velocity points forward in time for **{out['velocity_forward_fraction']:.1%}** of "
         "cells (ceiling for any velocity-based kernel). Per cell type: " +
         ", ".join(f"{c} {v:.0%}" for c, v in sorted(out["velocity_forward_by_celltype"].items(),
                                                     key=lambda x: x[1])), "",
         "Days gained after k steps (mean over cells), the same relative to the direction-blind "
         "symmetric control (Δ sym), and the fraction of cells moving forward:", "",
         "| kernel | " + " | ".join(f"days k={k} | Δ sym k={k} | fwd k={k}" for k in ks) + " |",
         "|---|" + "---|---|---|" * len(ks)]
    ctrl = out["kernels"].get("symmetric_d2")
    for name, r in out["kernels"].items():
        L.append(f"| {name} | " + " | ".join(
            f"{r[f'days_k{k}']:+.3f} | "
            + (f"{r[f'days_k{k}'] - ctrl[f'days_k{k}']:+.3f}" if ctrl else "–")
            + f" | {r[f'forward_k{k}']:.2f}" for k in ks) + " |")
    L += ["", "Per-stage days gained at k = 5:", ""]
    stages = list(next(iter(out["kernels"].values()))["by_stage_k5"])
    L += ["| kernel | " + " | ".join(stages) + " |", "|---|" + "---|" * len(stages)]
    for name, r in out["kernels"].items():
        L.append(f"| {name} | " + " | ".join(f"{r['by_stage_k5'][s]:+.3f}" for s in stages) + " |")
    open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "RESULTS.md"))


if __name__ == "__main__":
    main()
