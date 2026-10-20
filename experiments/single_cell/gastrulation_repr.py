"""Given cells and their RNA velocities, does a d-dimensional FlowSNE
representation preserve directed structure better than standard
representations do? Scored against REAL developmental time (mouse
gastrulation, embryonic stage E6.5-E8.5).

Replaces gastrulation_markov.py's comparison with scVelo's transition matrix,
which was circular (the velocities FlowSNE received were themselves computed
from that matrix). Here every representation gets the SAME independent drift:
scVelo's gene-space velocity, converted to log space and projected linearly
on our own PCA (``V_pca_direct``, see gastrulation_data.py).

Information checks first: how much of the velocity survives the PCA (share
of its norm in the first 30/50/100 PCs, from preprocessing) and whether it
still points forward in time (fraction of cells whose velocity points toward
later-stage neighbours) -- for scVelo's transition-matrix projection, and for
the direct projection at 30 and 100 PCs.

Representations, at d = 2 and 10 (positions Z, one direction f per cell):
  input30      no reduction: the 30-PC positions, f = direct velocity
  pca<d>       first d PCs, f = direct velocity's first d components (exact)
  sym<d>       symmetric neighbour embedding (no drift anywhere), f = the
               velocity projected onto it scVelo-style: sum_j p_ij u_ij -
               mean_j u_ij, u_ij the unit embedded displacement to input-space
               neighbour j, p_ij = softmax_j(10 cos(v_i, x_j - x_i))
  flowsne<d>   FlowSNE (Randers input affinity with the direct drift, flow
               fitted jointly, nu = 1), f = the fitted field -omega(z)

One chain rule for all, so only the representation varies: each cell's k
nearest cells IN THAT REPRESENTATION, weighted by exp(10 cos(f_i, z_j - z_i))
(scVelo's softmax form), row-normalised. Each representation also gets its
own direction-blind control (same neighbours, uniform weights); scores are
days gained relative to that control, so density effects cancel within each
representation. FlowSNE's native chain (its fitted Randers kernel) is
reported too, against the same control.

Scores: embryonic days gained after s steps (s in --steps) relative to the
control, on all cells and on winnable cell types (direct velocity forward
for at least half their cells), and the fraction of cells moving forward;
mean +- std over --seeds (seeds only change FlowSNE's flow initialisation).

Run (GPU):        python experiments/single_cell/gastrulation_repr.py --subset full --n-cells 30000
Quick test (CPU): python experiments/single_cell/gastrulation_repr.py --subset erythroid --n-cells 1500 \\
                      --iters 60 --seeds 0 --k-chain 30 --tag test
Writes: experiments/single_cell/gastrulation_results/<tag>/{repr.json, RESULTS.md, Z_*.npy, flow_*.pt}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_data import load  # noqa: E402
from gastrulation_markov import advances, normalise, output_chain, velocity_forward  # noqa: E402

SCALE = 10.0           # softmax scale of the cosine kernel (scVelo's form)


def knn(Z, k):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]


def cosine_rows(F, D):
    """cos(F_i, D_ij) for D (n, k, d)."""
    den = np.linalg.norm(D, axis=-1) * np.linalg.norm(F, axis=1)[:, None]
    return np.where(den > 0, (D * F[:, None]).sum(-1) / np.where(den > 0, den, 1), 0.0)


def chain(Z, F, k, directed=True):
    """k nearest cells in Z, weighted by exp(SCALE cos(F_i, z_j - z_i)) (or
    uniformly for the direction-blind control), row-normalised."""
    idx = knn(Z, k)
    w = np.exp(SCALE * cosine_rows(F, Z[idx] - Z[:, None])) if directed else np.ones(idx.shape)
    n = len(Z)
    return normalise(sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())),
                                   shape=(n, n)))


def project_velocity(X, V, Z, k=30):
    """scVelo-style projection of input-space velocities V onto an embedding Z:
    sum_j p_ij u_ij - mean_j u_ij, u_ij the unit embedded displacement to the
    input-space neighbour j, p_ij = softmax_j(SCALE cos(v_i, x_j - x_i))."""
    idx = knn(X, k)
    p = np.exp(SCALE * cosine_rows(V, X[idx] - X[:, None]))
    p /= p.sum(1, keepdims=True)
    U = Z[idx] - Z[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=-1, keepdims=True), 1e-12)
    return (p[..., None] * U).sum(1) - U.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-chain", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--tag", default="repr")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    res = os.path.join(HERE, "gastrulation_results", a.tag)
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "repr.json")

    d = load(a.subset, n_cells=a.n_cells, seed=0)
    ok = ~np.isnan(d["stage"])
    X = d["X_pca"][ok].astype(np.float64)
    X100 = d["X_pca100"][ok].astype(np.float64)
    Vd100 = d["V_pca_direct"][ok].astype(np.float64)
    V = Vd100[:, :X.shape[1]]                                  # direct velocity, 30 PCs
    stage, celltype = d["stage"][ok].astype(np.float64), d["celltype"][ok]
    n = len(X)
    backend = "keops" if (cuda or n > 6000) else None

    # --- information checks
    info = dict(vel_share_median={p: float(np.median(d[f"vel_share_{p}"][ok])) for p in (30, 50, 100)},
                cos_T_direct_median=float(np.median(d["cos_T_direct"][ok])))
    fwd = {"scvelo_T_projection_30pc": velocity_forward(X, d["V_pca"][ok].astype(np.float64), stage),
           "direct_30pc": velocity_forward(X, V, stage),
           "direct_100pc": velocity_forward(X100, Vd100, stage)}
    info["forward_fraction"] = {k: float((v > 0).mean()) for k, v in fwd.items()}
    by_type = {c: float((fwd["direct_30pc"][celltype == c] > 0).mean()) for c in np.unique(celltype)}
    winnable = np.isin(celltype, [c for c, f in by_type.items() if f >= 0.5])
    info.update(forward_by_celltype=by_type, winnable_fraction=float(winnable.mean()))
    print(f"{a.subset}: {n} cells; velocity share in 30/50/100 PCs "
          f"{'/'.join(f'{v:.2f}' for v in info['vel_share_median'].values())}; forward fractions "
          + ", ".join(f"{k} {v:.1%}" for k, v in info["forward_fraction"].items()), flush=True)

    out = dict(args=vars(a), n=n, info=info, scores={})
    if os.path.exists(path):
        out["scores"] = json.load(open(path)).get("scores", {})

    def score(name, k, seed, M, M0):
        adv, adv0 = advances(M, stage, a.steps), advances(M0, stage, a.steps)
        r = {}
        for s in adv:
            rel = adv[s] - adv0[s]
            r[f"rel_s{s}"] = float(rel.mean())
            r[f"rel_s{s}_winnable"] = float(rel[winnable].mean())
            r[f"fwd_s{s}"] = float((adv[s] > 0).mean())
        if 5 in adv:
            rel5 = adv[5] - adv0[5]
            r["by_celltype_s5"] = {c: float(rel5[celltype == c].mean()) for c in np.unique(celltype)}
        out["scores"].setdefault(name, {}).setdefault(f"k{k}", {})[f"seed{seed}"] = r
        json.dump(out, open(path, "w"), indent=1)
        print(f"  k={k:4d} seed={seed} {name:16s} " + "  ".join(
            f"s{s} {r[f'rel_s{s}']:+.3f}" for s in sorted(adv)), flush=True)

    # --- representations without fitting
    for k in a.k_chain:
        score("input30", k, 0, chain(X, V, k), chain(X, V, k, directed=False))
        for dim in a.dims:
            Z = X[:, :dim]
            score(f"pca{dim}", k, 0, chain(Z, V[:, :dim], k), chain(Z, V[:, :dim], k, directed=False))

    Xt = torch.tensor(X, dtype=dtype, device=device)
    Vt = torch.tensor(V, dtype=dtype, device=device)
    perp = a.perplexity if a.perplexity == "auto" else float(a.perplexity)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=perp, backend=backend)
    t0 = time.time()
    P, idx = EntropicRandersAffinity(omega=Vt, **kw)(Xt)
    P0, idx0 = EntropicRandersAffinity(omega=None, **kw)(Xt)
    print(f"  input affinities ({idx.shape[1]} neighbours per row)  ({time.time() - t0:.0f}s)", flush=True)
    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]

    def fitted(name, dim, seed, Pm, im, train_flow):
        torch.manual_seed(seed)
        flow = input_std_norm_flow(dim, hidden_dim=32) if train_flow else mlp_flow(dim)
        aff = RandersAffinity(n_components=dim, flow=flow, degrees_of_freedom=1.0,
                              backend=backend).to(device=device, dtype=dtype)
        zf, ff = os.path.join(res, f"Z_{name}_s{seed}.npy"), os.path.join(res, f"flow_{name}_s{seed}.pt")
        if os.path.exists(zf) and os.path.exists(ff):
            Z = torch.tensor(np.load(zf), dtype=dtype, device=device)
            aff.flow.load_state_dict(torch.load(ff, map_location=device))
        else:
            t1 = time.time()
            init = torch.tensor((X - X.mean(0)) @ pcs[:dim].T, dtype=dtype, device=device)
            Z = fit_embedding(aff, Pm, init, indices=im, max_iter=a.iters, lr=a.lr, seed=seed,
                              train_flow=train_flow)
            np.save(zf, Z.cpu().float().numpy())
            torch.save(aff.flow.state_dict(), ff)
            print(f"  {name} seed {seed}: fitted in {time.time() - t1:.0f}s", flush=True)
        return aff, Z

    for dim in a.dims:
        _, Zs = fitted(f"sym{dim}", dim, 0, P0, idx0, False)       # deterministic
        Zs_np = Zs.cpu().double().numpy()
        Fs = project_velocity(X, V, Zs_np)
        for k in a.k_chain:
            score(f"sym{dim}", k, 0, chain(Zs_np, Fs, k), chain(Zs_np, Fs, k, directed=False))
        for seed in a.seeds:
            aff, Z = fitted(f"flowsne{dim}", dim, seed, P, idx, True)
            Z_np = Z.cpu().double().numpy()
            with torch.no_grad():
                F = (-aff.omega_at(Z)).cpu().double().numpy()
            for k in a.k_chain:
                M0 = chain(Z_np, F, k, directed=False)
                score(f"flowsne{dim}", k, seed, chain(Z_np, F, k), M0)
                score(f"flowsne{dim}_native", k, seed, output_chain(aff, Z, k, 1.0, backend), M0)
    write_report(out, res)


def _agg(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():+.3f}" if len(v) == 1 else f"{v.mean():+.3f} ± {v.std():.3f}"


def write_report(out, res):
    S, info = out["scores"], out["info"]
    ks = sorted({int(k[1:]) for nm in S for k in S[nm]})
    steps = sorted({int(key.split("_s")[1]) for nm in S for k in S[nm] for sd in S[nm][k]
                    for key in S[nm][k][sd] if key.startswith("rel_s") and "winnable" not in key})
    L = [f"# Gastrulation: which representation preserves direction? ({out['args']['subset']}, "
         f"n = {out['n']})", "",
         "## Information checks", "",
         "Share of the (log-space) velocity's norm inside the first 30 / 50 / 100 PCs (median over "
         "cells): " + " / ".join(f"{v:.2f}" for v in info["vel_share_median"].values()) +
         f". Cosine between scVelo's transition-matrix projection and the direct projection "
         f"(median): {info['cos_T_direct_median']:+.2f}.", "",
         "Fraction of cells whose velocity points toward later-stage neighbours:", "",
         "| velocity | forward |", "|---|---|"]
    for k, v in info["forward_fraction"].items():
        L.append(f"| {k} | {v:.1%} |")
    L += ["", f"Winnable cell types (direct velocity forward for ≥ 50% of their cells) cover "
          f"{info['winnable_fraction']:.1%} of cells.", "",
          "## Days gained relative to each representation's own direction-blind control", "",
          "Same chain rule for every representation (k nearest cells in it, weights exp(10 cos)); "
          "`_native` = FlowSNE's own Randers kernel. Mean ± std over seeds.", ""]
    for k in ks:
        L += [f"### k = {k}", "",
              "| representation | " + " | ".join(f"s={s}" for s in steps) + " | " +
              " | ".join(f"s={s}, winnable" for s in steps) + f" | fwd s={steps[-1]} |",
              "|---|" + "---|" * (2 * len(steps) + 1)]
        for nm in S:
            rs = list(S[nm].get(f"k{k}", {}).values())
            if not rs:
                continue
            L.append(f"| {nm} | " + " | ".join(_agg([r[f'rel_s{s}'] for r in rs]) for s in steps) + " | "
                     + " | ".join(_agg([r[f'rel_s{s}_winnable'] for r in rs]) for s in steps)
                     + f" | {_agg([r[f'fwd_s{steps[-1]}'] for r in rs])} |")
        L.append("")
    open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "RESULTS.md"))


if __name__ == "__main__":
    main()
