"""SUPERSEDED by gastrulation_repr.py: the scVelo baseline here is circular
(FlowSNE's input drift is derived from scVelo's own transition matrix). Kept
for its chain helpers, which gastrulation_repr.py imports.

Does a FlowSNE embedding carry a useful directed Markov chain? Scored
against REAL developmental time (mouse gastrulation, embryonic stage
E6.5-E8.5; see gastrulation_data.py).

Kernels, each made a row-stochastic Markov chain on the same cells:
  scvelo            scVelo's velocity transition matrix (the CellRank-style
                    velocity kernel; restricted to the kept cells)
  input_randers     FlowSNE's input affinity alone (row-stochastic Randers,
                    sparse), no embedding
  flowsne_d<d>_nu<v>  FlowSNE's output kernel in d dimensions with v degrees
                    of freedom: each cell's nearest cells under the FITTED
                    Randers distance in the embedding, weighted by the Student
                    kernel (1 + d_F^2 / v)^(-(v+1)/2), row-normalised
  symmetric_d2      the same with a symmetric output kernel (no flow): a
                    direction-blind control

Fairness. "Days gained per step" grows with how far a step jumps, i.e. with
a chain's row support. So every chain is truncated to the SAME k neighbours
per row (its k largest entries, renormalised), for each k in --k-chain; and
days per unit of jump (days gained in one step divided by the mean step
length in PCA space) is reported as a support-free measure. scVelo's native
row sizes are reported too.

Scores, per k-chain and seed (mean +- std over --seeds in the report): run
the chain s steps from every cell (s in --steps) and measure the embryonic
days gained, mean(M^s stage - stage), raw and relative to the symmetric
control (which also drifts: toward dense regions, and at the boundary
stages); the fraction of cells moving forward; the same restricted to
WINNABLE cell types (where the input velocity itself points forward for at
least half the cells -- elsewhere, e.g. the erythroid lineage, no
velocity-based kernel can be right); and per cell type.

Each fit's embedding and flow network are saved; a re-run resumes from them.

Run (GPU, full data):  python experiments/single_cell/gastrulation_markov.py --subset full --n-cells 30000
Quick test (CPU):      python experiments/single_cell/gastrulation_markov.py --subset erythroid \\
                           --n-cells 1500 --iters 60 --configs 2:1 10:1 --seeds 0 1 \\
                           --k-chain 10 30 --tag test
Writes: experiments/single_cell/gastrulation_results/<tag>/{markov.json, RESULTS.md,
        Z_*.npy, flow_*.pt}
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

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, randers_knn  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_data import load  # noqa: E402


# ===========================================================================
# Chains
# ===========================================================================

def velocity_forward(X, V, stage, k=30):
    """Per cell: stage gained toward neighbours in the velocity direction minus
    against it (> 0: the velocity points forward in time)."""
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    D = X[idx] - X[:, None]
    cos = (D * V[:, None]).sum(-1) / (np.linalg.norm(D, axis=-1) * np.linalg.norm(V, axis=1)[:, None]
                                      + 1e-12)
    dst = stage[idx] - stage[:, None]
    w, wb = np.clip(cos, 0, None), np.clip(-cos, 0, None)
    return (w * dst).sum(1) / np.maximum(w.sum(1), 1e-12) - (wb * dst).sum(1) / np.maximum(
        wb.sum(1), 1e-12)


def normalise(M):
    rs = np.asarray(M.sum(1)).ravel()
    return (sp.diags(1.0 / np.clip(rs, 1e-300, None)) @ M).tocsr()


def chain_from_rows(values, indices, n):
    rows = np.repeat(np.arange(n), indices.shape[1])
    return normalise(sp.csr_matrix((values.ravel(), (rows, indices.ravel())), shape=(n, n)))


def truncate_rows(M, k):
    """Keep each row's k largest entries (all of them if it has fewer), renormalise."""
    M = M.tocsr()
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        v, c = M.data[a:b], M.indices[a:b]
        if len(v) > k:
            top = np.argpartition(-v, k - 1)[:k]
            v, c = v[top], c[top]
        rows.append(np.full(len(v), i)); cols.append(c); vals.append(v)
    return normalise(sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                            np.concatenate(cols))), shape=M.shape))


def output_chain(aff, Z, k, dof, backend):
    """k nearest cells under the fitted Randers distance in the embedding,
    Student-weighted, row-normalised."""
    with torch.no_grad():
        d, idx = randers_knn(Z, aff.omega_at(Z), k, mode=aff.drift_mode, backend=backend)
        w = (1 + d ** 2 / dof) ** (-(dof + 1) / 2)
    return chain_from_rows(w.cpu().double().numpy(), idx.cpu().numpy(), Z.shape[0])


def mean_jump(M, X):
    """Mean step length in PCA space: sum_ij M_ij ||x_j - x_i|| / n."""
    M = M.tocoo()
    return float((M.data * np.linalg.norm(X[M.col] - X[M.row], axis=1)).sum() / M.shape[0])


def advances(M, stage, steps):
    """{s: M^s stage - stage} for each s in steps."""
    x, out, done = stage.astype(np.float64).copy(), {}, 0
    for s in sorted(steps):
        while done < s:
            x = M @ x
            done += 1
        out[s] = x - stage
    return out


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--configs", nargs="+", default=["2:1", "10:1", "10:9"],
                    help="embedding dimension:degrees of freedom")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-chain", type=int, nargs="+", default=[30, 100, 300],
                    help="neighbours per row, the same for every chain")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--tag", default="fair")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    res = os.path.join(HERE, "gastrulation_results", a.tag)
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "markov.json")

    d = load(a.subset, n_cells=a.n_cells, seed=0)              # same cells for every seed
    ok = ~np.isnan(d["stage"])
    X_np, V_np, stage = (d["X_pca"][ok].astype(np.float64), d["V_pca"][ok].astype(np.float64),
                         d["stage"][ok].astype(np.float64))
    celltype = d["celltype"][ok]
    T = normalise(d["T_scvelo"][ok][:, ok])
    n = len(X_np)
    backend = "keops" if (cuda or n > 6000) else None
    nnz = np.diff(T.indptr)
    print(f"{a.subset}: {n} cells, backend {backend}, {dtype}; scVelo kernel row sizes: median "
          f"{np.median(nnz):.0f}, 90% {np.percentile(nnz, 90):.0f}, max {nnz.max()}", flush=True)

    gain = velocity_forward(X_np, V_np, stage)
    by_type = {c: float((gain[celltype == c] > 0).mean()) for c in np.unique(celltype)}
    winnable = np.isin(celltype, [c for c, f in by_type.items() if f >= 0.5])
    print(f"input velocity forward for {(gain > 0).mean():.1%} of cells; winnable cell types "
          f"(>= 50% forward) cover {winnable.mean():.1%} of cells", flush=True)
    out = dict(args=vars(a), n=n, velocity_forward_fraction=float((gain > 0).mean()),
               velocity_forward_by_celltype=by_type, winnable_fraction=float(winnable.mean()),
               scvelo_row_nnz=dict(median=float(np.median(nnz)), p90=float(np.percentile(nnz, 90)),
                                   max=int(nnz.max())),
               scores={})
    if os.path.exists(path):
        out["scores"] = json.load(open(path)).get("scores", {})

    def score(name, k, seed, M):
        adv = advances(M, stage, a.steps)
        r = dict(jump=mean_jump(M, X_np))
        for s, v in adv.items():
            r[f"days_s{s}"] = float(v.mean())
            r[f"fwd_s{s}"] = float((v > 0).mean())
            r[f"days_s{s}_winnable"] = float(v[winnable].mean())
        r["by_celltype_s5"] = {c: float(adv[5][celltype == c].mean()) for c in np.unique(celltype)} \
            if 5 in adv else {}
        out["scores"].setdefault(name, {}).setdefault(f"k{k}", {})[f"seed{seed}"] = r
        json.dump(out, open(path, "w"), indent=1)
        print(f"  k={k:4d} seed={seed} {name:18s} days/step {r['days_s1']:+.4f}  jump {r['jump']:.3f}  "
              + "  ".join(f"s{s} {r[f'days_s{s}']:+.3f}" for s in sorted(a.steps)), flush=True)

    X = torch.tensor(X_np, dtype=dtype, device=device)
    V = torch.tensor(V_np, dtype=dtype, device=device)
    perp = a.perplexity if a.perplexity == "auto" else float(a.perplexity)
    t0 = time.time()
    P, idx = EntropicRandersAffinity(omega=V, doubly_stochastic=False, sparsity=True,
                                     perplexity=perp, backend=backend)(X)
    P_chain = chain_from_rows(P.cpu().double().numpy(), idx.cpu().numpy(), n)
    print(f"  input affinity: {idx.shape[1]} neighbours per row  ({time.time() - t0:.0f}s)", flush=True)
    for k in a.k_chain:                                  # deterministic kernels: seed-independent
        score("scvelo", k, 0, truncate_rows(T, k))
        score("input_randers", k, 0, truncate_rows(P_chain, k))

    Xc = X_np - X_np.mean(0)
    pcs = np.linalg.svd(Xc, full_matrices=False)[2]
    runs = [("symmetric_d2", 2, 1.0, False)] + [
        (f"flowsne_d{c.split(':')[0]}_nu{c.split(':')[1]}", int(c.split(":")[0]),
         float(c.split(":")[1]), True) for c in a.configs]
    for seed in a.seeds:
        for name, dim, dof, train_flow in runs:
            torch.manual_seed(seed)
            flow = input_std_norm_flow(dim, hidden_dim=32) if train_flow else mlp_flow(dim)
            aff = RandersAffinity(n_components=dim, flow=flow, degrees_of_freedom=dof,
                                  backend=backend).to(device=device, dtype=dtype)
            zf = os.path.join(res, f"Z_{name}_s{seed}.npy")
            ff = os.path.join(res, f"flow_{name}_s{seed}.pt")
            if os.path.exists(zf) and os.path.exists(ff):          # resume
                Z = torch.tensor(np.load(zf), dtype=dtype, device=device)
                aff.flow.load_state_dict(torch.load(ff, map_location=device))
                print(f"  {name} seed {seed}: loaded saved fit", flush=True)
            else:
                init = torch.tensor(Xc @ pcs[:dim].T, dtype=dtype, device=device)
                t0 = time.time()
                Z = fit_embedding(aff, P, init, indices=idx, max_iter=a.iters, lr=a.lr, seed=seed,
                                  train_flow=train_flow)
                np.save(zf, Z.cpu().float().numpy())
                torch.save(aff.flow.state_dict(), ff)
                print(f"  {name} seed {seed}: fitted in {time.time() - t0:.0f}s", flush=True)
            for k in a.k_chain:
                score(name, k, seed, output_chain(aff, Z, k, dof, backend))
    write_report(out, res)


# ===========================================================================
# Report
# ===========================================================================

def _agg(vals):
    v = np.array(vals, dtype=float)
    return f"{v.mean():+.3f}" if len(v) == 1 else f"{v.mean():+.3f} ± {v.std():.3f}"


def write_report(out, res):
    S = out["scores"]
    ks = sorted({int(k[1:]) for name in S for k in S[name]})
    steps = sorted({int(key.split("_s")[1]) for name in S for k in S[name] for sd in S[name][k]
                    for key in S[name][k][sd] if key.startswith("days_s") and "winnable" not in key})
    names = list(S)
    ctrl = "symmetric_d2"

    def vals(name, k, key):
        return [r[key] for r in S[name].get(f"k{k}", {}).values() if key in r]

    def rel(name, k, key):
        c = np.mean(vals(ctrl, k, key)) if ctrl in S else 0.0
        return [v - c for v in vals(name, k, key)]

    by_type = out["velocity_forward_by_celltype"]
    L = [f"# Gastrulation Markov chains vs real stage — equal-support comparison "
         f"({out['args']['subset']}, n = {out['n']})", "",
         f"Input velocity points forward in time for **{out['velocity_forward_fraction']:.1%}** of "
         f"cells; winnable cell types (velocity forward for ≥ 50% of their cells) cover "
         f"{out['winnable_fraction']:.1%} of cells. scVelo kernel's native row sizes: median "
         f"{out['scvelo_row_nnz']['median']:.0f}, 90th percentile {out['scvelo_row_nnz']['p90']:.0f}, "
         f"max {out['scvelo_row_nnz']['max']}.", "",
         "Every chain is truncated to the same k neighbours per row. Mean ± std over seeds; seeds "
         "only change FlowSNE's flow initialisation (layouts start from PCA), so scVelo, the input "
         "affinity and the symmetric control are deterministic. Δ sym = days gained minus the "
         "direction-blind symmetric control's; days/jump = days gained in one step per unit of mean "
         "step length (PCA space).", ""]
    for k in ks:
        L += [f"## k = {k} neighbours per row", "",
              "| kernel | days/jump | " + " | ".join(f"Δ sym s={s}" for s in steps) + " | " +
              " | ".join(f"Δ sym s={s}, winnable" for s in steps) + f" | fwd s={steps[-1]} |",
              "|---|---|" + "---|" * (2 * len(steps) + 1)]
        for name in names:
            dj = [r["days_s1"] / max(r["jump"], 1e-12) for r in S[name].get(f"k{k}", {}).values()]
            if not dj:
                continue
            L.append(f"| {name} | {_agg(dj)} | "
                     + " | ".join(_agg(rel(name, k, f"days_s{s}")) for s in steps) + " | "
                     + " | ".join(_agg(rel(name, k, f"days_s{s}_winnable")) for s in steps)
                     + f" | {_agg(vals(name, k, f'fwd_s{steps[-1]}'))} |")
        L.append("")
    if 5 in steps and ks:
        k = ks[0]
        types = sorted(by_type, key=lambda c: by_type[c])
        L += [f"## Per cell type, k = {k}, 5 steps: days gained relative to the control", "",
              "| cell type | velocity forward | " + " | ".join(n_ for n_ in names if n_ != ctrl) + " |",
              "|---|---|" + "---|" * (len(names) - 1)]
        for c in types:
            row = []
            for name in names:
                if name == ctrl:
                    continue
                v = [r["by_celltype_s5"][c] for r in S[name][f"k{k}"].values()]
                cv = np.mean([r["by_celltype_s5"][c] for r in S[ctrl][f"k{k}"].values()])
                row.append(f"{np.mean(v) - cv:+.3f}")
            L.append(f"| {c} | {by_type[c]:.0%} | " + " | ".join(row) + " |")
        L.append("")
    open(os.path.join(res, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "RESULTS.md"))


if __name__ == "__main__":
    main()
