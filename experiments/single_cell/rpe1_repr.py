"""Cell cycle (RPE1-FUCCI): does a FlowSNE map keep the velocity's
circulation better than a symmetric map with the velocity projected on it?
Ground truth: the FUCCI cycle position (independent of the transcriptome).

The velocity circulates around the cycle (rpe1_circulation.py: net/abs +0.18
direct, +0.49 dynamo's projection, z 5-18), which no potential can represent
-- the regime where a Randers geometry could add something a landscape
cannot. Same protocol as gastrulation_repr.py, on the circle:

Representations, d = 2 and 10, for each velocity (direct linear projection,
dynamo's kernel projection):
  input30      30-PC positions, the velocity itself (reference, no reduction)
  pca<d>       first d PCs, the velocity's first d components
  sym<d>       symmetric embedding (no velocity), velocity projected
               scVelo-style onto it
  flowsne<d>   FlowSNE (Randers input affinity with that velocity, vector
               flow fitted jointly, row-stochastic, nu = 1), field -omega(z)
Scores:
  advance      cosine walk (k nearest cells in the representation, weights
               exp(10 cos(f_i, z_j - z_i))): mean FUCCI phase progress (unwrapped: summed per-step wrapped increments)
               after s steps, in fractions of a cycle, minus the same walk with
               uniform weights; on all cells and on the cells near the M -> G1
               seam (first and last phase bin, 1/12 each)
  native       FlowSNE's own Randers walk (k nearest under the fitted distance,
               Student weights) against the field switched off
  circulation  rpe1_circulation.py's loop statistic computed IN the
               representation (its own kNN graph, its own directions): does
               the map's direction field still go around the cycle?
Figure: the d = 2 maps coloured by FUCCI phase (cyclic colours) with grid
arrows of the direction field, one row per velocity.

Training: early exaggeration (12x attraction for the first 250 iterations)
and cosine learning-rate decay by default -- on RPE1 they let every map keep
the cycle's global structure (rpe1_diag10.py); --exaggeration 1
--lr-schedule constant gives the plain fit (tag repr: the first run).

Run:    python experiments/single_cell/rpe1_repr.py
Writes: experiments/single_cell/rpe1_results/repr/{repr.json, REPR.md, maps.png, Z_*.npy, flow_*.pt}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, randers_knn  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_hodge import edge_flow, edges_from_knn  # noqa: E402
from gastrulation_markov import chain_from_rows  # noqa: E402
from gastrulation_repr import chain, knn, project_velocity  # noqa: E402
from rpe1_circulation import build_loops, stats  # noqa: E402
from rpe1_hodge import TWO_PI, phase_progress, wrap  # noqa: E402

VELOCITIES = {"direct": "V_pca_direct", "kernel": "V_pca_kernel"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--velocities", nargs="+", default=list(VELOCITIES), choices=list(VELOCITIES))
    ap.add_argument("--k-chain", type=int, default=30)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--n-loops", type=int, default=200)
    ap.add_argument("--exaggeration", type=float, default=12.0)
    ap.add_argument("--exaggeration-iters", type=int, default=250)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--tag", default="repr_ee")
    a = ap.parse_args()
    res = os.path.join(HERE, "rpe1_results", a.tag)
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "repr.json")

    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X = d["X_pca"].astype(np.float64)
    pos = d["cycle_pos"]
    theta = TWO_PI * pos
    seam = (pos < 1 / 12) | (pos >= 11 / 12)
    n, k = len(X), a.k_chain
    dt = torch.float32
    settings = dict(iters=a.iters, lr=a.lr, exaggeration=a.exaggeration,
                    exaggeration_iters=a.exaggeration_iters, lr_schedule=a.lr_schedule)
    sfile = os.path.join(res, "fit_settings.json")
    if os.path.exists(sfile) and json.load(open(sfile)) != settings:
        raise SystemExit(f"{res} holds fits made with {json.load(open(sfile))}, not {settings}: "
                         "use another --tag")
    json.dump(settings, open(sfile, "w"), indent=1)
    out = dict(args=vars(a), n=n, seam_cells=int(seam.sum()), scores={}, circulation={}, fits={})

    def advance(M):
        return phase_progress(M, theta, a.steps)

    def score(name, seed, M, M0):
        adv, adv0 = advance(M), advance(M0)
        r = {}
        for s in a.steps:
            rel = adv[s] - adv0[s]
            r[f"adv_s{s}"], r[f"adv_s{s}_seam"] = float(rel.mean()), float(rel[seam].mean())
        out["scores"].setdefault(name, {})[f"seed{seed}"] = r
        json.dump(out, open(path, "w"), indent=1)
        print(f"  {name:22s} seed {seed}  " + "  ".join(
            f"s{s} {r[f'adv_s{s}']:+.4f} (seam {r[f'adv_s{s}_seam']:+.4f})" for s in a.steps), flush=True)

    rng = np.random.default_rng(0)

    def circulation(name, seed, Z, F):
        E = edges_from_knn(knn(Z, 30))
        loops = build_loops(Z, pos, E, a.n_loops, 12, rng)
        mean, frac, _ = stats(loops, edge_flow(Z, F, E), 12)
        out["circulation"].setdefault(name, {})[f"seed{seed}"] = dict(mean=mean, frac_pos=frac)

    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
    Xt = torch.tensor(X, dtype=dt)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=math.sqrt(n))
    P0, i0 = EntropicRandersAffinity(omega=None, **kw)(Xt)

    def fit(name, dim, seed, P, idx, train_flow):
        torch.manual_seed(seed)
        flow = input_std_norm_flow(dim, hidden_dim=32) if train_flow else mlp_flow(dim)
        aff = RandersAffinity(dim, flow=flow, degrees_of_freedom=1.0).to(dt)
        zf, ff = os.path.join(res, f"Z_{name}_s{seed}.npy"), os.path.join(res, f"flow_{name}_s{seed}.pt")
        if os.path.exists(zf) and os.path.exists(ff):
            Z = torch.tensor(np.load(zf), dtype=dt)
            aff.flow.load_state_dict(torch.load(ff))
        else:
            t0, hist = time.time(), []
            init = torch.tensor((X - X.mean(0)) @ pcs[:dim].T, dtype=dt)
            Z = fit_embedding(aff, P, init, indices=idx, max_iter=a.iters, lr=a.lr, seed=seed,
                              train_flow=train_flow, history=hist, log_every=500,
                              lr_schedule=a.lr_schedule, early_exaggeration=a.exaggeration,
                              exaggeration_iters=a.exaggeration_iters)
            np.save(zf, Z.numpy())
            torch.save(aff.flow.state_dict(), ff)
            out["fits"][f"{name}_s{seed}"] = [v for _, v in hist]
            print(f"  {name} seed {seed}: fitted in {time.time() - t0:.0f}s, loss "
                  + " ".join(f"{v:.4f}" for _, v in hist), flush=True)
        return aff, Z

    sym = {dim: fit(f"sym{dim}", dim, 0, P0, i0, False)[1].numpy().astype(np.float64) for dim in a.dims}
    maps = {}
    for vname in a.velocities:
        V = d[VELOCITIES[vname]].astype(np.float64)
        print(f"velocity: {vname}", flush=True)
        score(f"{vname}/input30", 0, chain(X, V, k), chain(X, V, k, directed=False))
        circulation(f"{vname}/input30", 0, X, V)
        P, idx = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dt), **kw)(Xt)
        for dim in a.dims:
            Z = X[:, :dim]
            score(f"{vname}/pca{dim}", 0, chain(Z, V[:, :dim], k), chain(Z, V[:, :dim], k, directed=False))
            circulation(f"{vname}/pca{dim}", 0, Z, V[:, :dim])
            Zs = sym[dim]
            Fs = project_velocity(X, V, Zs)
            score(f"{vname}/sym{dim}", 0, chain(Zs, Fs, k), chain(Zs, Fs, k, directed=False))
            circulation(f"{vname}/sym{dim}", 0, Zs, Fs)
            if dim == 2:
                maps[(vname, "sym")] = (Zs, Fs)
            for seed in a.seeds:
                aff, Zt = fit(f"{vname}_flowsne{dim}", dim, seed, P, idx, True)
                Zf = Zt.numpy().astype(np.float64)
                with torch.no_grad():
                    F = (-aff.omega_at(Zt)).numpy().astype(np.float64)
                    dist, nb = randers_knn(Zt, aff.omega_at(Zt), k)
                    dist0, nb0 = randers_knn(Zt, torch.zeros_like(Zt), k)
                M_nat = chain_from_rows((1 / (1 + dist ** 2)).numpy().astype(np.float64), nb.numpy(), n)
                M_nat0 = chain_from_rows((1 / (1 + dist0 ** 2)).numpy().astype(np.float64), nb0.numpy(), n)
                score(f"{vname}/flowsne{dim}", seed, chain(Zf, F, k), chain(Zf, F, k, directed=False))
                score(f"{vname}/flowsne{dim}_native", seed, M_nat, M_nat0)
                circulation(f"{vname}/flowsne{dim}", seed, Zf, F)
                if dim == 2 and seed == a.seeds[0]:
                    maps[(vname, "flowsne")] = (Zf, F)
        json.dump(out, open(path, "w"), indent=1)
    write_report(out, res)
    figure(maps, pos, out, res)


def _agg(v):
    v = np.array(v, dtype=float)
    return f"{v.mean():+.4f}" if len(v) == 1 else f"{v.mean():+.4f} ± {v.std():.4f}"


def write_report(out, res):
    steps = out["args"]["steps"]
    L = [f"# RPE1-FUCCI cell cycle: which map keeps the direction around the cycle? (n = {out['n']})", "",
         "Advance = mean FUCCI phase progress (unwrapped: summed per-step wrapped increments) after s steps of the cosine walk (fraction of a "
         "cycle), minus the same neighbours with uniform weights; seam = the "
         f"{out['seam_cells']} cells in the first and last twelfth of the cycle (M → G1). Circulation "
         "= mean net/abs of the map's own direction field around 200 loops through the 12 phase bins, "
         "on the map's own 30-NN graph (0 for any gradient). `_native` = FlowSNE's Randers walk vs "
         "the field switched off. Mean ± std over seeds.", "",
         "| representation | " + " | ".join(f"s={s}" for s in steps) + " | "
         + " | ".join(f"s={s}, seam" for s in steps) + " | circulation |",
         "|---|" + "---|" * (2 * len(steps) + 1)]
    for nm, by in out["scores"].items():
        rs = list(by.values())
        circ = out["circulation"].get(nm)
        c = _agg([v["mean"] for v in circ.values()]) if circ else "–"
        L.append(f"| {nm} | " + " | ".join(_agg([r[f'adv_s{s}'] for r in rs]) for s in steps) + " | "
                 + " | ".join(_agg([r[f'adv_s{s}_seam'] for r in rs]) for s in steps) + f" | {c} |")
    open(os.path.join(res, "REPR.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "REPR.md"))


def figure(maps, pos, out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flowsne.viz import default_bins, grid_arrows
    vels = sorted({v for v, _ in maps})
    fig, axes = plt.subplots(len(vels), 2, figsize=(12, 5.6 * len(vels)), constrained_layout=True,
                             squeeze=False)
    for r, vname in enumerate(vels):
        for c, kind in enumerate(("sym", "flowsne")):
            ax = axes[r, c]
            if (vname, kind) not in maps:
                ax.axis("off")
                continue
            Z, F = maps[(vname, kind)]
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=pos, cmap="twilight", vmin=0, vmax=1, s=5, rasterized=True)
            grid_arrows(ax, Z, F, bins=default_bins(len(Z)), key_label="|f|")
            nm = f"{vname}/{kind}2"
            circ = out["circulation"].get(nm, {})
            cval = np.mean([v["mean"] for v in circ.values()]) if circ else float("nan")
            ax.set_title(f"{'symmetric + projected velocity' if kind == 'sym' else 'FlowSNE, field −ω'}"
                         f" ({vname} velocity)\ncirculation {cval:+.2f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(sc, ax=axes, fraction=0.025, label="FUCCI cycle position")
    fig.savefig(os.path.join(res, "maps.png"), dpi=120)
    print("wrote", os.path.join(res, "maps.png"))


if __name__ == "__main__":
    main()
