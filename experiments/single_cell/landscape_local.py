"""Gradient (landscape) FlowSNE on the small single-cell datasets (pancreas,
dentate gyrus), on a CPU: fits, layout scores and level-set figures in d = 2.

Input (as in gastrulation_landscape.py): the velocity projected onto
gradients, psi = Hodge potential on the 30-PC kNN graph, scaled to be
0.95-Lipschitz over the affinity's candidate pairs; row-stochastic affinity
with the exact cost ||x_j - x_i|| - (psi_j - psi_i) (power 1). Velocity:
scVelo's transition-matrix projection shipped in data/<dataset>.npz (V_pca).

Output: a potential is a scalar per cell, so it needs no translation from the
data space to the map. The main variant ("psi") uses the input potential
itself on the map, h_i = c psi_i with one learned scale c (from 0), in the
exact distance ||z_j - z_i|| + h_j - h_i; admissibility then constrains the
layout (cells far apart in psi must be far apart on the map), pairs that
violate it get distance 0, and their fraction among map neighbours is
reported. With --with-net, also the 1-Lipschitz network h(z) ("net").
Compared with the symmetric embedding ("sym", no velocity) and the
vector-field FlowSNE ("vec", raw velocity, squared cost).

Since every map can show the same landscape psi, the question is which LAYOUT
reads best. Scores, on each map's k = 15 nearest neighbours:
  R2(psi)      1 - residual / total variance of psi when each cell's psi is
               predicted by its map neighbours' mean: psi varies smoothly
               across the map (clean level sets)
  R2(lineage)  the same for the lineage order known from the biology
               (independent of the velocity), on the lineage's cells:
                 pancreas      Ductal < Ngn3 low EP < Ngn3 high EP < Pre-endocrine
                               < Alpha, Beta, Delta, Epsilon
                 dentategyrus  Radial Glia-like < nIPC < Neuroblast
                               < Granule immature < Granule mature
  rho post hoc the Spearman correlation with the lineage of the potential
               recovered by the Hodge projection from the map's own positions
               and directions, where it has directions (sym: velocity
               projected scVelo-style; vec: -omega; net: -grad h)
psi's own Spearman correlation with the lineage is the reference.

Figure per dataset: every map with the SAME landscape drawn on it (bands
between level sets of psi, each band ~10% of cells, interpolated over the
map), cells coloured by cluster.

Outcome (Oct 2026, seed 0, 2000 iterations): the learned scale c comes out
negative, the correct sign (moving up psi cheaper), but the psi map is
slightly worse than the symmetric one -- R2(psi) 0.98 vs 1.00 and
R2(lineage) 0.96 vs 0.98 on pancreas, 0.93 vs 0.99 and 0.82 vs 0.90 on dentate
gyrus, with 2% / 9% of map-neighbour pairs inadmissible. The symmetric map
already shows psi cleanly; the earlier network variant (--with-net) also
flattened psi into a near-global tilt (-h vs lineage +0.79 / +0.32, against
+0.95 / +0.51 for psi).

Run: python experiments/single_cell/landscape_local.py --dataset pancreas dentategyrus
Writes: experiments/single_cell/landscape_local/<dataset>/{landscape.png, results.json}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, FixedPotential, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow, potential_flow  # noqa: E402
from flowsne.viz import potential_map, quantile_levels  # noqa: E402
from gastrulation_landscape import admissible_scale, hodge_potential, rho  # noqa: E402
from gastrulation_repr import knn, project_velocity  # noqa: E402

LINEAGE = {
    "pancreas": {"Ductal": 0, "Ngn3 low EP": 1, "Ngn3 high EP": 2, "Pre-endocrine": 3,
                 "Alpha": 4, "Beta": 4, "Delta": 4, "Epsilon": 4},
    "dentategyrus": {"Radial Glia-like": 0, "nIPC": 1, "Neuroblast": 2, "Granule immature": 3,
                     "Granule mature": 4},
}
TITLES = {"sym": "symmetric (no velocity)", "vec": "FlowSNE, vector field",
          "psi": "FlowSNE, input potential ψ", "net": "FlowSNE, potential network"}


def knn_r2(Z, y, k=15):
    """1 - SSE / SST of y predicted by the mean of each cell's k map
    neighbours (neighbours with y = NaN ignored; cells with y = NaN skipped)."""
    idx = knn(Z, k)
    ok = ~np.isnan(y)
    yhat = np.nanmean(np.where(ok[idx], y[idx], np.nan), axis=1)
    m = ok & ~np.isnan(yhat)
    return float(1 - ((y[m] - yhat[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())


def run(dataset, a):
    out_dir = os.path.join(HERE, "landscape_local", dataset)
    os.makedirs(out_dir, exist_ok=True)
    d = np.load(os.path.join(HERE, "data", f"{dataset}.npz"), allow_pickle=True)
    X = d["X_pca"].astype(np.float64)
    V = d["V_pca"].astype(np.float64)
    clusters = d["clusters"].astype(str)
    order = np.array([LINEAGE[dataset].get(c, np.nan) for c in clusters], dtype=float)
    lin = ~np.isnan(order)
    n = len(X)
    dt = torch.float32

    psi_raw, _, _ = hodge_potential(X, V, a.hodge_k)
    perp = math.sqrt(n) if a.perplexity == "auto" else float(a.perplexity)
    alpha, q = admissible_scale(X, psi_raw, min(n - 1, math.ceil(5 * perp)))
    psi = alpha * psi_raw
    res = dict(n=n, alpha=float(alpha), lipschitz_ratio_q50_q90_q99=q.tolist(),
               psi_vs_lineage=rho(psi[lin], order[lin]), maps={})
    print(f"{dataset}: {n} cells; |dpsi|/|dx| median {q[0]:.3f}, q99 {q[2]:.3f}; psi vs lineage "
          f"rho {res['psi_vs_lineage']:+.3f}", flush=True)

    Xt = torch.tensor(X, dtype=dt)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=perp)
    P0, i0 = EntropicRandersAffinity(omega=None, **kw)(Xt)
    P_vec, i_vec = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dt), **kw)(Xt)
    P_grad, i_grad = EntropicRandersAffinity(omega=torch.tensor(psi, dtype=dt), drift_mode="potential",
                                             cost_power=a.input_power, **kw)(Xt)
    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
    init = torch.tensor((X - X.mean(0)) @ pcs[:2].T, dtype=dt)
    affinity_for = {"sym": (P0, i0), "vec": (P_vec, i_vec), "psi": (P_grad, i_grad), "net": (P_grad, i_grad)}

    def make(kind):
        if kind == "psi":
            return RandersAffinity(2, flow=FixedPotential(torch.tensor(psi, dtype=dt)),
                                   drift_mode="potential", degrees_of_freedom=1.0)
        if kind == "net":
            return RandersAffinity(2, flow=potential_flow(2), drift_mode="potential",
                                   degrees_of_freedom=1.0)
        flow = input_std_norm_flow(2, hidden_dim=32) if kind == "vec" else mlp_flow(2)
        return RandersAffinity(2, flow=flow, degrees_of_freedom=1.0)

    maps = {}
    kinds = ["sym", "vec", "psi"] + (["net"] if a.with_net else [])
    for kind in kinds:
        torch.manual_seed(a.seed)
        aff = make(kind).to(dt)
        P, idx = affinity_for[kind]
        zf, ff = os.path.join(out_dir, f"Z_{kind}.npy"), os.path.join(out_dir, f"flow_{kind}.pt")
        hist = []
        if os.path.exists(zf) and os.path.exists(ff) and not a.refit:
            Z = torch.tensor(np.load(zf), dtype=dt)
            aff.flow.load_state_dict(torch.load(ff))
        else:
            Z = fit_embedding(aff, P, init, indices=idx, max_iter=a.iters, lr=a.lr, seed=a.seed,
                              train_flow=kind != "sym", history=hist, log_every=500)
            np.save(zf, Z.numpy())
            torch.save(aff.flow.state_dict(), ff)
        Z_np = Z.numpy().astype(np.float64)
        r = dict(loss_curve=[v for _, v in hist], R2_psi=knn_r2(Z_np, psi),
                 R2_lineage=knn_r2(Z_np, np.where(lin, order, np.nan)))
        if kind == "sym":
            F = project_velocity(X, V, Z_np)
        elif kind in ("vec", "net"):
            with torch.no_grad():
                F = (-aff.omega_at(Z)).numpy().astype(np.float64)
        else:
            F = None
        if F is not None:
            r["rho_post_hoc"] = rho(hodge_potential(Z_np, F, a.hodge_k)[0][lin], order[lin])
        if kind in ("psi", "net"):
            with torch.no_grad():
                h = aff.potential_at(Z).numpy().astype(np.float64)
            nb = knn(Z_np, 15)
            dz = np.linalg.norm(Z_np[nb] - Z_np[:, None], axis=-1)
            r["inadmissible_fraction_map_knn"] = float((np.abs(h[nb] - h[:, None]) >= dz).mean())
            if kind == "psi":
                r["scale_c"] = float(aff.flow.scale.detach())
            else:
                r["rho_-h_vs_lineage"] = rho(-h[lin], order[lin])
        res["maps"][kind] = r
        maps[kind] = Z_np
        print(f"  {kind:4s} loss {' '.join(f'{v:.4f}' for v in r['loss_curve'])}  "
              + "  ".join(f"{k} {v:+.3f}" for k, v in r.items() if k != "loss_curve"), flush=True)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    figure(dataset, out_dir, clusters, psi, maps, res)


def figure(dataset, out_dir, clusters, psi, maps, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from scipy.interpolate import LinearNDInterpolator

    cats = sorted(set(clusters))
    cmap_c = colormaps["tab20" if len(cats) > 10 else "tab10"]
    col = {c: cmap_c(i % cmap_c.N) for i, c in enumerate(cats)}
    levels = quantile_levels(psi)                     # the same landscape on every map
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps) + 1, 6), constrained_layout=True)
    for ax, (kind, Z) in zip(axes, maps.items()):
        sm, _ = potential_map(ax, LinearNDInterpolator(Z, psi), Z, levels, cmap="Greys", zorder=0)
        for c in cats:
            m = clusters == c
            ax.scatter(Z[m, 0], Z[m, 1], s=4, color=col[c], label=c, rasterized=True, zorder=1)
        r = res["maps"][kind]
        extra = f", ρ post hoc {r['rho_post_hoc']:+.2f}" if "rho_post_hoc" in r else ""
        extra += f", c = {r['scale_c']:.2f}" if "scale_c" in r else ""
        ax.set_title(f"{TITLES[kind]}\nR²(ψ) {r['R2_psi']:.2f}, R²(lineage) {r['R2_lineage']:.2f}"
                     f"{extra}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=7, markerscale=3, loc="best", frameon=True, framealpha=0.8)
    mid = 0.5 * (levels[:-1] + levels[1:])
    cb = fig.colorbar(sm, ax=axes, fraction=0.02, ticks=mid)
    cb.ax.set_yticklabels([f"{10 * i}–{10 * i + 10}%" for i in range(10)], fontsize=7)
    cb.set_label("band of the input potential ψ (share of cells, earliest first)")
    fig.suptitle(f"{dataset}: the same landscape ψ on each map (ψ vs lineage ρ = "
                 f"{res['psi_vs_lineage']:+.2f})", fontsize=12)
    path = os.path.join(out_dir, "landscape.png")
    fig.savefig(path, dpi=130)
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["pancreas", "dentategyrus"],
                    choices=["pancreas", "dentategyrus"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--input-power", type=float, default=1.0)
    ap.add_argument("--hodge-k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-net", action="store_true")
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    for ds in a.dataset:
        run(ds, a)


if __name__ == "__main__":
    main()
