"""Gradient (landscape) variant of FlowSNE on mouse gastrulation, against
the vector-field variant and standard representations, scored against the
real embryonic stage.

Why: the Hodge decomposition of the velocity (gastrulation_hodge.py) found
its forward-in-time content almost entirely in the gradient part; the rest
(curl + harmonic) carries none. For a gradient drift the Randers distance is
EXACT, ||y - x|| - (psi(y) - psi(x)), so this variant uses a potential on
both sides:

  input   psi = the velocity projected onto gradients: least-squares potential
          of the edge flow f_ij = (v_i + v_j)/2 . (x_j - x_i) on the 30-PC kNN
          graph (HodgeRank), scaled by one global alpha so that the largest
          |psi_j - psi_i| / ||x_j - x_i|| over the affinity's candidate pairs
          (each cell's 5 x perplexity Euclidean neighbours) is 0.95. Row-
          stochastic entropic affinity with cost ||x_j - x_i|| - (psi_j - psi_i)
          (power --input-power, default 1: then exactly the symmetric affinity
          with each row's targets reweighted by exp(psi_j / tau_i)).
  output  scalar network h(z), 1-Lipschitz by construction (spectrally clipped
          layers) times 0.95, exact distance ||dz|| + h(z_j) - h(z_i) in the
          Student kernel; -h is the forward landscape drawn on the map, -grad h
          its direction.

Compared at d = 2 and 10 (same velocity everywhere: the direct projection):
  pca<d>         first d PCs, velocity's first d components
  sym<d>         symmetric embedding, velocity projected scVelo-style
  vec<d>         FlowSNE, vector field (as in gastrulation_repr.py), squared cost
  grad<d>        FlowSNE, potential (this variant)
and in 30-PC space (no reduction): input30 (raw velocity), input30_grad
(its gradient part only), input_vec / input_grad (the two input affinities
themselves, truncated to k, against their own symmetric affinity).

Scores:
  walk        gastrulation_repr.py's cosine walk (k nearest cells in the
              representation, weights exp(10 cos(f_i, z_j - z_i))): days gained
              after s steps over the same neighbours with uniform weights.
  native      the fitted Randers kernel's own walk (k nearest under the fitted
              distance, Student weights) against the same rule with the field
              switched off (Euclidean neighbours, Student weights).
  pseudotime  Spearman correlation with stage of: psi (input ceiling); -h(z_i)
              (grad<d>); and, for every representation, the potential recovered
              post hoc by the same Hodge projection from its own positions and
              directions (what a landscape drawn on that map would show).
  fit checks  final loss, loss with the field switched off, loss curve, and
              |omega| (vec) or |grad h| (grad) quantiles against the 0.95 bound.

Run (GPU):        python experiments/single_cell/gastrulation_landscape.py --subset full --n-cells 30000
Quick test (CPU): python experiments/single_cell/gastrulation_landscape.py --subset erythroid \\
                      --n-cells 1500 --iters 100 --seeds 0 --k-chain 30 --tag landscape_test
Writes: experiments/single_cell/gastrulation_results/<tag>/{landscape.json, LANDSCAPE.md,
        Z_*.npy, flow_*.pt, hist_*.json}
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
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, randers_knn  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow, potential_flow  # noqa: E402
from gastrulation_data import load  # noqa: E402
from gastrulation_hodge import cell_vectors, edge_flow, edges_from_knn, hodge, incidence  # noqa: E402
from gastrulation_markov import advances, chain_from_rows, truncate_rows  # noqa: E402
from gastrulation_repr import chain, knn, project_velocity  # noqa: E402


def hodge_potential(Z, F, k):
    """Least-squares potential of the edge flow of (Z, F) on Z's kNN graph."""
    E = edges_from_knn(knn(Z, k))
    phi, g, r = hodge(incidence(E, len(Z)), edge_flow(Z, F, E))
    return phi, E, g


def admissible_scale(X, psi, k, bound=0.95):
    """alpha such that max |alpha (psi_j - psi_i)| / ||x_j - x_i|| over each
    point's k Euclidean neighbours is ``bound``; also that ratio's median,
    q90 and q99 after scaling."""
    nb = knn(X, k)
    ratio = np.abs(psi[nb] - psi[:, None]) / np.maximum(np.linalg.norm(X[nb] - X[:, None], axis=-1),
                                                       1e-12)
    alpha = bound / ratio.max()
    return alpha, np.quantile(alpha * ratio, [0.5, 0.9, 0.99])


def rho(a, b):
    return float(spearmanr(a, b)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-chain", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 25])
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--input-power", type=float, default=1.0)
    ap.add_argument("--hodge-k", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tag", default="landscape")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    res = os.path.join(HERE, "gastrulation_results", a.tag)
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "landscape.json")

    d = load(a.subset, n_cells=a.n_cells, seed=0)
    ok = ~np.isnan(d["stage"])
    X = d["X_pca"][ok].astype(np.float64)
    V = d["V_pca_direct"][ok].astype(np.float64)[:, :X.shape[1]]
    stage = d["stage"][ok].astype(np.float64)
    n = len(X)
    backend = "keops" if (cuda or n > 6000) else None
    out = dict(args=vars(a), n=n, input={}, fits={}, scores={}, pseudotime={})

    # --- input potential and its scale
    psi_raw, E30, g30 = hodge_potential(X, V, a.hodge_k)
    perp = math.sqrt(n) if a.perplexity == "auto" else float(a.perplexity)
    k_aff = min(n - 1, math.ceil(5 * perp))
    alpha, q = admissible_scale(X, psi_raw, k_aff)
    psi = alpha * psi_raw
    out["input"].update(alpha=float(alpha), lipschitz_ratio_q50_q90_q99=q.tolist(),
                        psi_stage_spearman=rho(psi, stage))
    out["pseudotime"]["input_psi (ceiling)"] = rho(psi, stage)
    print(f"{a.subset}: {n} cells; psi vs stage rho {rho(psi, stage):+.3f}; |dpsi|/|dx| over "
          f"{k_aff} neighbours: median {q[0]:.3f}, q90 {q[1]:.3f}, q99 {q[2]:.3f}, max 0.95", flush=True)

    def score(name, k, seed, M, M0):
        adv, adv0 = advances(M, stage, a.steps), advances(M0, stage, a.steps)
        r = {f"rel_s{s}": float((adv[s] - adv0[s]).mean()) for s in adv}
        r.update({f"fwd_s{s}": float((adv[s] > 0).mean()) for s in adv})
        out["scores"].setdefault(name, {}).setdefault(f"k{k}", {})[f"seed{seed}"] = r
        print(f"  k={k:4d} seed={seed} {name:18s} " + "  ".join(
            f"s{s} {r[f'rel_s{s}']:+.3f}" for s in sorted(adv)), flush=True)

    def save():
        json.dump(out, open(path, "w"), indent=1)

    # --- no reduction, PCA
    V_grad = cell_vectors(X, E30, g30)
    for k in a.k_chain:
        score("input30", k, 0, chain(X, V, k), chain(X, V, k, directed=False))
        score("input30_grad", k, 0, chain(X, V_grad, k), chain(X, V_grad, k, directed=False))
        for dim in a.dims:
            Z = X[:, :dim]
            score(f"pca{dim}", k, 0, chain(Z, V[:, :dim], k), chain(Z, V[:, :dim], k, directed=False))
    for dim in a.dims:
        out["pseudotime"][f"pca{dim} (post hoc)"] = rho(hodge_potential(X[:, :dim], V[:, :dim],
                                                                        a.hodge_k)[0], stage)

    # --- input affinities
    Xt = torch.tensor(X, dtype=dtype, device=device)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=perp, backend=backend)
    t0 = time.time()
    P_vec, i_vec = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dtype, device=device), **kw)(Xt)
    P0, i0 = EntropicRandersAffinity(omega=None, **kw)(Xt)
    psi_t = torch.tensor(psi, dtype=dtype, device=device)
    P_grad, i_grad = EntropicRandersAffinity(omega=psi_t, drift_mode="potential",
                                             cost_power=a.input_power, **kw)(Xt)
    P0_lin, i0_lin = EntropicRandersAffinity(omega=None, cost_power=a.input_power, **kw)(Xt)
    ig = i_grad.cpu().numpy()
    bad = (psi[ig] - psi[:, None]) > np.linalg.norm(X[ig] - X[:, None], axis=-1)
    out["input"]["inadmissible_fraction_of_kept_pairs"] = float(bad.mean())
    print(f"  input affinities ({time.time() - t0:.0f}s); inadmissible kept pairs: {bad.mean():.2e}",
          flush=True)
    del bad

    def as_chain(P, idx):
        return chain_from_rows(P.cpu().double().numpy(), idx.cpu().numpy(), n)
    C_vec, C0, C_grad, C0_lin = (as_chain(P_vec, i_vec), as_chain(P0, i0), as_chain(P_grad, i_grad),
                                 as_chain(P0_lin, i0_lin))
    for k in a.k_chain:
        score("input_vec", k, 0, truncate_rows(C_vec, k), truncate_rows(C0, k))
        score("input_grad", k, 0, truncate_rows(C_grad, k), truncate_rows(C0_lin, k))
    save()

    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]

    def make(dim, kind):
        if kind == "grad":
            flow, mode = potential_flow(dim, a.hidden, a.layers), "potential"
        else:
            flow, mode = (input_std_norm_flow(dim, hidden_dim=32) if kind == "vec" else mlp_flow(dim)), \
                "averaged"
        return RandersAffinity(n_components=dim, flow=flow, drift_mode=mode, degrees_of_freedom=1.0,
                               backend=backend).to(device=device, dtype=dtype)

    def fitted(kind, dim, seed, P, idx):
        torch.manual_seed(seed)
        aff = make(dim, kind)
        name = f"{kind}{dim}_s{seed}"
        zf, ff, hf = (os.path.join(res, f"{p}_{name}.{e}") for p, e in
                      (("Z", "npy"), ("flow", "pt"), ("hist", "json")))
        if os.path.exists(zf) and os.path.exists(ff):
            Z = torch.tensor(np.load(zf), dtype=dtype, device=device)
            aff.flow.load_state_dict(torch.load(ff, map_location=device))
            hist = json.load(open(hf)) if os.path.exists(hf) else []
        else:
            t1 = time.time()
            init = torch.tensor((X - X.mean(0)) @ pcs[:dim].T, dtype=dtype, device=device)
            hist = []
            Z = fit_embedding(aff, P, init, indices=idx, max_iter=a.iters, lr=a.lr, seed=seed,
                              train_flow=kind != "sym", history=hist, log_every=500)
            np.save(zf, Z.cpu().float().numpy())
            torch.save(aff.flow.state_dict(), ff)
            json.dump(hist, open(hf, "w"))
            print(f"  {name}: fitted in {time.time() - t1:.0f}s", flush=True)
        with torch.no_grad():
            loss = float(aff.cross_entropy(Z, P, idx))
            loss_off = float(make(dim, "grad" if kind == "grad" else "sym").cross_entropy(Z, P, idx))
            om = aff.omega_at(Z).norm(dim=1).cpu().double().numpy()
        out["fits"][name] = dict(loss=loss, loss_field_off=loss_off, history=hist,
                                 field_norm_q10_50_90_max=[float(np.quantile(om, p)) for p in (.1, .5, .9)]
                                 + [float(om.max())])
        save()
        return aff, Z

    def native(aff, Z, k, zero=False):
        with torch.no_grad():
            field = torch.zeros_like(aff._field(Z)) if zero else aff._field(Z)
            dist, idx = randers_knn(Z, field, k, mode=aff.drift_mode, backend=backend)
            w = 1.0 / (1.0 + dist ** 2)
        return chain_from_rows(w.cpu().double().numpy(), idx.cpu().numpy(), n)

    for dim in a.dims:
        _, Zs = fitted("sym", dim, 0, P0, i0)
        Zs = Zs.cpu().double().numpy()
        Fs = project_velocity(X, V, Zs)
        out["pseudotime"][f"sym{dim} (post hoc)"] = rho(hodge_potential(Zs, Fs, a.hodge_k)[0], stage)
        for k in a.k_chain:
            score(f"sym{dim}", k, 0, chain(Zs, Fs, k), chain(Zs, Fs, k, directed=False))
        for kind, P, idx in (("vec", P_vec, i_vec), ("grad", P_grad, i_grad)):
            for seed in a.seeds:
                aff, Z = fitted(kind, dim, seed, P, idx)
                Z_np = Z.cpu().double().numpy()
                with torch.no_grad():
                    F = (-aff.omega_at(Z)).cpu().double().numpy()
                pt = out["pseudotime"]
                pt.setdefault(f"{kind}{dim} (post hoc)", {})[f"seed{seed}"] = rho(
                    hodge_potential(Z_np, F, a.hodge_k)[0], stage)
                if kind == "grad":
                    with torch.no_grad():
                        h = aff.potential_at(Z).cpu().double().numpy()
                    pt.setdefault(f"grad{dim} (-h, fitted)", {})[f"seed{seed}"] = rho(-h, stage)
                for k in a.k_chain:
                    score(f"{kind}{dim}", k, seed, chain(Z_np, F, k), chain(Z_np, F, k, directed=False))
                    score(f"{kind}{dim}_native", k, seed, native(aff, Z, k), native(aff, Z, k, zero=True))
                save()
    write_report(out, res)


def _agg(v):
    v = np.array(v if isinstance(v, list) else [v], dtype=float)
    return f"{v.mean():+.3f}" if len(v) == 1 else f"{v.mean():+.3f} ± {v.std():.3f}"


def write_report(out, res):
    S, inp = out["scores"], out["input"]
    steps = out["args"]["steps"]
    q = inp["lipschitz_ratio_q50_q90_q99"]
    L = [f"# Gastrulation: landscape (gradient) FlowSNE ({out['args']['subset']}, n = {out['n']})", "",
         f"Input potential ψ = Hodge projection of the direct velocity (30-PC kNN graph), scaled by "
         f"α = {inp['alpha']:.3g} so that max |Δψ|/|Δx| over the affinity's candidate pairs = 0.95; "
         f"that ratio's median / q90 / q99: {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}. Input cost power "
         f"{out['args']['input_power']:g}. Inadmissible pairs among those kept: "
         f"{inp['inadmissible_fraction_of_kept_pairs']:.1e}.", "",
         "## Pseudotime: Spearman correlation with embryonic stage", "",
         "| landscape | ρ |", "|---|---|"]
    for nm, v in out["pseudotime"].items():
        L.append(f"| {nm} | {_agg(list(v.values()) if isinstance(v, dict) else v)} |")
    L += ["", "Post hoc = the same Hodge projection applied to the representation's own positions and "
          "directions.", "",
          "## Walk: days gained over each representation's own direction-blind control", "",
          "`_native` = the fitted Randers kernel's own walk vs the same rule with the field off. "
          "`input_vec`/`input_grad` = the input affinities (truncated to k) vs their symmetric "
          "counterpart. Mean ± std over seeds.", ""]
    ks = sorted({int(k[1:]) for nm in S for k in S[nm]})
    for k in ks:
        L += [f"### k = {k}", "", "| representation | " + " | ".join(f"s={s}" for s in steps)
              + f" | fwd s={steps[-1]} |", "|---|" + "---|" * (len(steps) + 1)]
        for nm in S:
            rs = list(S[nm].get(f"k{k}", {}).values())
            if rs:
                L.append(f"| {nm} | " + " | ".join(_agg([r[f'rel_s{s}'] for r in rs]) for s in steps)
                         + f" | {_agg([r[f'fwd_s{steps[-1]}'] for r in rs])} |")
        L.append("")
    L += ["## Fit checks", "", "| fit | loss | loss, field off | gain | field norm q10/50/90/max | "
          "loss curve (every 500 its) |", "|---|---|---|---|---|---|"]
    for nm, f in out["fits"].items():
        curve = " → ".join(f"{v:.4f}" for _, v in f["history"][::2]) if f["history"] else "–"
        L.append(f"| {nm} | {f['loss']:.5f} | {f['loss_field_off']:.5f} | "
                 f"{f['loss_field_off'] - f['loss']:+.5f} | "
                 + "/".join(f"{v:.2f}" for v in f["field_norm_q10_50_90_max"]) + f" | {curve} |")
    open(os.path.join(res, "LANDSCAPE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "LANDSCAPE.md"))


if __name__ == "__main__":
    main()
