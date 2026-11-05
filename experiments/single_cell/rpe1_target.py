"""RPE1-FUCCI, d = 10: is FlowSNE's flow fitting the VELOCITY's asymmetry,
or the asymmetry of the row-stochastic construction itself?

Even with no velocity, the row-stochastic perplexity affinity P0 is strongly
asymmetric (per-row bandwidths, one-way neighbourships: median
|log P_ij / P_ji| 1.17 on RPE1, against 1.4-1.6 with a velocity). A
symmetric output kernel never sees this (its loss depends on P + P^T only);
a Randers output kernel tries to reproduce it all -- a candidate cause of the
fitted field's saturation at the norm bound.

Fits (dense input affinities, so P0 is known on every pair; early
exaggeration 12x for 250 iterations, cosine decay, 2000 iterations, seed 0):
  no velocity     target P0 itself: does the flow saturate with no velocity?
  standard        target P (with the velocity), as FlowSNE does now
  velocity-only   target P~_ij = (P0_ij + P0_ji)/2 * P_ij / P0_ij: the
                  symmetrised no-velocity affinity, times the change the
                  velocity makes to each pair -- its only asymmetry is the
                  velocity's
  D-J affinity    (two bandwidths: their default, the std of edge distances,
                  and the median edge length) target = Debaussart-Joniec, Blanchard & Kalogeratos's
                  affinity (arXiv 2609.37649): Gaussian kernel of the Randers
                  distance on the symmetrised 10-NN graph, ONE global
                  bandwidth (std of edge distances), density-normalised on both
                  sides (theta = 1: W_theta = Q^-1 W Q^-1, Q = mean of in- and
                  out-degree) -- by construction its only asymmetry is the
                  drift's; normalised to sum 1
Baseline: their full method in d = 10 (leading eigenvectors of the symmetric
operator as layout, drift V = L^a Psi), same scores. Their operators.py is
loaded from FINSLER_EMBEDDING_SRC (a clone of Gwendal-Debaussart/
Finsler-Embedding, commit d10bbc0, src/ directory).
Each fitted map is scored as in rpe1_diag10.py (k = 30): FUCCI phase progress
over the direction-blind walk after s steps (all cells, seam cells), for the
learned field and for the velocity projected onto the map; circulation of the
direction field; |omega| quantiles; plus the target's own asymmetry.

Run:    python experiments/single_cell/rpe1_target.py
Writes: experiments/single_cell/rpe1_results/target/{target.json, TARGET.md, Z_*.npy, flow_*.pt}
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

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, randers_distance  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402
from gastrulation_hodge import edge_flow, edges_from_knn  # noqa: E402
from gastrulation_repr import chain, knn, project_velocity  # noqa: E402
from rpe1_circulation import build_loops, stats  # noqa: E402
from rpe1_hodge import TWO_PI, phase_progress  # noqa: E402

VELOCITIES = {"direct": "V_pca_direct", "kernel": "V_pca_kernel"}
DIM = 10


def asymmetry(P, knn_idx):
    """|log P_ij / P_ji| over each row's 30 nearest pairs (dense P)."""
    i = np.repeat(np.arange(P.shape[0]), knn_idx.shape[1])
    j = knn_idx.ravel()
    r = np.abs(np.log(np.maximum(P[i, j], 1e-300) / np.maximum(P[j, i], 1e-300)))
    return [float(np.quantile(r, q)) for q in (0.5, 0.9, 0.99)]


def dj_affinity(X, V, k=10, max_norm=0.95, eps_mode="std"):
    """Their W (symmetrised kNN, Gaussian of the Randers distance, global eps)
    and the theta = 1 density-normalised W_theta; drift scaled as in
    EntropicRandersAffinity (largest norm max_norm, favoured direction +V)."""
    from sklearn.neighbors import kneighbors_graph
    n = len(X)
    G = kneighbors_graph(X, n_neighbors=k, include_self=False)
    G = ((G + G.T) > 0).tocoo()
    Vt = torch.tensor(V, dtype=torch.float64)
    b = -max_norm * Vt / Vt.norm(dim=1).max()
    dF = randers_distance(torch.tensor(X, dtype=torch.float64), b, mode="averaged")
    rows, cols = torch.tensor(G.row), torch.tensor(G.col)
    d_e = dF[rows, cols]
    # "std": their default (compute_epsilon_empirical); on 30-PC single-cell data it is ~1/5 of
    # the edge lengths, so the kernel sits near exp(-27) and its asymmetry explodes.
    # "median": the median edge length (the usual diffusion-maps choice).
    eps = d_e.std().item() if eps_mode == "std" else d_e.median().item()
    W = torch.zeros(n, n, dtype=torch.float64)
    W[rows, cols] = torch.exp(-d_e ** 2 / eps ** 2)
    Q = 0.5 * (W.sum(1) + W.sum(0))
    return W, W / Q[:, None] / Q[None, :], eps


def their_operators():
    import importlib.util
    root = os.environ.get("FINSLER_EMBEDDING_SRC", os.path.join(HERE, "..", "..", "..", "Finsler-Embedding", "src"))
    spec = importlib.util.spec_from_file_location("fe_operators",
                                                  os.path.join(root, "finsler_embedding", "operators.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocities", nargs="+", default=list(VELOCITIES), choices=list(VELOCITIES))
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10])
    a = ap.parse_args()
    res = os.path.join(HERE, "rpe1_results", "target")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "target.json")

    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X = d["X_pca"].astype(np.float64)
    pos = d["cycle_pos"]
    theta = TWO_PI * pos
    seam = (pos < 1 / 12) | (pos >= 11 / 12)
    n, k = len(X), 30
    dt = torch.float64
    rng = np.random.default_rng(0)
    out = dict(args=vars(a), n=n, rows={})
    nb30 = knn(X, 30)

    def evaluate(name, Z, F, extra=None):
        r = {}
        adv = phase_progress(chain(Z, F, k), theta, a.steps)
        adv0 = phase_progress(chain(Z, F, k, directed=False), theta, a.steps)
        for s in a.steps:
            rel = adv[s] - adv0[s]
            r[f"s{s}"], r[f"s{s}_seam"] = float(rel.mean()), float(rel[seam].mean())
        E = edges_from_knn(knn(Z, 30))
        r["circulation"] = stats(build_loops(Z, pos, E, 200, 12, rng), edge_flow(Z, F, E), 12)[0]
        r.update(extra or {})
        out["rows"][name] = r
        json.dump(out, open(path, "w"), indent=1)
        print(f"  {name:48s} " + "  ".join(f"s{s} {r[f's{s}']:+.4f} (seam {r[f's{s}_seam']:+.4f})"
                                         for s in a.steps)
              + f"  circ {r['circulation']:+.3f}" + "".join(f"  {k_} {v}" for k_, v in (extra or {}).items()),
              flush=True)

    Xt = torch.tensor(X, dtype=dt)
    kw = dict(doubly_stochastic=False, perplexity=math.sqrt(n))
    P0 = EntropicRandersAffinity(omega=None, **kw)(Xt)
    P0_np = P0.numpy()
    out["asymmetry_no_velocity"] = asymmetry(P0_np, nb30)
    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
    init = torch.tensor((X - X.mean(0)) @ pcs[:DIM].T, dtype=dt)

    def fit(name, P):
        torch.manual_seed(0)
        aff = RandersAffinity(DIM, flow=input_std_norm_flow(DIM, hidden_dim=32), degrees_of_freedom=1.0).to(dt)
        zf, ff = os.path.join(res, f"Z_{name}.npy"), os.path.join(res, f"flow_{name}.pt")
        if os.path.exists(zf) and os.path.exists(ff):
            aff.flow.load_state_dict(torch.load(ff))
            return aff, torch.tensor(np.load(zf), dtype=dt)
        t0 = time.time()
        Z = fit_embedding(aff, P, init, max_iter=a.iters, lr=a.lr, seed=0, lr_schedule="cosine",
                          early_exaggeration=12.0, exaggeration_iters=250)
        np.save(zf, Z.numpy())
        torch.save(aff.flow.state_dict(), ff)
        print(f"  [{name}: fitted in {time.time() - t0:.0f}s]", flush=True)
        return aff, Z

    def field_info(aff, Z, P_target):
        with torch.no_grad():
            om = aff.omega_at(Z).norm(dim=1).numpy()
            loss = float(aff.cross_entropy(Z, P_target))
        return {"omega_q10_50_90": "/".join(f"{np.quantile(om, q):.2f}" for q in (.1, .5, .9)),
                "loss": round(loss, 4)}

    def score_fit(label, vname, aff, Z, P_target, V):
        Zn = Z.numpy()
        with torch.no_grad():
            F = (-aff.omega_at(Z)).numpy()
        evaluate(f"{vname} | {label}, learned field", Zn, F, field_info(aff, Z, P_target))
        evaluate(f"{vname} | {label}, projected velocity", Zn, project_velocity(X, V, Zn))

    print(f"no-velocity affinity asymmetry (median/q90/q99): {out['asymmetry_no_velocity']}", flush=True)
    aff0, Z0 = fit("novelocity", P0)
    for vname in a.velocities:
        V = d[VELOCITIES[vname]].astype(np.float64)
        print(f"velocity: {vname}", flush=True)
        P = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dt), **kw)(Xt)
        P_np = P.numpy()
        Pt_np = 0.5 * (P0_np + P0_np.T) * P_np / np.maximum(P0_np, 1e-300)
        np.fill_diagonal(Pt_np, 0.0)
        Pt_np /= Pt_np.sum()
        Pt = torch.tensor(Pt_np, dtype=dt)
        out[f"asymmetry_{vname}_standard"] = asymmetry(P_np, nb30)
        out[f"asymmetry_{vname}_velocity_only"] = asymmetry(Pt_np, nb30)
        print(f"  target asymmetry (median/q90/q99): standard {out[f'asymmetry_{vname}_standard']}, "
              f"velocity-only {out[f'asymmetry_{vname}_velocity_only']}", flush=True)
        score_fit("no velocity", vname, aff0, Z0, P0, V)
        aff, Z = fit(f"{vname}_standard", P)
        score_fit("standard target", vname, aff, Z, P, V)
        aff, Z = fit(f"{vname}_velocity_only", Pt)
        score_fit("velocity-only target", vname, aff, Z, Pt, V)
        for eps_mode in ("std", "median"):
            W, Wth, eps = dj_affinity(X, V, eps_mode=eps_mode)
            Pdj_np = (Wth / Wth.sum()).numpy()
            sup = Pdj_np > 0
            r = np.abs(np.log(Pdj_np[sup] / np.maximum(Pdj_np.T[sup], 1e-300)))
            key = f"asymmetry_{vname}_dj_{eps_mode}"
            out[key] = [float(np.quantile(r, q_)) for q_ in (0.5, 0.9, 0.99)]
            print(f"  D-J affinity, eps = {eps_mode} edge length ({eps:.3f}): {int(sup.sum())} edges, "
                  f"asymmetry (median/q90/q99) {out[key]}", flush=True)
            Pdj = torch.tensor(Pdj_np, dtype=dt)
            aff, Z = fit(f"{vname}_dj_{eps_mode}", Pdj)
            score_fit(f"D-J affinity target (eps = {eps_mode})", vname, aff, Z, Pdj, V)
            res_dj = their_operators().randers_approximation(W, eps, X_low=None, m=DIM)
            evaluate(f"{vname} | D-J full method, eps = {eps_mode} (spectral layout, V = L^a Psi)",
                     res_dj["X_low"].numpy(), res_dj["V"].numpy())
    write_report(out, res)


def write_report(out, res):
    steps = out["args"]["steps"]
    q = out["asymmetry_no_velocity"]
    L = ["# RPE1-FUCCI, d = 10: which asymmetry does the flow fit?", "",
         "Dense row-stochastic input affinities; early exaggeration + cosine decay, 2000 iterations, "
         "seed 0. Target asymmetry = |log P_ij / P_ji| over each cell's 30 nearest pairs "
         f"(median / q90 / q99); with no velocity: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}.", ""]
    for v in out["args"]["velocities"]:
        s, t = out.get(f"asymmetry_{v}_standard"), out.get(f"asymmetry_{v}_velocity_only")
        dj = out.get(f"asymmetry_{v}_dj_median")
        if s:
            L.append(f"- {v}: standard target {s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f}; velocity-only target "
                     f"{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f}"
                     + (f"; D-J affinity on its support, eps = median edge {dj[0]:.2f} / {dj[1]:.2f} / {dj[2]:.2f}, "
                        f"eps = their default {out[f'asymmetry_{v}_dj_std'][0]:.2f} / "
                        f"{out[f'asymmetry_{v}_dj_std'][1]:.2f} / {out[f'asymmetry_{v}_dj_std'][2]:.2f}" if dj else ""))
    L += ["", "Phase progress (fraction of a cycle) over the direction-blind walk, k = 30; seam = first "
          "and last twelfth of the cycle; circulation of the direction field on the map's own graph.", "",
          "| velocity | fit / field | " + " | ".join(f"s={s}" for s in steps) + " | "
          + " | ".join(f"s={s} seam" for s in steps) + " | circulation | |ω| q10/50/90 | loss |",
          "|---|---|" + "---|" * (2 * len(steps) + 3)]
    for name, r in out["rows"].items():
        v, what = (x.strip() for x in name.split("|"))
        L.append(f"| {v} | {what} | " + " | ".join(f"{r[f's{s}']:+.4f}" for s in steps) + " | "
                 + " | ".join(f"{r[f's{s}_seam']:+.4f}" for s in steps)
                 + f" | {r['circulation']:+.3f} | {r.get('omega_q10_50_90', '–')} | {r.get('loss', '–')} |")
    open(os.path.join(res, "TARGET.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "TARGET.md"))


if __name__ == "__main__":
    main()
