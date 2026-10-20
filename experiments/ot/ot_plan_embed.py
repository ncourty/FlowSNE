"""Visualising an exact OT plan between two high-dimensional distributions.

Source: 3 clusters; target: 3 clusters, far from the source, arranged so the
plan has a SPLIT (S0 -> T0 + T1) and a MERGE (S1 + S2 -> T2). The high-D
plan is exact EMD (uniform weights, n_src == n_tgt, so a permutation sigma);
velocities are the plan's displacements, y_sigma(i) - x_i on source points,
y_j - x_sigma^-1(j) on target points.

Embeddings of source U target compared:
  pca                 linear projection (displacements projected faithfully)
  sym                 symmetric entropic affinity on the union + Student kernel
                      (t-SNE-like; ignores the plan)
  drift_{ds,rs}       plan used as DRIFT only (EntropicRandersAffinity, balanced
                      or row-stochastic) + RandersAffinity flow
  cross_sym[a]        plan used as AFFINITY: P = (1-a) P_union + a P_cross,
                      P_cross = the plan as a directed source->target block;
                      symmetric output kernel
  cross_randers[a]    same P, Randers output kernel (direction expressible)
  crossdrift_randers[a]  P_union built with the drift too, + cross, Randers output
Randers outputs are fit both jointly and drift-only (flow on top of the
matching symmetric layout).

Metrics (all on the 2-D embedding):
  plan_point    fraction of source points whose 2-D exact-OT partner (EMD between
                embedded source and target) is their high-D partner
  plan_cluster  fraction of mass the 2-D plan sends to the right TARGET CLUSTER
  knn           10-NN preservation within each distribution (local structure)
  silhouette    6-cluster silhouette
  coherence     per (source cluster -> target cluster) pair, mean resultant length
                of the unit embedded displacements z_sigma(i) - z_i (1 = parallel)
  corresp_r2    per cluster pair, R^2 of a linear map centred source -> centred partner
                positions (does the fine point-to-point correspondence survive?)
  flow_align    Randers only: cos(fitted -omega(z), direction to the embedded
                partner), source and target points. Controls: flow_align_shuf (random
                partner), flow_align_wrongcl (source points, random WRONG-cluster target)

Run: conda run -n dev python experiments/ot/ot_plan_embed.py run --seed 0
     conda run -n dev python experiments/ot/ot_plan_embed.py report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import ot
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    input_std_norm_flow, fit_drift_only,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402
from ot_study import fit  # noqa: E402

RES = os.path.join(HERE, "results", "plan")
DT = torch.float64
ALPHAS = (0.05, 0.2, 0.5)
SRC_NAMES = ["S0", "S1", "S2"]
TGT_NAMES = ["T0", "T1", "T2"]


# ===========================================================================
# Data
# ===========================================================================

def make_problem(seed, n=450, d=50, gap=12.0, spread=8.0):
    """Source clusters S0 (40%), S1 (30%), S2 (30%); target clusters T0 (20%),
    T1 (20%), T2 (60%) placed so the intended plan is S0 -> {T0, T1},
    {S1, S2} -> T2. Everything is offset by a common vector of norm `gap`
    (source/target clearly separated); the plan itself is whatever exact EMD
    returns -- the intended structure is only how the data is laid out."""
    rng = np.random.default_rng(seed)
    offset = rng.normal(size=d); offset *= gap / np.linalg.norm(offset)
    cS = rng.normal(size=(3, d)) * spread / math.sqrt(2 * d)  # E|cS_a - cS_b| ~ spread
    split = rng.normal(size=d); split *= 0.5 * spread / np.linalg.norm(split)
    cT = np.stack([cS[0] + offset + split, cS[0] + offset - split,
                   0.5 * (cS[1] + cS[2]) + offset])
    n_s = [int(0.4 * n), int(0.3 * n)]; n_s.append(n - sum(n_s))
    n_t = [int(0.2 * n), int(0.2 * n)]; n_t.append(n - sum(n_t))

    def blob(c, m, width):
        L = rng.normal(size=(d, 3)) * width / math.sqrt(3)
        return c + rng.normal(size=(m, 3)) @ L.T + 0.25 * rng.normal(size=(m, d))
    Xs = np.concatenate([blob(cS[k], n_s[k], 1.0) for k in range(3)])
    Xt = np.concatenate([blob(cT[k], n_t[k], 0.8 if k < 2 else 1.2) for k in range(3)])
    ls = np.repeat(np.arange(3), n_s)
    lt = np.repeat(np.arange(3), n_t)
    return Xs, Xt, ls, lt


def exact_plan(A, B):
    n, m = len(A), len(B)
    M = ot.dist(A, B)  # squared Euclidean
    G = ot.emd(np.full(n, 1 / n), np.full(m, 1 / m), M)
    return G


def cluster_plan(G, ls, lt):
    C = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            C[a, b] = G[np.ix_(ls == a, lt == b)].sum()
    return C


# ===========================================================================
# Metrics
# ===========================================================================

def knn_preservation(Xh, Z, k=10):
    from sklearn.neighbors import NearestNeighbors
    a = NearestNeighbors(n_neighbors=k + 1).fit(Xh).kneighbors(Xh, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(Xh))]))


def metrics(Z, Xs, Xt, ls, lt, sigma, flow_dir=None, seed=0):
    from sklearn.metrics import silhouette_score
    n = len(Xs)
    Zs, Zt = Z[:n], Z[n:]
    G2 = exact_plan(Zs, Zt)
    sigma2 = G2.argmax(1)
    C_true = cluster_plan(np.eye(n)[sigma] / n, ls, lt)
    C_2d = cluster_plan(G2, ls, lt)
    out = dict(
        plan_point=float((sigma2 == sigma).mean()),
        plan_cluster=float(1 - 0.5 * np.abs(C_2d - C_true).sum()),
        knn=0.5 * (knn_preservation(Xs, Zs) + knn_preservation(Xt, Zt)),
        silhouette=float(silhouette_score(Z, np.r_[ls, 3 + lt])),
    )
    disp = Zt[sigma] - Zs
    u = disp / np.linalg.norm(disp, axis=1, keepdims=True).clip(1e-12)
    coh = []
    for a in range(3):
        for b in range(3):
            m = (ls == a) & (lt[sigma] == b)
            if m.sum() >= 10:
                coh.append((np.linalg.norm(u[m].mean(0)), m.sum()))
    out["coherence"] = float(sum(c * w for c, w in coh) / sum(w for _, w in coh))
    # fine correspondence: within each cluster pair, R^2 of a linear map from
    # centred embedded source positions to centred embedded partner positions
    r2 = []
    for a in range(3):
        for b in range(3):
            m = (ls == a) & (lt[sigma] == b)
            if m.sum() >= 10:
                A = Zs[m] - Zs[m].mean(0)
                B = Zt[sigma[m]] - Zt[sigma[m]].mean(0)
                W = np.linalg.lstsq(A, B, rcond=None)[0]
                r2.append((1 - ((B - A @ W) ** 2).sum() / (B ** 2).sum(), m.sum()))
    out["corresp_r2"] = float(sum(c * w for c, w in r2) / sum(w for _, w in r2))
    if flow_dir is not None:
        rng = np.random.default_rng(seed)
        inv = np.argsort(sigma)
        partner = np.r_[n + sigma, inv]
        sign = np.r_[np.ones(n), -np.ones(n)][:, None]
        e = sign * (Z[partner] - Z)
        cos = lambda F, E: (F * E).sum(1) / (np.linalg.norm(F, axis=1) * np.linalg.norm(E, axis=1) + 1e-12)
        c = cos(flow_dir, e)
        shuf = rng.permutation(2 * n)
        # control 2 (source points): a random target point from a WRONG target cluster
        wrong = np.array([n + rng.choice(np.where(lt != lt[sigma[i]])[0]) for i in range(n)])
        c_wrong = cos(flow_dir[:n], Z[wrong] - Z[:n])
        out.update(flow_align=float(c.mean()), flow_align_src=float(c[:n].mean()),
                   flow_align_tgt=float(c[n:].mean()),
                   flow_align_shuf=float(cos(flow_dir, sign * (Z[partner[shuf]] - Z)).mean()),
                   flow_align_wrongcl=float(c_wrong.mean()))
    return out


# ===========================================================================
# Runs
# ===========================================================================

def run(seed, quick=False):
    n = 120 if quick else 450
    it = (lambda m: 50) if quick else (lambda m: m)
    Xs_np, Xt_np, ls, lt = make_problem(seed, n=n)
    G = exact_plan(Xs_np, Xt_np)
    sigma = G.argmax(1)
    assert np.allclose(G.max(1), 1 / n), "EMD plan is not a permutation"
    C_true = cluster_plan(G, ls, lt)
    V_src = Xt_np[sigma] - Xs_np
    inv = np.argsort(sigma)
    V_tgt = Xt_np - Xs_np[inv]
    X_np = np.r_[Xs_np, Xt_np]
    X = torch.tensor(X_np, dtype=DT)
    V = torch.tensor(np.r_[V_src, V_tgt], dtype=DT)
    init_np = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(
        n_components=2, random_state=seed).fit_transform(X_np)

    D = np.linalg.norm(X_np[:, None] - X_np[None], axis=-1)
    info = dict(seed=seed, n=n, cluster_plan_true=C_true.round(4).tolist(),
                mean_disp=float(np.linalg.norm(V_src, axis=1).mean()),
                median_within_dist=float(np.median(D[:n, :n])),
                min_src_tgt_dist=float(D[:n, n:].min()))
    print(f"[seed {seed}] true cluster plan (rows S, cols T):\n{C_true.round(3)}\n"
          f"mean |displacement| {info['mean_disp']:.2f}, median within-source dist "
          f"{info['median_within_dist']:.2f}, min src-tgt dist {info['min_src_tgt_dist']:.2f}",
          flush=True)

    out = dict(info=info, runs=[])
    saved = dict(ls=ls, lt=lt, sigma=sigma)
    tag = f"plan{'_quick' if quick else ''}_seed{seed}"

    def record(name, Z, flow_dir=None, loss=None, extra=None, secs=0.0):
        Znp = Z.detach().numpy() if torch.is_tensor(Z) else Z
        m = metrics(Znp, Xs_np, Xt_np, ls, lt, sigma,
                    None if flow_dir is None else flow_dir, seed=seed)
        r = dict(method=name, loss=loss, secs=secs, **m, **(extra or {}))
        out["runs"].append(r)
        saved[name] = Znp
        if flow_dir is not None:
            saved[name + "_flow"] = flow_dir
        print(f"[seed {seed}] {name:28s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs")
            + f"  ({secs:.0f}s)", flush=True)
        os.makedirs(RES, exist_ok=True)
        json.dump(out, open(os.path.join(RES, tag + ".json"), "w"), indent=1)
        np.savez_compressed(os.path.join(RES, tag + ".npz"), **saved)

    def sym_fit(P):
        aff = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
        Z = fit(aff, P, init_np, AffinityMatcher, seed=seed, max_iter=it(1500), lr=0.1)
        with torch.no_grad():
            return Z, cross_entropy_loss(P, aff(Z, log=True), log=True).item()

    def randers_fits(prefix, P, Z_sym, loss_sym):
        for how in ("joint", "driftonly"):
            torch.manual_seed(seed)
            aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                                  degrees_of_freedom=1.0).double()
            t0 = time.time()
            try:
                if how == "joint":
                    Z = fit(aff, P, init_np, JointAffinityMatcher, seed=seed, max_iter=it(3000), lr=0.03)
                else:
                    Z = fit_drift_only(aff, P, Z_sym.detach(), seed=seed, max_iter=it(1500), lr=0.05)
                with torch.no_grad():
                    loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
                flow_dir = -aff.omega_at(Z.detach()).detach().numpy()
                record(f"{prefix}_{how}", Z, flow_dir, loss,
                       dict(beats_sym=bool(loss < loss_sym)), time.time() - t0)
            except Exception as e:
                out["runs"].append(dict(method=f"{prefix}_{how}", error=repr(e)))
                print(f"[seed {seed}] {prefix}_{how} FAILED: {e!r}", flush=True)

    # --- baselines
    record("pca", init_np)
    P_union = EntropicRandersAffinity(omega=None)(X).detach()
    t0 = time.time()
    Z_sym, loss_sym = sym_fit(P_union)
    record("sym", Z_sym, loss=loss_sym, secs=time.time() - t0)

    # --- plan as drift only
    for tagp, ds in (("ds", True), ("rs", False)):
        P = EntropicRandersAffinity(omega=V, doubly_stochastic=ds)(X).detach()
        Zs_, ls_ = sym_fit(P)
        randers_fits(f"drift_{tagp}", P, Zs_, ls_)

    # --- plan as affinity (directed cross block), +/- drift in the within part
    P_cross = torch.zeros(2 * n, 2 * n, dtype=DT)
    P_cross[torch.arange(n), n + torch.tensor(sigma)] = 1.0 / n
    P_union_drift = EntropicRandersAffinity(omega=V)(X).detach()
    for a in (ALPHAS[1:2] if quick else ALPHAS):
        P = (1 - a) * P_union + a * P_cross
        t0 = time.time()
        Zc, lc = sym_fit(P)
        record(f"cross_sym[{a}]", Zc, loss=lc, secs=time.time() - t0)
        randers_fits(f"cross_randers[{a}]", P, Zc, lc)
        Pd = (1 - a) * P_union_drift + a * P_cross
        Zd, ld = sym_fit(Pd)
        randers_fits(f"crossdrift_randers[{a}]", Pd, Zd, ld)


# ===========================================================================
# Report
# ===========================================================================

def report(quick=False):
    q = "_quick" if quick else ""
    files = sorted(f for f in os.listdir(RES) if f.startswith(f"plan{q}_seed") and f.endswith(".json"))
    E = [json.load(open(os.path.join(RES, f))) for f in files]
    if not E:
        print("no results"); return
    keys = ["plan_point", "plan_cluster", "corresp_r2", "knn", "silhouette", "coherence",
            "flow_align_src", "flow_align_wrongcl", "flow_align_tgt", "flow_align_shuf"]
    ms = lambda v: "–" if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
    info = E[0]["info"]
    L = ["# Exact OT plan visualisation — results", "",
         f"{len(E)} seed(s); {info['n']} source + {info['n']} target points, 50-D. "
         "Source S0/S1/S2 (40/30/30%), target T0/T1/T2 (20/20/60%), laid out for a split "
         "S0→{T0,T1} and a merge {S1,S2}→T2. High-D plan: exact EMD.", "",
         "True cluster-level plan, seed 0 (rows S0..S2, cols T0..T2):", "",
         "| | T0 | T1 | T2 |", "|---|---|---|---|"]
    for a, row in enumerate(info["cluster_plan_true"]):
        L.append(f"| S{a} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L += ["", f"Mean |displacement| {np.mean([e['info']['mean_disp'] for e in E]):.1f}; "
          f"median within-source distance {np.mean([e['info']['median_within_dist'] for e in E]):.1f}; "
          f"min source–target distance {np.mean([e['info']['min_src_tgt_dist'] for e in E]):.1f}.", "",
          "| method | loss | beats sym. layout | " + " | ".join(keys) + " |",
          "|---|---|---|" + "---|" * len(keys)]
    methods = [r["method"] for r in E[0]["runs"]]
    for m in methods:
        sel = [r for e in E for r in e["runs"] if r["method"] == m and "error" not in r]
        errs = [r for e in E for r in e["runs"] if r["method"] == m and "error" in r]
        loss = ms([r["loss"] for r in sel if r.get("loss") is not None])
        beats = (f"{sum(r['beats_sym'] for r in sel)}/{len(sel)}" if sel and "beats_sym" in sel[0] else "–")
        L.append(f"| {m} | {loss} | {beats} | " + " | ".join(
            ms([r[k] for r in sel if k in r]) for k in keys) + " |" + (f" ⚠ {len(errs)} failed" if errs else ""))
    L += ["", "Metrics: see the module docstring of `ot_plan_embed.py`. `plan_point` chance level "
          f"≈ 1/{info['n']}; `plan_cluster` = 1 means the 2-D plan moves the same mass between the "
          "same clusters as the high-D one.", "", "![embeddings](results/plan/plan.png)", ""]
    open(os.path.join(HERE, f"RESULTS_plan{q}.md"), "w", encoding="utf-8").write("\n".join(L))
    _figure(q)
    print(f"wrote {os.path.join(HERE, f'RESULTS_plan{q}.md')}")


def _figure(q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = os.path.join(RES, f"plan{q}_seed0.npz")
    if not os.path.exists(f):
        return
    S = np.load(f)
    want = ["pca", "sym", "drift_rs_driftonly", "cross_sym[0.2]", "cross_randers[0.2]_joint",
            "crossdrift_randers[0.2]_joint"]
    names = [k for k in want if k in S.files]
    ls, lt, sigma = S["ls"], S["lt"], S["sigma"]
    n = len(ls)
    src_cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    tgt_cols = ["#d62728", "#ff7f0e", "#8c564b"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.6))
    sub = np.random.default_rng(0).choice(n, size=min(n, 70), replace=False)
    for ax, k in zip(np.atleast_1d(axes), names):
        Z = S[k]
        for i in sub:
            ax.plot([Z[i, 0], Z[n + sigma[i], 0]], [Z[i, 1], Z[n + sigma[i], 1]],
                    color="gray", lw=0.4, alpha=0.45, zorder=1)
        ax.scatter(Z[:n, 0], Z[:n, 1], c=[src_cols[c] for c in ls], s=7, marker="o",
                   linewidths=0, zorder=2)
        ax.scatter(Z[n:, 0], Z[n:, 1], c=[tgt_cols[c] for c in lt], s=10, marker="^",
                   linewidths=0, zorder=2)
        if k + "_flow" in S.files:
            F = S[k + "_flow"]
            sc = 0.08 * (Z.max(0) - Z.min(0)).max() / max(np.abs(F).max(), 1e-9)
            ax.quiver(Z[sub, 0], Z[sub, 1], F[sub, 0], F[sub, 1], color="black",
                      scale=1 / sc, scale_units="xy", angles="xy", width=0.004, zorder=3)
        ax.set_title(k, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Exact OT plan, seed 0 — ● source (S0 blue, S1 green, S2 purple), "
                 "▲ target (T0 red, T1 orange, T2 brown), grey = plan, black = fitted flow",
                 fontsize=10)
    fig.savefig(os.path.join(RES, f"plan{q}.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    run(a.seed, a.quick) if a.cmd == "run" else report(a.quick)
