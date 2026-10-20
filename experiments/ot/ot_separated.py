"""Exact-OT plan visualisation, keeping source and target SEPARATED.

Follow-up to ot_plan_embed.py, where using the plan as a directed cross-affinity
block either barely shaped the flow (small alpha) or attracted every source point
onto its partner (large alpha: source and target collapse together). Two ways
to keep the plan's DIRECTION while removing that attraction:

  offset[a,D0]   P = (1-a) P_union + a P_cross as before, but in the OUTPUT
                 kernel every source<->target pair gets an artificial extra
                 distance D0:  q_ij ∝ 1 / (1 + (D0 + d_F(z_i->z_j))^2).
                 First order in 1/D0: log q ≈ const - (2/D0)(|dz| + omega.dz)
                 -- the offset weakens attraction and direction by the SAME
                 factor, so some partner pull is predicted to survive.
                 Symmetric (omega = 0) and Randers (jointly fit) outputs.
  dir[lam]       P = P_union only (no cross block); the plan enters through a
                 separate DIRECTION-ONLY term: for each source point, a softmax
                 over all target points of kappa * cos(flow(z_i), z_j - z_i),
                 cross-entropy against its OT partner (and the mirror term for
                 target points, arrival direction). Scale-free, so it cannot
                 attract; discriminative, so the flow must point at its OWN
                 partner rather than just "toward the target side". Warm-started
                 from the symmetric layout, fit jointly (Z + flow).
  dir_frozenZ    same term with the layout frozen (flow only): can the plain
                 symmetric layout already host a partner-specific flow?

Extra metric: st_mix = mean fraction of each point's 10 nearest embedded
neighbours that belong to the OTHER distribution (0 = fully separated).

Run: conda run -n dev python experiments/ot/ot_separated.py run --seed 0
     conda run -n dev python experiments/ot/ot_separated.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher, ConstantFlow,
    input_std_norm_flow, randers_distance,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402
from ot_study import fit  # noqa: E402
from ot_plan_embed import make_problem, exact_plan, metrics  # noqa: E402

RES = os.path.join(HERE, "results", "separated")
DT = torch.float64


class OffsetRandersAffinity(RandersAffinity):
    """RandersAffinity with an artificial extra distance ``offset`` on every
    pair flagged in ``cross`` (source<->target), added BEFORE squaring."""

    def __init__(self, cross: torch.Tensor, offset: float, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("cross", cross.to(DT))
        self.offset = offset

    def _distance_matrix(self, Z, k=None, return_indices=False):
        omega = self.omega_at(Z).to(dtype=Z.dtype, device=Z.device)
        d = randers_distance(Z, omega, mode=self.drift_mode) + self.offset * self.cross
        C = d ** 2
        if self.zero_diag:
            C = C.masked_fill(torch.eye(Z.shape[0], dtype=torch.bool, device=Z.device), 1e12)
        return C


def st_mix(Z, n, k=10):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    side = np.r_[np.zeros(n), np.ones(n)]
    return float((side[idx] != side[:, None]).mean())


def fit_directional(P, Z0, sigma, n, lam, kappa=10.0, iters=1500, seed=0, freeze_Z=False):
    torch.manual_seed(seed)
    Z = Z0.detach().clone().requires_grad_(not freeze_Z)
    flow = input_std_norm_flow(2, hidden_dim=32).double()
    last = [m for m in flow.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.normal_(last.weight, std=0.01)  # cos() needs a non-zero field at init
    sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
    groups = [{"params": flow.parameters(), "lr": 0.01}]
    if not freeze_Z:
        groups.append({"params": [Z], "lr": 0.1})
    opt = torch.optim.Adam(groups)
    tgt = torch.tensor(sigma, dtype=torch.long)
    src = torch.tensor(np.argsort(sigma), dtype=torch.long)
    cos = nn.functional.cosine_similarity
    for _ in range(iters):
        opt.zero_grad()
        L_emb = cross_entropy_loss(P, sym(Z, log=True), log=True)
        F = flow(Z)
        Zs, Zt = Z[:n], Z[n:]
        logit_s = kappa * cos(F[:n, None], Zt[None] - Zs[:, None], dim=-1)   # (src, tgt)
        logit_t = kappa * cos(F[n:, None], Zt[:, None] - Zs[None], dim=-1)   # (tgt, src)
        L_dir = 0.5 * (nn.functional.cross_entropy(logit_s, tgt)
                       + nn.functional.cross_entropy(logit_t, src))
        (L_emb + lam * L_dir).backward()
        opt.step()
    with torch.no_grad():
        L_emb = cross_entropy_loss(P, sym(Z, log=True), log=True).item()
        return Z.detach(), flow(Z).detach().numpy(), L_emb, L_dir.item()


def run(seed, quick=False):
    n = 120 if quick else 450
    it = (lambda m: 50) if quick else (lambda m: m)
    Xs_np, Xt_np, ls, lt = make_problem(seed, n=n)
    sigma = exact_plan(Xs_np, Xt_np).argmax(1)
    X_np = np.r_[Xs_np, Xt_np]
    X = torch.tensor(X_np, dtype=DT)
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X_np)

    out = dict(seed=seed, n=n, runs=[])
    saved = dict(ls=ls, lt=lt, sigma=sigma)
    tag = f"sep{'_quick' if quick else ''}_seed{seed}"

    def record(name, Z, flow_dir=None, loss=None, extra=None, secs=0.0):
        Znp = Z.detach().numpy() if torch.is_tensor(Z) else Z
        m = metrics(Znp, Xs_np, Xt_np, ls, lt, sigma, flow_dir, seed=seed)
        m["st_mix"] = st_mix(Znp, n)
        r = dict(method=name, loss=loss, secs=secs, **m, **(extra or {}))
        out["runs"].append(r)
        saved[name] = Znp
        if flow_dir is not None:
            saved[name + "_flow"] = flow_dir
        print(f"[seed {seed}] {name:26s} " + " ".join(
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

    P_union = EntropicRandersAffinity(omega=None)(X).detach()
    P_cross = torch.zeros(2 * n, 2 * n, dtype=DT)
    P_cross[torch.arange(n), n + torch.tensor(sigma)] = 1.0 / n
    cross = torch.zeros(2 * n, 2 * n, dtype=torch.bool)
    cross[:n, n:] = True; cross[n:, :n] = True

    # --- references (same as ot_plan_embed.py, re-run to get st_mix)
    t0 = time.time()
    Z_sym, loss_sym = sym_fit(P_union)
    record("sym", Z_sym, loss=loss_sym, secs=time.time() - t0)
    t0 = time.time()
    P05 = 0.95 * P_union + 0.05 * P_cross
    Zc, lc = sym_fit(P05)
    record("cross_sym[0.05]", Zc, loss=lc, secs=time.time() - t0)
    P20 = 0.8 * P_union + 0.2 * P_cross
    torch.manual_seed(seed)
    aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                          degrees_of_freedom=1.0).double()
    t0 = time.time()
    Z = fit(aff, P20, init_np, JointAffinityMatcher, seed=seed, max_iter=it(3000), lr=0.03)
    with torch.no_grad():
        l = cross_entropy_loss(P20, aff(Z, log=True), log=True).item()
    record("cross_randers[0.2]_joint", Z, -aff.omega_at(Z.detach()).detach().numpy(), l,
           secs=time.time() - t0)

    # --- (a) artificial source<->target offset in the output kernel
    for a in ((0.2,) if quick else (0.05, 0.2)):
        P = (1 - a) * P_union + a * P_cross
        for D0 in ((200.0,) if quick else (50.0, 200.0, 1000.0)):
            for kind in ("sym", "randers"):
                torch.manual_seed(seed)
                if kind == "sym":
                    fl = ConstantFlow(2)
                    fl.raw.requires_grad_(False)
                else:
                    fl = input_std_norm_flow(2, hidden_dim=32)
                aff = OffsetRandersAffinity(cross, D0, n_components=2, flow=fl,
                                            degrees_of_freedom=1.0).double()
                t0 = time.time()
                try:
                    if kind == "sym":
                        Z = fit(aff, P, init_np, AffinityMatcher, seed=seed, max_iter=it(1500), lr=0.1)
                    else:
                        Z = fit(aff, P, init_np, JointAffinityMatcher, seed=seed, max_iter=it(3000), lr=0.03)
                    with torch.no_grad():
                        l = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
                    fd = None if kind == "sym" else -aff.omega_at(Z.detach()).detach().numpy()
                    record(f"offset_{kind}[{a},{int(D0)}]", Z, fd, l, secs=time.time() - t0)
                except Exception as e:
                    out["runs"].append(dict(method=f"offset_{kind}[{a},{int(D0)}]", error=repr(e)))
                    print(f"[seed {seed}] offset_{kind}[{a},{int(D0)}] FAILED {e!r}", flush=True)

    # --- (b) direction-only cross term
    for lam, frz in [(1.0, True)] + [(l_, False) for l_ in ((0.5,) if quick else (0.1, 0.5, 2.0))]:
        name = "dir_frozenZ" if frz else f"dir[{lam}]"
        t0 = time.time()
        try:
            Z, F, l_emb, l_dir = fit_directional(P_union, Z_sym, sigma, n, lam, iters=it(1500),
                                                 seed=seed, freeze_Z=frz)
            record(name, Z, F, l_emb, dict(loss_dir=l_dir), secs=time.time() - t0)
        except Exception as e:
            out["runs"].append(dict(method=name, error=repr(e)))
            print(f"[seed {seed}] {name} FAILED {e!r}", flush=True)


def report(quick=False):
    q = "_quick" if quick else ""
    files = sorted(f for f in os.listdir(RES) if f.startswith(f"sep{q}_seed") and f.endswith(".json"))
    E = [json.load(open(os.path.join(RES, f))) for f in files]
    keys = ["st_mix", "knn", "silhouette", "plan_cluster", "corresp_r2", "coherence",
            "flow_align_src", "flow_align_wrongcl", "flow_align_tgt", "flow_align_shuf"]
    ms = lambda v: "–" if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
    L = ["# Separated OT-plan visualisation — results", "",
         f"{len(E)} seed(s), {E[0]['n']} + {E[0]['n']} points, same problem as `RESULTS_plan.md`. "
         "See `ot_separated.py` docstring for the methods. `st_mix` = fraction of 10-NN from the "
         "other distribution (0 = separated); `loss` = embedding cross-entropy (P_union part only "
         "for `dir*`).", "",
         "| method | loss | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for m in [r["method"] for r in E[0]["runs"]]:
        sel = [r for e in E for r in e["runs"] if r["method"] == m and "error" not in r]
        nerr = sum(1 for e in E for r in e["runs"] if r["method"] == m and "error" in r)
        L.append(f"| {m} | {ms([r['loss'] for r in sel if r.get('loss') is not None])} | "
                 + " | ".join(ms([r[k] for r in sel if k in r]) for k in keys) + " |"
                 + (f" ⚠ {nerr} failed" if nerr else ""))
    L += ["", "![embeddings](results/separated/sep.png)", ""]
    open(os.path.join(HERE, f"RESULTS_separated{q}.md"), "w", encoding="utf-8").write("\n".join(L))
    _figure(q)
    print(f"wrote RESULTS_separated{q}.md")


def _figure(q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = os.path.join(RES, f"sep{q}_seed0.npz")
    S = np.load(f)
    want = ["sym", "cross_randers[0.2]_joint", "offset_randers[0.2,200]", "offset_randers[0.2,1000]",
            "dir_frozenZ", "dir[0.5]", "dir[2.0]"]
    names = [k for k in want if k in S.files]
    ls, lt, sigma = S["ls"], S["lt"], S["sigma"]
    n = len(ls)
    src_cols = ["#1f77b4", "#2ca02c", "#9467bd"]
    tgt_cols = ["#d62728", "#ff7f0e", "#8c564b"]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.5))
    sub = np.random.default_rng(0).choice(n, size=min(n, 70), replace=False)
    for ax, k in zip(np.atleast_1d(axes), names):
        Z = S[k]
        for i in sub:
            ax.plot([Z[i, 0], Z[n + sigma[i], 0]], [Z[i, 1], Z[n + sigma[i], 1]],
                    color="gray", lw=0.4, alpha=0.4, zorder=1)
        ax.scatter(Z[:n, 0], Z[:n, 1], c=[src_cols[c] for c in ls], s=7, marker="o", linewidths=0, zorder=2)
        ax.scatter(Z[n:, 0], Z[n:, 1], c=[tgt_cols[c] for c in lt], s=10, marker="^", linewidths=0, zorder=2)
        if k + "_flow" in S.files:
            F = S[k + "_flow"]
            F = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)  # direction only
            sc = 0.06 * (Z.max(0) - Z.min(0)).max()
            ax.quiver(Z[sub, 0], Z[sub, 1], F[sub, 0], F[sub, 1], color="black",
                      scale=1 / sc, scale_units="xy", angles="xy", width=0.004, zorder=3)
        ax.set_title(k, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Exact OT plan, seed 0 — ● source (S0 blue, S1 green, S2 purple), ▲ target "
                 "(T0 red, T1 orange, T2 brown), grey = plan, black = fitted flow direction", fontsize=10)
    fig.savefig(os.path.join(RES, f"sep{q}.png"), dpi=130, bbox_inches="tight")


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
