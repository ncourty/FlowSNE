"""RPE1-FUCCI, d = 10: why does FlowSNE fail at the M -> G1 seam while
leading in the bulk of the cycle (rpe1_repr.py)? Field or layout?

Step 1 (no fitting; the saved 2000-iteration fits of rpe1_repr.py, tag repr)
  On each FlowSNE-10 map, score the learned field -omega(z) AND the raw
  velocity projected onto the map scVelo-style (as for the symmetric map).
  If the projected velocity crosses the seam on the FlowSNE layout, the
  layout is fine and the field is at fault; if not, the layout is.
  Layout check: seam closure = for the cells of the first and last twelfth
  of the cycle, the share of their 15 nearest cells (in the representation)
  that lie on the OTHER side of the seam -- does the map put the end of M
  next to the start of G1 (30-PC space as reference)?
Step 2 (layout): refits with early exaggeration (12x for 250 iterations) and
  cosine learning-rate decay: symmetric-10, and FlowSNE-10 with the flow at
  the same learning rate and at 1/10 of it.
Step 3 (field): the flow alone refitted on the frozen FlowSNE-10 layout
  (seed 0), cosine decay: the current network (input-std-norm, 32 units) at
  lr 0.03 and 0.003, and random-Fourier-feature networks (128 features, 32
  units) with sigma = c / std(Z), c in --rff-c (phase change 2 pi c per std of
  the layout: smooth to high-frequency).

Scores as in rpe1_repr.py (k = 30): phase advance after s steps over the
direction-blind walk, all cells and seam cells; circulation of the direction
field on the map's own graph; plus |omega| quantiles (saturation near 0.95)
and the final loss.

Run:    python experiments/single_cell/rpe1_diag10.py
Writes: experiments/single_cell/rpe1_results/diag10/{diag.json, DIAG.md, Z_*.npy, flow_*.pt}
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

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow, rff_flow  # noqa: E402
from gastrulation_hodge import edge_flow, edges_from_knn  # noqa: E402
from gastrulation_repr import chain, knn, project_velocity  # noqa: E402
from rpe1_circulation import build_loops, stats  # noqa: E402
from rpe1_hodge import TWO_PI, phase_progress, wrap  # noqa: E402

VELOCITIES = {"direct": "V_pca_direct", "kernel": "V_pca_kernel"}
DIM = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocities", nargs="+", default=list(VELOCITIES), choices=list(VELOCITIES))
    ap.add_argument("--src-tag", default="repr")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--rff-c", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--skip", nargs="*", default=[], choices=["1", "2", "3"])
    a = ap.parse_args()
    src = os.path.join(HERE, "rpe1_results", a.src_tag)
    res = os.path.join(HERE, "rpe1_results", "diag10")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "diag.json")

    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X = d["X_pca"].astype(np.float64)
    pos = d["cycle_pos"]
    theta = TWO_PI * pos
    lo, hi = pos < 1 / 12, pos >= 11 / 12
    seam = lo | hi
    n, k = len(X), 30
    dt = torch.float32
    rng = np.random.default_rng(0)
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.update(args=vars(a), n=n)
    out.setdefault("rows", {})

    def advance(M):
        return phase_progress(M, theta, a.steps)

    def closure(Z):
        nb = knn(Z, 15)
        other = np.where(lo[:, None], hi[nb], np.where(hi[:, None], lo[nb], False))
        return float(other[seam].mean())

    def evaluate(name, Z, F, extra=None):
        adv, adv0 = advance(chain(Z, F, k)), advance(chain(Z, F, k, directed=False))
        r = {}
        for s in a.steps:
            rel = adv[s] - adv0[s]
            r[f"s{s}"], r[f"s{s}_seam"] = float(rel.mean()), float(rel[seam].mean())
        E = edges_from_knn(knn(Z, 30))
        r["circulation"] = stats(build_loops(Z, pos, E, 200, 12, rng), edge_flow(Z, F, E), 12)[0]
        r["seam_closure"] = closure(Z)
        r.update(extra or {})
        out["rows"][name] = r
        json.dump(out, open(path, "w"), indent=1)
        print(f"  {name:44s} " + "  ".join(f"s{s} {r[f's{s}']:+.4f} (seam {r[f's{s}_seam']:+.4f})"
                                         for s in a.steps)
              + f"  circ {r['circulation']:+.3f}  closure {r['seam_closure']:.3f}"
              + "".join(f"  {k_} {v}" for k_, v in (extra or {}).items()), flush=True)

    def field_info(aff, Z, P, idx):
        with torch.no_grad():
            om = aff.omega_at(Z).norm(dim=1).numpy()
            loss = float(aff.cross_entropy(Z, P, idx))
        return {"omega_q10_50_90": "/".join(f"{np.quantile(om, q):.2f}" for q in (.1, .5, .9)),
                "loss": round(loss, 4)}

    Xt = torch.tensor(X, dtype=dt)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=math.sqrt(n))
    P0, i0 = EntropicRandersAffinity(omega=None, **kw)(Xt)
    pcs = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
    init = torch.tensor((X - X.mean(0)) @ pcs[:DIM].T, dtype=dt)
    out["reference_closure_30pc"] = closure(X)
    print(f"seam closure in 30-PC space: {out['reference_closure_30pc']:.3f}", flush=True)

    def flow_net(kind, Z=None, c=None):
        if kind == "rff":
            return rff_flow(DIM, hidden_dim=32, n_fourier=128, sigma=c / float(Z.std()), seed=0)
        return input_std_norm_flow(DIM, hidden_dim=32) if kind == "std" else mlp_flow(DIM)

    def fit(name, P, idx, flow, train_flow=True, train_embedding=True, Z0=None, **opts):
        torch.manual_seed(0)
        aff = RandersAffinity(DIM, flow=flow, degrees_of_freedom=1.0).to(dt)
        zf, ff = os.path.join(res, f"Z_{name}.npy"), os.path.join(res, f"flow_{name}.pt")
        if os.path.exists(zf) and os.path.exists(ff):
            aff.flow.load_state_dict(torch.load(ff))
            return aff, torch.tensor(np.load(zf), dtype=dt)
        t0 = time.time()
        Z = fit_embedding(aff, P, init if Z0 is None else Z0, indices=idx, max_iter=a.iters, lr=a.lr,
                          seed=0, train_flow=train_flow, train_embedding=train_embedding,
                          lr_schedule="cosine", **opts)
        np.save(zf, Z.numpy())
        torch.save(aff.flow.state_dict(), ff)
        print(f"  [{name}: fitted in {time.time() - t0:.0f}s]", flush=True)
        return aff, Z

    for vname in a.velocities:
        V = d[VELOCITIES[vname]].astype(np.float64)
        print(f"velocity: {vname}", flush=True)
        P, idx = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dt), **kw)(Xt)

        if "1" not in a.skip:
            evaluate(f"{vname} | 1 | input30 (velocity)", X, V)
            Zs = np.load(os.path.join(src, "Z_sym10_s0.npy")).astype(np.float64)
            evaluate(f"{vname} | 1 | sym10, projected velocity", Zs, project_velocity(X, V, Zs))
            for seed in (0, 1, 2):
                aff = RandersAffinity(DIM, flow=flow_net("std"), degrees_of_freedom=1.0).to(dt)
                aff.flow.load_state_dict(torch.load(os.path.join(src, f"flow_{vname}_flowsne10_s{seed}.pt")))
                Zt = torch.tensor(np.load(os.path.join(src, f"Z_{vname}_flowsne10_s{seed}.npy")), dtype=dt)
                Zf = Zt.numpy().astype(np.float64)
                with torch.no_grad():
                    F = (-aff.omega_at(Zt)).numpy().astype(np.float64)
                evaluate(f"{vname} | 1 | flowsne10 s{seed}, learned field", Zf, F, field_info(aff, Zt, P, idx))
                evaluate(f"{vname} | 1 | flowsne10 s{seed}, projected velocity", Zf, project_velocity(X, V, Zf))

        if "2" not in a.skip:
            ee = dict(early_exaggeration=12.0, exaggeration_iters=250)
            if vname == a.velocities[0]:
                _, Zs = fit("sym10_ee", P0, i0, flow_net("sym"), train_flow=False, **ee)
                Zs = Zs.numpy().astype(np.float64)
                for vn in a.velocities:
                    Vv = d[VELOCITIES[vn]].astype(np.float64)
                    evaluate(f"{vn} | 2 | sym10 EE+cosine, projected velocity", Zs, project_velocity(X, Vv, Zs))
            for label, flr in (("flow lr = lr", None), ("flow lr = lr/10", a.lr / 10)):
                tag = f"{vname}_flowsne10_ee" + ("" if flr is None else "_lowflow")
                aff, Zt = fit(tag, P, idx, flow_net("std"), flow_lr=flr, **ee)
                Zf = Zt.numpy().astype(np.float64)
                with torch.no_grad():
                    F = (-aff.omega_at(Zt)).numpy().astype(np.float64)
                evaluate(f"{vname} | 2 | flowsne10 EE+cosine, {label}, learned field", Zf, F,
                         field_info(aff, Zt, P, idx))
                evaluate(f"{vname} | 2 | flowsne10 EE+cosine, {label}, projected velocity", Zf,
                         project_velocity(X, V, Zf))

        if "3" not in a.skip:
            Zt = torch.tensor(np.load(os.path.join(src, f"Z_{vname}_flowsne10_s0.npy")), dtype=dt)
            Zf = Zt.numpy().astype(np.float64)
            variants = [("std net, lr 0.03", flow_net("std"), a.lr),
                        ("std net, lr 0.003", flow_net("std"), a.lr / 10)]
            variants += [(f"RFF c = {c:g}", flow_net("rff", Zt, c), a.lr) for c in a.rff_c]
            for label, flow, lr in variants:
                tag = f"{vname}_driftonly_" + label.replace(" ", "").replace(",", "_").replace("=", "")
                aff, _ = fit(tag, P, idx, flow, train_embedding=False, Z0=Zt, flow_lr=lr)
                with torch.no_grad():
                    F = (-aff.omega_at(Zt)).numpy().astype(np.float64)
                evaluate(f"{vname} | 3 | frozen layout s0, {label}", Zf, F, field_info(aff, Zt, P, idx))
    write_report(out, res)


def write_report(out, res):
    steps = out["args"]["steps"]
    L = ["# RPE1-FUCCI, d = 10: field or layout at the M → G1 seam?", "",
         "Advance = FUCCI phase progress (unwrapped) (fraction of a cycle) over the direction-blind walk, "
         "k = 30; seam = first and last twelfth of the cycle. Circulation of the direction field on "
         "the map's own graph. Seam closure = share of the seam cells' 15 nearest cells lying on the "
         f"other side of the seam (30-PC space: {out['reference_closure_30pc']:.3f}). Step 1: saved "
         "2000-iteration fits; step 2: refits with early exaggeration (12×, 250 its) + cosine decay; "
         "step 3: flow alone refitted on the frozen FlowSNE-10 layout (seed 0), cosine decay.", "",
         "| velocity | step | map / field | " + " | ".join(f"s={s}" for s in steps) + " | "
         + " | ".join(f"s={s} seam" for s in steps) + " | circulation | seam closure | |ω| q10/50/90 | loss |",
         "|---|---|---|" + "---|" * (2 * len(steps) + 4)]
    for name, r in out["rows"].items():
        v, step, what = (x.strip() for x in name.split("|"))
        L.append(f"| {v} | {step} | {what} | " + " | ".join(f"{r[f's{s}']:+.4f}" for s in steps) + " | "
                 + " | ".join(f"{r[f's{s}_seam']:+.4f}" for s in steps)
                 + f" | {r['circulation']:+.3f} | {r['seam_closure']:.3f} | {r.get('omega_q10_50_90', '–')}"
                 f" | {r.get('loss', '–')} |")
    open(os.path.join(res, "DIAG.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "DIAG.md"))


if __name__ == "__main__":
    main()
