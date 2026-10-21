"""Is FlowSNE's flow network well trained in gastrulation_repr.py's fits?
Reads the saved fits of a gastrulation_repr.py run (same --tag, --subset,
--n-cells) and reports, per embedding dimension and seed:

  loss      cross-entropy of the input Randers affinity P against the fitted
            output kernel: FlowSNE (layout + flow), the same layout with the
            flow switched off (omega = 0), and the symmetric layout. The flow's
            contribution is the gap between the first two.
  converge  the joint fit continued from its saved state for --extra
            iterations, loss logged every --every: still falling = not
            converged at --iters.
  field     |omega| quantiles (saturation near 1, or a near-zero field), and
            the cosine between the fitted field -omega(z_i) and the raw
            velocity projected onto the same layout (agreement, median).
  capacity  the flow refitted alone on the frozen FlowSNE layout, with the
            original network (1 x 32 tanh) and a bigger one (2 x 128), for
            --refit iterations: loss and the cosine-walk score (days gained
            after 25 steps relative to the direction-blind walk, k = 30) --
            same score as gastrulation_repr.py's flowsne<d> row.

Run (GPU): python experiments/single_cell/gastrulation_repr_diag.py --subset full --n-cells 30000 --tag repr
Writes: experiments/single_cell/gastrulation_results/<tag>/{diag.json, DIAG.md}
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_data import load  # noqa: E402
from gastrulation_markov import advances  # noqa: E402
from gastrulation_repr import chain, project_velocity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["full", "e75", "erythroid"], default="full")
    ap.add_argument("--n-cells", type=int, default=30000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--extra", type=int, default=3000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--refit", type=int, default=3000)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--tag", default="repr")
    a = ap.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    res = os.path.join(HERE, "gastrulation_results", a.tag)

    d = load(a.subset, n_cells=a.n_cells, seed=0)
    ok = ~np.isnan(d["stage"])
    X = d["X_pca"][ok].astype(np.float64)
    V = d["V_pca_direct"][ok].astype(np.float64)[:, :X.shape[1]]
    stage = d["stage"][ok].astype(np.float64)
    n = len(X)
    backend = "keops" if (cuda or n > 6000) else None
    Xt = torch.tensor(X, dtype=dtype, device=device)
    perp = a.perplexity if a.perplexity == "auto" else float(a.perplexity)
    P, idx = EntropicRandersAffinity(omega=torch.tensor(V, dtype=dtype, device=device),
                                     doubly_stochastic=False, sparsity=True, perplexity=perp,
                                     backend=backend)(Xt)

    def make(dim, flow):
        return RandersAffinity(n_components=dim, flow=flow, degrees_of_freedom=1.0,
                               backend=backend).to(device=device, dtype=dtype)

    def loss(aff, Z):
        with torch.no_grad():
            return float(aff.cross_entropy(Z, P, idx))

    def walk_score(aff, Z):
        Z_np = Z.cpu().double().numpy()
        with torch.no_grad():
            F = (-aff.omega_at(Z)).cpu().double().numpy()
        adv = advances(chain(Z_np, F, 30), stage, [25])[25]
        adv0 = advances(chain(Z_np, F, 30, directed=False), stage, [25])[25]
        return float((adv - adv0).mean())

    def field(aff, Z):
        Z_np = Z.cpu().double().numpy()
        with torch.no_grad():
            om = aff.omega_at(Z).cpu().double().numpy()
        nrm = np.linalg.norm(om, axis=1)
        Fp = project_velocity(X, V, Z_np)
        cos = (-om * Fp).sum(1) / np.maximum(nrm * np.linalg.norm(Fp, axis=1), 1e-12)
        return dict(omega_norm_q10_50_90_max=[float(np.quantile(nrm, q)) for q in (.1, .5, .9)]
                    + [float(nrm.max())], cos_field_vs_projected_velocity_median=float(np.median(cos)),
                    cos_positive_fraction=float((cos > 0).mean()))

    out = dict(args=vars(a), n=n, runs={})
    for dim in a.dims:
        Zs = torch.tensor(np.load(os.path.join(res, f"Z_sym{dim}_s0.npy")), dtype=dtype, device=device)
        for seed in a.seeds:
            name = f"flowsne{dim}_s{seed}"
            torch.manual_seed(seed)
            aff = make(dim, input_std_norm_flow(dim, hidden_dim=32))
            aff.flow.load_state_dict(torch.load(os.path.join(res, f"flow_flowsne{dim}_s{seed}.pt"),
                                                map_location=device))
            Z = torch.tensor(np.load(os.path.join(res, f"Z_flowsne{dim}_s{seed}.npy")), dtype=dtype,
                             device=device)
            r = dict(loss=dict(flowsne=loss(aff, Z), flowsne_layout_omega0=loss(make(dim, mlp_flow(dim)), Z),
                               sym_layout_omega0=loss(make(dim, mlp_flow(dim)), Zs)),
                     field=field(aff, Z), walk_score_saved=walk_score(aff, Z))
            print(f"{name}: loss {r['loss']}  field {r['field']}", flush=True)

            # convergence: continue the joint fit, logging the loss
            t0 = time.time()
            aff_c, Zc, curve = copy.deepcopy(aff), Z.clone(), []
            for _ in range(a.extra // a.every):
                Zc = fit_embedding(aff_c, P, Zc, indices=idx, max_iter=a.every, lr=a.lr, seed=seed,
                                   rescale_init=False)
                curve.append(loss(aff_c, Zc))
            r["continue_curve"] = curve
            r["continue_walk_score"] = walk_score(aff_c, Zc)
            print(f"  continued {a.extra} its ({time.time() - t0:.0f}s): loss "
                  + " ".join(f"{v:.5f}" for v in curve) + f"  walk {r['continue_walk_score']:+.3f}",
                  flush=True)

            # capacity: refit the flow alone on the frozen layout
            r["refit"] = {}
            for label, (h, layers) in {"1x32": (32, 1), "2x128": (128, 2)}.items():
                torch.manual_seed(seed)
                aff_r = make(dim, input_std_norm_flow(dim, hidden_dim=h, n_hidden_layers=layers))
                fit_embedding(aff_r, P, Z, indices=idx, max_iter=a.refit, lr=a.lr, seed=seed,
                              train_embedding=False)
                r["refit"][label] = dict(loss=loss(aff_r, Z), walk_score=walk_score(aff_r, Z),
                                         **field(aff_r, Z))
                print(f"  drift-only refit {label}: {r['refit'][label]}", flush=True)
            out["runs"][name] = r
            json.dump(out, open(os.path.join(res, "diag.json"), "w"), indent=1)
    write_report(out, res)


def write_report(out, res):
    L = [f"# FlowSNE fit diagnostics ({out['args']['subset']}, n = {out['n']})", "",
         "Loss = cross-entropy of the input Randers affinity against the output kernel (lower is "
         "better). Walk = days gained after 25 steps over the direction-blind walk (k = 30, "
         "cosine rule). cos = median cosine between the fitted field and the raw velocity "
         "projected onto the same layout.", "",
         "| run | loss | loss, ω = 0 | loss, sym layout | |ω| q10/50/90/max | cos | walk |",
         "|---|---|---|---|---|---|---|"]
    for nm, r in out["runs"].items():
        lo, f = r["loss"], r["field"]
        L.append(f"| {nm} | {lo['flowsne']:.5f} | {lo['flowsne_layout_omega0']:.5f} | "
                 f"{lo['sym_layout_omega0']:.5f} | "
                 + "/".join(f"{v:.2f}" for v in f["omega_norm_q10_50_90_max"])
                 + f" | {f['cos_field_vs_projected_velocity_median']:+.2f} | {r['walk_score_saved']:+.3f} |")
    L += ["", f"## Continuing the joint fit (loss every {out['args']['every']} iterations)", ""]
    for nm, r in out["runs"].items():
        L.append(f"- {nm}: " + " → ".join(f"{v:.5f}" for v in r["continue_curve"])
                 + f"; walk afterwards {r['continue_walk_score']:+.3f}")
    L += ["", f"## Flow refitted alone on the frozen layout ({out['args']['refit']} iterations)", "",
          "| run | network | loss | |ω| q10/50/90/max | cos | walk |", "|---|---|---|---|---|---|"]
    for nm, r in out["runs"].items():
        for lab, q in r["refit"].items():
            L.append(f"| {nm} | {lab} | {q['loss']:.5f} | "
                     + "/".join(f"{v:.2f}" for v in q["omega_norm_q10_50_90_max"])
                     + f" | {q['cos_field_vs_projected_velocity_median']:+.2f} | {q['walk_score']:+.3f} |")
    open(os.path.join(res, "DIAG.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", os.path.join(res, "DIAG.md"))


if __name__ == "__main__":
    main()
