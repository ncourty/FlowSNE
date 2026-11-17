"""Why does FlowSNE fold the noisy cycle's ring (Table 2 / Fig. 4) while the
symmetric fit of the same affinity draws a circle?  Refit seed 0 under
variants of the joint fit and score the ring's shape:

  convexity   area of the polygon through the 24 phase-bin centroids of the
              ring, over the area of its convex hull (1 = convex loop)
  turning     total absolute turning of that polygon / 2 pi (1 = convex loop)
plus the Table 2 metrics (direction, speed, kNN, kNN_dir).

    conda run -n dev python experiments/baselines/cycle_fold.py --variant ee --key ds
Writes results/fold/{variant}_{key}.json / .npz; `report` tabulates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finsler_embedding_comparison import (  # noqa: E402
    DT, RES, RHO, EntropicRandersAffinity, RandersAffinity, evaluate, input_std_norm_flow,
    load_dataset)
from flowsne.embedding import fit_embedding  # noqa: E402

OUT = os.path.join(RES, "fold")
VARIANTS = {
    "base":     dict(),
    "ee":       dict(early_exaggeration=12.0, exaggeration_iters=250),
    "ee_cos":   dict(early_exaggeration=12.0, exaggeration_iters=250, lr_schedule="cosine"),
    "rho05":    dict(rho_out=0.5),
    "syminit":  dict(init="sym"),
    "flowlr":   dict(flow_lr=0.003),
    "warmup":   dict(warmup_iters=500),
    "linear":   dict(lift="linear"),
    "syminit_rho07": dict(init="sym", rho_out=0.7),      # no sine features: is the fold the lift's doing?
}


def ring_shape(Z, phase, branch, B=24):
    ring = ~branch
    bins = (phase[ring] * B).astype(int) % B
    C = np.array([Z[ring][bins == b].mean(0) for b in range(B)])
    def area(P):
        x, y = P[:, 0], P[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    from scipy.spatial import ConvexHull
    hull = ConvexHull(C).volume
    e = np.roll(C, -1, 0) - C
    ang = np.arctan2(e[:, 1], e[:, 0])
    turn = np.abs((np.diff(np.r_[ang, ang[0]]) + np.pi) % (2 * np.pi) - np.pi).sum()
    return dict(convexity=float(area(C) / hull), turning=float(turn / (2 * np.pi)))


def run(variant, key, seed=0, iters=3000):
    opts = dict(VARIANTS[variant])
    if opts.pop("lift", None) == "linear":
        import finsler_embedding_comparison as C
        C.CYCLE_FEATURES = 0
    D = load_dataset("cycle", seed)
    X, om = D["X"], D["omega"]
    Xt, omt = torch.tensor(X, dtype=DT), torch.tensor(om, dtype=DT)
    P = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO, doubly_stochastic=(key == "ds"))(Xt).detach()
    S = np.load(os.path.join(RES, f"cycle_seed{seed}.npz"))
    if variant == "linear":        # the symmetric fit of this input, for the shape reference
        from torchdr import AffinityMatcher, NormalizedStudentAffinity
        from finsler_embedding_comparison import fit_matcher
        from sklearn.decomposition import PCA
        sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
        Zs = fit_matcher(sym, P, PCA(2, random_state=seed).fit_transform(X), AffinityMatcher,
                         seed, 1500, 0.1).detach().numpy()
        print(f"[fold] sym_{key}_linear       " + " ".join(f"{k}={v:.3f}" for k, v in
              dict(knn=evaluate(D, Zs, np.zeros_like(Zs))["knn"],
                   **ring_shape(Zs, D["color"], D["branch"])).items()), flush=True)
    if opts.pop("init", None) == "sym":
        init, rescale = torch.tensor(S[f"sym_{key}"], dtype=DT), False
    else:
        from sklearn.decomposition import PCA
        init, rescale = torch.tensor(PCA(2, random_state=seed).fit_transform(X), dtype=DT), True
    rho_out = opts.pop("rho_out", 0.9)
    torch.manual_seed(seed)
    aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                          degrees_of_freedom=1.0, max_omega_norm=rho_out).double()
    t0 = time.time()
    Z = fit_embedding(aff, P, init, max_iter=iters, lr=0.03, seed=seed, rescale_init=rescale,
                      log_every=1000, **opts).detach()
    with torch.no_grad():
        F = -aff.omega_at(Z).numpy()
        loss = aff.cross_entropy(Z, P).item()
    name = f"flowsne_{key}_{variant}"
    r = dict(method=name, variant=variant, key=key, secs=time.time() - t0, loss=loss,
             **evaluate(D, Z.numpy(), F, name), **ring_shape(Z.numpy(), D["color"], D["branch"]),
             omega_med=float(np.median(np.linalg.norm(F, axis=1))))
    print(f"[fold] {name:22s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()
                                             if isinstance(v, float) and k != "secs"), flush=True)
    os.makedirs(OUT, exist_ok=True)
    sfx = f"_s{seed}" if seed else ""
    r["seed"] = seed
    json.dump(r, open(os.path.join(OUT, f"{variant}_{key}{sfx}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{variant}_{key}{sfx}.npz"), Z=Z.numpy(), drift=F)


def report():
    D = load_dataset("cycle", 0)
    S = np.load(os.path.join(RES, "cycle_seed0.npz"))
    rows = []
    for m in ("sym_ds", "sym_rs", "finsler_mds", "theirs_isomap", "flowsne_ds_joint", "flowsne_rs_joint"):
        r = dict(method=m, **ring_shape(S[m], D["color"], D["branch"]))
        rows.append(r)
    for f in sorted(os.listdir(OUT)) if os.path.exists(OUT) else []:
        if f.endswith(".json"):
            rows.append(json.load(open(os.path.join(OUT, f))))
    keys = ["cos_ref", "speed_spearman", "knn", "knn_dir", "loss", "omega_med", "convexity", "turning"]
    print(f"{'method':24s} " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for r in rows:
        r = dict(r, method=r["method"] + (f"_s{r['seed']}" if r.get("seed") else ""))
        print(f"{r['method']:24s} " + " ".join(f"{r[k]:9.3f}" if k in r else f"{'':>9s}" for k in keys))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "report"])
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--key", default="ds", choices=["ds", "rs"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run(a.variant, a.key, a.seed) if a.cmd == "run" else report()
