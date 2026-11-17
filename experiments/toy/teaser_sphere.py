"""One-column teaser: the sphere vortex in R^3 (left) and its FlowSNE
embedding with the fitted drift (right), both coloured by azimuth.

n = 3000 points, fitted here with the protocol of
experiments/toy/stream_sphere.py (free flow input_std_norm_flow(2, 32),
balanced input, rho = 0.9, joint fit, Adam lr 0.03, 3000 its, PCA init).
Arrows on the right are -omega_theta, the direction the fitted Randers
kernel favours.

Run:    conda run -n dev python experiments/toy/teaser_sphere.py fit --seed 0 --norm ds
        conda run -n dev python experiments/toy/teaser_sphere.py
Writes: experiments/toy/results/teaser_sphere_n3000.npz,
        article/ICML_FlowSNE/figures/teaser_sphere.{pdf,png}
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "experiments", "baselines"))

N, SEED = 3000, 0
OUT = os.path.join(REPO, "article", "ICML_FlowSNE", "figures", "teaser_sphere")
ELEV, AZIM = 48, -60
CMAP = "coolwarm"  # height z: red north, blue south

plt.rcParams.update({"font.size": 7, "font.family": "serif", "mathtext.fontset": "cm"})


def sphere(n, seed):
    """Same construction as load_dataset("sphere"), any n."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    X = v / np.linalg.norm(v, axis=1, keepdims=True)
    V = np.stack([-X[:, 1], X[:, 0], np.zeros(n)], axis=1)
    return X, V, np.arctan2(X[:, 1], X[:, 0])


def fit(seed, norm, iters=6000, lr=0.03, flow_lr=None, schedule="constant", warmup=0,
        tol=0.0, tag_suffix=""):
    """Joint fit with the loss logged every 100 iterations (fit_embedding
    reproduces TorchDR's matcher step for step), then direction / speed /
    kNN against the push-forward of the true drift and the vortex centres."""
    import json
    import time
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(REPO, "src"))
    from flowsne import EntropicRandersAffinity
    from flowsne.embedding import fit_embedding
    from finsler_embedding_comparison import RHO, evaluate
    from sklearn.decomposition import PCA
    from stream_sphere import critical_points, make
    X, V, _ = sphere(N, seed)
    t0 = time.time()
    P = EntropicRandersAffinity(omega=torch.tensor(V), max_omega_norm=RHO,
                                doubly_stochastic=(norm == "ds"))(torch.tensor(X)).detach()
    init = PCA(n_components=2, random_state=seed).fit_transform(X)
    aff = make("flow", seed)
    hist, diag = [], []
    Z = fit_embedding(aff, P, init, max_iter=iters, lr=lr, flow_lr=flow_lr, seed=seed,
                      history=hist, log_every=100, lr_schedule=schedule, warmup_iters=warmup,
                      tol=tol, diagnostics=diag).detach()
    with torch.no_grad():
        F = -aff.omega_at(Z).numpy()
    Z = Z.numpy()
    r = evaluate(dict(X=X, omega=V, tan_dim=2), Z, F)
    cps = critical_points(Z, F, X[:, 2])
    cen = [c for c in cps if c["kind"] == "centre"]
    r.update(secs=time.time() - t0, history=[(int(i), float(l)) for i, l in hist],
             centres=[round(c["absz"], 2) for c in cen],
             others=sum(c["kind"] != "centre" for c in cps), diagnostics=diag,
             iters_run=hist[-1][0] + 1,
             config=dict(lr=lr, flow_lr=flow_lr, schedule=schedule, warmup=warmup, tol=tol,
                         max_iter=iters))
    tag = f"teaser_sphere_n{N}_{norm}_seed{seed}{tag_suffix}"
    np.savez_compressed(os.path.join(HERE, "results", tag + ".npz"), Z=Z, F=F)
    json.dump(r, open(os.path.join(HERE, "results", tag + ".json"), "w"), indent=1)
    print(tag, {k: v for k, v in r.items() if k not in ("history", "diagnostics")}, flush=True)


def main(seed=SEED, norm="ds", out=OUT, tag_suffix=""):
    X, V, _ = sphere(N, seed)
    c = X[:, 2]  # latitude colouring
    d = np.load(os.path.join(HERE, "results", f"teaser_sphere_n{N}_{norm}_seed{seed}{tag_suffix}.npz"))
    Z, F = d["Z"], d["F"]

    fig = plt.figure(figsize=(3.3, 1.75))
    ax3 = fig.add_axes([-0.04, -0.02, 0.54, 0.98], projection="3d")
    axz = fig.add_axes([0.53, 0.02, 0.46, 0.86])

    # 3-D: points coloured by azimuth, arrows on the hemisphere facing the viewer
    el, az = np.radians(ELEV), np.radians(AZIM)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    front = X @ view > 0.05
    ax3.scatter(*X[~front].T, c=c[~front], cmap=CMAP, vmin=-1, vmax=1, s=1.0,
                alpha=0.15, linewidths=0, depthshade=False)
    ax3.scatter(*X[front].T, c=c[front], cmap=CMAP, vmin=-1, vmax=1, s=2.4,
                linewidths=0, depthshade=False)
    rng = np.random.default_rng(1)
    for lat in (-0.5, 0.0, 0.5):  # faint latitude circles: the drift's orbits
        t = np.linspace(0, 2 * np.pi, 200)
        r = np.sqrt(1 - lat ** 2)
        P = np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, lat)], 1)
        vis = P @ view > 0
        P[~vis] = np.nan
        ax3.plot(*P.T, color="0.55", lw=0.4, zorder=0)
    idx = np.flatnonzero(X @ view > 0.3)
    idx = idx[rng.choice(len(idx), 110, replace=False)]
    ax3.quiver(*X[idx].T, *V[idx].T, length=0.2, normalize=False, color="k",
               linewidth=0.9, arrow_length_ratio=0.45)
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_xlim(-0.72, 0.72); ax3.set_ylim(-0.72, 0.72); ax3.set_zlim(-0.72, 0.72)
    ax3.set_axis_off()
    fig.text(0.25, 0.97, r"data in $\mathbb{R}^3$, velocity $e_z\times x$", ha="center", va="top")

    # 2-D FlowSNE embedding with the fitted drift
    axz.scatter(Z[:, 0], Z[:, 1], c=c, cmap=CMAP, vmin=-1, vmax=1, s=1.8,
                linewidths=0, rasterized=True)
    sel = rng.choice(len(Z), 200, replace=False)
    span = (Z.max(0) - Z.min(0)).max()
    axz.quiver(Z[sel, 0], Z[sel, 1], F[sel, 0], F[sel, 1], color="k", angles="xy",
               scale_units="xy", scale=np.linalg.norm(F, axis=1).max() / (0.06 * span),
               width=0.008, headwidth=3.2, headlength=3.6, headaxislength=3.2)
    axz.set_aspect("equal"); axz.set_xticks([]); axz.set_yticks([])
    for s in axz.spines.values():
        s.set_visible(False)
    fig.text(0.76, 0.97, r"FlowSNE, fitted drift $-\omega_\theta$", ha="center", va="top")

    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=300)
    print("wrote", out + ".{pdf,png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="figure", choices=["fit", "figure"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["ds", "rs"], default="ds")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--flow-lr", type=float, default=None)
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="suffix of the result files")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a.seed, a.norm, a.iters, a.lr, a.flow_lr, a.schedule, a.warmup, a.tol, a.tag)
    else:
        main(a.seed, a.norm, a.out, a.tag)
