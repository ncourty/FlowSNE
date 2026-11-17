"""Integers 0..359 in GPT-2 small's residual stream (template means), embedded
at several depths with one of two drifts:
  --drift successor  omega_n = h_l(n+1) - h_l(n)   (the number-line relation)
  --drift residual   omega_n = h_{l+1}(n) - h_l(n) (what layer l does to n)
``fit_points`` is generic (any points + drift + successor map) and is reused by
``cyclic_embed.py`` (days, months).

Methods: symmetric t-SNE (balanced input), Finsler t-SNE (constant drift,
alpha on its own cross-entropy, row-stochastic input), FlowSNE joint fit
(balanced, row-stochastic). Input space: the layer's template-mean states,
centred, PCA to 50 dims (the drift through the same projection).

    conda run -n dev python experiments/transformer/numbers/numbers_embed.py fit --layer 6
    conda run -n dev python experiments/transformer/numbers/numbers_embed.py figure
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
from flowsne import (EntropicRandersAffinity, JointAffinityMatcher, RandersAffinity,  # noqa: E402
                     finsler_tsne, input_std_norm_flow, select_alpha)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402

RES = os.path.join(HERE, "results")
DT = torch.float64
RHO = 0.9
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8)


def project(H, drift, n_pca=50):
    """Centre, PCA to n_pca dims; the drift goes through the same linear map."""
    from sklearn.decomposition import PCA
    mu = H.mean(0)
    pca = PCA(n_components=n_pca, random_state=0).fit(H - mu)
    return pca.transform(H - mu), drift @ pca.components_.T


def load(layer, drift="successor"):
    d = np.load(os.path.join(HERE, "data", "numbers_gpt2.npz"))
    Hm = d["H"].mean(1).astype(np.float64)               # (L+1, 361, 768)
    if drift == "successor":
        om = Hm[layer, 1:] - Hm[layer, :-1]
    else:
        om = Hm[layer + 1, :-1] - Hm[layer, :-1]
    X, om = project(Hm[layer, :-1], om)
    num = d["numbers"][:-1]
    nxt = np.append(np.arange(1, len(num)), -1)          # successor index, -1: none
    return X, om, num, nxt


def fit_matcher(aff, P, init_np, cls, seed, max_iter, lr):
    torch.manual_seed(seed)
    dr = cls(affinity_in="precomputed", affinity_out=aff, n_components=init_np.shape[1],
             loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
             init=init_np, init_scaling=1.0, random_state=seed, verbose=False)
    return dr.fit_transform(P).detach()


def succ_cos(Z, F, nxt):
    """mean cos between a method's drift at z_i and the step to its successor."""
    has = nxt >= 0
    S = Z[nxt[has]] - Z[has]
    F = F[has]
    c = (S * F).sum(1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(F, axis=1)).clip(1e-12)
    return float(c.mean())


def knn(X, Z, k=10):
    from sklearn.neighbors import NearestNeighbors
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


def fit(layer, seed=0, iters=3000, drift="successor", dim=2):
    X, om, num, nxt = load(layer, drift)
    tag = (f"layer{layer}" + ("" if drift == "successor" else f"_{drift}")
           + ("" if dim == 2 else f"_{dim}d"))
    fit_points(X, om, nxt, dict(numbers=num), tag, seed, iters, dim=dim)


def fit_points(X, om, nxt, saved, tag, seed=0, iters=3000, res=RES, methods=None, dim=2,
               alphas=ALPHAS, finsler_iters=None):
    """methods: which of sym, finsler_tsne, flowsne_ds, flowsne_rs to run, in
    that order (default all, in this order). Results are written after every
    method, so a long run can be read before it ends. Finsler t-SNE searches
    ``alphas`` with ``finsler_iters`` iterations (default ``iters``).
    FlowSNE runs also save the flow's parameter vector and the layout's global
    std, for ``flowsne.viz.field_from_flow``."""
    methods = methods or ("sym", "finsler_tsne", "flowsne_ds", "flowsne_rs")
    Xt, omt = torch.tensor(X, dtype=DT), torch.tensor(om, dtype=DT)
    from sklearn.decomposition import PCA
    init = PCA(n_components=dim, random_state=seed).fit_transform(X)
    P = {}

    def affinity(key):
        if key not in P:
            t0 = time.time()
            P[key] = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO,
                                             doubly_stochastic=key == "ds")(Xt).detach()
            print(f"[{tag}] P_{key} {time.time() - t0:.0f}s", flush=True)
        return P[key]

    out = dict(tag=tag, seed=seed, runs=[])
    saved["nxt"] = nxt
    os.makedirs(res, exist_ok=True)

    def record(name, Z, F, t0, **extra):
        Z = np.asarray(Z, np.float64)
        r = dict(method=name, secs=time.time() - t0, knn=knn(X, Z), **extra)
        if F is not None:
            r["succ_cos"] = succ_cos(Z, F, nxt)
            saved[name + "_drift"] = F
        saved[name] = Z
        out["runs"].append(r)
        print(f"[{tag}] {name:16s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)), flush=True)
        json.dump(out, open(os.path.join(res, f"{tag}_seed{seed}.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(res, f"{tag}_seed{seed}.npz"), **saved)

    for meth in methods:
        t0 = time.time()
        if meth == "sym":
            sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True,
                                            normalization_dim=(0, 1))
            record("sym", fit_matcher(sym, affinity("ds"), init, AffinityMatcher, seed, 1500,
                                      0.1).numpy(), None, t0)
        elif meth == "finsler_tsne":
            Prs = affinity("rs")
            a, Z, loss, grid = select_alpha(
                lambda a: finsler_tsne(Prs, a, init, n_components=dim,
                                       max_iter=finsler_iters or iters, seed=seed), alphas)
            F = np.zeros((len(X), dim))
            F[:, -1] = -a                               # flow direction = -alpha e_d
            record("finsler_tsne", Z.detach().numpy(), F if a > 0 else None, t0, alpha=a,
                   loss=loss)
        else:
            key = meth.split("_")[1]
            Pk = affinity(key)
            t0 = time.time()
            torch.manual_seed(seed)
            aff = RandersAffinity(n_components=dim, flow=input_std_norm_flow(dim, hidden_dim=32),
                                  degrees_of_freedom=1.0).double()
            Z = fit_matcher(aff, Pk, init, JointAffinityMatcher, seed, iters, 0.03)
            with torch.no_grad():
                F = -aff.omega_at(Z).numpy()
                loss = float(aff.cross_entropy(Z, Pk))
            saved[f"{meth}_flow"] = torch.nn.utils.parameters_to_vector(
                aff.flow.parameters()).detach().numpy()
            saved[f"{meth}_zstd"] = float(Z.std())
            record(meth, Z.numpy(), F, t0, loss=loss)


def figure(layers=(0, 4, 8, 12), seed=0, drift="successor"):
    suffix = "" if drift == "successor" else f"_{drift}"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("sym", "symmetric t-SNE"), ("finsler_tsne", "Finsler t-SNE"),
            ("flowsne_ds", "FlowSNE balanced"), ("flowsne_rs", "FlowSNE row-stoch.")]
    layers = [l for l in layers if os.path.exists(os.path.join(RES, f"layer{l}{suffix}_seed{seed}.npz"))]
    fig, axes = plt.subplots(2 * len(layers), len(cols), figsize=(2.6 * len(cols), 5.0 * len(layers)),
                             squeeze=False)
    for r, l in enumerate(layers):
        S = np.load(os.path.join(RES, f"layer{l}{suffix}_seed{seed}.npz"))
        num = S["numbers"]
        for j, (m, title) in enumerate(cols):
            Z = S[m]
            for k, (c, cmap, lab) in enumerate(((num % 10, "tab10", "n mod 10"),
                                                (num, "viridis", "n"))):
                ax = axes[2 * r + k, j]
                ax.plot(Z[:, 0], Z[:, 1], color="0.85", lw=0.4, zorder=0)
                ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=6, linewidths=0)
                if k == 0 and m + "_drift" in S.files:
                    F = S[m + "_drift"]
                    Fu = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
                    sub = np.arange(0, len(Z), 3)
                    sc = 0.04 * np.ptp(Z, 0).max()
                    ax.quiver(Z[sub, 0], Z[sub, 1], Fu[sub, 0], Fu[sub, 1], angles="xy",
                              scale_units="xy", scale=1 / sc, width=0.004, color="k")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                ax.set_title(f"layer {l} - {title} ({lab})", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"numbers{suffix}_seed{seed}.png"), dpi=110)
    print("wrote", os.path.join(RES, f"numbers{suffix}_seed{seed}.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "figure"])
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--drift", default="successor", choices=["successor", "residual"])
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 4, 8, 12])
    ap.add_argument("--dim", type=int, default=2)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "fit":
        fit(a.layer, a.seed, a.iters, a.drift, a.dim)
    else:
        figure(tuple(a.layers), seed=a.seed, drift=a.drift)
