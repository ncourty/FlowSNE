"""Baseline comparison against Debaussart-Joniec, Blanchard & Kalogeratos,
"A Finslerian Approach for Embedding Directed Data" (arXiv 2609.37649):
spectral / diffusion-maps construction, drift in the embedding estimated as
V = L^a[Psi] (antisymmetric graph operator applied to the embedding
coordinates). Their code (MIT): github.com/Gwendal-Debaussart/Finsler-Embedding
-- only its self-contained `operators.py` is used; point FINSLER_EMBEDDING_SRC
at a clone's `src/` directory.

Same input for every method: points X and a per-point drift omega, turned into
a Randers ground cost with the averaged closed-form distance (distances.py),
admissibility bound rho = 0.9 (their experiments' beta), favoured direction =
+omega. Their graph: symmetrised 10-NN edges, Gaussian kernel, eps = std of
edge distances (their `compute_epsilon_empirical`), theta = 1, m = 2.

Methods:
  theirs_isomap        their estimator on an Isomap embedding (their default)
  theirs_eig           their estimator on the leading eigenvectors of W^s
  theirs_on_flowsne    their estimator on FlowSNE's symmetric layout
  flowsne_{ds,rs}_{joint,driftonly}   FlowSNE, balanced / row-stochastic input
  flowsne_{ds,rs}_on_isomap           FlowSNE's flow fit on the Isomap layout
The two hybrids separate "which layout" from "which drift estimator".

Datasets:
  sphere    S^2, rotation drift e_z x x (solenoidal, speed = sin colatitude)
  swiss     sklearn Swiss roll (noise 0.1), drift = unit tangent along the
            roll -- a GRADIENT field on the manifold (of arclength), the case
            where Sinkhorn balancing is predicted to hide the drift
  pancreas  RNA velocity (scVelo), stratified subsample

Reference drift in each embedding (method-agnostic): the push-forward J_i
omega_i, J_i estimated by local linear regression of embedded neighbour
offsets on tangent-projected input offsets (their Table 1 uses the same
J c comparison). Pancreas adds scVelo's own velocity-embedding projection of
its transition matrix onto the same layout, and alignment with the local
velocity-pseudotime gradient -- agreement measures, not ground truth.

Run: FINSLER_EMBEDDING_SRC=/path/to/Finsler-Embedding/src \
     conda run -n dev python experiments/baselines/finsler_embedding_comparison.py run --dataset sphere --seed 0
     conda run -n dev python experiments/baselines/finsler_embedding_comparison.py report
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
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "single_cell"))
sys.path.insert(0, os.environ.get("FINSLER_EMBEDDING_SRC",
                                  os.path.join(REPO, "..", "Finsler-Embedding", "src")))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    input_std_norm_flow, fit_drift_only, randers_distance,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

RES = os.path.join(HERE, "results")
DT = torch.float64
RHO = 0.9


# ===========================================================================
# Data
# ===========================================================================

def load_dataset(name, seed, quick=False):
    rng = np.random.default_rng(seed)
    if name == "sphere":
        n = 300 if quick else 1500
        v = rng.standard_normal((n, 3))
        X = v / np.linalg.norm(v, axis=1, keepdims=True)
        om = np.stack([-X[:, 1], X[:, 0], np.zeros(n)], axis=1)
        return dict(X=X, omega=om, color=np.arctan2(X[:, 1], X[:, 0]), tan_dim=2)
    if name == "swiss":
        from sklearn.datasets import make_swiss_roll
        n = 300 if quick else 1500
        X, t = make_swiss_roll(n_samples=n, noise=0.1, random_state=seed)
        tang = np.stack([np.cos(t) - t * np.sin(t), np.zeros(n), np.sin(t) + t * np.cos(t)], 1)
        om = tang / np.linalg.norm(tang, axis=1, keepdims=True)
        return dict(X=X, omega=om, color=t, tan_dim=2)
    if name == "pancreas":
        from pancreas_data import load
        d = load(subsample=40 if quick else 150, seed=0)  # fixed cells; seed varies the fits
        return dict(X=d["X_pca"], omega=d["V_pca"], color=d["pseudotime"], tan_dim=5,
                    pseudotime=d["pseudotime"], T=d["T_scvelo"], clusters=d["clusters"])
    raise ValueError(name)


# ===========================================================================
# Reference drifts + metrics
# ===========================================================================

def pushforward(X, Z, omega, tan_dim, k=None):
    """J_i omega_i by local linear regression of embedded offsets on
    tangent-projected input offsets."""
    from sklearn.neighbors import NearestNeighbors
    k = k or (15 if tan_dim <= 2 else 30)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    out = np.zeros((len(X), Z.shape[1]))
    for i in range(len(X)):
        dX = X[idx[i]] - X[i]
        dZ = Z[idx[i]] - Z[i]
        _, _, Vt = np.linalg.svd(dX - dX.mean(0), full_matrices=False)
        T = Vt[:tan_dim].T                       # (D, tan_dim) local tangent basis
        A = np.linalg.lstsq(dX @ T, dZ, rcond=None)[0]   # (tan_dim, 2)
        out[i] = (omega[i] @ T) @ A
    return out


def scvelo_projection(T, Z):
    """scVelo's velocity_embedding: sum_j T_ij u_ij - mean_{j in N(i)} u_ij,
    u_ij the unit embedded offset."""
    T = T.tocsr()
    out = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        cols = T.indices[T.indptr[i]:T.indptr[i + 1]]
        w = T.data[T.indptr[i]:T.indptr[i + 1]]
        if len(cols) == 0:
            continue
        u = Z[cols] - Z[i]
        u /= np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
        out[i] = w @ u - u.mean(0)
    return out


def local_gradient(Z, f, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    g = np.zeros_like(Z)
    for i in range(len(Z)):
        dZ = Z[idx[i]] - Z[i]
        g[i] = np.linalg.lstsq(dZ, f[idx[i]] - f[i], rcond=None)[0]
    return g


def _cos(A, B):
    """Row-wise cosine, scale-free (an absolute epsilon in the denominator
    silently zeroes it for embeddings with a tiny overall scale, e.g. raw
    eigenvectors); rows where either vector is exactly 0 give 0."""
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
    return np.where(den > 0, (A * B).sum(1) / np.where(den > 0, den, 1.0), 0.0)


def knn_preservation(X, Z, k=10):
    from sklearn.neighbors import NearestNeighbors
    a = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(X))]))


def evaluate(D, Z, drift):
    from scipy.stats import spearmanr
    ref = pushforward(D["X"], Z, D["omega"], D["tan_dim"])
    ref_norm = np.linalg.norm(ref, axis=1)
    keep = ref_norm >= np.quantile(ref_norm, 0.2)
    c = _cos(drift, ref)
    out = dict(cos_ref=float(c[keep].mean()), cos_ref_median=float(np.median(c[keep])),
               speed_spearman=float(spearmanr(np.linalg.norm(drift, axis=1), ref_norm)[0]),
               knn=knn_preservation(D["X"], Z))
    if "T" in D:
        out["cos_scvelo"] = float(_cos(drift, scvelo_projection(D["T"], Z)).mean())
        out["pt_align"] = float(_cos(drift, local_gradient(Z, D["pseudotime"])).mean())
        out["scvelo_vs_ref"] = float(_cos(scvelo_projection(D["T"], Z), ref)[keep].mean())
    return out


# ===========================================================================
# Methods
# ===========================================================================

def _their_operators():
    """Load their `operators.py` by path: the package __init__ imports their
    private `geodesic_toolbox`, which operators.py itself does not need."""
    import importlib.util
    for root in sys.path:
        f = os.path.join(root, "finsler_embedding", "operators.py")
        if os.path.exists(f):
            spec = importlib.util.spec_from_file_location("fe_operators", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("set FINSLER_EMBEDDING_SRC to a Finsler-Embedding clone's src/")


def their_drift(W, eps, X_low):
    randers_approximation = _their_operators().randers_approximation
    res = randers_approximation(W.float(), eps, X_low=None if X_low is None else
                                torch.as_tensor(X_low, dtype=torch.float32), m=2)
    return res["X_low"].double().numpy(), res["V"].double().numpy()


def their_graph(X, omega, k=10):
    """Symmetrised k-NN edges, directed Randers distance on each edge (favoured
    direction +omega, same cost as FlowSNE's input), Gaussian kernel with
    eps = std(edge distances) -- their `get_knn_graph` / `compute_epsilon_empirical`
    / `apply_gaussian_kernel`."""
    from sklearn.neighbors import kneighbors_graph
    n = len(X)
    G = kneighbors_graph(X, n_neighbors=k, include_self=False)
    G = ((G + G.T) > 0).tocoo()
    Xt = torch.tensor(X, dtype=DT)
    om = torch.tensor(omega, dtype=DT)
    b = -RHO * om / om.norm(dim=1).max()  # d = |dx| + b.dx, favours +omega
    dF = randers_distance(Xt, b, mode="averaged")
    rows, cols = torch.tensor(G.row), torch.tensor(G.col)
    d_e = dF[rows, cols]
    eps = d_e.std().item()
    W = torch.zeros(n, n, dtype=DT)
    W[rows, cols] = torch.exp(-d_e ** 2 / eps ** 2)
    return W, eps


def fit_matcher(aff, P, init_np, cls, seed, max_iter, lr):
    torch.manual_seed(seed)
    dr = cls(affinity_in="precomputed", affinity_out=aff, n_components=2,
             loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
             init=init_np, init_scaling=1.0, random_state=seed, verbose=False)
    return dr.fit_transform(P)


def run(dataset, seed, quick=False):
    it = (lambda m: 50) if quick else (lambda m: m)
    D = load_dataset(dataset, seed, quick)
    X, om = D["X"], D["omega"]
    n = len(X)
    Xt = torch.tensor(X, dtype=DT)
    omt = torch.tensor(om, dtype=DT)
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X)
    Z_iso = Isomap(n_components=2, n_neighbors=10).fit_transform(X)

    tag = f"{dataset}{'_quick' if quick else ''}_seed{seed}"
    out = dict(dataset=dataset, seed=seed, n=n, runs=[])
    saved = dict(color=D["color"])

    def record(name, Z, drift, secs, extra=None):
        Z = np.asarray(Z, dtype=np.float64)
        drift = np.asarray(drift, dtype=np.float64)
        r = dict(method=name, secs=secs, **evaluate(D, Z, drift), **(extra or {}))
        out["runs"].append(r)
        saved[name] = Z
        saved[name + "_drift"] = drift
        print(f"[{dataset} s{seed}] {name:24s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs")
            + f"  ({secs:.0f}s)", flush=True)
        os.makedirs(RES, exist_ok=True)
        json.dump(out, open(os.path.join(RES, tag + ".json"), "w"), indent=1)
        np.savez_compressed(os.path.join(RES, tag + ".npz"), **saved)

    # --- FlowSNE symmetric layout (shared by a hybrid)
    P_ds = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO)(Xt).detach()
    P_rs = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO, doubly_stochastic=False)(Xt).detach()
    sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
    Z_sym = {}
    for key, P in (("ds", P_ds), ("rs", P_rs)):
        Z_sym[key] = fit_matcher(sym, P, init_np, AffinityMatcher, seed, it(1500), 0.1).detach()

    # --- theirs
    W, eps = their_graph(X, om)
    for name, low in (("theirs_isomap", Z_iso), ("theirs_eig", None),
                      ("theirs_on_flowsne", Z_sym["ds"].numpy())):
        t0 = time.time()
        Zl, V = their_drift(W, eps, low)
        record(name, Zl, V, time.time() - t0)

    # --- FlowSNE
    for key, P in (("ds", P_ds), ("rs", P_rs)):
        for how in ("joint", "driftonly"):
            torch.manual_seed(seed)
            aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                                  degrees_of_freedom=1.0).double()
            t0 = time.time()
            if how == "joint":
                Z = fit_matcher(aff, P, init_np, JointAffinityMatcher, seed, it(3000), 0.03)
            else:
                Z = fit_drift_only(aff, P, Z_sym[key], seed=seed, max_iter=it(1500), lr=0.05)
            Z = Z.detach()
            with torch.no_grad():
                loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
                F = -aff.omega_at(Z).numpy()
            record(f"flowsne_{key}_{how}", Z.numpy(), F, time.time() - t0, dict(loss=loss))
        # hybrid: FlowSNE's flow on the Isomap layout
        torch.manual_seed(seed)
        aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                              degrees_of_freedom=1.0).double()
        t0 = time.time()
        Zi = torch.tensor(Z_iso, dtype=DT)
        fit_drift_only(aff, P, Zi, seed=seed, max_iter=it(1500), lr=0.05)
        with torch.no_grad():
            F = -aff.omega_at(Zi).numpy()
        record(f"flowsne_{key}_on_isomap", Z_iso, F, time.time() - t0)


# ===========================================================================
# Report
# ===========================================================================

def report():
    files = sorted(f for f in os.listdir(RES) if f.endswith(".json") and "_quick" not in f)
    E = [json.load(open(os.path.join(RES, f))) for f in files]
    # recompute every metric from the saved embeddings with the current evaluate()
    for e in E:
        D = load_dataset(e["dataset"], e["seed"])
        S = np.load(os.path.join(RES, f"{e['dataset']}_seed{e['seed']}.npz"))
        for r in e["runs"]:
            r.update(evaluate(D, S[r["method"]], S[r["method"] + "_drift"]))
    ms = lambda v: "–" if not v else (f"{np.mean(v):+.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:+.3f}")
    L = ["# Baseline: FlowSNE vs Finsler directed embedding (Debaussart-Joniec et al. 2026)", "",
         "See `finsler_embedding_comparison.py` docstring. `cos_ref` = cosine between the estimated "
         "drift and the push-forward of the true input drift into that same embedding (points above "
         "the 20th percentile of reference speed). `speed ρ` = Spearman(|estimated|, |reference|). "
         "`knn` = 10-NN preservation of the layout. Mean ± std over seeds.", ""]
    for ds in ("sphere", "swiss", "pancreas"):
        sel = [e for e in E if e["dataset"] == ds]
        if not sel:
            continue
        keys = ["cos_ref", "cos_ref_median", "speed_spearman", "knn"]
        if ds == "pancreas":
            keys += ["cos_scvelo", "pt_align", "scvelo_vs_ref"]
        L += [f"## {ds} (n={sel[0]['n']}, {len(sel)} seed(s))", "",
              "| method | " + " | ".join(keys) + " | secs |", "|---|" + "---|" * (len(keys) + 1)]
        for m in [r["method"] for r in sel[0]["runs"]]:
            rr = [r for e in sel for r in e["runs"] if r["method"] == m]
            L.append(f"| {m} | " + " | ".join(ms([r[k] for r in rr if k in r]) for k in keys)
                     + f" | {np.mean([r['secs'] for r in rr]):.0f} |")
        L.append("")
    L += ["![embeddings](results/comparison.png)", ""]
    open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L))
    _figure()
    print("wrote RESULTS.md")


def _figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [ds for ds in ("sphere", "swiss", "pancreas")
            if os.path.exists(os.path.join(RES, f"{ds}_seed0.npz"))]
    cols = ["theirs_isomap", "theirs_eig", "theirs_on_flowsne", "flowsne_ds_joint",
            "flowsne_rs_joint", "flowsne_rs_on_isomap"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.6 * len(cols), 3.6 * len(rows)),
                             squeeze=False)
    for r, ds in enumerate(rows):
        S = np.load(os.path.join(RES, f"{ds}_seed0.npz"))
        c = S["color"]
        sub = np.random.default_rng(0).choice(len(c), size=min(len(c), 200), replace=False)
        for j, m in enumerate(cols):
            ax = axes[r, j]
            if m not in S.files:
                ax.axis("off"); continue
            Z, F = S[m], S[m + "_drift"]
            ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="twilight" if ds == "sphere" else "viridis", s=4)
            Fu = F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
            sc = 0.05 * (Z.max(0) - Z.min(0)).max()
            ax.quiver(Z[sub, 0], Z[sub, 1], Fu[sub, 0], Fu[sub, 1], color="black",
                      scale=1 / sc, scale_units="xy", angles="xy", width=0.004)
            ax.set_title(f"{ds}: {m}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("Seed 0 — estimated drift direction (unit arrows, 200 random points)", fontsize=11)
    fig.savefig(os.path.join(RES, "comparison.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--dataset", default="sphere")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    run(a.dataset, a.seed, a.quick) if a.cmd == "run" else report()
