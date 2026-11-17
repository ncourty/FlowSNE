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

Constant-drift baselines (`run-finsler`, appended to the same result files;
flowsne.baselines):
  sym_{ds,rs}          symmetric Student fit of the same input (no drift)
  finsler_mds          Finsler MDS (Dages et al. 2025, their corrected SMACOF) on
                       the directed graph geodesics of the same Randers edge costs
  finsler_tsne         Finsler t-SNE (Dages et al. 2026): Student kernel on the
                       canonical Randers cost, constant drift, row-stochastic input
  theirs_on_{fmds,ftsne}  Debaussart-Joniec et al.'s estimator on those layouts
alpha (the constant drift's norm) is picked on each method's own objective
(stress / cross-entropy; t-SNE: 1000-iteration fits) over ALPHAS. The drift of
both is the constant -alpha e_2: direction is scored as for the others, speed
is undefined (constant).

Datasets:
  sphere    S^2, rotation drift e_z x x (solenoidal, speed = sin colatitude)
  swiss     sklearn Swiss roll (noise 0.1), drift = unit tangent along the
            roll -- a GRADIENT field on the manifold (of arclength), the case
            where Sinkhorn balancing is predicted to hide the drift
  cycle     attracting ring + exit branch (2-D latent) lifted to R^50 by a
            smooth nonlinear map, noise on points and velocities: a mixed
            drift (circulation + gradient); reference from the clean latent
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
    randers_geodesics, finsler_mds, finsler_tsne, constant_drift, select_alpha,
    rescale_admissible,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

RES = os.path.join(HERE, "results")
DT = torch.float64
RHO = 0.9
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)


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
    if name == "cycle":
        return noisy_cycle(seed, n=300 if quick else 1500)
    if name in ("pancreas", "pancreas_full"):
        from pancreas_data import load
        sub = 40 if quick else (None if name == "pancreas_full" else 150)
        d = load(subsample=sub, seed=0)  # fixed cells; seed varies the fits
        return dict(X=d["X_pca"], omega=d["V_pca"], color=d["pseudotime"], tan_dim=5,
                    pseudotime=d["pseudotime"], T=d["T_scvelo"], clusters=d["clusters"])
    raise ValueError(name)


CYCLE_DIM, CYCLE_FEATURES = 50, 6
CYCLE_NOISE_X, CYCLE_NOISE_V = 1.5, 1.0   # noise norm / median 10-NN spacing, / median |v|


def cycle_latent(rng, n):
    """2-D latent: an attracting ring (2/3 of the points; rotation, plus a pull
    -2 (r - 1) towards r = 1) and a branch leaving it tangentially at (1, 0),
    the curve (1 + 0.6 s^2, s), s in [0, 1.5], travelled at unit speed."""
    nc = 2 * n // 3
    th = rng.uniform(0, 2 * np.pi, nc)
    r = 1 + 0.08 * rng.standard_normal(nc)
    U_c = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    V_c = np.stack([-np.sin(th), np.cos(th)], 1) - 2 * (r - 1)[:, None] * U_c / r[:, None]
    sb = rng.uniform(0, 1.5, n - nc)
    tan = np.stack([1.2 * sb, np.ones_like(sb)], 1)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([tan[:, 1], -tan[:, 0]], 1)
    U_b = np.stack([1 + 0.6 * sb ** 2, sb], 1) + 0.08 * rng.standard_normal(n - nc)[:, None] * nrm
    U = np.concatenate([U_c, U_b])
    V = np.concatenate([V_c, tan])
    phase = np.concatenate([th / (2 * np.pi), sb / 1.5])
    branch = np.r_[np.zeros(nc, bool), np.ones(n - nc, bool)]
    return U, V, phase, branch


def noisy_cycle(seed, n=1500):
    """Attracting ring + exit branch (cycle_latent) lifted to R^50 by a smooth
    nonlinear map f(u) = Q [u, 0.5 sin(W u + b)] (Q a random orthonormal
    50 x 8 frame, the features fixed across seeds), velocity J_f(u) v; then
    isotropic noise on the points and, independently, on the velocities. The
    drift mixes a circulation (around the ring) and gradients (the pull to
    the ring, the branch). Metrics use the clean latent: U, V_latent."""
    from sklearn.neighbors import NearestNeighbors
    frng = np.random.default_rng(1234)          # the map is part of the dataset
    W = frng.standard_normal((CYCLE_FEATURES, 2)) * 1.2
    b = frng.uniform(0, 2 * np.pi, CYCLE_FEATURES)
    Q = np.linalg.qr(frng.standard_normal((CYCLE_DIM, 2 + CYCLE_FEATURES)))[0]
    rng = np.random.default_rng(seed)
    U, V, phase, branch = cycle_latent(rng, n)
    arg = U @ W.T + b
    F = np.concatenate([U, 0.5 * np.sin(arg)], 1)
    JV = np.concatenate([V, 0.5 * np.cos(arg) * (V @ W.T)], 1)
    X0, om0 = F @ Q.T, JV @ Q.T
    d10 = np.median(NearestNeighbors(n_neighbors=11).fit(X0).kneighbors(X0)[0][:, 10])
    X = X0 + CYCLE_NOISE_X * d10 / np.sqrt(CYCLE_DIM) * rng.standard_normal(X0.shape)
    vs = np.median(np.linalg.norm(om0, axis=1))
    om = om0 + CYCLE_NOISE_V * vs / np.sqrt(CYCLE_DIM) * rng.standard_normal(om0.shape)
    return dict(X=X, omega=om, color=phase, branch=branch, tan_dim=2, U=U, V_latent=V,
                X_clean=X0, omega_clean=om0)


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


def directed_knn_preservation(X, omega_in, Z, omega_out, k=10):
    """Overlap of each point's k nearest TARGETS under the input Randers cost
    (averaged, drift rescaled to RHO, moving with omega_in cheaper -- the cost
    every method receives) and under the method's own output cost
    ||dz|| + mean(omega_out)^T dz (omega_out = 0: Euclidean)."""
    Xt = torch.tensor(X, dtype=DT)
    b = rescale_admissible(-torch.tensor(omega_in, dtype=DT), RHO)
    Ci = randers_distance(Xt, b, mode="averaged").numpy()
    Co = randers_distance(torch.tensor(Z, dtype=DT), torch.tensor(omega_out, dtype=DT),
                          mode="averaged").numpy()
    for C in (Ci, Co):
        np.fill_diagonal(C, np.inf)
    a = np.argpartition(Ci, k, axis=1)[:, :k]
    o = np.argpartition(Co, k, axis=1)[:, :k]
    return float(np.mean([len(set(a[i]) & set(o[i])) / k for i in range(len(X))]))


def output_omega(method, drift):
    """The drift of a method's own output cost: -drift for the Randers-kernel
    methods (they store the flow direction -omega), none for the spectral
    estimator, whose arrows are not an admissible drift of any cost."""
    return np.zeros_like(drift) if method.startswith("theirs") else -np.asarray(drift)


def evaluate(D, Z, drift, method=None):
    from scipy.stats import spearmanr
    if "U" in D:   # synthetic with a known latent: push the clean latent velocity forward
        ref = pushforward(D["U"], Z, D["V_latent"], 2)
    else:
        ref = pushforward(D["X"], Z, D["omega"], D["tan_dim"])
    ref_norm = np.linalg.norm(ref, axis=1)
    keep = ref_norm >= np.quantile(ref_norm, 0.2)
    c = _cos(drift, ref)
    out = dict(cos_ref=float(c[keep].mean()), cos_ref_median=float(np.median(c[keep])),
               speed_spearman=float(spearmanr(np.linalg.norm(drift, axis=1), ref_norm)[0]),
               knn=knn_preservation(D["X"], Z))
    if method is not None:
        out["knn_dir"] = directed_knn_preservation(D["X"], D["omega"], Z,
                                                   output_omega(method, drift))
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
        r = dict(method=name, secs=secs, **evaluate(D, Z, drift, name), **(extra or {}))
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

    # --- theirs (skipped when their code is not available)
    try:
        _their_operators()
        W, eps = their_graph(X, om)
        for name, low in (("theirs_isomap", Z_iso), ("theirs_eig", None),
                          ("theirs_on_flowsne", Z_sym["ds"].numpy())):
            t0 = time.time()
            Zl, V = their_drift(W, eps, low)
            record(name, Zl, V, time.time() - t0)
    except ImportError:
        print("Debaussart-Joniec code not found: skipping theirs_* (set FINSLER_EMBEDDING_SRC)")

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


def run_finsler(dataset, seed, quick=False, only=None):
    """Constant-drift baselines + symmetric layouts, merged into the result
    files written by ``run`` (same data, same input affinities)."""
    it = (lambda m: 50) if quick else (lambda m: m)
    D = load_dataset(dataset, seed, quick)
    X, om = D["X"], D["omega"]
    n = len(X)
    Xt, omt = torch.tensor(X, dtype=DT), torch.tensor(om, dtype=DT)
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X)
    tag = f"{dataset}{'_quick' if quick else ''}_seed{seed}"
    jf, nf = os.path.join(RES, tag + ".json"), os.path.join(RES, tag + ".npz")
    out = json.load(open(jf)) if os.path.exists(jf) else dict(dataset=dataset, seed=seed, n=n, runs=[])
    saved = dict(np.load(nf)) if os.path.exists(nf) else dict(color=D["color"])

    def record(name, Z, drift, secs, extra=None):
        Z, drift = np.asarray(Z, dtype=np.float64), np.asarray(drift, dtype=np.float64)
        r = dict(method=name, secs=secs, **evaluate(D, Z, drift, name), **(extra or {}))
        out["runs"] = [q for q in out["runs"] if q["method"] != name] + [r]
        saved[name], saved[name + "_drift"] = Z, drift
        print(f"[{dataset} s{seed}] {name:24s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs")
            + f"  ({secs:.0f}s)", flush=True)
        os.makedirs(RES, exist_ok=True)
        json.dump(out, open(jf, "w"), indent=1)
        np.savez_compressed(nf, **saved)

    P_ds = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO)(Xt).detach()
    P_rs = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO, doubly_stochastic=False)(Xt).detach()
    sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
    todo = lambda name: only is None or name in only  # noqa: E731
    prev = {r["method"]: r for r in out["runs"]}
    for key, P in (("ds", P_ds), ("rs", P_rs)):
        if not todo("sym"):
            continue
        t0 = time.time()
        Z = fit_matcher(sym, P, init_np, AffinityMatcher, seed, it(1500), 0.1).detach().numpy()
        record(f"sym_{key}", Z, np.zeros_like(Z), time.time() - t0)

    try:
        _their_operators()
        W, eps = their_graph(X, om)
        theirs = lambda name, Zl: record(name, *their_drift(W, eps, Zl), 0.0)  # noqa: E731
    except ImportError:
        print("Debaussart-Joniec code not found: skipping theirs_on_* (set FINSLER_EMBEDDING_SRC)")
        theirs = lambda name, Zl: None  # noqa: E731

    if todo("finsler_mds"):
        t0 = time.time()
        Dg = randers_geodesics(X, om, k=10, max_omega_norm=RHO)
        a, Y, stress, grid = select_alpha(
            lambda a: finsler_mds(Dg, a, n_init=4, max_iter=it(300), seed=seed), ALPHAS)
        record("finsler_mds", Y, constant_drift(n, a), time.time() - t0,
               dict(alpha=a, stress=stress, alpha_grid={str(k): v for k, v in grid.items()}))
        theirs("theirs_on_fmds", Y)

    if todo("finsler_tsne"):
        # selection fits already on file are reused (same protocol), so extending
        # ALPHAS only runs the new values; the final fit is redone only if the
        # selected alpha changes
        t0 = time.time()
        old = prev.get("finsler_tsne", {})
        grid = {float(k): v for k, v in old.get("alpha_grid", {}).items()}
        for a in ALPHAS:
            if a not in grid:
                grid[a] = finsler_tsne(P_rs, a, init_np, max_iter=it(1000), seed=seed)[1]
        a = min(grid, key=grid.get)
        if old.get("alpha") == a and "finsler_tsne" in saved:
            Z, loss, secs = saved["finsler_tsne"], old["loss"], old["secs"]
        else:
            Z, loss = finsler_tsne(P_rs, a, init_np, max_iter=it(3000), seed=seed)
            Z, secs = Z.numpy(), time.time() - t0
        record("finsler_tsne", Z, constant_drift(n, a), secs,
               dict(alpha=a, loss=loss, alpha_grid={str(k): v for k, v in grid.items()}))
        theirs("theirs_on_ftsne", Z)


def run_ws(dataset, seed, iters=3000):
    """FlowSNE joint fits WARM-STARTED from the symmetric layout of the same
    affinity (``sym_{key}`` on file, no rescaling) instead of PCA, merged into
    the result files as ``flowsne_{key}_joint_ws``. Motivation: on the noisy
    cycle the PCA-started joint fit settles in a more folded layout with a
    higher loss than the warm-started one (cycle_fold.py)."""
    from flowsne.embedding import fit_embedding
    D = load_dataset(dataset, seed)
    X, om = D["X"], D["omega"]
    Xt, omt = torch.tensor(X, dtype=DT), torch.tensor(om, dtype=DT)
    tag = f"{dataset}_seed{seed}"
    jf, nf = os.path.join(RES, tag + ".json"), os.path.join(RES, tag + ".npz")
    out, saved = json.load(open(jf)), dict(np.load(nf))
    for key in ("ds", "rs"):
        P = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO,
                                    doubly_stochastic=(key == "ds"))(Xt).detach()
        torch.manual_seed(seed)
        aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                              degrees_of_freedom=1.0).double()
        t0 = time.time()
        Z = fit_embedding(aff, P, torch.tensor(saved[f"sym_{key}"], dtype=DT), max_iter=iters,
                          lr=0.03, seed=seed, rescale_init=False, log_every=1000).detach()
        with torch.no_grad():
            F = -aff.omega_at(Z).numpy()
            loss = aff.cross_entropy(Z, P).item()
        name = f"flowsne_{key}_joint_ws"
        r = dict(method=name, secs=time.time() - t0, loss=loss, **evaluate(D, Z.numpy(), F, name))
        out["runs"] = [q for q in out["runs"] if q["method"] != name] + [r]
        saved[name], saved[name + "_drift"] = Z.numpy(), F
        print(f"[{dataset} s{seed}] {name:24s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs")
            + f"  ({r['secs']:.0f}s)", flush=True)
        json.dump(out, open(jf, "w"), indent=1)
        np.savez_compressed(nf, **saved)


def run_theirs(dataset, seed):
    """Debaussart-Joniec et al.'s estimator only (seconds), merged into the result
    files: on Isomap, on W's eigenvectors, and on the symmetric / Finsler MDS /
    Finsler t-SNE layouts already on file -- for runs made without their code."""
    D = load_dataset(dataset, seed)
    X, om = D["X"], D["omega"]
    tag = f"{dataset}_seed{seed}"
    jf, nf = os.path.join(RES, tag + ".json"), os.path.join(RES, tag + ".npz")
    out, saved = json.load(open(jf)), dict(np.load(nf))
    from sklearn.manifold import Isomap
    _their_operators()
    W, eps = their_graph(X, om)
    for name, low in (("theirs_isomap", Isomap(n_components=2, n_neighbors=10).fit_transform(X)),
                      ("theirs_eig", None), ("theirs_on_flowsne", saved.get("sym_ds")),
                      ("theirs_on_fmds", saved.get("finsler_mds")),
                      ("theirs_on_ftsne", saved.get("finsler_tsne"))):
        if name != "theirs_eig" and low is None:
            print(f"{name}: layout not on file, skipped")
            continue
        t0 = time.time()
        Zl, V = their_drift(W, eps, low)
        r = dict(method=name, secs=time.time() - t0, **evaluate(D, Zl, V, name))
        out["runs"] = [q for q in out["runs"] if q["method"] != name] + [r]
        saved[name], saved[name + "_drift"] = Zl, V
        print(f"[{dataset} s{seed}] {name:24s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs"), flush=True)
    json.dump(out, open(jf, "w"), indent=1)
    np.savez_compressed(nf, **saved)


# ===========================================================================
# Report
# ===========================================================================

def run_perp(dataset, seed, perplexity, key, quick=False):
    """FlowSNE joint fit at a given input perplexity (default elsewhere: sqrt(n)),
    written to its own file so several can run in parallel:
    ``{dataset}_seed{seed}_p{perplexity}_{key}.{json,npz}``."""
    it = (lambda m: 50) if quick else (lambda m: m)
    D = load_dataset(dataset, seed, quick)
    X, om = D["X"], D["omega"]
    Xt, omt = torch.tensor(X, dtype=DT), torch.tensor(om, dtype=DT)
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X)
    P = EntropicRandersAffinity(omega=omt, max_omega_norm=RHO, perplexity=perplexity,
                                doubly_stochastic=(key == "ds"))(Xt).detach()
    torch.manual_seed(seed)
    aff = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                          degrees_of_freedom=1.0).double()
    t0 = time.time()
    Z = fit_matcher(aff, P, init_np, JointAffinityMatcher, seed, it(3000), 0.03).detach()
    with torch.no_grad():
        loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
        F = -aff.omega_at(Z).numpy()
    name = f"flowsne_{key}_joint_p{perplexity:g}"
    r = dict(method=name, dataset=dataset, seed=seed, secs=time.time() - t0, loss=loss, perplexity=perplexity,
             **evaluate(D, Z.numpy(), F, name))
    sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
    Zs = fit_matcher(sym, P, init_np, AffinityMatcher, seed, it(1500), 0.1).detach().numpy()
    es = evaluate(D, Zs, np.zeros_like(Zs), "sym")
    r.update(sym_knn=es["knn"], sym_knn_dir=es["knn_dir"])
    print(f"[{dataset} s{seed}] {name:24s} " + " ".join(
        f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs"), flush=True)
    base = os.path.join(RES, f"{dataset}{'_quick' if quick else ''}_seed{seed}_p{perplexity:g}_{key}")
    json.dump(r, open(base + ".json", "w"), indent=1)
    np.savez_compressed(base + ".npz", Z=Z.numpy(), drift=F, Z_sym=Zs)


def report_extra():
    """Warm-started joint fits next to the PCA-started ones (mean over seeds),
    and the perplexity sweep (seed 0)."""
    files = sorted(f for f in os.listdir(RES) if f.endswith(".json") and "_quick" not in f)
    E = [json.load(open(os.path.join(RES, f))) for f in files]
    keys = ["cos_ref", "speed_spearman", "knn", "knn_dir", "loss"]
    fmt = lambda v: f"{np.mean(v):.3f}" if v else "    –"  # noqa: E731
    print("== warm start (ws) vs PCA init, mean over seeds")
    print(f"{'dataset':8s} {'method':22s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + "  seeds")
    for ds in ("sphere", "swiss", "cycle", "pancreas", "pancreas_full"):
        sel = [e for e in E if e.get("dataset") == ds and "runs" in e]
        for m in ("flowsne_ds_joint", "flowsne_ds_joint_ws", "flowsne_rs_joint", "flowsne_rs_joint_ws"):
            rr = [r for e in sel for r in e["runs"] if r["method"] == m]
            if rr:
                print(f"{ds:8s} {m:22s} " + " ".join(f"{fmt([r[k] for r in rr if k in r]):>9s}" for k in keys)
                      + f"  {len(rr)}")
    print("\n== perplexity sweep, seed 0 (default: sqrt(n) = 38.7)")
    print(f"{'dataset':8s} {'key':3s} {'perp':>6s} " + " ".join(f"{k[:9]:>9s}" for k in keys)
          + f" {'sym_knn':>8s} {'sym_kd':>7s}")
    for e in sorted((e for e in E if "perplexity" in e), key=lambda e: (e["method"], e["perplexity"])):
        key = e["method"].split("_")[1]
        print(f"{e.get('dataset', '?'):8s} {key:3s} {e['perplexity']:6g} " + " ".join(f"{e.get(k, float('nan')):9.3f}" for k in keys)
              + f" {e.get('sym_knn', float('nan')):8.3f} {e.get('sym_knn_dir', float('nan')):7.3f}")


def report():
    files = sorted(f for f in os.listdir(RES) if f.endswith(".json") and "_quick" not in f)
    E = [json.load(open(os.path.join(RES, f))) for f in files]
    E = [e for e in E if "runs" in e]          # run-perp files hold a single run
    # recompute every metric from the saved embeddings with the current evaluate()
    for e in E:
        D = load_dataset(e["dataset"], e["seed"])
        S = np.load(os.path.join(RES, f"{e['dataset']}_seed{e['seed']}.npz"))
        for r in e["runs"]:
            r.update(evaluate(D, S[r["method"]], S[r["method"] + "_drift"], r["method"]))
    ms = lambda v: "–" if not v else (f"{np.mean(v):+.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:+.3f}")
    L = ["# Baseline: FlowSNE vs Finsler directed embedding (Debaussart-Joniec et al. 2026)", "",
         "See `finsler_embedding_comparison.py` docstring. `cos_ref` = cosine between the estimated "
         "drift and the push-forward of the true input drift into that same embedding (points above "
         "the 20th percentile of reference speed). `speed ρ` = Spearman(|estimated|, |reference|). "
         "`knn` = 10-NN preservation of the layout, Euclidean in both spaces (direction-blind). "
         "`knn_dir` = overlap of the 10 nearest targets under the input Randers cost and under the "
         "method's own output cost (fitted / constant drift; Euclidean for the spectral estimator "
         "and symmetric fits). Mean ± std over seeds.", ""]
    for ds in ("sphere", "swiss", "cycle", "pancreas", "pancreas_full"):
        sel = [e for e in E if e["dataset"] == ds]
        if not sel:
            continue
        keys = ["cos_ref", "cos_ref_median", "speed_spearman", "knn", "knn_dir"]
        if ds.startswith("pancreas"):
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
    rows = [ds for ds in ("sphere", "swiss", "cycle", "pancreas", "pancreas_full")
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
    ap.add_argument("cmd", choices=["run", "run-finsler", "run-perp", "run-theirs", "run-ws", "report", "report-extra"])
    ap.add_argument("--dataset", default="sphere")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run-finsler: subset of sym, finsler_mds, finsler_tsne")
    ap.add_argument("--perplexity", type=float, default=None)
    ap.add_argument("--key", default="rs", choices=["ds", "rs"])
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.cmd == "run":
        run(a.dataset, a.seed, a.quick)
    elif a.cmd == "run-perp":
        run_perp(a.dataset, a.seed, a.perplexity, a.key, a.quick)
    elif a.cmd == "run-ws":
        run_ws(a.dataset, a.seed)
    elif a.cmd == "report-extra":
        report_extra()
    elif a.cmd == "run-theirs":
        run_theirs(a.dataset, a.seed)
    elif a.cmd == "run-finsler":
        run_finsler(a.dataset, a.seed, a.quick, a.only)
    else:
        report()
