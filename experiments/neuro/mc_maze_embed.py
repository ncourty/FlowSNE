"""MC_Maze neighbour embeddings with the population velocity as drift.

Points: every (condition, time) state of ``mc_maze_data.py``'s trajectories,
cross-condition mean removed, PCA to --d dims, sampled every --step ms in
[--t0, --t1]; drift dx/dt through the same projection; successor = the same
condition one sample later. Methods and fitting: ``numbers_embed.fit_points``
(symmetric t-SNE, Finsler t-SNE, FlowSNE balanced / row-stochastic).

    conda run -n dev python experiments/neuro/mc_maze_embed.py fit
    conda run -n dev python experiments/neuro/mc_maze_embed.py score
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "transformer", "numbers"))
from mc_maze_rotation import trajectories  # noqa: E402
from numbers_embed import fit_points  # noqa: E402

DATA = os.path.join(HERE, "data")
RES = os.path.join(HERE, "results")
METHODS = ("sym", "finsler_tsne", "flowsne_ds", "flowsne_rs")


def load(file="full", d=20, step=20, t0=-250, t1=450, centre=True):
    D = np.load(os.path.join(DATA, f"mc_maze_{file}.npz"))
    R, t = D["R"].astype(np.float64), D["t"]
    X, V, c, tt, _ = trajectories(R, t, d, centre=centre)
    keep = (tt >= t0) & (tt <= t1) & ((tt - t0) % step == 0)
    X, V, c, tt = X[keep], V[keep], c[keep], tt[keep]
    T = len(np.unique(tt))
    nxt = np.arange(len(X)) + 1
    nxt[tt == tt.max()] = -1
    hv = D["hand_vel"][:, np.isin(t, np.unique(tt))].reshape(-1, 2)
    return X, V / np.abs(V).max(), c, tt, hv, T, nxt


def tag(a):
    return f"{a.file}_d{a.d}_step{a.step}" + ("" if a.centre else "_raw")


def turning(Z, c, tt, window=(-150, 150)):
    """Signed angle swept by each condition's trajectory around its own centroid
    in the window, in turns (2-D layouts)."""
    out = []
    for k in np.unique(c):
        m = (c == k) & (tt >= window[0]) & (tt <= window[1])
        Zk = Z[m][np.argsort(tt[m])]
        ang = np.arctan2(*(Zk - Zk.mean(0)).T[::-1])
        d = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
        out.append(d.sum() / (2 * np.pi))
    return np.array(out)


def drift_turn(Z, F, c, tt, window=(-150, 150)):
    s = []
    for k in np.unique(c):
        m = (c == k) & (tt >= window[0]) & (tt <= window[1])
        Rk = Z[m] - Z[m].mean(0)
        s.append(np.sign(np.mean(Rk[:, 0] * F[m, 1] - Rk[:, 1] * F[m, 0])))
    return np.array(s)


def score(a):
    S = np.load(os.path.join(RES, f"{tag(a)}_seed0.npz"))
    runs = {r["method"]: r for r in json.load(open(os.path.join(RES, f"{tag(a)}_seed0.json")))["runs"]}
    c, tt = S["cond"], S["time"]
    from mc_maze_rotation import jpca_r2
    from sklearn.decomposition import PCA
    X = S["X"]
    X6 = PCA(6, random_state=0).fit_transform(X)
    m = (tt >= -150) & (tt <= 150)
    # input reference: the jPCA plane (top-rotation plane of the skew fit)
    _, _, Ms = jpca_r2(X6[m], S["V"][m] @ PCA(6, random_state=0).fit(X).components_.T)
    w, U = np.linalg.eig(Ms)
    u = U[:, np.argmax(w.imag)]
    P2 = np.stack([u.real, u.imag], 1)
    ref = turning(X6 @ P2, c, tt)
    print(f"[{tag(a)}] jPCA plane: turns per condition median {np.median(ref):+.2f}, "
          f"same sign {np.mean(np.sign(ref) == np.sign(np.median(ref))):.2f}")
    print("method        knn   succ_cos  median turns  same-sign  drift agrees")
    for meth in METHODS:
        if meth not in S.files:
            continue
        Z = S[meth]
        tr = turning(Z, c, tt)
        sgn = np.sign(np.median(tr))
        line = (f"{meth:12s} {runs[meth]['knn']:.3f}  {runs[meth].get('succ_cos', float('nan')):+.3f}"
                f"    {np.median(tr):+.2f}        {np.mean(np.sign(tr) == sgn):.2f}")
        if meth + "_drift" in S.files:
            dt = drift_turn(Z, S[meth + "_drift"], c, tt)
            line += f"       {np.mean(dt == np.sign(tr)):.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "score"])
    ap.add_argument("--file", default="full")
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--raw", dest="centre", action="store_false")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    ap.add_argument("--finsler-iters", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    if a.cmd == "score":
        return score(a)
    X, V, c, tt, hv, T, nxt = load(a.file, a.d, a.step, centre=a.centre)
    print(f"{len(X)} points ({len(np.unique(c))} conditions x {T} times), d={a.d}", flush=True)
    fit_points(X, V, nxt, dict(X=X, V=V, cond=c, time=tt, hand_vel=hv), tag(a), iters=a.iters, res=RES,
               methods=a.methods, alphas=tuple(a.alphas), finsler_iters=a.finsler_iters)


if __name__ == "__main__":
    main()
