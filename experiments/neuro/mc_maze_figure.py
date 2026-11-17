"""MC_Maze layouts: the jPCA plane (input reference) and each method's map,
every condition's trajectory from -150 to +150 ms around movement onset,
coloured by reach direction (circular hue), dots at movement onset; drift
arrows (flow direction) for the methods that have one.

    conda run -n dev python experiments/neuro/mc_maze_figure.py --d 6
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mc_maze_rotation import jpca_r2  # noqa: E402

RES = os.path.join(HERE, "results")
TITLES = {"jpca": "jPCA plane (input)", "sym": "symmetric t-SNE", "finsler_tsne": "Finsler t-SNE",
          "flowsne_ds": "FlowSNE (balanced)", "flowsne_rs": "FlowSNE (row-stoch.)"}


def jpca_plane(X, V, tt, window=(-150, 150)):
    pca = PCA(6, random_state=0).fit(X)
    X6, V6 = pca.transform(X), (V @ pca.components_.T)
    m = (tt >= window[0]) & (tt <= window[1])
    _, _, Ms = jpca_r2(X6[m], V6[m])
    w, U = np.linalg.eig(Ms)
    u = U[:, np.argmax(w.imag)]
    P2 = np.stack([u.real, u.imag], 1)
    P2 /= np.linalg.norm(P2, axis=0)
    return X6 @ P2, V6 @ P2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_d6_step20")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    S = np.load(os.path.join(RES, f"{a.tag}_seed0.npz"))
    c, tt = S["cond"], S["time"]
    hv = S["hand_vel"]
    # reach direction: angle of the mean hand velocity over [0, +300] ms
    ang = np.zeros(c.max() + 1)
    for k in np.unique(c):
        m = (c == k) & (tt >= 0) & (tt <= 300)
        v = hv[m].mean(0)
        ang[k] = np.arctan2(v[1], v[0])
    col = plt.cm.hsv((ang + np.pi) / (2 * np.pi))
    panels = {"jpca": jpca_plane(S["X"], S["V"], tt)}
    for meth in ("sym", "finsler_tsne", "flowsne_ds", "flowsne_rs"):
        panels[meth] = (S[meth], S[meth + "_drift"] if meth + "_drift" in S.files else None)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.5))
    win = (tt >= -150) & (tt <= 150)
    for ax, (meth, (Z, F)) in zip(axes, panels.items()):
        ax.scatter(*Z.T, s=1, c="0.85", lw=0)
        for k in np.unique(c):
            m = (c == k) & win
            o = np.argsort(tt[m])
            ax.plot(*Z[m][o].T, color=col[k], lw=0.6)
            s = (c == k) & (tt == 0)
            ax.plot(*Z[s].T, "o", ms=1.8, color=col[k], mec="none")
        if F is not None and meth != "jpca":
            sel = win & ((tt + 150) % 60 == 0)
            Fn = F[sel] / np.linalg.norm(F[sel], axis=1, keepdims=True).clip(1e-12)
            L = 0.04 * np.ptp(Z, axis=0).max()
            ax.quiver(*Z[sel].T, *(Fn * L).T, angles="xy", scale_units="xy", scale=1,
                      width=0.004, color="k", alpha=0.6)
        ax.set_title(TITLES[meth], fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    out = a.out or os.path.join(RES, f"{a.tag}.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
