"""Is the integer layout a coil (one turn per decade)? For 2-D and 3-D results.

- nn10: share of a layout's 10-NN pairs at an offset that is a nonzero
  multiple of 10 (same last digit); the input value is printed for reference.
- turn: per decade k = 1..34, the path 10k -> 10k+1 -> ... -> 10k+10 seen
  around the decade's centroid -- in 3-D projected on the plane orthogonal to
  the local axis c_{k+1} - c_{k-1} (decade centroids) -- and its winding
  (total turning angle / 2 pi, rounded); share of decades with |w| = 1.
- agree: for methods with a drift, share of those decades where the drift
  turns the same way, sign((z - c) x F . axis) vs sign(w).

    conda run -n dev python experiments/transformer/numbers/numbers_coil.py
"""
import os
import sys

import numpy as np
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from numbers_embed import RES, load  # noqa: E402

METHODS = ("sym", "finsler_tsne", "flowsne_ds", "flowsne_rs")


def nn10(Z, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    off = np.abs(idx - np.arange(len(Z))[:, None])
    return float(np.mean(off % 10 == 0))


def plane(a):
    a = a / np.linalg.norm(a)
    e1 = np.linalg.svd(np.eye(3) - np.outer(a, a))[0][:, 0]
    return a, e1, np.cross(a, e1)        # e1 x e2 = a: positive angle = counter-clockwise about a


def decades(Z, F=None):
    d = Z.shape[1]
    cent = np.array([Z[10 * k:10 * k + 10].mean(0) for k in range(36)])
    w, agree = [], []
    for k in range(1, 35):
        P = Z[10 * k:10 * k + 11]
        c = P[:-1].mean(0)
        R = P - c
        if d == 3:
            a, e1, e2 = plane(cent[k + 1] - cent[k - 1])
            R2 = np.stack([R @ e1, R @ e2], 1)
        else:
            R2 = R
        ang = np.arctan2(R2[:, 1], R2[:, 0])
        dd = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
        wk = int(np.round(dd.sum() / (2 * np.pi)))
        w.append(wk)
        if F is not None and abs(wk) == 1:
            Fk = F[10 * k:10 * k + 10]
            if d == 3:
                t = np.mean(np.cross(R[:-1], Fk) @ a)
            else:
                t = np.mean(R[:-1, 0] * Fk[:, 1] - R[:-1, 1] * Fk[:, 0])
            agree.append(np.sign(t) == np.sign(wk))
    w = np.array(w)
    return float(np.mean(np.abs(w) == 1)), (float(np.mean(agree)) if agree else float("nan"))


def main():
    for l in (0, 4, 8):
        X = load(l)[0]
        print(f"layer {l}: input nn10 = {nn10(X):.2f}")
        for dim in (2, 3):
            f = os.path.join(RES, f"layer{l}{'' if dim == 2 else '_3d'}_seed0.npz")
            if not os.path.exists(f):
                continue
            S = np.load(f)
            for m in METHODS:
                F = S[m + "_drift"] if m + "_drift" in S.files else None
                turn, agree = decades(S[m], F)
                print(f"   {dim}-D {m:13s} nn10 {nn10(S[m]):.2f}  turn {turn:.2f}"
                      + (f"  agree {agree:.2f}" if F is not None else ""))


if __name__ == "__main__":
    main()
