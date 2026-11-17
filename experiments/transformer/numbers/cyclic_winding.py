"""Do the cyclic concepts form rings in a layout, per context? For each
context, the K points' angles around their own centroid; winding number of
the path c_1 -> ... -> c_K -> c_1 (|w| = 1: the context is a ring in calendar
order). For methods with a drift, the drift's turning direction around that
ring, sign(mean of (z - centre) x F), compared with the winding's sign.

    conda run -n dev python experiments/transformer/numbers/cyclic_winding.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "cyclic")
METHODS = ("sym", "finsler_tsne", "flowsne_ds", "flowsne_rs")


def winding(Z, ctx, c):
    out, turn = [], []
    for k in np.unique(ctx):
        m = ctx == k
        Zk = Z[m][np.argsort(c[m])]
        a = np.arctan2(*(Zk - Zk.mean(0)).T[::-1])
        d = np.diff(np.append(a, a[0]))
        d = (d + np.pi) % (2 * np.pi) - np.pi
        out.append(int(np.round(d.sum() / (2 * np.pi))))
    return np.array(out)


def drift_turn(Z, F, ctx):
    s = []
    for k in np.unique(ctx):
        m = ctx == k
        R = Z[m] - Z[m].mean(0)
        s.append(np.sign(np.mean(R[:, 0] * F[m, 1] - R[:, 1] * F[m, 0])))
    return np.array(s)


def main():
    print("tag | " + " | ".join(f"{m}: |w|=1, drift agrees" for m in METHODS))
    for f in sorted(os.listdir(RES)):
        if not f.endswith("_seed0.npz"):
            continue
        S = np.load(os.path.join(RES, f))
        row = []
        for m in METHODS:
            w = winding(S[m], S["context"], S["concept"])
            txt = f"{np.mean(np.abs(w) == 1):.2f}"
            if m + "_drift" in S.files:
                t = drift_turn(S[m], S[m + "_drift"], S["context"])
                ok = np.abs(w) == 1
                txt += f", {np.mean(t[ok] == np.sign(w[ok])):.2f}" if ok.any() else ", -"
            else:
                txt += ", -"
            row.append(f"{txt:>12s}")
        print(f"{f[:-10]:24s} | " + " | ".join(row))


if __name__ == "__main__":
    main()
