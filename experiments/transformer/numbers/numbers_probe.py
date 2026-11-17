"""Does GPT-2 small carry the number helix? Per layer, a ridge probe from the
residual stream to n and to (cos, sin)(2 pi n / T), T in {2, 5, 10, 100},
5-fold cross-validated over NUMBERS (all templates of a held-out number are
held out). Reports held-out R^2, per-template points and template means.

    conda run -n dev python experiments/transformer/numbers/numbers_probe.py
"""
import os

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PERIODS = (2, 5, 10, 100)


def targets(n):
    cols = {"n": n[:, None].astype(float)}
    for T in PERIODS:
        a = 2 * np.pi * n / T
        cols[f"T{T}"] = np.stack([np.cos(a), np.sin(a)], 1)
    return cols


def r2(X, Y, groups):
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, groups):
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[tr], Y[tr])
        pred[te] = m.predict(X[te]).reshape(len(te), -1)
    return 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


def main():
    d = np.load(os.path.join(HERE, "data", "numbers_gpt2.npz"))
    H, num = d["H"], d["numbers"]
    L, nt, nn, D = H.shape
    print("layer | " + " ".join(f"{k:>6s}" for k in targets(num)) + " | template-mean: "
          + " ".join(f"{k:>6s}" for k in targets(num)) + " | median norm")
    for l in range(L):
        X = H[l].reshape(-1, D)
        X = X - X.mean(0)
        n_all = np.tile(num, nt)
        tg = targets(n_all)
        a = [r2(X, Y, n_all) for Y in tg.values()]
        Xm = H[l].mean(0); Xm = Xm - Xm.mean(0)
        b = [r2(Xm, Y, num) for Y in targets(num).values()]
        print(f"{l:5d} | " + " ".join(f"{v:6.2f}" for v in a) + " |                " +
              " ".join(f"{v:6.2f}" for v in b) + f" | {np.median(np.linalg.norm(H[l], axis=-1)):.0f}",
              flush=True)


if __name__ == "__main__":
    main()
