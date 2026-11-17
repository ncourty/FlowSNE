"""How rotational are MC_Maze's population dynamics, before any embedding?

Input: the condition-averaged trajectories of ``mc_maze_data.py``, the
cross-condition mean removed (as in jPCA), PCA; drift = dx/dt (central
differences along each condition's trajectory), through the same projection.

  jpca   on the top 6 PCs, in the movement epoch [-150, +150] ms: least-squares
         dX = X M with M unconstrained and with M skew-symmetric (a pure
         rotation); R^2 of each, and the ratio (Churchland et al. 2012: the
         skew fit captures most of what the free one does)
  hodge  the drift as an edge flow on the kNN graph of all (condition, time)
         points in d PCs, f_ij = (v_i + v_j)/2 . (x_j - x_i), split into
         grad(phi) + rest; share of the rest (non-gradient); also Spearman of
         phi with time (is the potential just elapsed time?)
  null   the same with each neuron's condition labels permuted independently
         (keeps every neuron's own condition-averaged time course, destroys
         the population structure: the "rotations from single-neuron
         sequences" alternative), 10 draws

    conda run -n dev python experiments/neuro/mc_maze_rotation.py --file full
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def trajectories(R, t, d, centre=True, window=None):
    """(C, T, N) rates -> points X (C*T, d), drift V (C*T, d), cond and time labels."""
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])
        R, t = R[:, m], t[m]
    if centre:
        R = R - R.mean(0, keepdims=True)
    dR = np.gradient(R, t, axis=1) * 1000.0                 # per second
    C, T, N = R.shape
    mu = R.reshape(-1, N).mean(0)
    pca = PCA(n_components=d, random_state=0).fit(R.reshape(-1, N) - mu)
    X = pca.transform(R.reshape(-1, N) - mu)
    V = dR.reshape(-1, N) @ pca.components_.T
    return X, V, np.repeat(np.arange(C), T), np.tile(t, C), pca


def jpca_r2(X, V):
    M = np.linalg.lstsq(X, V, rcond=None)[0]
    d = X.shape[1]
    # skew-symmetric M = sum_k a_k (E_ij - E_ji): linear in the d(d-1)/2 parameters
    I, J = np.triu_indices(d, 1)
    A = np.zeros((X.shape[0] * d, len(I)))
    for k, (i, j) in enumerate(zip(I, J)):
        B = np.zeros((d, d))
        B[i, j], B[j, i] = 1, -1
        A[:, k] = (X @ B).ravel()
    a = np.linalg.lstsq(A, V.ravel(), rcond=None)[0]
    Ms = np.zeros((d, d))
    Ms[I, J], Ms[J, I] = a, -a
    ss = ((V - V.mean(0)) ** 2).sum()
    r2 = lambda M: 1 - ((V - X @ M) ** 2).sum() / ss  # noqa: E731
    return r2(M), r2(Ms), Ms


def hodge_share(X, V, tt, k=15):
    n = len(X)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    E = np.unique(np.sort(np.stack([np.repeat(np.arange(n), k), idx.ravel()], 1), 1), axis=0)
    m = len(E)
    B = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)],
                       (np.r_[np.arange(m), np.arange(m)], np.r_[E[:, 0], E[:, 1]])), shape=(m, n))
    i, j = E[:, 0], E[:, 1]
    f = 0.5 * ((V[i] + V[j]) * (X[j] - X[i])).sum(1)
    L = (B.T @ B + 1e-8 * sp.eye(n)).tocsr()
    phi = cg(L, B.T @ f, rtol=1e-10, maxiter=20000)[0]
    g = B @ phi
    return float(((f - g) ** 2).sum() / (f ** 2).sum()), float(spearmanr(phi, tt)[0])


def report(R, t, label, d=20, window=(-150, 150)):
    X6, V6, _, _, pca6 = trajectories(R, t, 6, window=window)
    full, skew, Ms = jpca_r2(X6, V6)
    X, V, c, tt, pca = trajectories(R, t, d)
    rot, rho = hodge_share(X, V, tt)
    ev = np.linalg.eigvals(Ms).imag
    print(f"{label:10s} var6={pca6.explained_variance_ratio_.sum():.2f} "
          f"var{d}={pca.explained_variance_ratio_.sum():.2f} | jPCA R2 free={full:.3f} "
          f"skew={skew:.3f} ratio={skew / full:.2f} | top rotation {ev.max() / (2 * np.pi):.2f} Hz"
          f" | hodge non-grad share={rot:.3f}, spearman(phi, t)={rho:+.2f}", flush=True)
    return full, skew, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="full", choices=["small", "full"])
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=10)
    a = ap.parse_args()
    D = np.load(os.path.join(DATA, f"mc_maze_{a.file}.npz"))
    R, t = D["R"].astype(np.float64), D["t"]
    print(f"{R.shape[0]} conditions, {R.shape[1]} times, {R.shape[2]} units")
    report(R, t, "data", a.d)
    rng = np.random.default_rng(0)
    nulls = []
    for s in range(a.n_null):
        Rn = np.stack([R[rng.permutation(R.shape[0]), :, u] for u in range(R.shape[2])], 2)
        nulls.append(report(Rn, t, f"null{s}", a.d))
    nulls = np.array(nulls)
    print(f"null mean: free={nulls[:, 0].mean():.3f} skew={nulls[:, 1].mean():.3f} "
          f"ratio={np.mean(nulls[:, 1] / nulls[:, 0]):.2f} non-grad={nulls[:, 2].mean():.3f}")


if __name__ == "__main__":
    main()
