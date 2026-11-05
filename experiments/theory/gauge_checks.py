"""Numerical checks of the gauge / Helmholtz theory (article/ICML).

Self-contained (numpy + torch): the costs needed here (flat-torus metric,
exact potential term, per-row bandwidths on a linear cost) are not exposed by
flowsne's public API.

Checks
  1. finite-sample identities: balanced coupling invariant to separable cost
     changes; cyclic part of log(P/P^T) = -(2/eps) * cyclic part of the cost's
     antisymmetric part; row-stochastic coupling not invariant.
  2. continuum transport field on the flat torus: balanced chain moves along
     -(d+1) times the p-solenoidal part of the drift, row-stochastic along
     -(d+1) times the whole drift (linear Randers cost, small drift).
  3. leakage of a gradient drift vs drift strength rho for three costs:
     linear + global eps (0), squared + global eps (O(rho^3)),
     linear + per-row eps (O(rho)); rotational drift for reference (O(rho)).

Run: conda run -n dev python experiments/theory/gauge_checks.py [--n 3000]
"""
import argparse, json, math, os, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
TWO_PI = 2 * math.pi
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


# ---------------------------------------------------------------- Sinkhorn
def sinkhorn_log(logK, iters=20000, tol=1e-11):
    """Balanced coupling with uniform marginals 1/n, log domain."""
    n = logK.shape[0]
    la = torch.full((n,), -math.log(n))
    f = torch.zeros(n); g = torch.zeros(n)
    for it in range(iters):
        f = la - torch.logsumexp(logK + g[None, :], dim=1)
        g = la - torch.logsumexp(logK + f[:, None], dim=0)
        if it % 25 == 0:
            r = torch.logsumexp(logK + f[:, None] + g[None, :], dim=1).exp()
            err = (r * n - 1).abs().max().item()
            if err < tol:
                break
    return logK + f[:, None] + g[None, :], err


def row_stochastic_log(logK):
    return logK - torch.logsumexp(logK, dim=1, keepdim=True) - math.log(logK.shape[0])


def calibrate_rows(C, perp, iters=60):
    """Per-row inverse bandwidth beta_i so that row i of exp(-beta_i C) has perplexity perp."""
    n = C.shape[0]
    lo = torch.full((n,), 1e-3); hi = torch.full((n,), 1e4)
    target = math.log(perp)
    for _ in range(iters):
        b = (lo * hi).sqrt()
        L = -b[:, None] * C
        L = L - torch.logsumexp(L, 1, keepdim=True)
        H = -(L.exp() * L).sum(1)
        big = H > target          # too flat -> increase beta
        lo = torch.where(big, b, lo); hi = torch.where(big, hi, b)
    return (lo * hi).sqrt()


# ---------------------------------------------------------- torus geometry
def sample_density(n, rng):
    def p(x):
        return 1 + 0.6 * np.cos(TWO_PI * x[:, 0]) * np.cos(TWO_PI * (x[:, 1] - 0.15)) \
                 + 0.25 * np.sin(TWO_PI * (x[:, 0] + x[:, 1]))
    pts = []
    while sum(len(a) for a in pts) < n:
        x = rng.random((4 * n, 2))
        keep = rng.random(4 * n) * 1.9 < p(x)
        pts.append(x[keep])
    X = np.concatenate(pts)[:n]
    return X, p


def fields(X, p):
    """Gradient part grad(phi) and p-solenoidal part w = (1/p) rot(chi), each with max norm 1."""
    x, y = X[:, 0], X[:, 1]
    # phi = sin(2 pi x)/(2pi) + 0.5 sin(2 pi (y + 0.2))/(2 pi)
    phi = np.sin(TWO_PI * x) / TWO_PI + 0.5 * np.sin(TWO_PI * (y + 0.2)) / TWO_PI
    gphi = np.stack([np.cos(TWO_PI * x), 0.5 * np.cos(TWO_PI * (y + 0.2))], 1)
    # chi = sin(2 pi x) sin(2 pi y) / (2 pi);  rot chi = (d_y chi, -d_x chi)
    rot = np.stack([np.sin(TWO_PI * x) * np.cos(TWO_PI * y),
                    -np.cos(TWO_PI * x) * np.sin(TWO_PI * y)], 1)
    w = rot / p(X)[:, None]
    # normalise on a dense grid so the max norm is exactly 1
    g = np.stack(np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400)), -1).reshape(-1, 2)
    mg = np.sqrt(np.cos(TWO_PI * g[:, 0]) ** 2 + 0.25 * np.cos(TWO_PI * (g[:, 1] + 0.2)) ** 2).max()
    rg = np.stack([np.sin(TWO_PI * g[:, 0]) * np.cos(TWO_PI * g[:, 1]),
                   -np.cos(TWO_PI * g[:, 0]) * np.sin(TWO_PI * g[:, 1])], 1) / p(g)[:, None]
    mw = np.linalg.norm(rg, axis=1).max()
    return phi / mg, gphi / mg, w / mw


def torus_delta(X):
    D = X[None, :, :] - X[:, None, :]          # D[i, j] = x_j - x_i
    return D - np.round(D)


# ------------------------------------------------ weighted Helmholtz on a grid
def weighted_solenoidal(q, V, N):
    """Torus grid N x N. Return u = V - grad(Psi) with div(q u) = 0 (cell centres).

    q: (N, N) positive weight; V: (N, N, 2) field. Finite volumes, periodic."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    hgrid = 1.0 / N
    idx = np.arange(N * N).reshape(N, N)            # idx[ix, iy]
    rows, cols, vals, rhs = [], [], [], np.zeros(N * N)
    for ax in (0, 1):
        nb = np.roll(idx, -1, axis=ax)              # +1 neighbour along ax
        qf = 0.5 * (q + np.roll(q, -1, axis=ax))     # face weight
        Vf = 0.5 * (V[..., ax] + np.roll(V[..., ax], -1, axis=ax))
        flux = qf * Vf                              # q V . n on the + face
        a, b = idx.ravel(), nb.ravel(); w = (qf / hgrid ** 2).ravel()
        rows += [a, a, b, b]; cols += [a, b, b, a]; vals += [w, -w, w, -w]
        np.add.at(rhs, a, -(flux / hgrid).ravel()); np.add.at(rhs, b, (flux / hgrid).ravel())
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N * N, N * N))          # L = -div(q grad .)
    L = L + sp.eye(N * N) * 1e-10
    Psi = spl.spsolve(L.tocsc(), rhs).reshape(N, N)   # div(q grad Psi) = div(q V)
    g = np.stack([(np.roll(Psi, -1, 0) - np.roll(Psi, 1, 0)) / (2 * hgrid),
                  (np.roll(Psi, -1, 1) - np.roll(Psi, 1, 1)) / (2 * hgrid)], -1)
    return V - g


def interp_periodic(F, X):
    """Bilinear interpolation of a cell-centred periodic grid field F (N, N, k) at X (n, 2)."""
    N = F.shape[0]
    u = X * N - 0.5
    i0 = np.floor(u).astype(int); t = u - i0
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            wgt = (t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
            out = out + wgt[:, None] * F[(i0[:, 0] + dx) % N, (i0[:, 1] + dy) % N]
    return out


def predict_transport(X, p, beta, rho_g, rho_w, h, N=128, smooth=0.04):
    """Small-drift prediction of the balanced chain's mean displacement (units of h),
    linear Randers cost exp(-beta_i C): E[Delta] = -(d+1) u / beta^2,
    u = solenoidal part of beta*omega in L^2(p / beta^2)."""
    c = (np.arange(N) + 0.5) / N
    G = np.stack(np.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
    _, gphi_g, w_g = fields(G, p)
    omega_g = rho_g * gphi_g + rho_w * w_g
    if np.ndim(beta) == 0:
        beta_g = np.full(len(G), float(beta))
    else:                                           # Nadaraya-Watson of 1/beta onto the grid
        D = G[:, None, :] - X[None, :, :]; D -= np.round(D)
        K = np.exp(-(D ** 2).sum(-1) / (2 * smooth ** 2))
        beta_g = 1.0 / ((K @ (1.0 / beta)) / K.sum(1))
    q = (p(G) / beta_g ** 2).reshape(N, N)
    V = (beta_g[:, None] * omega_g).reshape(N, N, 2)
    u = weighted_solenoidal(q, V, N)
    disp = -(2 + 1) * u / (beta_g.reshape(N, N, 1) ** 2)
    return interp_periodic(disp, X) / h


# ---------------------------------------------------------------- measures
def transport(logP, Delta, h):
    """Mean displacement of the chain P/rowsum, in units of h: (n, 2)."""
    P = logP.exp()
    P = P / P.sum(1, keepdim=True)
    return torch.einsum("ij,ijk->ik", P, Delta) / h


def cyclic_rms(logP, knn_idx):
    """Weighted RMS of the cyclic (non-gradient) part of log(P_ij/P_ji) on a kNN graph."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    n = logP.shape[0]
    L = (logP - logP.T).numpy()
    rows = np.repeat(np.arange(n), knn_idx.shape[1]); cols = knn_idx.ravel()
    m = rows < cols
    a = np.concatenate([rows[m], cols[~m]]); b = np.concatenate([cols[m], rows[~m]])
    e = np.unique(np.stack([a, b], 1), axis=0)
    f = L[e[:, 0], e[:, 1]]
    B = sp.csr_matrix((np.concatenate([-np.ones(len(e)), np.ones(len(e))]),
                       (np.tile(np.arange(len(e)), 2), np.concatenate([e[:, 0], e[:, 1]]))),
                      shape=(len(e), n))
    phi = spl.lsqr(B, f, atol=1e-14, btol=1e-14, iter_lim=20000)[0]
    r = f - B @ phi
    return float(np.sqrt(np.mean(r ** 2)))


# ------------------------------------------------------------------ checks
def check_identities(rng):
    n, eps = 400, 0.3
    X = rng.normal(size=(n, 5))
    S = torch.cdist(torch.tensor(X), torch.tensor(X))
    phi = torch.tensor(np.tanh(X[:, 0]) + 0.3 * X[:, 1] ** 2) * 0.2
    G = phi[None, :] - phi[:, None]                     # gradient (separable) part
    R = torch.tensor(rng.normal(size=(n, n))) * 0.1
    A = (R - R.T) / 2                                   # generic antisymmetric part
    out = {}
    lP0, _ = sinkhorn_log(-S / eps)
    lPg, _ = sinkhorn_log(-(S + G) / eps)
    out["ds_gradient_change"] = ((lPg.exp() - lP0.exp()).abs().sum() / lP0.exp().sum()).item()
    lR0 = row_stochastic_log(-S / eps); lRg = row_stochastic_log(-(S + G) / eps)
    out["rs_gradient_change"] = ((lRg.exp() - lR0.exp()).abs().sum() / lR0.exp().sum()).item()
    # cyclic identity on the complete graph: cyc(log P - log P^T) = -(2/eps) cyc(A)
    lPa, _ = sinkhorn_log(-(S + A + G) / eps)
    Lr = lPa - lPa.T
    def cyc(M):                                         # complete-graph Hodge, unweighted
        psi = -M.mean(1)                                # LS fit M_ij ~ psi_j - psi_i
        return M - (psi[None, :] - psi[:, None])
    lhs, rhs = cyc(Lr), cyc(-2 * A / eps)
    out["cyclic_identity_rel_err"] = ((lhs - rhs).norm() / rhs.norm()).item()
    out["logratio_gradient_part_nonzero"] = ((Lr - lhs).norm() / Lr.norm()).item()
    return out


def build_costs(Delta, phi, w, rho_g, rho_w):
    r = Delta.norm(dim=-1)
    wbar = 0.5 * (w[:, None, :] + w[None, :, :])
    a = rho_g * (phi[None, :] - phi[:, None]) + rho_w * (wbar * Delta).sum(-1)
    return r + a                                        # linear Randers cost, >= 0 if admissible


def run_torus(n, h, rng, rhos, perp=30.0, leakage=True, save_fields=True):
    X, p = sample_density(n, rng)
    phi, gphi, w = fields(X, p)
    Delta = torch.tensor(torus_delta(X))
    phi_t, w_t = torch.tensor(phi), torch.tensor(w)
    r = Delta.norm(dim=-1)
    knn = torch.topk(-r, 31, dim=1).indices[:, 1:].numpy()
    d = 2
    res = {"n": n, "h": h}

    # baselines without drift
    C0 = r
    lDS0, _ = sinkhorn_log(-C0 / h)
    b0_ds = transport(lDS0, Delta, h)
    b0_rs = transport(row_stochastic_log(-C0 / h), Delta, h)

    # --- check 2: transport field, mixed drift, small rho
    def corr(b, target):
        b, t = b.numpy().ravel(), target.ravel()
        cos = b @ t / np.linalg.norm(b) / np.linalg.norm(t)
        slope = b @ t / (t @ t)
        return float(cos), float(slope)
    tr = {}
    for rho in [0.1, 0.3]:
        C = build_costs(Delta, phi_t, w_t, rho, rho)
        lDS, err = sinkhorn_log(-C / h)
        bds = transport(lDS, Delta, h) - b0_ds
        brs = transport(row_stochastic_log(-C / h), Delta, h) - b0_rs
        omega = rho * (gphi + w)
        tr[str(rho)] = {
            "sinkhorn_err": err,
            "DS_vs_pred_solenoidal(cos,slope)": corr(bds, -(d + 1) * rho * w),
            "DS_vs_whole_drift(cos)": corr(bds, -(d + 1) * omega)[0],
            "DS_vs_gradient_part(cos)": corr(bds, -(d + 1) * rho * gphi)[0],
            "RS_vs_whole_drift(cos,slope)": corr(brs, -(d + 1) * omega),
            "RS_vs_solenoidal(cos)": corr(brs, -(d + 1) * rho * w)[0],
        }
        print("transport", rho, tr[str(rho)], flush=True)
    # same check through the grid solver (global bandwidth), and for per-row bandwidths
    beta = calibrate_rows(r, perp=perp)
    for rho in [0.1, 0.3]:
        pred_g = predict_transport(X, p, 1.0 / h, rho, rho, h)
        C = build_costs(Delta, phi_t, w_t, rho, rho)
        lDS, _ = sinkhorn_log(-C / h)
        bds = transport(lDS, Delta, h) - b0_ds
        tr[str(rho)]["DS_vs_solver_global(cos,slope)"] = corr(bds, pred_g)
        lB0, _ = sinkhorn_log(-beta[:, None] * r)
        bpr0 = transport(lB0, Delta, h)
        Cg = build_costs(Delta, phi_t, w_t, rho, 0.0)
        lB, _ = sinkhorn_log(-beta[:, None] * Cg)
        bpr = transport(lB, Delta, h) - bpr0
        pred_pr = predict_transport(X, p, beta.numpy(), rho, 0.0, h)
        tr[str(rho)]["perrow_gradient_DS_vs_pred(cos,slope)"] = corr(bpr, pred_pr)
        tr[str(rho)]["perrow_gradient_DS_rms"] = float(bpr.pow(2).sum(1).mean().sqrt())
        print("transport+", rho, tr[str(rho)], flush=True)
    res["transport"] = tr
    res["beta_h_quantiles"] = (beta * h).quantile(torch.tensor([0.1, 0.5, 0.9])).tolist()
    if save_fields:
        np.savez(os.path.join(OUT, "torus_fields.npz"), X=X, gphi=gphi, w=w, p=p(X),
                 bds=bds.numpy(), brs=brs.numpy(), rho=0.3)

    if not leakage:
        return res
    # --- check 3: leakage vs rho
    leak = {k: {"rho": [], "transport_rms": [], "cyclic_rms": []} for k in
            ["grad_linear_global", "grad_squared_global", "grad_linear_perrow",
             "rot_linear_global", "rot_squared_global", "rot_linear_perrow"]}
    base = {"linear_global": (lambda C: -C / h),
            "squared_global": (lambda C: -C ** 2 / (2 * h ** 2)),
            "linear_perrow": (lambda C: -beta[:, None] * C)}
    base_ds = {}
    for name, fk in base.items():
        lP, _ = sinkhorn_log(fk(r))
        base_ds[name] = (transport(lP, Delta, h), cyclic_rms(lP, knn))
    for rho in rhos:
        for kind in ["grad", "rot"]:
            C = build_costs(Delta, phi_t, w_t, rho if kind == "grad" else 0.0,
                            rho if kind == "rot" else 0.0)
            for name, fk in base.items():
                lP, err = sinkhorn_log(fk(C))
                b = transport(lP, Delta, h) - base_ds[name][0]
                key = f"{kind}_{name}"
                leak[key]["rho"].append(rho)
                leak[key]["transport_rms"].append(float(b.pow(2).sum(1).mean().sqrt()))
                leak[key]["cyclic_rms"].append(abs(cyclic_rms(lP, knn) - base_ds[name][1]))
        print("leak", rho, {k: (v["transport_rms"][-1], v["cyclic_rms"][-1]) for k, v in leak.items()}, flush=True)
    for k, v in leak.items():
        x, y = np.log(v["rho"]), np.log(np.maximum(v["transport_rms"], 1e-300))
        v["slope_transport"] = float(np.polyfit(x[:4], y[:4], 1)[0]) if min(v["transport_rms"]) > 1e-12 else None
    res["leakage"] = leak
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--h", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perp", type=float, default=30.0)
    ap.add_argument("--h-sweep", type=float, nargs="*", default=[])
    ap.add_argument("--sweep-point", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.sweep_point is not None:                 # one bandwidth of the sweep, saved alone
        hh = args.sweep_point
        nn = int(round(args.n * (args.h / hh) ** 2))
        sub = run_torus(nn, hh, np.random.default_rng(args.seed + 1), rhos=[0.1, 0.2, 0.4],
                        perp=args.perp, save_fields=False)
        with open(os.path.join(OUT, f"gauge_sweep_h{hh:g}_n{nn}_s{args.seed}.json"), "w") as f:
            json.dump({"h": hh, "n": nn, "leakage": sub["leakage"], "transport": sub["transport"]}, f, indent=1)
        return
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    res = {"identities": check_identities(rng)}
    print("identities", res["identities"], flush=True)
    res["torus"] = run_torus(args.n, args.h, rng, rhos=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
                             perp=args.perp)
    # bandwidth sweep (constant neighbours per bandwidth): squared-cost leakage should be O(h rho)
    sweep = []
    for hh in args.h_sweep:
        nn = int(round(args.n * (args.h / hh) ** 2))
        rr = np.random.default_rng(args.seed + 1)
        sub = run_torus(nn, hh, rr, rhos=[0.1, 0.2, 0.4], perp=args.perp)
        sweep.append({"h": hh, "n": nn, "leakage": sub["leakage"], "transport": sub["transport"]})
        print("sweep", hh, nn, flush=True)
    res["h_sweep"] = sweep
    res["seconds"] = time.time() - t0
    with open(os.path.join(OUT, f"gauge_checks_n{args.n}_s{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("done", res["seconds"])


if __name__ == "__main__":
    main()
