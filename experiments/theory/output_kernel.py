"""What does the OUTPUT (embedding) kernel's drift capture? Tests of the
gauge argument on the target side (article/ICML, discussion).

The loss -sum P_ij log Q_ij reads the input's asymmetry only through the net
flow N = P - P^T. A balanced P has N 1 = 0 (Prop. netflow), so N is
orthogonal to every gradient edge flow: the gradient part of the output's
log-asymmetry is not rewarded by the attraction term. Exactly so for a
kernel linear in the output Randers distance, approximately for FlowSNE's
Student kernel on its square. A row-stochastic P has N 1 != 0 (its column
sums), which the gradient part of the output drift can fit -- but a
per-point target weight exp(-h_j) fits it without any geometry.

  test 1  loss sensitivity to adding a gradient field grad h vs a rotated one
          J grad h (same norm) to the output drift: (a) at the symmetric fit,
          omega = 0, first-order derivative, Student and Laplace kernels;
          (b) at the fitted FlowSNE model, quadratic fit of L(omega* + t g).
  test 2  graph-Hodge decomposition of the fitted flow direction -omega on the
          INPUT kNN graph (a frame common to all fits): gradient share, its
          potential vs the input velocity's, cyclic part vs the input's, and
          agreement across seeds.
  test 3  target kernels, same input, same init: symmetric; FlowSNE flow
          (input_std_norm_flow); stream function omega = J grad psi
          (divergence-free); per-point weight exp(-h_j); stream + weight.
          Scored by the loss and by the progress of the walk on Q (truncated
          to 30 entries per row) over the symmetric fit's (pancreas:
          pseudotime; RPE1: unwrapped FUCCI phase, fraction of a cycle).

Run:   conda run -n dev python experiments/theory/output_kernel.py fit --data rpe1 --norm DS --model flow --seed 0
       conda run -n dev python experiments/theory/output_kernel.py all       # every fit, then analyse
       conda run -n dev python experiments/theory/output_kernel.py analyse
Writes: experiments/theory/results/output_kernel/{fits/*.pt, P_*.npy, output_kernel.{json,md}}
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "..", "single_cell")
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SC)

from flowsne import EntropicRandersAffinity, RandersAffinity  # noqa: E402
from flowsne.distances import randers_distance  # noqa: E402
from flowsne.embedding import fit_embedding  # noqa: E402
from flowsne.flows import InputStdNorm, input_std_norm_flow, mlp_flow  # noqa: E402
from gastrulation_hodge import edge_flow, edges_from_knn, hodge, incidence  # noqa: E402
from gastrulation_repr import knn  # noqa: E402
from rpe1_hodge import TWO_PI, phase_progress  # noqa: E402

OUT = os.path.join(HERE, "results", "output_kernel")
FITS = os.path.join(OUT, "fits")
DT = torch.float64
MODELS = ["sym", "flow", "stream", "weight", "stream_weight"]
SEEDS = [0, 1, 2]
STEPS = [1, 5, 10]
Q = 2.0  # input cost power (--cost-power); results for q != 2 go to output_kernel_q<q>


# ---------------------------------------------------------------- data
def load(name, n_sub):
    if name == "pancreas":
        d = np.load(os.path.join(SC, "data", "pancreas.npz"), allow_pickle=True)
        X, V, truth, kind = d["X_pca"], d["V_pca"], d["pseudotime"], "linear"
    else:
        d = np.load(os.path.join(SC, "data", "rpe1.npz"))
        X, V, truth, kind = d["X_pca"], d["V_pca_direct"], TWO_PI * d["cycle_pos"], "circle"
    idx = np.sort(np.random.default_rng(0).choice(len(X), min(n_sub, len(X)), replace=False))
    return dict(X=X[idx].astype(np.float64), V=V[idx].astype(np.float64),
                truth=truth[idx].astype(np.float64), kind=kind)


def input_affinity(name, norm, d):
    path = os.path.join(OUT, f"P_{name}_{norm}.npy")
    if os.path.exists(path):
        return np.load(path)
    aff = EntropicRandersAffinity(perplexity="auto", omega=torch.tensor(d["V"]),
                                  doubly_stochastic=(norm == "DS"), cost_power=Q)
    P = aff(torch.tensor(d["X"])).detach().double().numpy()
    P = P / P.sum()
    np.save(path, P)
    return P


# ---------------------------------------------------------------- target kernels
class TargetKernel(nn.Module):
    """Student-t kernel on the squared averaged Randers distance, normalised
    over all pairs (as RandersAffinity), with an optional stream-function
    drift omega = c J grad psi (J the +90 degree rotation; c a single global
    factor c = 0.95 / (1 + max |J grad psi|), so omega stays divergence-free
    and admissible) and an optional per-point target weight exp(-h_j)."""

    def __init__(self, n, stream=False, weight=False, hidden=64, max_norm=0.95, dof=1.0):
        super().__init__()
        self.stream, self.weight, self.max_norm, self.dof = stream, weight, max_norm, dof
        if stream:
            last = nn.Linear(hidden, 1)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.psi = nn.Sequential(InputStdNorm(), nn.Linear(2, hidden), nn.Tanh(), last)
        if weight:
            self.h = nn.Parameter(torch.zeros(n, dtype=DT))

    def omega_at(self, Z):
        if not self.stream:
            return torch.zeros_like(Z)
        with torch.enable_grad():
            Zg = Z if Z.requires_grad else Z.detach().requires_grad_(True)
            g = torch.autograd.grad(self.psi(Zg).sum(), Zg, create_graph=True)[0]
        raw = torch.stack([-g[:, 1], g[:, 0]], 1)
        return self.max_norm * raw / (1.0 + raw.norm(dim=1).max())

    def log_kernel(self, Z, omega):
        n = Z.shape[0]
        C = randers_distance(Z, omega, mode="averaged") ** 2
        C = C.masked_fill(torch.eye(n, dtype=torch.bool), 1e12)
        lk = -0.5 * (self.dof + 1) * (C / self.dof + 1).log()
        return lk - self.h[None, :] if self.weight else lk

    def cross_entropy(self, Z, P, indices=None, exaggeration=1.0):
        lk = self.log_kernel(Z, self.omega_at(Z))
        return -exaggeration * (P * lk).sum() + P.sum() * torch.logsumexp(lk.reshape(-1), 0)


def lib_log_kernel(model, Z, omega):
    return model._log_kernel(model._cost(Z, omega))


def make_model(kind, n):
    if kind == "sym":
        return RandersAffinity(2, flow=mlp_flow(2), degrees_of_freedom=1.0).to(DT)
    if kind == "flow":
        return RandersAffinity(2, flow=input_std_norm_flow(2, hidden_dim=32), degrees_of_freedom=1.0).to(DT)
    return TargetKernel(n, stream="stream" in kind, weight="weight" in kind).to(DT)


def log_kernel(model, Z, omega):
    return model.log_kernel(Z, omega) if isinstance(model, TargetKernel) else lib_log_kernel(model, Z, omega)


def loss_at(model, Z, P, omega):
    lk = log_kernel(model, Z, omega)
    return -(P * lk).sum() + P.sum() * torch.logsumexp(lk.reshape(-1), 0)


# ---------------------------------------------------------------- fitting
def fit_path(name, norm, kind, seed):
    return os.path.join(FITS, f"{name}_{norm}_{kind}_s{seed}.pt")


def fit_one(name, norm, kind, seed, n_sub, iters):
    path = fit_path(name, norm, kind, seed)
    if os.path.exists(path):
        return
    d = load(name, n_sub)
    P = torch.tensor(input_affinity(name, norm, d), dtype=DT)
    n = len(P)
    X = d["X"] - d["X"].mean(0)
    pcs = np.linalg.svd(X, full_matrices=False)[2][:2]
    init = X @ pcs.T
    init = init + 0.05 * init.std() * np.random.default_rng(seed).standard_normal(init.shape)
    torch.manual_seed(seed)
    model = make_model(kind, n)
    hist, t0 = [], time.time()
    Z = fit_embedding(model, P, torch.tensor(init, dtype=DT), max_iter=iters, lr=0.1, seed=seed,
                      train_flow=(kind != "sym"), history=hist, log_every=250)
    torch.save(dict(Z=Z, state=model.state_dict(), hist=hist, seconds=time.time() - t0), path)
    print(f"{name} {norm} {kind} s{seed}: {time.time() - t0:.0f}s, loss "
          + " ".join(f"{v:.4f}" for _, v in hist), flush=True)


def load_fit(name, norm, kind, seed, n):
    f = torch.load(fit_path(name, norm, kind, seed))
    model = make_model(kind, n)
    model.load_state_dict(f["state"])
    return model, f["Z"]


# ---------------------------------------------------------------- analysis helpers
def random_fields(Z, K, rng, n_terms=16):
    """K pairs (grad h, J grad h) of smooth random fields on the embedding,
    h = sum_t a_t cos(b_t . z / s + c_t), each normalised to rms norm 1."""
    s = Z.std(0).mean()
    out = []
    for _ in range(K):
        b = rng.normal(size=(n_terms, 2)) * rng.uniform(0.5, 3.0, size=(n_terms, 1))
        a, c = rng.normal(size=n_terms), rng.uniform(0, TWO_PI, size=n_terms)
        arg = Z @ b.T / s + c
        g = -(np.sin(arg) * a) @ b / s
        g = g / np.sqrt((g ** 2).sum(1).mean())
        out.append((g, np.stack([-g[:, 1], g[:, 0]], 1)))
    return out


def laplace_loss(Z, P, omega, scale, potential=None):
    n = Z.shape[0]
    if potential is None:
        D = randers_distance(Z, omega, mode="averaged")
    else:
        D = randers_distance(Z, potential, mode="potential")
    lk = (-D / scale).masked_fill(torch.eye(n, dtype=torch.bool), -1e12)
    return -(P * lk).sum() + torch.logsumexp(lk.reshape(-1), 0)


def first_order(Z, P, fields, kernel):
    """d/dt L(t * field) at t = 0 for each field (omega = 0)."""
    Zt = torch.tensor(Z, dtype=DT)
    Pt = torch.tensor(P, dtype=DT)
    om = torch.zeros_like(Zt, requires_grad=True)
    if kernel == "student":
        model = make_model("sym", len(Z))
        L = loss_at(model, Zt, Pt, om)
    else:
        scale = float((Pt * torch.cdist(Zt, Zt)).sum())
        L = laplace_loss(Zt, Pt, om, scale)
    G = torch.autograd.grad(L, om)[0].numpy()
    return [float((G * f).sum()) for f in fields]


def walk_progress(Q, truth, kind, k=30):
    idx = np.argpartition(-Q, k, axis=1)[:, :k]
    vals = np.take_along_axis(Q, idx, 1)
    n = len(Q)
    M = sp.csr_matrix((vals.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    M = (sp.diags(1 / np.asarray(M.sum(1)).ravel()) @ M).tocsr()
    if kind == "circle":
        return {s: float(v.mean()) for s, v in phase_progress(M, truth, STEPS).items()}
    x, out = truth.copy(), {}
    for s in range(1, max(STEPS) + 1):
        x = M @ x
        if s in STEPS:
            out[s] = float((x - truth).mean())
    return out


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------- analysis
def analyse(n_sub, datasets, K=40):
    rng = np.random.default_rng(1)
    res = dict(n_sub=n_sub, test1a={}, test1b={}, test2={}, test3={})
    for name in datasets:
        d = load(name, n_sub)
        n = len(d["X"])
        EX = edges_from_knn(knn(d["X"], 30))
        BX = incidence(EX, n)
        phiV, gV, rV = hodge(BX, edge_flow(d["X"], d["V"], EX))
        res["test2"].setdefault(name, {})["input_gradient_share"] = float((gV ** 2).sum() / ((gV + rV) ** 2).sum())
        for norm in ("DS", "RS"):
            P = input_affinity(name, norm, d)
            key = f"{name}/{norm}"
            Pt = torch.tensor(P, dtype=DT)
            Nmat = P - P.T
            res["test1a"][key] = dict(net_divergence_rel=float(np.abs(Nmat.sum(1)).sum() / np.abs(Nmat).sum()))
            # ---- test 1a: omega = 0 at the symmetric fit
            _, Zs = load_fit(name, norm, "sym", 0, n)
            Zs_np = Zs.numpy()
            fields = random_fields(Zs_np, K, rng)
            for kernel in ("student", "laplace"):
                dg = np.abs(first_order(Zs_np, P, [g for g, _ in fields], kernel))
                dr = np.abs(first_order(Zs_np, P, [r for _, r in fields], kernel))
                res["test1a"][key][kernel] = dict(grad=float(np.median(dg)), rot=float(np.median(dr)),
                                                  ratio=float(np.median(dg) / np.median(dr)))
            # exact potential drift term h_j - h_i, Laplace kernel: zero for a balanced P
            scale = float((Pt * torch.cdist(Zs, Zs)).sum())
            hs = []
            for _ in range(K):
                h = torch.zeros(n, dtype=DT, requires_grad=True)
                L = laplace_loss(Zs, Pt, None, scale, potential=h)
                gh = torch.autograd.grad(L, h)[0].numpy()
                hv = rng.normal(size=n)
                hs.append(abs(float(gh @ hv)) / np.linalg.norm(hv))
            res["test1a"][key]["laplace_exact_potential"] = float(np.median(hs))
            # ---- test 1b: at the fitted FlowSNE model
            ts = np.array([-0.02, -0.01, 0.01, 0.02])
            coef = {"grad": [], "rot": []}
            for seed in SEEDS:
                model, Z = load_fit(name, norm, "flow", seed, n)
                with torch.no_grad():
                    om = model.omega_at(Z)
                    L0 = float(loss_at(model, Z, Pt, om))
                    for g, r in random_fields(Z.numpy(), K // 2, rng):
                        for lab, f in (("grad", g), ("rot", r)):
                            ft = torch.tensor(f, dtype=DT)
                            dL = np.array([float(loss_at(model, Z, Pt, om + t * ft)) - L0 for t in ts])
                            A = np.stack([ts, ts ** 2], 1)
                            a, b = np.linalg.lstsq(A, dL, rcond=None)[0]
                            coef[lab].append((a, b))
            res["test1b"][key] = {lab: dict(abs_slope=float(np.median(np.abs([c[0] for c in v]))),
                                            curvature=float(np.median([c[1] for c in v])))
                                  for lab, v in coef.items()}
            # ---- test 2: Hodge of the fitted flow direction on the input graph
            phis, rests, rows = [], [], []
            for seed in SEEDS:
                model, Z = load_fit(name, norm, "flow", seed, n)
                with torch.no_grad():
                    F = (-model.omega_at(Z)).numpy()
                Znp = Z.numpy()
                f = edge_flow(Znp, F, EX)
                phi, g, r = hodge(BX, f)
                phis.append(phi)
                rests.append(r)
                fc = edge_flow(Znp, F - F.mean(0), EX)
                phic, gc, _ = hodge(BX, fc)
                rows.append(dict(gradient_share=float((g ** 2).sum() / (f ** 2).sum()),
                                 mean_field_share=float((F.mean(0) ** 2).sum() / (F ** 2).sum(1).mean()),
                                 gradient_share_centred=float((gc ** 2).sum() / (fc ** 2).sum()),
                                 phi_centred_vs_input_phi=corr(phic, phiV),
                                 phi_vs_input_phi=corr(phi, phiV),
                                 phi_vs_truth=corr(phi, d["truth"]) if d["kind"] == "linear" else float("nan"),
                                 cyclic_vs_input_cyclic=corr(r, rV),
                                 full_vs_input_full=corr(f, gV + rV)))
            agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            agg["phi_across_seeds"] = float(np.mean([corr(a, b) for a, b in itertools.combinations(phis, 2)]))
            agg["cyclic_across_seeds"] = float(np.mean([corr(a, b) for a, b in itertools.combinations(rests, 2)]))
            res["test2"][name][norm] = agg
            # ---- test 3: target kernels
            _, Z0 = load_fit(name, norm, "sym", 0, n)
            sym_model = make_model("sym", n)
            with torch.no_grad():
                lk0 = log_kernel(sym_model, Z0, torch.zeros_like(Z0))
                L_sym = float(loss_at(sym_model, Z0, Pt, torch.zeros_like(Z0)))
            prog0 = walk_progress(torch.softmax(lk0, 1).numpy(), d["truth"], d["kind"])
            t3 = {"sym": dict(loss=L_sym, progress={s: 0.0 for s in STEPS})}
            for kind in MODELS[1:]:
                losses, progs = [], []
                for seed in SEEDS:
                    model, Z = load_fit(name, norm, kind, seed, n)
                    with torch.no_grad():
                        om = model.omega_at(Z) if kind == "flow" else None
                    if om is None:
                        om = model.omega_at(Z).detach()
                    with torch.no_grad():
                        lk = log_kernel(model, Z, om)
                        losses.append(float(-(Pt * lk).sum() + torch.logsumexp(lk.reshape(-1), 0)))
                    pr = walk_progress(torch.softmax(lk, 1).numpy(), d["truth"], d["kind"])
                    progs.append({s: pr[s] - prog0[s] for s in STEPS})
                t3[kind] = dict(loss=float(np.mean(losses)), loss_sd=float(np.std(losses)),
                                progress={s: float(np.mean([p[s] for p in progs])) for s in STEPS},
                                progress_sd={s: float(np.std([p[s] for p in progs])) for s in STEPS})
            res["test3"][key] = t3
            print(f"[{key}] done", flush=True)
            json.dump(res, open(os.path.join(OUT, "output_kernel.json"), "w"), indent=1)
    write_report(res)


def write_report(res):
    L = ["# Output kernel: what the target drift captures", "",
         f"Subsample n = {res['n_sub']} per dataset, 2-D embeddings, input affinity with cost power q = {Q:g} "
         "(perplexity sqrt(n)), averaged drift on both sides. See `output_kernel.py`.", "",
         "## Test 1a: first-order loss sensitivity at the symmetric fit (omega = 0)", "",
         "Median |dL/dt| over random fields, gradient grad h vs rotated J grad h of the same rms norm. "
         "`net div` = sum_i |(N 1)_i| / sum |N_ij|, N = P - P^T.", "",
         "| input | net div | Student: grad | rot | ratio | Laplace: grad | rot | ratio | Laplace, exact h_j - h_i |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in res["test1a"].items():
        s, l = v["student"], v["laplace"]
        L.append(f"| {k} | {v['net_divergence_rel']:.3f} | {s['grad']:.2e} | {s['rot']:.2e} | {s['ratio']:.3f} | "
                 f"{l['grad']:.2e} | {l['rot']:.2e} | {l['ratio']:.3f} | {v['laplace_exact_potential']:.1e} |")
    L += ["", "## Test 1b: at the fitted FlowSNE model, L(omega* + t f) - L(omega*) = a t + b t^2", "",
          "| input | grad: median abs a | median b | rot: median abs a | median b |", "|---|---|---|---|---|"]
    for k, v in res["test1b"].items():
        L.append(f"| {k} | {v['grad']['abs_slope']:.2e} | {v['grad']['curvature']:.2e} | "
                 f"{v['rot']['abs_slope']:.2e} | {v['rot']['curvature']:.2e} |")
    L += ["", "## Test 2: Hodge decomposition of the fitted flow direction on the input kNN graph", "",
          "Means over 3 seeds; correlations are over edges (cyclic parts, full flows) or cells (potentials).", "",
          "`mean` = share of the field's mean squared norm carried by its mean vector (a constant field "
          "is an exact gradient of c.z); `centred` = the same decomposition after removing it.", "",
          "| input | mean | gradient share | centred | potential vs input potential | centred | "
          "potential vs pseudotime | cyclic vs input cyclic | full vs input full | potential across seeds | "
          "cyclic across seeds |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in res["test2"].items():
        for norm in ("DS", "RS"):
            if norm not in v:
                continue
            a = v[norm]
            L.append(f"| {name}/{norm} (input share {v['input_gradient_share']:.2f}) | {a['mean_field_share']:.2f} | "
                     f"{a['gradient_share']:.2f} | {a['gradient_share_centred']:.2f} | "
                     f"{a['phi_vs_input_phi']:+.2f} | {a['phi_centred_vs_input_phi']:+.2f} | "
                     f"{a['phi_vs_truth']:+.2f} | {a['cyclic_vs_input_cyclic']:+.2f} | "
                     f"{a['full_vs_input_full']:+.2f} | {a['phi_across_seeds']:+.2f} | {a['cyclic_across_seeds']:+.2f} |")
    L += ["", "## Test 3: target kernels", "",
          "Loss = cross-entropy (mean over 3 seeds; symmetric: seed 0). Progress = walk on Q minus the "
          "symmetric fit's, s = 10 steps (pancreas: pseudotime; RPE1: fraction of a cycle).", "",
          "| input | " + " | ".join(MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for k, v in res["test3"].items():
        L.append(f"| {k} loss | " + " | ".join(f"{v[m]['loss']:.4f}" for m in MODELS) + " |")
        L.append(f"| {k} progress | " + " | ".join(
            f"{v[m]['progress'][10]:+.4f}" + (f" ± {v[m]['progress_sd'][10]:.4f}" if m != "sym" else "")
            for m in MODELS) + " |")
    open(os.path.join(OUT, "output_kernel.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "all", "analyse"])
    ap.add_argument("--data", default="rpe1")
    ap.add_argument("--norm", default="DS")
    ap.add_argument("--model", default="flow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sub", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--datasets", nargs="+", default=["pancreas", "rpe1"])
    ap.add_argument("--cost-power", type=float, default=2.0)
    a = ap.parse_args()
    global Q, OUT, FITS
    Q = a.cost_power
    if Q != 2:
        OUT = os.path.join(HERE, "results", f"output_kernel_q{Q:g}")
        FITS = os.path.join(OUT, "fits")
    os.makedirs(FITS, exist_ok=True)
    if a.cmd == "fit":
        fit_one(a.data, a.norm, a.model, a.seed, a.n_sub, a.iters)
    elif a.cmd == "all":
        for name, norm, kind in itertools.product(a.datasets, ("DS", "RS"), MODELS):
            for seed in ([0] if kind == "sym" else SEEDS):
                fit_one(name, norm, kind, seed, a.n_sub, a.iters)
        analyse(a.n_sub, a.datasets)
    else:
        analyse(a.n_sub, a.datasets)


if __name__ == "__main__":
    main()
