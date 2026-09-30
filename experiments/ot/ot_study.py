"""Can a Randers self-affinity visualise an optimal-transport flow?

OT displacements are gradient fields (Brenier: T = grad u, u convex, so
T(x) - x = grad(u - |x|^2/2)). For a gradient drift omega = grad(phi) the
Randers distance is exact, d_F(x->y) = |y-x| + phi(y) - phi(x) -- but that
potential difference is a row/column scaling of exp(-d_F/eps), i.e. a gauge
that Sinkhorn's own scalings absorb. This script measures how much of a
drift each input-affinity construction actually sees, then whether an
embedding recovers a known OT flow.

Subcommands:
  gauge            Study A: affinity-only, drift = gradient (convex /
                   non-convex) / rotation / mix, many input-affinity
                   variants. Writes results/gauge.json.
  embed --seed s   Study B: source + Brenier-map target, drift from an
                   entropic coupling, several output kernels. Writes
                   results/embed_seed{s}.json (+ .npz embeddings).
  report           Aggregates both into RESULTS.md + embed.png.

Add --quick for a tiny smoke-test configuration.

Run: conda run -n dev python experiments/ot/ot_study.py gauge
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, InputStdNorm, input_std_norm_flow, fit_drift_only, randers_distance,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

RES = os.path.join(HERE, "results")
DT = torch.float64


# ===========================================================================
# Data + fields
# ===========================================================================

def make_clusters(n, d, k, rng):
    """k anisotropic clusters (3-dim-ish structure each) in R^d."""
    centers = rng.normal(size=(k, d)) * 1.0
    lab = rng.integers(0, k, size=n)
    Ls = rng.normal(size=(k, d, 3))
    X = centers[lab] + np.einsum("ndk,nk->nd", Ls[lab], rng.normal(size=(n, 3))) \
        + 0.3 * rng.normal(size=(n, d))
    return X, lab


def convex_potential(d, rng, n_ridges=6, rank=5):
    """h(x) = 0.5 x^T B x + sum_k softplus(a_k.x - b_k), B PSD: convex."""
    Q = rng.normal(size=(d, rank)) / math.sqrt(d)
    B = Q @ Q.T
    A = rng.normal(size=(n_ridges, d)) / math.sqrt(d)
    b = rng.normal(size=n_ridges)

    def h(X):
        return 0.5 * np.einsum("nd,de,ne->n", X, B, X) + np.logaddexp(0, X @ A.T - b).sum(1)

    def grad(X):
        s = 1 / (1 + np.exp(-(X @ A.T - b)))
        return X @ B + s @ A
    return h, grad


def nonconvex_potential(d, rng, n_ridges=6):
    """h(x) = sum_k sin(a_k.x + b_k): smooth, not convex."""
    A = rng.normal(size=(n_ridges, d)) * 0.3 / math.sqrt(d) * 3
    b = rng.uniform(0, 2 * np.pi, n_ridges)
    return (lambda X: np.sin(X @ A.T + b).sum(1),
            lambda X: np.cos(X @ A.T + b) @ A)


def rotation_field(X):
    """omega = J (x - mean), J skew in the top-2 PCA plane: divergence- and
    gradient-free (no potential), acting within the data's visible structure."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    u, v = Vt[0], Vt[1]
    J = np.outer(v, u) - np.outer(u, v)
    return Xc @ J.T


def unit_max(V):
    return V / np.linalg.norm(V, axis=1).max()


# ===========================================================================
# Study A: input-affinity gauge study
# ===========================================================================

def exact_randers_P(X, phi_scaled, perplexity, doubly_stochastic):
    """EntropicRandersAffinity's own construction, with the EXACT Randers
    distance of a gradient drift. Physical drift = grad(phi); the affinity
    uses -omega (see affinity.py), so d(x_i->x_j) = |dx| - (phi_j - phi_i)."""
    aff = EntropicRandersAffinity(perplexity=perplexity, doubly_stochastic=doubly_stochastic)
    n = X.shape[0]
    D = torch.cdist(X, X)
    dF = D - (phi_scaled[None, :] - phi_scaled[:, None])
    n_neg = int((dF < 0).sum())
    C = dF.clamp_min(0.0) ** 2
    eye = torch.eye(n, dtype=torch.bool)
    C = C.masked_fill(eye, float("inf"))
    target = math.sqrt(n) if perplexity == "auto" else float(perplexity)
    tau = aff._calibrate_bandwidth(C, target)
    K = torch.exp(-(C - C.min(1, keepdim=True).values) / tau[:, None]).masked_fill(eye, 0.0)
    P = aff._sinkhorn(K) if doubly_stochastic else (K / K.sum(1, keepdim=True)) / n
    return P, n_neg


def linear_global_P(Cost, eps, iters=3000):
    """Balanced Sinkhorn with a LINEAR (unsquared) cost and one global eps
    -- the construction under which a potential difference is exactly a
    gauge."""
    n = Cost.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    C = Cost.masked_fill(eye, float("inf"))
    logK = -(C - C[~eye].min()) / eps
    loga = torch.full((n,), -math.log(n), dtype=DT)
    f = torch.zeros(n, dtype=DT); g = torch.zeros(n, dtype=DT)
    for _ in range(iters):
        f = loga - torch.logsumexp(logK + g[None], 1)
        g = loga - torch.logsumexp(logK + f[:, None], 0)
    return torch.exp(logK + f[:, None] + g[None])


def affinity_metrics(P, P_sym, X, omega_phys):
    """rel_change, asymmetry, and flow alignment: the SHIFT in each point's
    mean outgoing displacement (vs the symmetric affinity), cosine with the
    physical drift. Using the shift removes density effects both share."""
    Pn = P / P.sum(1, keepdim=True)
    Sn = P_sym / P_sym.sum(1, keepdim=True)
    m = Pn @ X - X
    m0 = Sn @ X - X
    dm = m - m0
    cos = torch.nn.functional.cosine_similarity(dm, omega_phys, dim=1)
    strong = omega_phys.norm(dim=1) >= omega_phys.norm(dim=1).median()
    return dict(
        rel_change=((P - P_sym).abs().sum() / P_sym.abs().sum()).item(),
        asym=((P - P.T).abs().sum() / P.abs().sum()).item(),
        asym_sym=((P_sym - P_sym.T).abs().sum() / P_sym.abs().sum()).item(),
        align=cos.mean().item(),
        align_strong=cos[strong].mean().item(),
    )


def study_gauge(quick=False):
    n, d, k = (200, 10, 3) if quick else (1000, 50, 6)
    seeds = (0,) if quick else (0, 1, 2)
    rhos = (0.6,) if quick else (0.3, 0.6, 0.95)
    rows, approx_rows = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_np, _ = make_clusters(n, d, k, rng)
        hc, gc = convex_potential(d, rng)
        hn, gn = nonconvex_potential(d, rng)
        rot = rotation_field(X_np)
        fields = {
            "grad_convex": (unit_max(gc(X_np)), hc(X_np) / np.linalg.norm(gc(X_np), axis=1).max()),
            "grad_nonconvex": (unit_max(gn(X_np)), hn(X_np) / np.linalg.norm(gn(X_np), axis=1).max()),
            "rotation": (unit_max(rot), None),
        }
        fields["mix"] = (unit_max(fields["grad_convex"][0] + unit_max(rot)), None)

        X = torch.tensor(X_np, dtype=DT)
        D = torch.cdist(X, X)
        eps_lin = 0.1 * D.median().item()
        base = {
            "ds": EntropicRandersAffinity(omega=None)(X),
            "rs": EntropicRandersAffinity(omega=None, doubly_stochastic=False)(X),
            "lin": linear_global_P(D, eps_lin),
        }
        for fname, (om_unit, phi_unit) in fields.items():
            om = torch.tensor(om_unit, dtype=DT)
            for rho in rhos:
                t0 = time.time()
                res = {}
                res["ds_avg"] = (EntropicRandersAffinity(omega=om, max_omega_norm=rho)(X), "ds")
                res["rs_avg"] = (EntropicRandersAffinity(omega=om, max_omega_norm=rho,
                                                         doubly_stochastic=False)(X), "rs")
                res["ds_frozen"] = (EntropicRandersAffinity(omega=om, max_omega_norm=rho,
                                                            drift_mode="frozen")(X), "ds")
                # linear cost, global eps, averaged approximation (sign: affinity uses -omega)
                dF_avg = randers_distance(X, -rho * om, mode="averaged")
                res["lin_avg"] = (linear_global_P(dF_avg, eps_lin), "lin")
                n_neg = None
                if phi_unit is not None:
                    phi = torch.tensor(rho * phi_unit, dtype=DT)
                    P_ex, n_neg = exact_randers_P(X, phi, "auto", True)
                    res["ds_exact"] = (P_ex, "ds")
                    res["rs_exact"] = (exact_randers_P(X, phi, "auto", False)[0], "rs")
                    dF_ex = (D - (phi[None] - phi[:, None])).clamp_min(0.0)
                    res["lin_exact"] = (linear_global_P(dF_ex, eps_lin), "lin")
                    # closed-form approximation error vs exact, on each row's sqrt(n) nearest pairs
                    kk = int(math.sqrt(n))
                    nn_idx = D.masked_fill(torch.eye(n, dtype=torch.bool), float("inf")).topk(
                        kk, largest=False).indices
                    mask = torch.zeros(n, n, dtype=torch.bool)
                    mask.scatter_(1, nn_idx, True)
                    exact = D - (phi[None] - phi[:, None])
                    for mode in ("frozen", "averaged"):
                        approx = D + torch.einsum("ijd,ijd->ij", X[None] - X[:, None],
                                                  -rho * (om[:, None] if mode == "frozen"
                                                          else 0.5 * (om[:, None] + om[None])))
                        err = (approx - exact).abs() / exact.abs().clamp_min(1e-9)
                        approx_rows.append(dict(seed=seed, field=fname, rho=rho, mode=mode,
                                                rel_err_knn=err[mask].median().item(),
                                                rel_err_all=err[~torch.eye(n, dtype=torch.bool)]
                                                .median().item()))
                for vname, (P, bkey) in res.items():
                    r = dict(seed=seed, field=fname, rho=rho, variant=vname,
                             **affinity_metrics(P, base[bkey], X, om))
                    if vname == "ds_exact":
                        r["n_negative_dF"] = n_neg
                    rows.append(r)
                print(f"[gauge] seed={seed} {fname:15s} rho={rho}  ({time.time()-t0:.1f}s)  " +
                      "  ".join(f"{r['variant']}:chg={r['rel_change']:.3f}/al={r['align']:+.2f}"
                                for r in rows[-len(res):]), flush=True)
                os.makedirs(RES, exist_ok=True)
                json.dump(dict(rows=rows, approx=approx_rows),
                          open(os.path.join(RES, "gauge_quick.json" if quick else "gauge.json"), "w"),
                          indent=1)


# ===========================================================================
# Study B: embedding a known OT flow
# ===========================================================================

class GradFlow(nn.Module):
    """omega(z) = grad_z psi(z), psi a scalar Tanh MLP on std-normalised z,
    zero-initialised last layer (so omega == 0 at init). NOTE RandersAffinity
    then applies bounded_direction, a norm-dependent rescaling, so the final
    field is grad-psi's DIRECTION with a squashed magnitude -- a gradient
    inductive bias, not an exactly curl-free field."""

    def __init__(self, n_components, hidden_dim=32):
        super().__init__()
        self.norm = InputStdNorm()
        last = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.psi = nn.Sequential(nn.Linear(n_components, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), last)

    def forward(self, z):
        with torch.enable_grad():
            z_ = z if z.requires_grad else z.detach().requires_grad_(True)
            psi = self.psi(self.norm(z_)).sum()
            return torch.autograd.grad(psi, z_, create_graph=True)[0]


def entropic_coupling(Xs, Xt, eps_frac=0.01, iters=3000):
    C = 0.5 * torch.cdist(Xs, Xt) ** 2
    eps = eps_frac * C.mean().item()
    n, m = C.shape
    loga = torch.full((n,), -math.log(n), dtype=DT); logb = torch.full((m,), -math.log(m), dtype=DT)
    f = torch.zeros(n, dtype=DT); g = torch.zeros(m, dtype=DT)
    for _ in range(iters):
        f = -eps * torch.logsumexp((g[None] - C) / eps + logb[None], 1)
        g = -eps * torch.logsumexp((f[:, None] - C) / eps + loga[:, None], 0)
    return torch.exp((f[:, None] + g[None] - C) / eps + loga[:, None] + logb[None])


def fit(affinity_out, P, init_np, matcher_cls, seed=0, max_iter=1500, lr=0.1):
    torch.manual_seed(seed)
    dr = matcher_cls(
        affinity_in="precomputed", affinity_out=affinity_out, n_components=2,
        loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
        init=init_np, init_scaling=1.0, random_state=seed, verbose=False,
    )
    return dr.fit_transform(P)


def study_embed(seed, quick=False):
    n, d, k = (120, 10, 3) if quick else (600, 50, 6)
    it = (lambda m: 60) if quick else (lambda m: m)
    rng = np.random.default_rng(100 + seed)
    Xs_np, lab = make_clusters(n, d, k, rng)
    h, gh = convex_potential(d, rng)
    V_true = gh(Xs_np)
    V_true = 4.0 * V_true / np.median(np.linalg.norm(V_true, axis=1))  # typical shift ~ 4
    Xt_np = Xs_np + V_true   # T = grad(|x|^2/2 + c h), c h convex: OT pairing is i <-> i
    Xs, Xt = torch.tensor(Xs_np, dtype=DT), torch.tensor(Xt_np, dtype=DT)

    pi = entropic_coupling(Xs, Xt)
    V_src = (pi @ Xt) / pi.sum(1, keepdim=True) - Xs                # barycentric, forward
    V_tgt = Xt - (pi.T @ Xs) / pi.sum(0)[:, None]                   # barycentric, backward
    Vt = torch.tensor(V_true, dtype=DT)
    cos_cpl = torch.nn.functional.cosine_similarity(V_src, Vt, dim=1)
    out = dict(seed=seed, n=n, d=d,
               coupling_cos_src=cos_cpl.mean().item(),
               coupling_relerr_src=((V_src - Vt).norm(dim=1) / Vt.norm(dim=1)).median().item(),
               coupling_cos_tgt=torch.nn.functional.cosine_similarity(V_tgt, Vt, dim=1).mean().item(),
               frac_diag_mass=(pi.diagonal().sum() / pi.sum()).item(),
               runs=[])
    print(f"[embed seed={seed}] coupling: cos(src)={out['coupling_cos_src']:.3f} "
          f"relerr={out['coupling_relerr_src']:.3f} diag mass={out['frac_diag_mass']:.3f}", flush=True)

    X = torch.cat([Xs, Xt]); V = torch.cat([V_src, V_tgt])
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X.numpy())
    partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    sign = torch.cat([torch.ones(n), -torch.ones(n)]).to(DT)  # tgt: partner is behind
    shuffled = partner[torch.randperm(2 * n, generator=torch.Generator().manual_seed(seed))]

    saved = dict(labels=np.concatenate([lab, lab]), side=np.r_[np.zeros(n), np.ones(n)])
    for in_name, ds in (("ds", True), ("rs", False)):
        P = EntropicRandersAffinity(omega=V, doubly_stochastic=ds)(X).detach()
        P_sym = EntropicRandersAffinity(omega=None, doubly_stochastic=ds)(X).detach()
        in_m = affinity_metrics(P, P_sym, X, V)
        print(f"[embed seed={seed}] input {in_name}: {in_m}", flush=True)

        def score(Z, flow_dir):
            e = Z[partner] - Z
            c = torch.nn.functional.cosine_similarity(flow_dir, sign[:, None] * e, dim=1)
            e0 = Z[shuffled] - Z
            c0 = torch.nn.functional.cosine_similarity(flow_dir, sign[:, None] * e0, dim=1)
            return c.mean().item(), c[:n].mean().item(), c0.mean().item()

        sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
        t0 = time.time()
        Z_sym = fit(sym, P, init_np, AffinityMatcher, seed=seed, max_iter=it(1500), lr=0.1)
        with torch.no_grad():
            loss_sym = cross_entropy_loss(P, sym(Z_sym, log=True), log=True).item()
        out["runs"].append(dict(input=in_name, output="symmetric", loss=loss_sym,
                                secs=time.time() - t0, **{f"in_{k}": v for k, v in in_m.items()}))
        saved[f"{in_name}_symmetric"] = Z_sym.detach().numpy()

        torch.manual_seed(seed)
        configs = [
            ("global", lambda: ConstantFlow(2), "joint", 1500, 0.1),
            ("stdnorm32_joint", lambda: input_std_norm_flow(2, hidden_dim=32), "joint", 3000, 0.03),
            ("stdnorm32_driftonly", lambda: input_std_norm_flow(2, hidden_dim=32), "drift", 1500, 0.05),
            ("gradnet_joint", lambda: GradFlow(2, hidden_dim=32), "joint", 3000, 0.03),
            ("gradnet_driftonly", lambda: GradFlow(2, hidden_dim=32), "drift", 1500, 0.05),
        ]
        for name, make, how, iters, lr in configs:
            torch.manual_seed(seed)
            aff = RandersAffinity(n_components=2, flow=make(), degrees_of_freedom=1.0).double()
            t0 = time.time()
            try:
                if how == "joint":
                    Z = fit(aff, P, init_np, JointAffinityMatcher, seed=seed, max_iter=it(iters), lr=lr)
                else:
                    Z = fit_drift_only(aff, P, Z_sym.detach(), seed=seed, max_iter=it(iters), lr=lr)
                with torch.no_grad():
                    loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
                flow_dir = -aff.omega_at(Z.detach()).detach()
                Zd = Z.detach()
                a_all, a_src, a_ctrl = score(Zd, flow_dir)
                r = dict(input=in_name, output=name, loss=loss, beats_sym=loss < loss_sym,
                         align_partner=a_all, align_partner_src=a_src, align_shuffled=a_ctrl,
                         mean_flow_norm=flow_dir.norm(dim=1).mean().item(), secs=time.time() - t0)
                saved[f"{in_name}_{name}"] = Zd.numpy()
                saved[f"{in_name}_{name}_flow"] = flow_dir.numpy()
            except Exception as e:  # keep the night going; record the failure
                r = dict(input=in_name, output=name, error=repr(e), secs=time.time() - t0)
            out["runs"].append(r)
            print(f"[embed seed={seed}] {in_name} {name}: " +
                  " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in r.items() if k not in ("input", "output")), flush=True)
            os.makedirs(RES, exist_ok=True)
            tag = f"embed{'_quick' if quick else ''}_seed{seed}"
            json.dump(out, open(os.path.join(RES, tag + ".json"), "w"), indent=1)
            np.savez_compressed(os.path.join(RES, tag + ".npz"), **saved)


# ===========================================================================
# Report
# ===========================================================================

def _ms(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "–"
    return f"{np.mean(vals):+.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:+.3f}"


def report(quick=False):
    q = "_quick" if quick else ""
    lines = ["# OT-flow study results", "",
             "Generated by `experiments/ot/ot_study.py report`. Mean ± std over seeds.", ""]
    gpath = os.path.join(RES, f"gauge{q}.json")
    if os.path.exists(gpath):
        G = json.load(open(gpath))
        rows = G["rows"]
        lines += ["## Study A — how much of the drift does each input affinity see?", "",
                  "`chg` = ‖P − P_sym‖₁/‖P_sym‖₁ (vs the symmetric affinity of the same "
                  "construction). `align` = cosine between the shift in each point's mean outgoing "
                  "displacement and the physical drift (0 = blind, 1 = perfectly along the flow).",
                  ""]
        variants = ["ds_avg", "ds_frozen", "ds_exact", "rs_avg", "rs_exact", "lin_avg", "lin_exact"]
        for rho in sorted({r["rho"] for r in rows}):
            lines += [f"### ‖ω‖max = {rho}", "",
                      "| field | " + " | ".join(variants) + " |",
                      "|---|" + "---|" * len(variants)]
            for field in ("grad_convex", "grad_nonconvex", "rotation", "mix"):
                cells = []
                for v in variants:
                    sel = [r for r in rows if r["rho"] == rho and r["field"] == field and r["variant"] == v]
                    if not sel:
                        cells.append("–"); continue
                    cells.append(f"chg {np.mean([r['rel_change'] for r in sel]):.3f} / "
                                 f"align {np.mean([r['align'] for r in sel]):+.2f}")
                lines.append(f"| {field} | " + " | ".join(cells) + " |")
            lines.append("")
        neg = [r.get("n_negative_dF") for r in rows if r["variant"] == "ds_exact"]
        lines += [f"Exact-distance pairs clamped at 0 (|∇φ| exceeding ρ between samples): "
                  f"max {max(neg) if neg else 0} per run.", "",
                  "### Closed-form approximation error vs exact distance (gradient fields)", "",
                  "Median relative error, over each row's √n nearest pairs / over all pairs.", "",
                  "| field | ρ | frozen (kNN) | averaged (kNN) | frozen (all) | averaged (all) |",
                  "|---|---|---|---|---|---|"]
        A = G["approx"]
        for field in ("grad_convex", "grad_nonconvex"):
            for rho in sorted({r["rho"] for r in A}):
                g = lambda m, key: np.mean([r[key] for r in A if r["field"] == field
                                            and r["rho"] == rho and r["mode"] == m])
                lines.append(f"| {field} | {rho} | {g('frozen','rel_err_knn'):.2e} | "
                             f"{g('averaged','rel_err_knn'):.2e} | {g('frozen','rel_err_all'):.2e} | "
                             f"{g('averaged','rel_err_all'):.2e} |")
        lines.append("")

    E = [json.load(open(os.path.join(RES, f))) for f in sorted(os.listdir(RES))
         if f.startswith(f"embed{q}_seed") and f.endswith(".json")] if os.path.isdir(RES) else []
    if E:
        lines += ["## Study B — embedding a known OT flow", "",
                  f"Source: {E[0]['n']} points, {E[0]['d']}-D clusters; target = convex Brenier map "
                  "of source (true pairing i ↔ i). Drift from an entropic coupling's barycentric "
                  "projection.", "",
                  f"Coupling quality: cos(barycentric, true displacement) = "
                  f"{_ms([e['coupling_cos_src'] for e in E])}, median relative error = "
                  f"{_ms([e['coupling_relerr_src'] for e in E])}, diagonal mass = "
                  f"{_ms([e['frac_diag_mass'] for e in E])}.", "",
                  "`align` = cosine between the fitted flow at a point and the direction to its true "
                  "OT partner in the embedding (src → tgt; reversed for tgt points). `shuffled` = same "
                  "with random partners (should be ≈ 0).", "",
                  "| input | output | loss | beats symmetric | align | align (src only) | shuffled |",
                  "|---|---|---|---|---|---|---|"]
        for in_name in ("ds", "rs"):
            for e0 in E[0]["runs"]:
                if e0["input"] != in_name:
                    continue
                name = e0["output"]
                sel = [r for e in E for r in e["runs"] if r["input"] == in_name and r["output"] == name]
                errs = [r["error"] for r in sel if "error" in r]
                ok = [r for r in sel if "error" not in r]
                if name == "symmetric":
                    lines.append(f"| {in_name} | symmetric | {_ms([r['loss'] for r in ok])} | – | – | – | – |")
                    continue
                beats = f"{sum(r['beats_sym'] for r in ok)}/{len(ok)}"
                lines.append(f"| {in_name} | {name} | {_ms([r['loss'] for r in ok])} | {beats} | "
                             f"{_ms([r['align_partner'] for r in ok])} | "
                             f"{_ms([r['align_partner_src'] for r in ok])} | "
                             f"{_ms([r['align_shuffled'] for r in ok])} |"
                             + (f" ⚠ {len(errs)} failed: {errs[0][:80]}" if errs else ""))
        lines.append("")
        in_rows = [r for e in E for r in e["runs"] if r["output"] == "symmetric"]
        lines += ["Input-side (affinity only): " + "; ".join(
            f"{i}: chg {_ms([r['in_rel_change'] for r in in_rows if r['input']==i])}, "
            f"align {_ms([r['in_align'] for r in in_rows if r['input']==i])}" for i in ("ds", "rs")), ""]
        _embed_figure(q)
        lines += ["![embeddings](results/embed.png)", ""]
    path = os.path.join(HERE, f"RESULTS{q}.md")
    open(path, "w").write("\n".join(lines))
    print(f"wrote {path}")


def _embed_figure(q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = os.path.join(RES, f"embed{q}_seed0.npz")
    if not os.path.exists(f):
        return
    S = np.load(f)
    names = [k for k in ("ds_symmetric", "ds_global", "ds_stdnorm32_joint", "ds_gradnet_joint",
                         "ds_stdnorm32_driftonly") if k in S.files]
    lab, side = S["labels"], S["side"]
    n = int((side == 0).sum())
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.6))
    axes = np.atleast_1d(axes)
    sub = np.random.default_rng(0).choice(n, size=min(n, 60), replace=False)
    for ax, k in zip(axes, names):
        Z = S[k]
        for s, mk in ((0, "o"), (1, "^")):
            m = side == s
            ax.scatter(Z[m, 0], Z[m, 1], c=lab[m], cmap="tab10", vmin=0, vmax=9, s=8,
                       marker=mk, alpha=0.35 if s == 0 else 0.8, linewidths=0)
        for i in sub:  # true OT partner segments
            ax.plot([Z[i, 0], Z[n + i, 0]], [Z[i, 1], Z[n + i, 1]], color="gray", lw=0.4, alpha=0.5)
        if k + "_flow" in S.files:
            F = S[k + "_flow"]
            sc = 0.08 * (Z.max(0) - Z.min(0)).max() / max(np.abs(F).max(), 1e-9)
            ax.quiver(Z[sub, 0], Z[sub, 1], F[sub, 0], F[sub, 1], color="crimson",
                      scale=1 / sc, scale_units="xy", angles="xy", width=0.004)
        ax.set_title(k.replace("ds_", ""), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.suptitle("OT flow, seed 0, balanced input (o = source, ▲ = target, grey = true pairing, "
                 "red = fitted −ω)", fontsize=11)
    fig.savefig(os.path.join(RES, f"embed{q}.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gauge", "embed", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    {"gauge": lambda: study_gauge(a.quick), "embed": lambda: study_embed(a.seed, a.quick),
     "report": lambda: report(a.quick)}[a.cmd]()
