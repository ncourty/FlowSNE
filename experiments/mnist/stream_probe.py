"""MNIST joint-fit probe: free flow vs divergence-free (stream) drift.

The legacy finding (mnist.py): a plain joint (Z, flow) fit is unstable on
MNIST -- every learning rate gave a worse-than-symmetric loss -- hence
fit_drift_only. The drift there, omega = -grad_a loss (each sample's own
loss gradient w.r.t. its activation), is close to a gradient field within a
class. Hypothesis (article/ICML): under a BALANCED input most of the input
asymmetry is gauge, the loss does not reward the gradient part of the output
drift to first order, and a free flow spends itself on an unidentified
direction; a stream drift (RandersAffinity(drift_mode="stream")) removes it.

Per seed and input normalisation (ds / rs), all with flowsne.fit_embedding
(the embedding at lr 1.0, as mnist.py's symmetric fit; PCA init; 1500 its):
  sym                       symmetric Student kernel (flow frozen at 0)
  {flow,stream}_joint_lr<f> joint fit, flow learning rate f in FLOW_LRS
                            (f = 1.0: one rate for both, the legacy setting)
  {flow,stream}_driftonly   flow on the frozen symmetric layout (lr 0.05, 2000 its)
flow = input_std_norm_flow(2, hidden_dim=16) (as mnist.py); stream =
stream_bound="global" (scalar_flow, one global rescaling -- the best stream
variant on the sphere vortex, experiments/toy/stream_sphere.py). Metrics: loss and loss minus
the symmetric fit's, the loss curve, same-class mass of the output kernel
(misclassified samples and all), RMS divergence / curl of the fitted field.
`hodge`: gradient share of the drift's edge flow on the activations' 30-NN
graph, and the asymmetry ||P - P^T|| / ||P|| each normalisation keeps.

Run:    conda run -n dev python experiments/mnist/stream_probe.py hodge
        conda run -n dev python experiments/mnist/stream_probe.py run --seed 0 --norm ds
        conda run -n dev python experiments/mnist/stream_probe.py report
Writes: experiments/mnist/stream_results/{hodge.json, probe_<norm>_seed<s>.json, STREAM_PROBE.md}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "single_cell"))
sys.path.insert(0, os.path.join(HERE, "..", "toy"))

from mnist_data import load  # noqa: E402
from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, fit_drift_only, fit_embedding, input_std_norm_flow,
)
from stream_sphere import div_curl  # noqa: E402

RES = os.path.join(HERE, "stream_results")
DT = torch.float64
SEEDS = [0, 1, 2]
FLOW_LRS = [0.003, 0.01, 0.03, 0.1, 1.0]
LR, ITERS, DRIFT_ITERS = 1.0, 1500, 2000


def data():
    d = load()
    X = torch.tensor(d["activations"], dtype=DT)
    V = torch.tensor(-d["grad"], dtype=DT)                      # descent direction
    return d, X, V


def make(kind, seed):
    torch.manual_seed(seed)
    if kind == "flow":
        return RandersAffinity(2, flow=input_std_norm_flow(2, hidden_dim=16), degrees_of_freedom=1.0).double()
    return RandersAffinity(2, drift_mode="stream", stream_bound="global", degrees_of_freedom=1.0).double()


def same_class_mass(Q, labels):
    row = Q / Q.sum(1, keepdims=True).clip(min=1e-300)
    return (row * (labels[:, None] == labels[None, :])).sum(1)


def run(seed, norm):
    d, X, V = data()
    labels, wrong = d["labels"], d["pred"] != d["labels"]
    from sklearn.decomposition import PCA
    init = PCA(n_components=2, random_state=seed).fit_transform(d["activations"])
    P = EntropicRandersAffinity(perplexity="auto", omega=V, doubly_stochastic=(norm == "ds"))(X).detach()
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"probe_{norm}_seed{seed}.json")
    out = dict(seed=seed, norm=norm, n=len(X), runs={})

    def record(name, aff, Z, hist, secs):
        Z = Z.detach()
        with torch.no_grad():
            loss = aff.cross_entropy(Z, P).item()
            scm = same_class_mass(aff(Z).double().numpy(), labels)
        r = dict(loss=loss, scm_wrong=float(scm[wrong].mean()), scm_all=float(scm.mean()),
                 hist=hist, secs=secs)
        if name != "sym":
            r["div_rms"], r["curl_rms"] = div_curl(aff, Z)
            r["omega_median"] = float(aff.omega_at(Z).norm(dim=1).median())
        out["runs"][name] = r
        print(f"[{norm} s{seed}] {name:24s} loss={loss:.4f} scm_wrong={r['scm_wrong']:.4f} "
              f"({secs:.0f}s)", flush=True)
        json.dump(out, open(path, "w"), indent=1)
        return Z

    t0, hist = time.time(), []
    sym = make("flow", seed)
    Z_sym = fit_embedding(sym, P, init, max_iter=ITERS, lr=LR, seed=seed, train_flow=False, history=hist)
    Z_sym = record("sym", sym, Z_sym, hist, time.time() - t0)
    for kind in ("flow", "stream"):
        for flr in FLOW_LRS:
            aff, hist, t0 = make(kind, seed), [], time.time()
            Z = fit_embedding(aff, P, init, max_iter=ITERS, lr=LR, flow_lr=flr, seed=seed, history=hist)
            record(f"{kind}_joint_lr{flr:g}", aff, Z, hist, time.time() - t0)
        aff, t0 = make(kind, seed), time.time()
        Z = fit_drift_only(aff, P, Z_sym, seed=seed, max_iter=DRIFT_ITERS, lr=0.05)
        record(f"{kind}_driftonly", aff, Z, [], time.time() - t0)


def hodge_check():
    from gastrulation_hodge import edge_flow, edges_from_knn, hodge, incidence
    from gastrulation_repr import knn
    d, X, V = data()
    Xn, Vn = X.numpy(), V.numpy()
    n = len(Xn)
    E = edges_from_knn(knn(Xn, 30))
    f = edge_flow(Xn, Vn, E)
    _, g, r = hodge(incidence(E, n), f)
    res = dict(n=n, gradient_share=float((g ** 2).sum() / (f ** 2).sum()),
               mean_share=float((Vn.mean(0) ** 2).sum() / (Vn ** 2).sum(1).mean()))
    # per class: the drift is the gradient of one loss per label
    shares = []
    for c in range(10):
        m = d["labels"] == c
        idx = np.flatnonzero(m)
        Ec = edges_from_knn(knn(Xn[idx], 15))
        fc = edge_flow(Xn[idx], Vn[idx], Ec)
        _, gc, _ = hodge(incidence(Ec, len(idx)), fc)
        shares.append(float((gc ** 2).sum() / (fc ** 2).sum()))
    res["gradient_share_within_class"] = shares
    for norm in ("ds", "rs"):
        P = EntropicRandersAffinity(perplexity="auto", omega=V, doubly_stochastic=(norm == "ds"))(X).detach()
        P0 = EntropicRandersAffinity(perplexity="auto", omega=None, doubly_stochastic=(norm == "ds"))(X).detach()
        a = lambda M: float((M - M.T).norm() / M.norm())
        res[f"asym_{norm}"], res[f"asym_{norm}_driftfree"] = a(P), a(P0)
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "hodge.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def report():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "probe_*_seed*.json"))):
        d = json.load(open(f))
        s0 = d["runs"]["sym"]["loss"]
        for m, r in d["runs"].items():
            rows.setdefault((d["norm"], m), []).append(dict(r, dloss=r["loss"] - s0))
    methods = ["sym"] + [f"{k}_joint_lr{f:g}" for k in ("flow", "stream") for f in FLOW_LRS] + \
              ["flow_driftonly", "stream_driftonly"]
    keys = ["loss", "dloss", "scm_wrong", "scm_all", "omega_median", "div_rms", "curl_rms"]
    L = ["# MNIST joint-fit probe: free flow vs stream (divergence-free) drift", "",
         f"See `stream_probe.py`. Mean ± sd over seeds; `dloss` = loss minus the symmetric fit's "
         f"(same seed and input), negative = better than symmetric; `worse` = number of seeds with "
         f"dloss > 0. Embedding lr {LR}, {ITERS} its; drift-only: lr 0.05, {DRIFT_ITERS} its.", ""]
    hp = os.path.join(RES, "hodge.json")
    if os.path.exists(hp):
        h = json.load(open(hp))
        L += [f"Input: gradient share of the drift's edge flow on the 30-NN graph "
              f"{h['gradient_share']:.3f} (within classes, 15-NN: median "
              f"{np.median(h['gradient_share_within_class']):.3f}); common-mean share "
              f"{h['mean_share']:.3f}. Asymmetry ||P - P^T||/||P||: balanced {h['asym_ds']:.4f} "
              f"(drift-free {h['asym_ds_driftfree']:.4f}), row-stochastic {h['asym_rs']:.4f} "
              f"(drift-free {h['asym_rs_driftfree']:.4f}).", ""]
    for norm in ("ds", "rs"):
        L += [f"## {'balanced' if norm == 'ds' else 'row-stochastic'} input", "",
              "| method | seeds | worse | " + " | ".join(keys) + " |", "|---" * (len(keys) + 3) + "|"]
        for m in methods:
            rs = rows.get((norm, m))
            if not rs:
                continue
            cells = [f"{np.mean([r[k] for r in rs]):.4f} ± {np.std([r[k] for r in rs]):.4f}"
                     if all(k in r for r in rs) else "" for k in keys]
            L.append(f"| {m} | {len(rs)} | {sum(r['dloss'] > 0 for r in rs)} | " + " | ".join(cells) + " |")
        L.append("")
    open(os.path.join(RES, "STREAM_PROBE.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "hodge", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["ds", "rs"], default="ds")
    a = ap.parse_args()
    {"run": lambda: run(a.seed, a.norm), "hodge": hodge_check, "report": report}[a.cmd]()
