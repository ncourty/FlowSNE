"""Divergence-free (stream) drift vs the free flow network on the single-cell
velocity datasets: pancreas and dentate gyrus (RNA velocity, scVelo),
stratified subsample of 150 cells per cluster (pancreas n = 1112, dentate
gyrus n = 1108). Same protocol as experiments/toy/stream_sphere.py and the
baseline comparison (experiments/baselines/finsler_embedding_comparison.py):
input EntropicRandersAffinity (rho = 0.9) balanced (ds) or row-stochastic
(rs); sym / {flow,stream_global}_joint (lr 0.03, 3000 its) /
{flow,stream_global}_driftonly (lr 0.05, 1500 its), stream_global =
stream_bound="global"; metrics: loss, cos_ref and speed_spearman against the
push-forward of the velocity, knn, cos_scvelo (scVelo's own embedding
projection), pt_align (local pseudotime gradient), RMS div / curl.

Expectation (article/ICML): these are mostly gradient (differentiation)
velocities, so under a balanced input little asymmetry is left and the stream
mode should match the free flow; under a row-stochastic input the free flow
can use the gradient channel the stream mode lacks.

Run:    conda run -n dev python experiments/single_cell/stream_sc.py run --dataset pancreas --seed 0 --norm ds
        conda run -n dev python experiments/single_cell/stream_sc.py report
Writes: experiments/single_cell/stream_results/{<dataset>_<norm>_seed<s>.{json,npz}, STREAM_SC.md}
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
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "baselines"))
sys.path.insert(0, os.path.join(REPO, "experiments", "toy"))

from flowsne import EntropicRandersAffinity, JointAffinityMatcher, fit_drift_only  # noqa: E402
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402
from finsler_embedding_comparison import RHO, evaluate, fit_matcher, knn_preservation  # noqa: E402
from stream_sphere import div_curl, make  # noqa: E402

RES = os.path.join(HERE, "stream_results")
DT = torch.float64
DATASETS = ["pancreas", "dentategyrus"]
SEEDS = [0, 1, 2]
METHODS = ["sym", "flow_joint", "stream_global_joint", "flow_driftonly", "stream_global_driftonly"]


def load_sc(name):
    if name == "pancreas":
        from pancreas_data import load
    else:
        from dentategyrus_data import load
    d = load(subsample=150, seed=0)               # fixed cells; the seed varies the fits
    return dict(X=d["X_pca"], omega=d["V_pca"], color=d["pseudotime"], tan_dim=5,
                pseudotime=d["pseudotime"], T=d["T_scvelo"], clusters=d["clusters"])


def run(dataset, seed, norm):
    D = load_sc(dataset)
    X, om = torch.tensor(D["X"], dtype=DT), torch.tensor(D["omega"], dtype=DT)
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(D["X"])
    P = EntropicRandersAffinity(omega=om, max_omega_norm=RHO, doubly_stochastic=(norm == "ds"))(X).detach()
    out = dict(dataset=dataset, seed=seed, norm=norm, n=len(X), runs={})
    saved = dict(pseudotime=D["pseudotime"], clusters=D["clusters"])
    path = os.path.join(RES, f"{dataset}_{norm}_seed{seed}")
    os.makedirs(RES, exist_ok=True)

    def record(name, Z, aff, secs):
        Z = Z.detach()
        with torch.no_grad():
            loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
        r = dict(loss=loss, secs=secs)
        if name == "sym":
            r["knn"] = knn_preservation(D["X"], Z.numpy())
        else:
            with torch.no_grad():
                F = -aff.omega_at(Z).numpy()
            r.update(evaluate(D, Z.numpy(), F))
            r["div_rms"], r["curl_rms"] = div_curl(aff, Z)
            saved[name + "_drift"] = F
            if aff.drift_mode == "stream":
                with torch.no_grad():
                    saved[name + "_h"] = aff.stream_function_at(Z).numpy()
        saved[name] = Z.numpy()
        out["runs"][name] = r
        print(f"[{dataset} {norm} s{seed}] {name:17s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()),
              flush=True)
        json.dump(out, open(path + ".json", "w"), indent=1)
        np.savez_compressed(path + ".npz", **saved)

    t0 = time.time()
    sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
    Z_sym = fit_matcher(sym, P, init_np, AffinityMatcher, seed, 1500, 0.1).detach()
    record("sym", Z_sym, sym, time.time() - t0)
    for kind in ("flow", "stream_global"):
        aff, t0 = make(kind, seed), time.time()
        Z = fit_matcher(aff, P, init_np, JointAffinityMatcher, seed, 3000, 0.03)
        record(f"{kind}_joint", Z, aff, time.time() - t0)
        aff, t0 = make(kind, seed), time.time()
        Z = fit_drift_only(aff, P, Z_sym, seed=seed, max_iter=1500, lr=0.05)
        record(f"{kind}_driftonly", Z, aff, time.time() - t0)


def report():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "*_seed*.json"))):
        d = json.load(open(f))
        for m, r in d["runs"].items():
            rows.setdefault((d["dataset"], d["norm"], m), []).append(
                dict(r, dloss=r["loss"] - d["runs"]["sym"]["loss"]))
    keys = ["dloss", "cos_ref", "speed_spearman", "cos_scvelo", "pt_align", "knn", "div_rms", "curl_rms"]
    L = ["# Stream (divergence-free) drift vs free flow: single-cell velocity", "",
         "See `stream_sc.py`. Mean ± sd over seeds; `dloss` = loss minus the symmetric fit's.", ""]
    for ds in DATASETS:
        for norm in ("ds", "rs"):
            L += [f"## {ds}, {'balanced' if norm == 'ds' else 'row-stochastic'} input", "",
                  "| method | seeds | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
            for m in METHODS:
                rs = rows.get((ds, norm, m))
                if not rs:
                    continue
                cells = [f"{np.mean([r[k] for r in rs]):.3f} ± {np.std([r[k] for r in rs]):.3f}"
                         if all(k in r for r in rs) else "" for k in keys]
                L.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + " |")
            L.append("")
    open(os.path.join(RES, "STREAM_SC.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--dataset", choices=DATASETS, default="pancreas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["ds", "rs"], default="ds")
    a = ap.parse_args()
    run(a.dataset, a.seed, a.norm) if a.cmd == "run" else report()
