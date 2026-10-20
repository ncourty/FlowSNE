"""FlowSNE in more than two dimensions: does the embedding still keep local
structure, and is the drift still recovered, at d = 5 or 10?

Neighbour embeddings are mostly used in 2-D (t-SNE's Student kernel with one
degree of freedom is tuned against the 2-D crowding problem; van der Maaten
2009 suggests d - 1 degrees of freedom in d dimensions), but higher-d
embeddings are used as features for downstream tasks. FlowSNE's repulsion is
exact (dense or KeOps), so its cost grows only linearly with d.

Grid: datasets {sphere vortex (1500 points), pancreas (baseline subsample)}
x d in {2, 5, 10} x degrees of freedom in {1, d - 1}; for each, FlowSNE (flow
fitted jointly) and a symmetric baseline (same input affinity, zero flow).
Input affinity: row-stochastic Randers (recommended for velocity data).

Metrics (same definitions as experiments/baselines/finsler_embedding_comparison.py):
  knn            10-NN preservation, input space -> embedding
  cos_ref        cosine between the fitted field and the true velocity pushed
                 into the embedding (local linear regression)
  speed_spearman Spearman(|fitted field|, |pushed-forward velocity|)
  pancreas only: cos_scvelo (vs scVelo's own projection), pt_align (vs the
                 local pseudotime gradient), celltype_knn_acc (10-NN cell-type
                 classification accuracy in the embedding, leave-one-out)

Run:    conda run -n dev python experiments/higher_dim/higher_dim.py run --dataset sphere
        (resumes: fits already in results/<dataset>.json are skipped)
        conda run -n dev python experiments/higher_dim/higher_dim.py report
Writes: experiments/higher_dim/results/<dataset>.json, experiments/higher_dim/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "baselines"))
sys.path.insert(0, os.path.join(HERE, "..", "single_cell"))

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow, mlp_flow  # noqa: E402
from finsler_embedding_comparison import evaluate, load_dataset  # noqa: E402

RES = os.path.join(HERE, "results")
DT = torch.float64


def celltype_knn_acc(Z, labels, k=10):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    votes = labels[idx]
    pred = np.array([max(set(v), key=list(v).count) for v in votes])
    return float((pred == labels).mean())


def run(dataset, seed=0, dims=(2, 5, 10), iters=3000, lr=0.03, quick=False):
    D = load_dataset(dataset, seed=0, quick=quick)
    X = torch.tensor(D["X"], dtype=DT)
    V = torch.tensor(D["omega"], dtype=DT)
    n = len(X)
    P = EntropicRandersAffinity(omega=V, max_omega_norm=0.9, doubly_stochastic=False)(X).detach()
    Xc = D["X"] - D["X"].mean(0)
    pcs = np.linalg.svd(Xc, full_matrices=False)[2]
    os.makedirs(RES, exist_ok=True)
    path = os.path.join(RES, f"{dataset}{'_quick' if quick else ''}.json")
    out = dict(dataset=dataset, n=n, seed=seed, runs=[])
    if os.path.exists(path):                            # resume: keep finished fits
        out = json.load(open(path))
    done = {(r["d"], r["dof"], r["method"]) for r in out["runs"]}
    for d in dims:
        init = Xc @ pcs[:d].T                           # at most min(d, input dim) columns
        if init.shape[1] < d:                           # pad (e.g. 3-D sphere, d = 10)
            rng = np.random.default_rng(seed)
            pad = 0.01 * init[:, 0].std() * rng.standard_normal((n, d - init.shape[1]))
            init = np.concatenate([init, pad], axis=1)
        for dof in sorted({1.0, float(max(d - 1, 1))}):
            for name, make_flow, train_flow in (
                    ("symmetric", lambda: mlp_flow(d), False),
                    ("flowsne", lambda: input_std_norm_flow(d, hidden_dim=32), True)):
                if (d, dof, name) in done:
                    continue
                torch.manual_seed(seed)                 # seed BEFORE building the flow: each
                aff = RandersAffinity(n_components=d, flow=make_flow(),  # fit reproducible alone
                                      degrees_of_freedom=dof).double()
                t0 = time.time()
                Z = fit_embedding(aff, P, init, max_iter=iters, lr=lr, seed=seed,
                                  train_flow=train_flow)
                with torch.no_grad():
                    loss = aff.cross_entropy(Z, P).item()
                    F = -aff.omega_at(Z)
                Zn, Fn = Z.numpy(), F.numpy()
                with warnings.catch_warnings():         # symmetric: zero field, speed undefined
                    warnings.simplefilter("ignore")
                    m = evaluate(D, Zn, Fn)
                if name == "symmetric":                 # no field: only layout metrics apply
                    m = {k: v for k, v in m.items() if k == "knn"}
                if "clusters" in D:
                    m["celltype_knn_acc"] = celltype_knn_acc(Zn, np.asarray(D["clusters"]))
                r = dict(d=d, dof=dof, method=name, loss=loss, secs=round(time.time() - t0, 1), **m)
                out["runs"].append(r)
                print(f"[{dataset}] d={d:2d} dof={dof:4.1f} {name:9s} " + "  ".join(
                    f"{k}={v:+.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in r.items() if k not in ("d", "dof", "method")), flush=True)
                json.dump(out, open(path, "w"), indent=1)


def report():
    files = sorted(f for f in os.listdir(RES) if f.endswith(".json") and "_quick" not in f)
    L = ["# FlowSNE in more than two dimensions", "",
         "See `higher_dim.py`. Row-stochastic Randers input affinity; FlowSNE = flow fitted "
         "jointly, symmetric = same input affinity with zero flow. `knn` = 10-NN preservation; "
         "`cos_ref` / `speed ρ` = recovery of the true velocity pushed into the embedding.", ""]
    for f in files:
        E = json.load(open(os.path.join(RES, f)))
        keys = ["loss", "knn", "cos_ref", "speed_spearman"]
        if any("cos_scvelo" in r for r in E["runs"]):
            keys += ["cos_scvelo", "pt_align", "celltype_knn_acc"]
        L += [f"## {E['dataset']} (n = {E['n']})", "",
              "| d | ν | method | " + " | ".join(keys) + " | secs |", "|---|---|---|" + "---|" * (len(keys) + 1)]
        for r in E["runs"]:
            L.append(f"| {r['d']} | {r['dof']:g} | {r['method']} | " + " | ".join(
                f"{r[k]:.3f}" if k in r else "–" for k in keys) + f" | {r['secs']:.0f} |")
        L.append("")
    open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", os.path.join(HERE, "RESULTS.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--dataset", choices=["sphere", "swiss", "pancreas"], default="sphere")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.dataset, iters=(60 if a.quick else a.iters), quick=a.quick)
    else:
        report()
