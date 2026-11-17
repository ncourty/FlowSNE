"""Single-cell results table for the paper (pancreas + gastrulation), and the
GPU-side pancreas fits it needs.

    python experiments/single_cell/sc_table.py pancreas --seeds 0 1 2      # fits (GPU if available)
    python experiments/single_cell/sc_table.py table [--gastrulation-tags repr_scvelo repr_direct]

``pancreas``: the full pancreas dataset (n = 3696, velocity from scVelo), the
same rows and metrics as the toy comparison (``experiments/baselines/
finsler_embedding_comparison.py``, Table 1 of the paper) but fitted with
``flowsne.embedding.fit_embedding`` on the available device (the baselines
script goes through TorchDR's matcher on the CPU in float64; at 3.7k cells
dense that is ~1 h per FlowSNE fit, a few minutes on a GPU):
  sym_{ds,rs}                 symmetric t-SNE on the balanced / row-stochastic
                              Randers affinity (direction-blind), velocity
                              projected scVelo-style for the field columns
  flowsne_{ds,rs}_joint       FlowSNE, joint fit
  flowsne_{ds,rs}_driftonly   FlowSNE, field fitted on the frozen sym_* layout
  theirs_isomap               Debaussart-Joniec et al. (only if their code is
                              available: FINSLER_EMBEDDING_SRC)
Metrics (``evaluate`` of the baselines script): cos_scvelo = agreement of the
method's field with the velocity projected onto its own layout (scVelo's
projection, the standard practice); pt_align = agreement with the local
gradient of scVelo's velocity pseudotime on the layout; cos_ref = agreement
with the push-forward of the 30-PC velocity through the layout's local
tangent map; knn = 10-NN preservation (Euclidean); knn_dir = overlap of the
10 nearest targets under the input Randers cost and the method's own output
cost. Results go to experiments/baselines/results/pancreas_full_seed<s>.{json,npz},
the format of the toy runs, so ``finsler_embedding_comparison.py report``
includes them.

``table``: gathers the pancreas_full runs and the gastrulation runs of
``gastrulation_repr.py`` (one tag per input velocity) into
experiments/single_cell/SC_TABLE.md and the LaTeX rows
article/ICML_FlowSNE/tables/single_cell_rows.tex. Gastrulation columns:
days of embryo age gained after 25 steps of a direction-following chain
(k = 30) over the same chain ignoring direction, on all cells and on the
"winnable" cell types; the fraction of cells moving forward; 10-NN
preservation.
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
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "baselines"))

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding, fit_drift_only  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402
import finsler_embedding_comparison as fec  # noqa: E402

RES = fec.RES
GRES = os.path.join(HERE, "gastrulation_results")
TABLE_MD = os.path.join(HERE, "SC_TABLE.md")
TABLE_TEX = os.path.join(ROOT, "article", "ICML_FlowSNE", "tables", "single_cell_rows.tex")


# ---------------------------------------------------------------------------
# pancreas, full, on the available device
def run_pancreas(seed, iters=3000, quick=False, dataset="pancreas_full"):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32 if cuda else torch.float64
    D = fec.load_dataset(dataset, seed, quick)
    X, om = D["X"], D["omega"]
    n = len(X)
    it = (lambda m: 50) if quick else (lambda m: m)
    tag = f"{dataset}{'_quick' if quick else ''}_seed{seed}"
    print(f"[{tag}] n = {n}, device {device}, {dtype}", flush=True)

    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(X)
    init = torch.tensor(init_np, dtype=dtype, device=device)
    Xt = torch.tensor(X, dtype=dtype, device=device)
    omt = torch.tensor(om, dtype=dtype, device=device)

    out = dict(dataset=dataset, seed=seed, n=n, runs=[])
    saved = dict(color=D["color"])

    def record(name, Z, drift, secs, extra=None):
        Z = np.asarray(Z, dtype=np.float64)
        drift = np.asarray(drift, dtype=np.float64)
        r = dict(method=name, secs=secs, **fec.evaluate(D, Z, drift, name), **(extra or {}))
        out["runs"].append(r)
        saved[name], saved[name + "_drift"] = Z, drift
        print(f"[{tag}] {name:24s} " + " ".join(
            f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float) and k != "secs")
            + f"  ({secs:.0f}s)", flush=True)
        os.makedirs(RES, exist_ok=True)
        json.dump(out, open(os.path.join(RES, tag + ".json"), "w"), indent=1)
        np.savez_compressed(os.path.join(RES, tag + ".npz"), **saved)

    t0 = time.time()
    P = {"ds": EntropicRandersAffinity(omega=omt, max_omega_norm=fec.RHO)(Xt).detach(),
         "rs": EntropicRandersAffinity(omega=omt, max_omega_norm=fec.RHO,
                                       doubly_stochastic=False)(Xt).detach()}
    print(f"[{tag}] input affinities ({time.time() - t0:.0f}s)", flush=True)

    def new_aff():
        return RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                               degrees_of_freedom=1.0).to(device=device, dtype=dtype)

    Z_sym = {}
    for key in ("ds", "rs"):
        torch.manual_seed(seed)
        aff = new_aff()                                   # zero flow, not trained: symmetric Student
        t0 = time.time()
        Z = fit_embedding(aff, P[key], init, max_iter=it(1500), lr=0.1, seed=seed,
                          train_flow=False).detach()
        Z_sym[key] = Z
        Zn = Z.cpu().double().numpy()
        F = fec.scvelo_projection(D["T"], Zn)             # the standard practice, for the field columns
        record(f"sym_{key}", Zn, F, time.time() - t0)

    try:
        fec._their_operators()
        from sklearn.manifold import Isomap
        Z_iso = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
        W, eps = fec.their_graph(X, om)
        t0 = time.time()
        Zl, V = fec.their_drift(W, eps, Z_iso)
        record("theirs_isomap", Zl, V, time.time() - t0)
    except ImportError:
        print("Debaussart-Joniec code not found: skipping theirs_isomap (set FINSLER_EMBEDDING_SRC)")

    for key in ("ds", "rs"):
        for how in ("joint", "driftonly"):
            torch.manual_seed(seed)
            aff = new_aff()
            t0 = time.time()
            if how == "joint":
                Z = fit_embedding(aff, P[key], init, max_iter=it(iters), lr=0.03, seed=seed)
            else:
                Z = fit_drift_only(aff, P[key], Z_sym[key], seed=seed, max_iter=it(1500), lr=0.05)
            Z = Z.detach()
            with torch.no_grad():
                loss = aff.cross_entropy(Z, P[key]).item()
                F = (-aff.omega_at(Z)).cpu().double().numpy()
            record(f"flowsne_{key}_{how}", Z.cpu().double().numpy(), F, time.time() - t0,
                   dict(loss=loss))


# ---------------------------------------------------------------------------
# the table
PANCREAS_ROWS = [("sym_ds", "symmetric t-SNE, balanced + projected $v$"),
                 ("sym_rs", "symmetric t-SNE, row-stoch. + projected $v$"),
                 ("theirs_isomap", "Debaussart-Joniec et al., Isomap"),
                 ("flowsne_ds_driftonly", "FlowSNE, balanced, frozen layout"),
                 ("flowsne_ds_joint", "FlowSNE, balanced"),
                 ("flowsne_rs_driftonly", "FlowSNE, row-stochastic, frozen layout"),
                 ("flowsne_rs_joint", "FlowSNE, row-stochastic")]
PANCREAS_COLS = [("cos_scvelo", "vs projected $v$"), ("pt_align", "vs pseudotime"),
                 ("cos_ref", "vs push-forward"), ("knn", "kNN"), ("knn_dir", "kNN$_\\to$")]
GASTRU_ROWS = [("input30", "input (30 PCs) + $v$"), ("pca10", "PCA, 10-D + $v$"),
               ("sym2", "symmetric t-SNE, 2-D + projected $v$"), ("flowsne2", "FlowSNE, 2-D"),
               ("flowsne2_native", "\\quad its Randers chain"),
               ("sym10", "symmetric t-SNE, 10-D + projected $v$"), ("flowsne10", "FlowSNE, 10-D"),
               ("flowsne10_native", "\\quad its Randers chain")]


def _ms(v, prec=2):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "--", "–"
    m = np.mean(v)
    md = f"{m:+.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{m:+.3f}"
    return f"{m:.{prec}f}".replace("-0.", "$-$0.").replace("+", ""), md


def pancreas_table(dataset="pancreas_full"):
    files = sorted(glob.glob(os.path.join(RES, f"{dataset}_seed*.json")))
    E = [json.load(open(f)) for f in files]
    if not E:
        return [], []
    md = [f"## Pancreas (n = {E[0]['n']}, {len(E)} seed(s))", "",
          "cos vs projected v: cosine between the method's field and scVelo's projection of the "
          "velocity onto the method's own layout; vs pseudotime: vs the local gradient of scVelo's "
          "velocity pseudotime on the layout; vs push-forward: vs the 30-PC velocity pushed through "
          "the layout's local tangent map; kNN: 10-NN preservation; kNN->: directed, under the "
          "input Randers cost and the method's own output cost.", "",
          "| method | " + " | ".join(c[1] for c in PANCREAS_COLS) + " |", "|---|" + "---|" * len(PANCREAS_COLS)]
    tex = [f"% pancreas, n = {E[0]['n']}, mean over {len(E)} seeds; columns: "
           + ", ".join(c[0] for c in PANCREAS_COLS)]
    for key, label in PANCREAS_ROWS:
        rr = [r for e in E for r in e["runs"] if r["method"] == key]
        if not rr:
            continue
        cells = [_ms([r.get(c) for r in rr]) for c, _ in PANCREAS_COLS]
        if key.startswith("sym"):              # its field IS the projection: the column is trivial
            cells[0] = ("--", "–")
        md.append(f"| {key} | " + " | ".join(c[1] for c in cells) + " |")
        tex.append(f"    {label} & " + " & ".join(c[0] for c in cells) + " \\\\")
    return md + [""], tex


def gastrulation_table(tag, k=30, step=25):
    path = os.path.join(GRES, tag, "repr.json")
    if not os.path.exists(path):
        return [], []
    J = json.load(open(path))
    S, K = J["scores"], J.get("knn", {})
    vel = J["args"].get("velocity", "direct")
    md = [f"## Gastrulation (n = {J['n']}, input velocity: {vel}, tag `{tag}`)", "",
          f"Days of embryo age gained after {step} steps of the direction-following chain "
          f"(k = {k}) over the direction-blind chain on the same representation: all cells, "
          "and the winnable cell types (velocity forward for >= 50% of their cells, "
          f"{J['info']['winnable_fraction']:.0%} of cells); fwd: fraction of cells moving forward; "
          "kNN: 10-NN preservation vs the 30 PCs.", "",
          "| representation | days | days, winnable | fwd | kNN |", "|---|---|---|---|---|"]
    tex = [f"% gastrulation, n = {J['n']}, velocity {vel}, k = {k}, s = {step}; columns: "
           "days, days winnable, fwd, knn"]
    for key, label in GASTRU_ROWS:
        rs = list(S.get(key, {}).get(f"k{k}", {}).values())
        if not rs:
            continue
        kn = list(K.get(key.replace("_native", ""), {}).values())
        cells = [_ms([r[f"rel_s{step}"] for r in rs]), _ms([r[f"rel_s{step}_winnable"] for r in rs]),
                 _ms([r[f"fwd_s{step}"] for r in rs]), _ms(kn)]
        md.append(f"| {key} | " + " | ".join(c[1] for c in cells) + " |")
        tex.append(f"    {label} & " + " & ".join(c[0] for c in cells) + " \\\\")
    return md + [""], tex


def table(gastrulation_tags):
    md = ["# Single-cell results (paper Table: pancreas + gastrulation)", "",
          "Generated by `sc_table.py table`; LaTeX rows in "
          "`article/ICML_FlowSNE/tables/single_cell_rows.tex`.", ""]
    tex = ["% generated by experiments/single_cell/sc_table.py table -- do not edit by hand"]
    m, t = pancreas_table()
    md += m
    tex += t + [""]
    for tag in gastrulation_tags:
        m, t = gastrulation_table(tag)
        md += m
        tex += t + [""]
    open(TABLE_MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    os.makedirs(os.path.dirname(TABLE_TEX), exist_ok=True)
    open(TABLE_TEX, "w", encoding="utf-8").write("\n".join(tex) + "\n")
    print("\n".join(md))
    print("wrote", TABLE_MD, "and", TABLE_TEX)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pancreas", "table"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="40 cells per cluster, 50 iterations")
    ap.add_argument("--gastrulation-tags", nargs="*", default=["repr_scvelo", "repr_direct"])
    a = ap.parse_args()
    if a.cmd == "pancreas":
        for s in a.seeds:
            run_pancreas(s, a.iters, a.quick)
    else:
        table(a.gastrulation_tags)
