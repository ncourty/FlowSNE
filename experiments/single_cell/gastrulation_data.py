"""Mouse gastrulation (Pijuan-Sala et al. 2019, via scVelo's built-in
loaders) -- a large single-cell dataset with spliced/unspliced counts for RNA
velocity: 116,312 cells (as documented by scVelo) from mouse embryos at nine
time points, E6.5 to E8.5, across many lineages.

What it adds over pancreas (3.7k cells): ~30x more cells, and REAL collection
times (embryonic stage). The stage is an independent check of direction --
not derived from the velocity, unlike scVelo's velocity pseudotime.

Subsets (scVelo datasets):
  full        gastrulation            ~4.7 GB download, 116k cells
  e75         gastrulation_e75        ~390 MB, E7.5 only
  erythroid   gastrulation_erythroid  ~485 MB, erythroid lineage only

``preprocess()`` runs the same scVelo pipeline as pancreas_data.py
(filter/normalise, HVGs, moments, stochastic-mode velocity, velocity graph,
PCA-space velocity -- scVelo's PCA projection needs the velocity graph, and
using it keeps the velocities comparable with pancreas) and caches PCA
coordinates, velocity in that space, cell type, stage, scVelo's velocity
transition matrix (a CellRank-style velocity kernel), and optionally its
velocity pseudotime (off by default for the full set, where it is slow).

The obs column names for cell type and stage are detected (the files are not
guaranteed to use the same ones); what was found is printed and stored.

Run:    conda run -n dev python experiments/single_cell/gastrulation_data.py --subset erythroid
Writes: experiments/single_cell/data/gastrulation_<subset>.npz
        (raw download cached under data/Gastrulation/)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")
LOADERS = {"full": "gastrulation", "e75": "gastrulation_e75", "erythroid": "gastrulation_erythroid"}
CELLTYPE_KEYS = ("celltype", "cell_type", "clusters", "celltype_extended")
STAGE_KEYS = ("stage", "time", "day", "embryo_stage")


def cache_path(subset):
    return os.path.join(DATA, f"gastrulation_{subset}.npz")


def _pick(obs, keys, what):
    for k in keys:
        if k in obs.columns:
            return k
    print(f"  !! no {what} column among {keys}; obs columns are: {list(obs.columns)}")
    return None


def _stage_to_float(values):
    """'E7.25' -> 7.25; anything unparsable (e.g. 'mixed_gastrulation') -> NaN."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    for i, v in enumerate(values):
        s = str(v).strip().lstrip("Ee")
        try:
            out[i] = float(s)
        except ValueError:
            pass
    return out


def preprocess(subset="full", n_pcs=30, n_top_genes=2000, mode="stochastic",
               with_pseudotime=None):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    os.makedirs(DATA, exist_ok=True)
    with_pseudotime = (subset != "full") if with_pseudotime is None else with_pseudotime
    t0 = time.time()

    raw = os.path.join(DATA, "Gastrulation", f"{LOADERS[subset]}.h5ad")
    adata = getattr(scv.datasets, LOADERS[subset])(file_path=raw)
    print(f"  loaded {adata.n_obs} cells x {adata.n_vars} genes; layers {list(adata.layers)}; "
          f"obs columns {list(adata.obs.columns)}  ({time.time() - t0:.0f}s)", flush=True)
    if not {"spliced", "unspliced"} <= set(adata.layers):
        raise RuntimeError("no spliced/unspliced layers: RNA velocity cannot be computed")
    ct_key = _pick(adata.obs, CELLTYPE_KEYS, "cell type")
    st_key = _pick(adata.obs, STAGE_KEYS, "stage")

    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    sc.pp.log1p(adata)                       # see pancreas_data.py: layers stay linear
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=30)
    scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="pca")
    print(f"  velocity + velocity graph done  ({time.time() - t0:.0f}s)", flush=True)

    out = dict(
        X_pca=np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float32),
        V_pca=np.asarray(adata.obsm["velocity_pca"][:, :n_pcs], dtype=np.float32),
        celltype=(adata.obs[ct_key].astype(str).values if ct_key
                  else np.array(["?"] * adata.n_obs)),
        stage_label=(adata.obs[st_key].astype(str).values if st_key
                     else np.array(["?"] * adata.n_obs)),
        celltype_key=np.array(ct_key or ""), stage_key=np.array(st_key or ""),
    )
    out["stage"] = _stage_to_float(out["stage_label"])
    if "X_umap" in adata.obsm:
        out["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    # scVelo's velocity transition matrix (CellRank-style velocity kernel): cheap
    # once the velocity graph exists, always kept (baseline for Markov-chain tests)
    T = scv.tl.transition_matrix(adata).tocsr()
    out.update(T_scvelo_data=T.data, T_scvelo_indices=T.indices, T_scvelo_indptr=T.indptr,
               T_scvelo_shape=np.array(T.shape))
    if with_pseudotime:
        scv.tl.velocity_pseudotime(adata)
        out["pseudotime"] = np.asarray(adata.obs["velocity_pseudotime"], dtype=np.float32)
        print(f"  pseudotime done  ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(cache_path(subset), **out)
    print(f"  cached {cache_path(subset)}  ({out['X_pca'].shape[0]} cells, {n_pcs} PCs, "
          f"{time.time() - t0:.0f}s total)", flush=True)


def load(subset="full", subsample=None, seed=0, n_cells=None):
    """Dict with X_pca, V_pca (n, n_pcs), celltype, stage_label (str),
    stage (float, embryonic day; NaN if unparsable), T_scvelo, optionally
    X_umap and pseudotime. ``subsample``: at most this many cells per cell
    type (keeps rare types). ``n_cells``: a uniform random subset of this
    size (keeps stage and cell-type proportions in expectation). Either way
    T_scvelo is restricted to the kept cells and row-renormalised (which
    drops transitions to removed cells)."""
    path = cache_path(subset)
    if not os.path.exists(path):
        preprocess(subset)
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files if not k.startswith("T_scvelo") and not k.endswith("_key")}
    if "T_scvelo_data" in d.files:
        import scipy.sparse as sp
        out["T_scvelo"] = sp.csr_matrix((d["T_scvelo_data"], d["T_scvelo_indices"],
                                         d["T_scvelo_indptr"]), shape=tuple(d["T_scvelo_shape"]))
    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = np.sort(np.concatenate([
            (lambda ci: rng.choice(ci, size=subsample, replace=False) if len(ci) > subsample
             else ci)(np.flatnonzero(out["celltype"] == c))
            for c in np.unique(out["celltype"])]))
        out = {k: (v[idx][:, idx] if k == "T_scvelo" else v[idx]) for k, v in out.items()}
    if n_cells is not None and n_cells < len(out["X_pca"]):
        idx = np.sort(np.random.default_rng(seed).choice(len(out["X_pca"]), size=n_cells,
                                                          replace=False))
        out = {k: (v[idx][:, idx] if k == "T_scvelo" else v[idx]) for k, v in out.items()}
    if "T_scvelo" in out:
        import scipy.sparse as sp
        rs = np.asarray(out["T_scvelo"].sum(1)).ravel()
        out["T_scvelo"] = (sp.diags(1.0 / np.clip(rs, 1e-12, None)) @ out["T_scvelo"]).tocsr()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(LOADERS), default="full")
    ap.add_argument("--with-pseudotime", action="store_true",
                    help="also compute scVelo's velocity pseudotime and transition matrix "
                         "(default: on for subsets, off for 'full')")
    a = ap.parse_args()
    if not os.path.exists(cache_path(a.subset)):
        preprocess(a.subset, with_pseudotime=True if a.with_pseudotime else None)
    d = load(a.subset)
    n = d["X_pca"].shape[0]
    print(f"\ncells: {n}   PCs: {d['X_pca'].shape[1]}")
    print("\ncell types:")
    for c, k in sorted(zip(*np.unique(d["celltype"], return_counts=True)), key=lambda x: -x[1]):
        print(f"  {c:32s} {k:7d}")
    print("\nstages:")
    for c, k in zip(*np.unique(d["stage_label"], return_counts=True)):
        print(f"  {c:20s} {k:7d}")
    print(f"\nvelocity norm (PCA space): mean {np.linalg.norm(d['V_pca'], axis=1).mean():.3f}")
