"""Pancreatic endocrinogenesis (Bastidas-Ponce et al. 2019, via `scvelo`'s
built-in loader) -- the canonical scVelo/CellRank benchmark with *real*
spliced/unspliced RNA velocity (unlike LARRY, see CLAUDE.md's correction).

Known biology (the validated differentiation hierarchy, used as an
independent check below, not fit by anything here):

    Ductal -> Ngn3 low EP -> Ngn3 high EP -> Pre-endocrine -> {Alpha, Beta,
                                                                Delta, Epsilon}

``load()`` runs scVelo's standard pipeline (filter/normalise, moments,
stochastic-mode velocity, PCA-space velocity projection) and returns the
PCA representation, the velocity in that same space, cluster labels, and
scVelo's own velocity-pseudotime (a cheap, independent ordering to sanity
check directionality against -- not something this project fits, just
another consumed signal, same spirit as the velocity itself). Cached to
``data/pancreas.npz``; the raw download (~50MB) is cached by scvelo itself
under ``data/Pancreas/``.
"""

import os

import numpy as np
import scipy.sparse as sp

DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(DATA, "pancreas.npz")


def preprocess(n_pcs=30, n_top_genes=2000, mode="stochastic", seed=0):
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 1
    os.makedirs(DATA, exist_ok=True)

    adata = scv.datasets.pancreas()
    # scv.pp.filter_and_normalize no longer forwards n_top_genes (that kwarg
    # was dropped from normalize_per_cell in this scvelo/scanpy combo) --
    # do HVG selection as its own step via scanpy, the modern equivalent.
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    # HVG selection (scanpy's default "seurat" flavor) expects log data on
    # .X; log1p only touches .X, not the spliced/unspliced *layers* that
    # scv.pp.moments / scv.tl.velocity need to stay on the linear scale for.
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=30)
    scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)
    scv.tl.velocity_embedding(adata, basis="pca")
    # scVelo/CellRank's own heuristic baseline: cosine similarity between
    # each cell's velocity and its neighbours' displacement, softmax
    # row-normalised into a genuine Markov transition matrix -- exactly the
    # "heuristic cosine-similarity transition kernel" this project's
    # introduction contrasts FinslerSinkhornAffinity against. Kept sparse
    # (it is exactly as sparse as the kNN graph) rather than densified.
    T_scvelo = scv.tl.transition_matrix(adata).tocsr()

    X_pca = np.asarray(adata.obsm["X_pca"][:, :n_pcs], dtype=np.float32)
    V_pca = np.asarray(adata.obsm["velocity_pca"][:, :n_pcs], dtype=np.float32)
    X_umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    clusters = adata.obs["clusters"].astype(str).values
    clusters_coarse = adata.obs["clusters_coarse"].astype(str).values
    pseudotime = np.asarray(adata.obs["velocity_pseudotime"], dtype=np.float32)

    np.savez_compressed(
        CACHE, X_pca=X_pca, V_pca=V_pca, X_umap=X_umap,
        clusters=clusters, clusters_coarse=clusters_coarse, pseudotime=pseudotime,
        T_scvelo_data=T_scvelo.data, T_scvelo_indices=T_scvelo.indices,
        T_scvelo_indptr=T_scvelo.indptr, T_scvelo_shape=np.array(T_scvelo.shape),
    )
    print(f"  cached {CACHE}  ({X_pca.shape[0]} cells, {n_pcs} PCs)", flush=True)


def load(subsample=None, seed=0):
    """Returns a dict: X_pca, V_pca (n, n_pcs), X_umap (n, 2), clusters,
    clusters_coarse (str arrays), pseudotime (n,) in [0, 1], T_scvelo
    (n, n sparse CSR) -- scVelo's own cosine-similarity transition matrix
    (`scv.tl.transition_matrix`), the baseline `pancreas_scvelo_baseline.py`
    compares FinslerSinkhornAffinity against.

    ``subsample``: if given, a stratified subsample of at most this many
    cells *per cluster* (keeps every cluster, including the rare ones --
    Epsilon n=142, Delta n=70 in the full data -- rather than a uniform
    random draw that could nearly wipe them out). ``T_scvelo`` is
    row/column-subset to the same cells and re-row-normalised -- this
    changes what it means (neighbours outside the subsample are dropped),
    so prefer the full dataset (``subsample=None``) when comparing against
    it.
    """
    if not os.path.exists(CACHE):
        print("preprocessing pancreas (one-time; runs scVelo's pipeline) ...",
             flush=True)
        preprocess()
    d = np.load(CACHE, allow_pickle=True)
    T_scvelo = sp.csr_matrix(
        (d["T_scvelo_data"], d["T_scvelo_indices"], d["T_scvelo_indptr"]),
        shape=tuple(d["T_scvelo_shape"]),
    )
    out = dict(X_pca=d["X_pca"], V_pca=d["V_pca"], X_umap=d["X_umap"],
              clusters=d["clusters"].astype(str),
              clusters_coarse=d["clusters_coarse"].astype(str),
              pseudotime=d["pseudotime"], T_scvelo=T_scvelo)

    if subsample is not None:
        rng = np.random.default_rng(seed)
        idx = []
        for c in np.unique(out["clusters"]):
            ci = np.flatnonzero(out["clusters"] == c)
            if len(ci) > subsample:
                ci = rng.choice(ci, size=subsample, replace=False)
            idx.append(ci)
        idx = np.sort(np.concatenate(idx))
        T_sub = out["T_scvelo"][idx][:, idx].tocsr()
        row_sum = np.asarray(T_sub.sum(1)).ravel()
        T_sub = sp.diags(1.0 / np.clip(row_sum, 1e-12, None)) @ T_sub
        out = {k: (v[idx] if k != "T_scvelo" else T_sub.tocsr())
              for k, v in out.items()}

    return out


if __name__ == "__main__":
    d = load()
    n = d["X_pca"].shape[0]
    print(f"\ncells: {n}   PCs: {d['X_pca'].shape[1]}")
    print("\nclusters:")
    for c in sorted(set(d["clusters"])):
        print(f"  {c:16s} {(d['clusters'] == c).sum():6d}")
    print(f"\npseudotime range: [{d['pseudotime'].min():.3f}, {d['pseudotime'].max():.3f}]")
    print(f"velocity norm (PCA space): mean {np.linalg.norm(d['V_pca'], axis=1).mean():.3f}")
