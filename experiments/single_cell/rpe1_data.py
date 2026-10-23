"""RPE1-FUCCI cell cycle (scEU-seq, Battich et al. 2020): preprocessing with
dynamo, exported to a plain .npz for FlowSNE's own env.

Runs in a SEPARATE conda env (dynamo's dependencies are pinned and heavy):
    conda create -n dynamo python=3.10 && conda run -n dynamo pip install dynamo-release statsmodels==0.14.4
    conda run -n dynamo python experiments/single_cell/rpe1_data.py
Downloads ~520 MB once (dyn.sample_data.scEU_seq_rpe1, into data/rpe1/data/).

Pipeline (dynamo's own kinetic tutorial): the pulse-labelled cells (labelling
times 15 min - 3 h; DMSO controls dropped), new = labelled RNA, total = all
RNA, dyn.tl.recipe_kin_data (normalisation, gene selection, PCA, moments,
kinetic parameters, velocity). Velocity: dynamo's labelling-based velocity
of total RNA ("velocity_T", measured from new RNA, not inferred from
splicing). Gene IDs are prefixed so dynamo skips its Ensembl->symbol
conversion (needs pyensembl and a genome download; symbols are not used).

Exported (data/rpe1.npz), as for gastrulation: positions = dynamo's PCA;
velocity projected LINEARLY onto it (log space: v / (1 + M_t), the moment of
total expression, then the PCA loadings), plus dynamo's own kernel-based
projection (cell_velocities, basis pca) for reference, and the share of the
velocity's norm the PCA keeps.

Ground truth, independent of the transcriptome: the FUCCI reporters.
Cell_cycle_relativePos in [0, 1) (Battich et al.'s position along the cycle
from the two fluorescence channels), RFP / GFP intensities.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    warnings.filterwarnings("ignore")
    import dynamo as dyn
    import scipy.sparse as sp

    raw_dir = os.path.join(DATA, "rpe1")
    os.makedirs(raw_dir, exist_ok=True)
    os.chdir(raw_dir)                       # sample_data downloads into ./data/
    a = dyn.sample_data.scEU_seq_rpe1()
    dyn.convert2float(a, ["Cell_cycle_possition", "Cell_cycle_relativePos"])
    a.var_names = "g_" + a.var_names        # skip the Ensembl -> symbol conversion

    k = a[a.obs.exp_type == "Pulse", :].copy()
    k.obs["time"] = k.obs["time"].astype(str)
    k = k[k.obs.time != "dmso", :].copy()
    k.obs["time"] = k.obs["time"].astype(float) / 60          # hours
    k.layers["new"] = k.layers["ul"] + k.layers["sl"]
    k.layers["total"] = k.layers["su"] + k.layers["sl"] + k.layers["uu"] + k.layers["ul"]
    for layer in ("uu", "ul", "su", "sl"):
        del k.layers[layer]
    dyn.tl.recipe_kin_data(adata=k, keep_filtered_genes=True, keep_raw_layers=True,
                           del_2nd_moments=False, tkey="time")
    dyn.tl.cell_velocities(k, enforce=True, vkey="velocity_T", ekey="M_t", basis="pca")

    def dense(m):
        return np.asarray(m.toarray() if sp.issparse(m) else m, dtype=np.float64)

    X_pca = np.asarray(k.obsm["X_pca"], dtype=np.float64)
    pca_genes = k.var["use_for_pca"].values.astype(bool)
    PCs = np.asarray(k.uns["PCs"], dtype=np.float64)
    if PCs.shape[0] != pca_genes.sum():
        raise RuntimeError(f"PCs {PCs.shape} vs {pca_genes.sum()} PCA genes")
    vel = dense(k.layers["velocity_T"])
    bad_genes = ~np.isfinite(vel).all(0)
    vel[~np.isfinite(vel)] = 0.0             # genes whose kinetic fit failed (infinite rates)
    M_t = dense(k.layers["M_t"])
    print(f"genes with non-finite velocity (set to 0): {int(bad_genes[pca_genes].sum())} of the "
          f"PCA genes; normalisation: X {k.uns['pp'].get('X_norm_method')}, layers "
          f"{k.uns['pp'].get('layers_norm_method')}; X max {float(dense(k.X[:50]).max()):.2f}, "
          f"M_t max {M_t.max():.1f}")
    v_log = vel[:, pca_genes] / (1.0 + M_t[:, pca_genes])
    V_direct = v_log @ PCs
    share = np.linalg.norm(V_direct, axis=1) / np.maximum(np.linalg.norm(v_log, axis=1), 1e-30)
    V_kernel = np.asarray(k.obsm["velocity_pca"], dtype=np.float64)
    d = min(V_kernel.shape[1], V_direct.shape[1])
    cos = (V_kernel[:, :d] * V_direct[:, :d]).sum(1) / np.maximum(
        np.linalg.norm(V_kernel[:, :d], axis=1) * np.linalg.norm(V_direct[:, :d], axis=1), 1e-30)
    print(f"{k.n_obs} cells, {pca_genes.sum()} PCA genes, "
          f"{int((np.abs(vel[:, pca_genes]).sum(0) > 0).sum())} of them with a velocity; "
          f"PCA dims {X_pca.shape[1]}; velocity norm share in the PCA: median {np.median(share):.2f}; "
          f"cos(kernel projection, direct) median {np.median(cos):+.2f}")
    out = dict(X_pca=X_pca, n_bad_genes=np.array(int(bad_genes[pca_genes].sum())), V_pca_direct=V_direct, V_pca_kernel=V_kernel, vel_share=share,
               cos_kernel_direct=cos,
               cycle_pos=k.obs["Cell_cycle_relativePos"].values.astype(np.float64),
               cycle_pos_raw=k.obs["Cell_cycle_possition"].values.astype(np.float64),
               rfp=k.obs["RFP_log10_corrected"].values.astype(np.float64),
               gfp=k.obs["GFP_log10_corrected"].values.astype(np.float64),
               time_h=k.obs["time"].values.astype(np.float64))
    path = os.path.join(DATA, "rpe1.npz")
    np.savez_compressed(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
