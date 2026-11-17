"""Paper figure: RPE1-FUCCI cell cycle, 2-D, direct velocity. Four maps coloured
by FUCCI cycle position (cyclic colour map), arrows = mean field per grid cell:
  (a) FlowSNE, row-stochastic input (joint fit): the field saturates and
      converges on the dense centre -- it fits the in-degree excess (hubs)
  (b) FlowSNE, balanced input (joint fit): an unsaturated field circulating
      around the cycle
  (c) the balanced symmetric layout with the velocity projected onto it
      (scVelo-style)
  (d) FlowSNE with the layout frozen at (c): only the field is fitted, to the
      same balanced affinity as (b) (drift-only)
Arrow length: each panel's own scale, the reference value in its key (the
row-stochastic field saturates at ~0.9, the balanced ones sit near 0.1, so a
shared scale would make the latter invisible). Title: the field's
circulation around the cycle (rpe1_circulation.py's loop statistic on the
map's own 30-NN graph).

    conda run -n dev python experiments/single_cell/rpe1_figure.py [--rs-tag rs2]
Writes article/ICML_FlowSNE/figures/rpe1_maps.{pdf,png}
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
from flowsne import RandersAffinity  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402
from flowsne.viz import grid_arrows  # noqa: E402
from gastrulation_hodge import edge_flow, edges_from_knn  # noqa: E402
from gastrulation_repr import knn, project_velocity  # noqa: E402
from rpe1_circulation import build_loops, stats  # noqa: E402

RES = os.path.join(HERE, "rpe1_results")
FIG = os.path.join(HERE, "..", "..", "article", "ICML_FlowSNE", "figures")


def field(tag, name, Z, rho=0.95):
    aff = RandersAffinity(2, flow=input_std_norm_flow(2, hidden_dim=32), degrees_of_freedom=1.0,
                          max_omega_norm=rho)
    aff.flow.load_state_dict(torch.load(os.path.join(RES, tag, f"flow_{name}.pt")))
    with torch.no_grad():
        return (-aff.omega_at(torch.tensor(Z, dtype=torch.float32))).numpy().astype(np.float64)


def circulation(Z, F, pos, rng):
    E = edges_from_knn(knn(Z, 30))
    loops = build_loops(Z, pos, E, 200, 12, rng)
    return stats(loops, edge_flow(Z, F, E), 12)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs-tag", default="rs2" if os.path.exists(os.path.join(RES, "rs2", "Z_direct_flowsne2_s0.npy"))
                    else "repr")
    ap.add_argument("--ds-tag", default="ds2")
    a = ap.parse_args()
    d = np.load(os.path.join(HERE, "data", "rpe1.npz"))
    X, V, pos = d["X_pca"].astype(np.float64), d["V_pca_direct"].astype(np.float64), d["cycle_pos"]
    rng = np.random.default_rng(0)

    Z_rs = np.load(os.path.join(RES, a.rs_tag, "Z_direct_flowsne2_s0.npy")).astype(np.float64)
    Z_ds = np.load(os.path.join(RES, a.ds_tag, "Z_direct_flowsne2_s0.npy")).astype(np.float64)
    Z_sym = np.load(os.path.join(RES, a.ds_tag, "Z_sym2_s0.npy")).astype(np.float64)
    panels = [
        ("FlowSNE, row-stochastic", Z_rs, field(a.rs_tag, "direct_flowsne2_s0", Z_rs), True),
        ("FlowSNE, balanced", Z_ds, field(a.ds_tag, "direct_flowsne2_s0", Z_ds), True),
        ("symmetric + projected $v$", Z_sym, project_velocity(X, V, Z_sym), False),
        ("FlowSNE, frozen layout", Z_sym, field(a.ds_tag, "direct_driftonly_s0", Z_sym), True),
    ]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    fig, axes = plt.subplots(2, 2, figsize=(3.25, 3.45))
    for ax, (title, Z, F, is_omega), letter in zip(axes.ravel(), panels, "abcd"):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=pos, cmap="twilight", vmin=0, vmax=1, s=1.2, lw=0,
                        alpha=0.75, rasterized=True)
        grid_arrows(ax, Z, F, bins=10, min_count=3, ref=None, ref_cells=1.0,
                    key=True, key_label="|ω|" if is_omega else "|v|", width=0.015, key_size=5.5)
        c = circulation(Z, F, pos, rng)
        ax.set_title(f"({letter}) {title}\ncirculation {c:+.2f}", pad=2, linespacing=1.1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal", "datalim")
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    fig.subplots_adjust(left=0.01, right=0.86, top=0.93, bottom=0.01, wspace=0.05, hspace=0.2)
    cax = fig.add_axes([0.88, 0.2, 0.025, 0.6])
    cb = fig.colorbar(sc, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cax.set_title("FUCCI\nposition", fontsize=6, pad=3)
    cb.ax.tick_params(labelsize=6)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"rpe1_maps.{ext}"), dpi=300)
    print("wrote", os.path.join(FIG, "rpe1_maps.pdf"), f"(RS from {a.rs_tag})")


if __name__ == "__main__":
    main()
