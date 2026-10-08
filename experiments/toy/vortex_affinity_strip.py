"""FlowSNE version of the draft paper's Figure 1 (`FinslerAffinity/article/
sections/results.tex`, `fig:vortex-affinity`): a symmetric self-referential
affinity gives a direction-blind neighbourhood of a probe point, while a
Randers ground cost makes the *outgoing* (who a point sends mass to) and
*incoming* (who sends mass to it) neighbourhoods visibly different arcs of
the ring, offset in the direction of rotation.

Laid out as a single row of 4 panels (the original was a 2x2 grid), using
FlowSNE's own EntropicRandersAffinity in place of the paper's heat-kernel
FinslerSinkhornAffinity:

  [1] the vortex data + drift field omega (dense quiver, clearly visible
      at this point count/scale -- not just a token few arrows)
  [2] symmetric affinity (EntropicRandersAffinity, omega=None -- the
      ablation, not a different method)
  [3] Randers affinity, OUTGOING row from the probe point
  [4] Randers affinity, INCOMING column to the probe point

Run: conda run -n dev python experiments/toy/vortex_affinity_strip.py
Writes: experiments/toy/vortex_affinity_strip.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from flowsne import EntropicRandersAffinity  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "vortex_affinity_strip.png")


def make_vortex(n=300, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 1.0 + 0.12 * rng.standard_normal(n)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=1)  # CCW drift
    return X, tangent, theta


def mean_direction(weights, X_np, i):
    d = X_np - X_np[i]
    v = (weights[:, None] * d).sum(0)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def main():
    n = 300
    X_np, omega_np, theta = make_vortex(n=n, seed=0)
    X = torch.tensor(X_np, dtype=torch.float64)
    omega = torch.tensor(omega_np, dtype=torch.float64)

    P_fin = EntropicRandersAffinity(perplexity=15, omega=omega)(X).detach().numpy()
    P_sym = EntropicRandersAffinity(perplexity=15, omega=None)(X).detach().numpy()

    asym = lambda P: np.linalg.norm(P - P.T) / np.linalg.norm(P)
    asym_fin, asym_sym = asym(P_fin), asym(P_sym)
    print(f"asymmetry ||P-P^T||/||P||  Randers={asym_fin:.3f}  symmetric={asym_sym:.3f}")

    tangent_unit = omega_np / np.linalg.norm(omega_np, axis=1, keepdims=True)
    cos_sym = np.array([mean_direction(P_sym[i], X_np, i) @ tangent_unit[i] for i in range(n)])
    cos_fin = np.array([mean_direction(P_fin[i], X_np, i) @ tangent_unit[i] for i in range(n)])
    print(f"population mean-direction cosine with true drift:  "
         f"symmetric={cos_sym.mean():+.3f}+-{cos_sym.std():.3f}  "
         f"Randers={cos_fin.mean():+.3f}+-{cos_fin.std():.3f}")

    i0 = int(np.argmin(np.abs(theta)))  # probe point near angle 0
    row_sym = P_sym[i0]
    row_out = P_fin[i0, :]
    row_in = P_fin[:, i0]

    dir_sym = mean_direction(row_sym, X_np, i0)
    dir_out = mean_direction(row_out, X_np, i0)
    dir_in = -mean_direction(row_in, X_np, i0)  # direction FROM i0 TO its sources
    dir_true = tangent_unit[i0]

    vmax = np.percentile(np.concatenate([row_sym, row_out, row_in]), 99.5)
    half_win = 0.4

    fig, axes = plt.subplots(1, 4, figsize=(19, 5.2))
    fig.subplots_adjust(top=0.80, wspace=0.45)

    # [1] data + drift field -- dense enough quiver to be clearly visible,
    # not just a handful of token arrows.
    ax = axes[0]
    sca = ax.scatter(X_np[:, 0], X_np[:, 1], c=theta, cmap="twilight", s=28, zorder=2)
    step = 4
    ax.quiver(X_np[::step, 0], X_np[::step, 1], omega_np[::step, 0], omega_np[::step, 1],
             color="black", scale=6, width=0.009, headwidth=4, headlength=5,
             zorder=3, alpha=0.95)
    ax.scatter(*X_np[i0], marker="*", s=320, c="black", zorder=3, label="probe $i_0$")
    ax.add_patch(plt.Rectangle((X_np[i0, 0] - half_win, X_np[i0, 1] - half_win),
                               2 * half_win, 2 * half_win, fill=False,
                               linestyle="--", color="black", linewidth=1))
    ax.set_title("Vortex data + drift $\\omega$", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.colorbar(sca, ax=ax, shrink=0.7, pad=0.06, label="angle $\\theta$")

    panels = [
        (axes[1], row_sym, dir_sym, f"symmetric\nasym={asym_sym:.3f}"),
        (axes[2], row_out, dir_out, f"Randers: OUTGOING\nasym={asym_fin:.3f}"),
        (axes[3], row_in, dir_in, "Randers: INCOMING\n(arrow = source direction)"),
    ]
    mappable = None
    arrow_scale = 0.28
    for ax, vals, mdir, title in panels:
        mappable = ax.scatter(X_np[:, 0], X_np[:, 1], c=vals, cmap="inferno",
                              vmin=0.0, vmax=vmax, s=110, zorder=2, edgecolor="none")
        ax.scatter(*X_np[i0], marker="*", s=400, facecolor="none",
                  edgecolor="cyan", linewidth=1.8, zorder=4)
        ax.annotate("", xy=X_np[i0] + arrow_scale * dir_true, xytext=X_np[i0], zorder=3,
                   arrowprops=dict(arrowstyle="-|>", color="lightgray", linestyle="--", linewidth=2))
        ax.annotate("", xy=X_np[i0] + arrow_scale * mdir, xytext=X_np[i0], zorder=5,
                   arrowprops=dict(arrowstyle="-|>", color="deepskyblue", linewidth=2.5))
        ax.set_xlim(X_np[i0, 0] - half_win, X_np[i0, 0] + half_win)
        ax.set_ylim(X_np[i0, 1] - half_win, X_np[i0, 1] + half_win)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    fig.colorbar(mappable, ax=list(axes[1:]), shrink=0.7, pad=0.02,
                label="affinity $P_{i_0,\\cdot}$ (shared scale)")
    fig.text(0.5, 0.90,
            "gray dashed = true local drift $\\omega_{i_0}$    "
            "blue = affinity-weighted mean direction    (zoom = dashed box, panel 1)",
            ha="center", fontsize=9, style="italic")
    fig.suptitle("Self-referential entropic OT: symmetric vs. Randers ground cost "
                "(FlowSNE, closed-form averaged distance)", fontsize=13, y=0.98)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
