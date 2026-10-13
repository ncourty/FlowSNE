"""Render the MNIST learning-dynamics frames (dynamics.py) with the fitted
flow drawn as STREAMLINES of the continuous field, instead of arrows at the
samples.

The field is FlowSNE's flow network itself, evaluated on a grid: frames.npz
stores each frame's network weights. The network starts with InputStdNorm,
which divides its input by that input's own std -- so the inner network is
evaluated on grid / std(Z_t), i.e. exactly the function fitted at frame t,
not one rescaled by the grid's spread. Field = -bounded_direction(raw, rho),
the "flow direction" (see RandersAffinity.omega_at).

Grid cells far from any sample are masked (the network extrapolates
meaninglessly there). Line width AND grey level follow the field's speed on
ONE scale for all frames (no minimum width), so the fading as training
settles stays visible. Framing is
identical to dynamics.py's arrow animation (fixed axes from all frames).

Run:    conda run -n dev python experiments/mnist/animate_streamlines.py --dirs n1000_lam0 n1000_lam0.1
Writes: <dir>/dynamics_streamlines.mp4 (or .gif), and a contact sheet
        dynamics_results/streamlines_vs_arrows_n<n>.png comparing both styles.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from flowsne.distances import bounded_direction  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402

RESULTS = os.path.join(HERE, "dynamics_results")
MAX_OMEGA_NORM = 0.95            # RandersAffinity default, used by dynamics.py
# Greys scale for relative speed in [0, 1], starting at a light grey so slow
# lines fade without vanishing on white
_NORM = Normalize(vmin=-0.15, vmax=1.0, clip=True)


def load(run_dir):
    S = np.load(os.path.join(run_dir, "frames.npz"))
    M = json.load(open(os.path.join(run_dir, "metrics.json")))["metrics"]
    if "flow_params" not in S.files:
        sys.exit(f"{run_dir}: no flow weights in frames.npz; re-run dynamics.py")
    return S, M


def frame_axes(Z):
    lo, hi = np.percentile(Z.reshape(-1, 2), [0.5, 99.5], axis=0)
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def grid_field(S, t, lo, hi, res=70, mask_radius=2.0):
    """Field on a res x res grid at frame t, masked away from the samples."""
    flow = input_std_norm_flow(2, hidden_dim=int(S["flow_hidden_dim"])).double()
    torch.nn.utils.vector_to_parameters(torch.tensor(S["flow_params"][t]), flow.parameters())
    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    GX, GY = np.meshgrid(xs, ys)
    G = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1))
    with torch.no_grad():
        raw = flow[1](G / float(S["z_std"][t]))       # skip InputStdNorm: use Z_t's std
        field = -bounded_direction(raw, MAX_OMEGA_NORM).numpy()
    U = field[:, 0].reshape(res, res)
    V = field[:, 1].reshape(res, res)
    # mask cells with no sample within mask_radius grid cells
    from scipy.ndimage import maximum_filter
    Z = S["Z"][t]
    cnt, _, _ = np.histogram2d(Z[:, 1], Z[:, 0], bins=res,
                               range=[[lo[1], hi[1]], [lo[0], hi[0]]])
    near = maximum_filter(cnt, size=int(2 * mask_radius + 1)) > 0
    U = np.ma.array(U, mask=~near)
    V = np.ma.array(V, mask=~near)
    return xs, ys, U, V


def draw(ax, S, M, t, lo, hi, vmax, style, colors, sub):
    Z = S["Z"][t]
    n = Z.shape[0]
    size, alpha = (5, 0.55) if n <= 2000 else (max(0.3, 5000 / n), 0.35)  # 60k points: no blob
    ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=size, linewidths=0, alpha=alpha, zorder=1)
    if style == "streamlines":
        xs, ys, U, V = grid_field(S, t, lo, hi)
        speed = np.ma.array(np.hypot(U.data, V.data), mask=U.mask)   # avoid masked-array sqrt quirk
        rel = (speed / vmax).filled(0.0)            # ONE scale for all frames
        ax.streamplot(xs, ys, U, V, density=1.4, color=rel, cmap="Greys", norm=_NORM,
                      arrowsize=0.7, linewidth=2.4 * rel, zorder=2)
    else:
        F = S["F"][t]
        arrow = 0.08 * (hi - lo).max() / vmax
        ax.quiver(Z[sub, 0], Z[sub, 1], F[sub, 0] * arrow, F[sub, 1] * arrow, angles="xy",
                  scale_units="xy", scale=1, width=0.004, color="black", alpha=0.7, zorder=2)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    m = M[t]
    ax.set_title(f"iteration {m['step']}  acc {m['acc']:.2f}  |grad| {m['mean_grad_norm']:.1e}",
                 fontsize=9)


def animate(run_dir, fps=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    S, M = load(run_dir)
    lo, hi = frame_axes(S["Z"])
    vmax = max(np.linalg.norm(S["F"], axis=2).max(), 1e-12)
    colors = np.array(matplotlib.colormaps["tab10"].colors)[S["labels"]]
    fig, ax = plt.subplots(figsize=(6.4, 6.4))

    def update(t):
        ax.clear()
        draw(ax, S, M, t, lo, hi, vmax, "streamlines", colors, None)

    anim = animation.FuncAnimation(fig, update, frames=S["Z"].shape[0], blit=False)
    path = os.path.join(run_dir, "dynamics_streamlines.mp4")
    if shutil.which("ffmpeg"):
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    else:
        path = path[:-4] + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def contact_sheet(run_dirs, picks=(0, 2, 5, 10, 20, -1)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(d, style) for d in run_dirs for style in ("arrows", "streamlines")]
    fig, axes = plt.subplots(len(rows), len(picks), figsize=(3.1 * len(picks), 3.2 * len(rows)),
                             squeeze=False)
    for r, (d, style) in enumerate(rows):
        S, M = load(d)
        lo, hi = frame_axes(S["Z"])
        vmax = max(np.linalg.norm(S["F"], axis=2).max(), 1e-12)
        colors = np.array(matplotlib.colormaps["tab10"].colors)[S["labels"]]
        sub = np.random.default_rng(0).choice(S["Z"].shape[1], size=min(150, S["Z"].shape[1]),
                                              replace=False)
        for c, t in enumerate(picks):
            draw(axes[r, c], S, M, t % S["Z"].shape[0], lo, hi, vmax, style, colors, sub)
        axes[r, 0].set_ylabel(f"{os.path.basename(d)}\n{style}", fontsize=10)
    tag = os.path.basename(run_dirs[0]).split("_lam")[0]
    path = os.path.join(RESULTS, f"streamlines_vs_arrows_{tag}.png")
    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["n1000_lam0", "n1000_lam0.1"],
                    help="run directories under dynamics_results/")
    ap.add_argument("--no-animation", action="store_true")
    a = ap.parse_args()
    dirs = [os.path.join(RESULTS, d) for d in a.dirs]
    print("contact sheet:", contact_sheet(dirs))
    if not a.no_animation:
        for d in dirs:
            print("animation:", animate(d))
