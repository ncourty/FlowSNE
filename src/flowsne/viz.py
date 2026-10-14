"""Ways to draw a fitted flow over an embedding that stay readable when the
embedding has thousands of points (streamlines get cluttered).

- ``grid_arrows``: one arrow per cell of a coarse grid, the mean field of
  the samples in that cell, faded by the cell's point count (scVelo-style
  "grid" view), with a legend giving the field value of a one-cell arrow;
  ``default_bins(n)`` picks the grid size.
- ``direction_colors`` / ``direction_wheel``: colour each point by its
  flow direction (hue) and speed (saturation); no glyphs at all.
- ``divergence_map``: the divergence of the fitted field, computed exactly
  with autograd from the flow network, as a background heatmap -- sources
  (where points flow away from) in red, sinks (where they converge) in
  blue, faded away from the data.

``field_from_flow`` rebuilds the field -omega(z) (the "flow direction", see
``RandersAffinity.omega_at``) from a saved flow network, so it can be
evaluated anywhere. It assumes ``input_std_norm_flow``: that network divides
its input by the input's OWN std, so it must be evaluated on
``points / std(Z)`` -- the function fitted on Z -- not on raw grid points.

Matplotlib is imported lazily; this module is optional for the library.
"""

from __future__ import annotations

import numpy as np
import torch

from .distances import bounded_direction
from .flows import input_std_norm_flow


def field_from_flow(flow_params, z_std, hidden_dim=32, max_omega_norm=0.95):
    """Callable ``field(points (m, 2) array) -> (m, 2) array`` for a saved
    ``input_std_norm_flow`` (parameter vector + std of the embedding it was
    fitted on); ``field.divergence(points) -> (m,)`` gives its divergence."""
    flow = input_std_norm_flow(2, hidden_dim=hidden_dim).double()
    torch.nn.utils.vector_to_parameters(torch.as_tensor(flow_params, dtype=torch.float64),
                                        flow.parameters())
    inner, scale = flow[1], float(z_std)

    def _f(G):
        return -bounded_direction(inner(G / scale), max_omega_norm)

    def field(points):
        with torch.no_grad():
            return _f(torch.as_tensor(np.asarray(points), dtype=torch.float64)).numpy()

    def divergence(points):
        G = torch.as_tensor(np.asarray(points), dtype=torch.float64).clone().requires_grad_(True)
        F = _f(G)
        d0 = torch.autograd.grad(F[:, 0].sum(), G, retain_graph=True)[0][:, 0]
        d1 = torch.autograd.grad(F[:, 1].sum(), G)[0][:, 1]
        return (d0 + d1).detach().numpy()

    field.divergence = divergence
    return field


def _bounds(Z, pad=0.06, lo=None, hi=None):
    if lo is None or hi is None:
        lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
        p = pad * (hi - lo)
        lo, hi = lo - p, hi + p
    return np.asarray(lo, float), np.asarray(hi, float)


def default_bins(n):
    """Grid cells across for ``grid_arrows``: ~sqrt(n)/2.6, i.e. 12 at
    n = 1000, ~25 at 10k, capped to [8, 30] so arrows stay legible."""
    return int(np.clip(round(np.sqrt(n) / 2.6), 8, 30))


def grid_field(Z, F, bins=20, min_count=5, lo=None, hi=None):
    """Per occupied cell of a bins x bins grid: centroid of its samples, mean
    of their field F, and count (cells with fewer than ``min_count`` samples
    dropped). Returns (pos (m, 2), vec (m, 2), counts (m,), cell size (2,))."""
    lo, hi = _bounds(Z, lo=lo, hi=hi)
    cell = (hi - lo) / bins
    ij = np.floor((Z - lo) / cell).astype(int)
    ok = np.all((ij >= 0) & (ij < bins), axis=1)
    uniq, inv, cnt = np.unique(ij[ok, 0] * bins + ij[ok, 1], return_inverse=True,
                               return_counts=True)
    pos = np.zeros((len(uniq), 2))
    vec = np.zeros((len(uniq), 2))
    np.add.at(pos, inv, Z[ok])
    np.add.at(vec, inv, F[ok])
    keep = cnt >= min_count
    return pos[keep] / cnt[keep, None], vec[keep] / cnt[keep, None], cnt[keep], cell


def grid_arrows(ax, Z, F, bins=20, min_count=5, lo=None, hi=None, ref=None, ref_cells=1.0,
                key=True, key_label="|ω|", color="black", width=0.009, halo=True, zorder=3):
    """One arrow per occupied grid cell (see ``grid_field``), at the cell's
    sample centroid; opacity grows with the cell's count.

    Length has a meaning: a cell-mean field of norm ``ref`` is drawn
    ``ref_cells`` cells long (default ref: the 90th percentile of this panel's
    cell-mean norms -- pass a shared ref to compare panels), and ``key``
    adds a legend arrow labelled with that value. For FlowSNE fields the unit
    is the Randers drift omega (0 = symmetric, 0.95 = maximum asymmetry).
    Averaging shortens arrows where directions within a cell disagree, so
    length reads as strength x coherence. ``halo`` draws a white outline so
    arrows stay visible over a coloured background. Returns ref."""
    from matplotlib.colors import to_rgba
    pos, vec, c, cell = grid_field(Z, F, bins=bins, min_count=min_count, lo=lo, hi=hi)
    if not len(c):
        return ref
    norms = np.linalg.norm(vec, axis=1)
    ref = ref or max(np.percentile(norms, 90), 1e-12)
    scale = ref_cells * cell.min() / ref
    U, V = vec[:, 0] * scale, vec[:, 1] * scale
    kw = dict(angles="xy", scale_units="xy", scale=1, headwidth=3.2, headlength=3.6,
              headaxislength=3.2)
    if halo:
        ax.quiver(pos[:, 0], pos[:, 1], U, V, color=(1, 1, 1, 0.85), width=width * 1.9,
                  zorder=zorder, **kw)
    rgba = np.tile(np.array(to_rgba(color)), (len(c), 1))
    rgba[:, 3] = 0.55 + 0.45 * np.sqrt(c / c.max())
    q = ax.quiver(pos[:, 0], pos[:, 1], U, V, color=rgba, width=width, zorder=zorder + 0.1, **kw)
    if key:
        ax.quiverkey(q, 0.80, 0.04, ref * scale, f"{key_label} = {ref:.2f}", labelpos="W",
                     coordinates="axes", color=color, fontproperties={"size": 8})
    return ref


def direction_colors(F, vmax=None, min_saturation=0.15):
    """RGB per point: hue = direction of F, saturation = speed / vmax."""
    from matplotlib.colors import hsv_to_rgb
    ang = (np.arctan2(F[:, 1], F[:, 0]) + np.pi) / (2 * np.pi)
    sp = np.linalg.norm(F, axis=1)
    vmax = vmax or max(sp.max(), 1e-12)
    sat = min_saturation + (1 - min_saturation) * np.clip(sp / vmax, 0, 1)
    return hsv_to_rgb(np.stack([ang, sat, np.full_like(ang, 0.9)], axis=1))


def direction_wheel(ax_inset, n=120):
    """Draw the hue legend for ``direction_colors`` in a small inset axis."""
    from matplotlib.colors import hsv_to_rgb
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.hypot(x, y)
    ang = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    rgb = hsv_to_rgb(np.stack([ang, np.clip(r, 0, 1) * 0.85 + 0.15, np.full_like(r, 0.9)], -1))
    rgba = np.concatenate([rgb, (r <= 1)[..., None].astype(float)], -1)
    ax_inset.imshow(rgba, origin="lower", extent=(-1, 1, -1, 1))
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    for s in ax_inset.spines.values():
        s.set_visible(False)


def divergence_map(ax, field, Z, res=120, lo=None, hi=None, vmax=None, density_floor=0.08,
                   cmap="RdBu_r", zorder=0):
    """Background heatmap of the field's divergence (red = source, blue =
    sink), faded where there are no data: opacity is the square root of a
    smoothed point density relative to its 95th percentile, cut to zero where
    that relative density is below ``density_floor`` (where the divergence is
    the network extrapolating between clusters). Returns (image, vmax) so panels can share the colour
    scale."""
    from scipy.ndimage import gaussian_filter
    lo, hi = _bounds(Z, lo=lo, hi=hi)
    xs, ys = np.linspace(lo[0], hi[0], res), np.linspace(lo[1], hi[1], res)
    GX, GY = np.meshgrid(xs, ys)
    div = field.divergence(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(res, res)
    dens, _, _ = np.histogram2d(Z[:, 1], Z[:, 0], bins=res, range=[[lo[1], hi[1]], [lo[0], hi[0]]])
    dens = gaussian_filter(dens, sigma=res / 60)
    rel = np.clip(dens / max(np.percentile(dens[dens > 0], 95), 1e-12), 0, 1)
    alpha = np.sqrt(rel)                        # boost moderate densities ...
    alpha[rel < density_floor] = 0.0            # ... but nothing where data are absent
    if vmax is None:
        vmax = max(np.percentile(np.abs(div[alpha > 0.2]), 98) if (alpha > 0.2).any() else 1e-12,
                   1e-12)
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    rgba = colormaps[cmap](Normalize(-vmax, vmax, clip=True)(div))
    rgba[..., 3] = alpha * 0.85
    im = ax.imshow(rgba, origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]), aspect="auto",
                   zorder=zorder, interpolation="bilinear")
    return im, vmax
