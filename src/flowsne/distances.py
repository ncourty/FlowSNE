"""Closed-form Randers distance approximations.

A Randers metric F(x, v) = ||v|| + omega(x)^T v induces a distance
d_F(x -> y) defined as an infimum over paths, which has no closed form in
general when omega varies with position. This module implements the two
closed-form *approximations* to that true distance used throughout this
project, both exact when omega is spatially constant and increasingly
accurate as x and y get close:

- ``frozen``: freezes omega at the source point x.
      dhat_F(x->y) = ||y-x|| + omega(x)^T (y-x)
  First-order accurate: error is O(L * r^2), L a Lipschitz bound on omega,
  r = ||y-x||.

- ``averaged``: averages omega at both endpoints x and y.
      bdF(x->y) = ||y-x|| + ((omega(x)+omega(y))/2)^T (y-x)
  One order more accurate: error is O((L^2 + L2) * r^3), L2 a bound on
  omega's second derivative -- exact whenever omega is affine (not just
  constant), since the trapezoidal rule integrates linear functions
  exactly. Costs nothing extra: both endpoints' omega values are already
  available (given per-point on the input side, cheap to evaluate anywhere
  for a network on the output side).

No heat-kernel or graph-diffusion approximation of the Finsler distance is
implemented here -- this project intentionally keeps only these two
closed-form approximations.
"""

from __future__ import annotations

from typing import Literal

import torch

DriftMode = Literal["frozen", "averaged"]


def pairwise_delta(Z: torch.Tensor) -> torch.Tensor:
    """delta[i, j] = Z[j] - Z[i], shape (n, n, d). Materialises O(n^2 d)
    memory -- not used by ``randers_distance`` / ``drift_term``, which
    only ever form (n, n) matrices; kept as a reference for tests and
    small-scale inspection."""
    return Z.unsqueeze(0) - Z.unsqueeze(1)


def drift_term(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    mode: DriftMode = "averaged",
) -> torch.Tensor:
    """The omega^T (z_j - z_i) term of the Randers distance, computed
    from (n, n) matrix products only (never the (n, n, d) difference
    tensor). With ``M = omega_field @ Z.T`` and ``s_i = omega_i . z_i``:

    - frozen:   (z_j - z_i) . omega_i                 = M_ij - s_i
    - averaged: (z_j - z_i) . (omega_i + omega_j) / 2 = (M_ij - M_ji + s_j - s_i) / 2

    The averaged term is exactly antisymmetric (A_ij = -A_ji), so
    ``bdF`` is a symmetric distance plus an antisymmetric tilt.

    Parameters
    ----------
    Z : Tensor, shape (n, d)
    omega_field : Tensor, shape (n, d)
        ``omega_field[i]`` is the drift evaluated at ``z_i``.
    mode : {"frozen", "averaged"}
        Which closed-form approximation to use.

    Returns
    -------
    Tensor, shape (n, n)
    """
    M = omega_field @ Z.T
    s = (omega_field * Z).sum(-1)
    if mode == "frozen":
        return M - s.unsqueeze(1)
    if mode == "averaged":
        return 0.5 * (M - M.T + s.unsqueeze(0) - s.unsqueeze(1))
    raise ValueError(f"mode must be 'frozen' or 'averaged', got {mode!r}")


def randers_distance(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    mode: DriftMode = "averaged",
    eps: float = 1e-24,
) -> torch.Tensor:
    """Dense pairwise Randers distance matrix d_F(z_i -> z_j), shape (n, n).

    Not symmetric in general. O(n^2) memory: the squared Euclidean part is
    expanded as |z_i|^2 + |z_j|^2 - 2 z_i.z_j, clamped at 0 (the expansion
    can go slightly negative from cancellation), with its diagonal zeroed
    explicitly (cancellation leaves an O(machine-eps * |z|^2) residue
    there otherwise). ``eps`` inside the square root keeps the gradient
    finite at the (exactly zero) diagonal, where a plain ``.norm()`` has a
    well-defined forward value but a 0/0 gradient.
    """
    n = Z.shape[0]
    sq = (Z * Z).sum(-1)
    sqdist = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (Z @ Z.T)).clamp_min(0.0)
    sqdist = sqdist.masked_fill(torch.eye(n, dtype=torch.bool, device=Z.device), 0.0)
    norm = torch.sqrt(sqdist + eps)
    return (norm + drift_term(Z, omega_field, mode=mode)).clamp_min(0.0)


def bounded_direction(raw: torch.Tensor, max_norm: float, dim: int = -1) -> torch.Tensor:
    """Reparametrise an unconstrained tensor into the open ball of radius
    ``max_norm``: ``max_norm * r / (1 + ||r||)`` -- smooth and singularity
    free everywhere including ``r = 0``, unlike clamping a norm after the
    fact. Enforces the Randers admissibility bound ``||omega|| < 1``."""
    norm = torch.sqrt((raw ** 2).sum(dim=dim, keepdim=True) + 1e-24)
    return max_norm * raw / (1.0 + norm)


def rescale_admissible(omega: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
    """Globally rescale a GIVEN (not fitted) drift field so its largest
    norm equals ``max_norm``, preserving relative magnitude across points
    (unlike per-point unit-normalisation, which would discard it). Used
    for the input-side affinity, where omega is measured data (e.g. RNA
    velocity), not a learned/reparametrised network output."""
    scale = omega.norm(dim=-1).max().clamp_min(1e-12)
    return max_norm * omega / scale
