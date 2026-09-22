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
    """delta[i, j] = Z[j] - Z[i], shape (n, n, d)."""
    return Z.unsqueeze(0) - Z.unsqueeze(1)


def drift_term(
    delta: torch.Tensor,
    omega_field: torch.Tensor,
    mode: DriftMode = "averaged",
) -> torch.Tensor:
    """The omega^T (y - x) term of the Randers distance.

    Parameters
    ----------
    delta : Tensor, shape (n, n, d)
        ``delta[i, j] = z_j - z_i``.
    omega_field : Tensor, shape (n, d)
        ``omega_field[i]`` is the drift evaluated at ``z_i``.
    mode : {"frozen", "averaged"}
        Which closed-form approximation to use.

    Returns
    -------
    Tensor, shape (n, n)
    """
    if mode == "frozen":
        return torch.einsum("ijd,id->ij", delta, omega_field)
    if mode == "averaged":
        omega_avg = 0.5 * (omega_field.unsqueeze(1) + omega_field.unsqueeze(0))
        return torch.einsum("ijd,ijd->ij", delta, omega_avg)
    raise ValueError(f"mode must be 'frozen' or 'averaged', got {mode!r}")


def randers_distance(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    mode: DriftMode = "averaged",
    eps: float = 1e-24,
) -> torch.Tensor:
    """Dense pairwise Randers distance matrix d_F(z_i -> z_j), shape (n, n).

    Not symmetric in general. ``eps`` inside the square root keeps the
    gradient finite at the (exactly zero) diagonal, where a plain
    ``.norm()`` has a well-defined forward value but a 0/0 gradient.
    """
    delta = pairwise_delta(Z)
    norm = torch.sqrt((delta ** 2).sum(-1) + eps)
    return (norm + drift_term(delta, omega_field, mode=mode)).clamp_min(0.0)


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
