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

A third mode is EXACT rather than approximate, for a gradient drift
omega = grad(phi): the drift term integrates to phi(y) - phi(x) along any
path, so the infimum is the straight line and

- ``potential``: d_F(x->y) = ||y-x|| + phi(y) - phi(x)

In this mode the ``omega_field`` argument of every function below is the
potential itself, one value per point (shape (n,) or (n, 1)), not a vector
field. Admissibility (d_F >= 0) needs phi to be 1-Lipschitz; distances are
clamped at 0 as in the other modes.

No heat-kernel or graph-diffusion approximation of the Finsler distance is
implemented here -- this project intentionally keeps only these two
closed-form approximations.
"""

from __future__ import annotations

from typing import Literal

import torch

DriftMode = Literal["frozen", "averaged", "potential"]


def _phi(omega_field: torch.Tensor) -> torch.Tensor:
    """The potential, flattened to (n,) (``mode="potential"``)."""
    return omega_field.reshape(omega_field.shape[0])


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
    - potential: phi_j - phi_i, with ``omega_field`` the potential phi (n,)

    The averaged term is exactly antisymmetric (A_ij = -A_ji), so
    ``bdF`` is a symmetric distance plus an antisymmetric tilt.

    Parameters
    ----------
    Z : Tensor, shape (n, d)
    omega_field : Tensor, shape (n, d)
        ``omega_field[i]`` is the drift evaluated at ``z_i``.
    mode : {"frozen", "averaged", "potential"}
        Which closed-form approximation to use (``potential``: exact, and
        ``omega_field`` is the potential, shape (n,) or (n, 1)).

    Returns
    -------
    Tensor, shape (n, n)
    """
    if mode == "potential":
        phi = _phi(omega_field)
        return phi.unsqueeze(0) - phi.unsqueeze(1)
    M = omega_field @ Z.T
    s = (omega_field * Z).sum(-1)
    if mode == "frozen":
        return M - s.unsqueeze(1)
    if mode == "averaged":
        return 0.5 * (M - M.T + s.unsqueeze(0) - s.unsqueeze(1))
    raise ValueError(f"mode must be 'frozen', 'averaged' or 'potential', got {mode!r}")


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


def randers_distance_lazy(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    mode: DriftMode = "averaged",
    eps: float = 1e-24,
):
    """``randers_distance`` as a symbolic KeOps LazyTensor of shape (n, n):
    the same formula, evaluated on the fly inside reductions, O(n) memory.
    The diagonal is NOT excluded here (callers add their own mask)."""
    from pykeops.torch import LazyTensor

    z_i, z_j = LazyTensor(Z[:, None, :]), LazyTensor(Z[None, :, :])
    dz = z_j - z_i
    norm = (dz.sqnorm2() + eps).sqrt()
    if mode == "potential":
        phi = _phi(omega_field).contiguous()
        drift = LazyTensor(phi[None, :, None]) - LazyTensor(phi[:, None, None])
    elif mode == "frozen":
        drift = (LazyTensor(omega_field[:, None, :]) | dz)
    elif mode == "averaged":
        w_i, w_j = LazyTensor(omega_field[:, None, :]), LazyTensor(omega_field[None, :, :])
        drift = 0.5 * ((w_i + w_j) | dz)
    else:
        raise ValueError(f"mode must be 'frozen', 'averaged' or 'potential', got {mode!r}")
    return (norm + drift).relu()


def randers_distance_indexed(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    indices: torch.Tensor,
    mode: DriftMode = "averaged",
    eps: float = 1e-24,
    chunk_elems: int = 2 ** 25,
) -> torch.Tensor:
    """``d_F(z_i -> z_{indices[i, m]})`` for a (n, k) index array, shape (n, k).
    Differentiable, and processed in row chunks so the (rows, k, d) gather
    never exceeds ``chunk_elems`` entries."""
    n, k = indices.shape
    d = Z.shape[1]
    rows = max(1, chunk_elems // max(1, k * d))
    out = []
    for s in range(0, n, rows):
        idx = indices[s:s + rows]
        dz = Z[idx] - Z[s:s + rows, None, :]                       # (r, k, d)
        norm = torch.sqrt((dz ** 2).sum(-1) + eps)
        if mode == "potential":
            phi = _phi(omega_field)
            drift = phi[idx] - phi[s:s + rows, None]
        elif mode == "frozen":
            drift = (omega_field[s:s + rows, None, :] * dz).sum(-1)
        elif mode == "averaged":
            drift = (0.5 * (omega_field[s:s + rows, None, :] + omega_field[idx]) * dz).sum(-1)
        else:
            raise ValueError(f"mode must be 'frozen', 'averaged' or 'potential', got {mode!r}")
        out.append((norm + drift).clamp_min(0.0))
    return torch.cat(out, dim=0)


def randers_distance_pairs(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    mode: DriftMode = "averaged",
    eps: float = 1e-24,
    chunk_elems: int = 2 ** 25,
) -> torch.Tensor:
    """``d_F(z_rows[e] -> z_cols[e])`` for an edge list, shape (E,).
    Differentiable, processed in chunks of at most ``chunk_elems`` gathered
    entries."""
    d = Z.shape[1]
    step = max(1, chunk_elems // max(1, d))
    out = []
    for s in range(0, rows.shape[0], step):
        r, c = rows[s:s + step], cols[s:s + step]
        dz = Z[c] - Z[r]
        norm = torch.sqrt((dz ** 2).sum(-1) + eps)
        if mode == "potential":
            phi = _phi(omega_field)
            drift = phi[c] - phi[r]
        elif mode == "frozen":
            drift = (omega_field[r] * dz).sum(-1)
        elif mode == "averaged":
            drift = (0.5 * (omega_field[r] + omega_field[c]) * dz).sum(-1)
        else:
            raise ValueError(f"mode must be 'frozen', 'averaged' or 'potential', got {mode!r}")
        out.append((norm + drift).clamp_min(0.0))
    return torch.cat(out) if out else Z.new_zeros(0)


def randers_knn(
    Z: torch.Tensor,
    omega_field: torch.Tensor,
    k: int,
    mode: DriftMode = "averaged",
    backend: str | None = None,
):
    """Each row's ``k`` nearest points under the DIRECTED cost ``d_F(z_i -> .)``,
    self excluded. Returns ``(d (n, k), indices (n, k))``, ``d`` increasing
    along each row.

    ``backend=None``: dense (n, n) cost then ``topk`` (O(n^2) memory).
    ``backend="keops"``: lazy cost and ``argKmin`` (O(n) memory). Either way
    the returned distances are recomputed from the indices with
    ``randers_distance_indexed``, so they are exact, differentiable and
    identical across backends."""
    n = Z.shape[0]
    if not 0 < k < n:
        raise ValueError(f"k must be in [1, n-1] = [1, {n - 1}], got {k}.")
    if backend is None:
        with torch.no_grad():
            D = randers_distance(Z.detach(), omega_field.detach(), mode=mode)
            D = D.masked_fill(torch.eye(n, dtype=torch.bool, device=Z.device), float("inf"))
            indices = D.topk(k, dim=1, largest=False).indices
    elif backend == "keops":
        from torchdr.utils import identity_matrix

        with torch.no_grad():
            D = randers_distance_lazy(Z.detach(), omega_field.detach(), mode=mode)
            D = D + identity_matrix(n, keops=True, device=Z.device, dtype=Z.dtype) * 1e12
            indices = D.argKmin(k, dim=1).long()
    else:
        raise ValueError(f"backend must be None or 'keops', got {backend!r}")
    d = randers_distance_indexed(Z, omega_field, indices, mode=mode)
    order = d.detach().argsort(dim=1)                 # argKmin order is not guaranteed
    return d.gather(1, order), indices.gather(1, order)


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
