"""TorchDR-compatible affinities built on the closed-form Randers distance
approximations in ``distances.py``.

Two classes, matching a self-referential-entropic-OT dimensionality
reduction pipeline's two halves:

- ``EntropicRandersAffinity`` (input side): a self-referential entropic
  affinity over the DATA space, with an asymmetric Randers ground cost
  built from a given per-point drift (e.g. RNA velocity) -- the direct
  Finsler generalisation of TorchDR's own entropic-affinity /
  ``SinkhornAffinity`` constructions, following ``torchdr.affinity.base
  .LogAffinity``'s calling convention exactly.
- ``RandersAffinity`` (output/embedding side): an asymmetric Student-t-like
  kernel over the EMBEDDING space, with the same Randers ground cost, now
  driven by a pluggable ``flow`` network (``flowsne.flows``) instead of a
  fixed drift -- fit jointly with the embedding coordinates via
  ``flowsne.embedding.JointAffinityMatcher``.

Both support ``drift_mode="frozen"`` or ``"averaged"`` (``distances.py``),
the only two Randers-distance approximations this project uses -- no
graph/heat-kernel diffusion proxy.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torchdr.affinity.base import LogAffinity
from torchdr.affinity.entropic import NormalizedStudentAffinity
from torchdr.utils import to_torch

from .distances import (
    DriftMode, bounded_direction, randers_distance, randers_distance_indexed,
    randers_distance_lazy, randers_distance_pairs, randers_knn, rescale_admissible,
)
from .flows import mlp_flow, potential_flow


def _segment_logsumexp(values: torch.Tensor, segments: torch.Tensor, n: int) -> torch.Tensor:
    """``out[s] = logsumexp(values[segments == s])`` for s in range(n)."""
    m = values.new_full((n,), -math.inf).scatter_reduce(
        0, segments, values, reduce="amax", include_self=True)
    s = values.new_zeros(n).scatter_add(0, segments, torch.exp(values - m[segments]))
    return m + s.log()
from .flows import mlp_flow


# ===========================================================================
# Input-side: self-referential entropic affinity with a Randers ground cost
# ===========================================================================

class EntropicRandersAffinity(LogAffinity):
    r"""Self-referential entropic affinity over the data space, with an
    asymmetric Randers ground cost tilted by a given per-point drift
    ``omega`` (e.g. RNA velocity): the Finsler generalisation of a plain
    entropic/SNE affinity.

    Construction, per row ``i``: a per-point bandwidth ``tau_i`` is
    calibrated (by bisection) so the row distribution
    ``p_{j|i} = softmax_j(-d_F(x_i->x_j)^2 / tau_i)`` has the target
    perplexity (the standard t-SNE-style calibration, generalised to an
    asymmetric cost) -- self-transport (``j=i``) is always excluded. If
    ``doubly_stochastic=True`` (default), the resulting kernel is then
    Sinkhorn-balanced to uniform marginals :math:`a=b=\tfrac1n\mathbf1`,
    matching SNEkhorn's own construction (this is what fixes the
    heterogeneous-density / hub problem a plain row-normalised affinity
    does not address); ``doubly_stochastic=False`` skips that step (a
    cheaper, un-regularised alternative -- useful as an ablation to isolate
    how much of any downstream effect comes from the asymmetric ground cost
    alone versus from marginal control, not a recommended default).

    Both modes return an affinity with the SAME overall scale convention
    (entries summing to 1 over the whole matrix, uniform ~1/n row/column
    sums), so either is a drop-in ``affinity_in`` for an embedding fit.

    Parameters
    ----------
    perplexity : float or "auto", optional
        Target perplexity for the per-row bandwidth calibration. "auto"
        (default) uses ``sqrt(n)`` -- a scale-adaptive default that avoids
        the failure mode of a FIXED target (e.g. a fixed neighbour count)
        becoming an ever-shrinking fraction of the dataset as ``n`` grows.
    drift_mode : {"frozen", "averaged", "potential"}, optional
        Which closed-form Randers distance approximation to use (see
        ``distances.py``). Default "averaged" (one order more accurate,
        same cost). "potential": the drift is a gradient, ``omega`` is then
        a FORWARD potential psi (shape (n,), higher = later), and the cost is
        the exact ``||x_j - x_i|| - (psi_j - psi_i)`` -- see ``omega_scale``.
    cost_power : float, optional
        The ground cost is ``d_F ** cost_power``. Default 2 (SNE's squared
        distance). With 1 and ``drift_mode="potential"``, the row-stochastic
        affinity is EXACTLY the symmetric one with each row's targets
        reweighted by ``exp(psi_j / tau_i)`` (the ``-psi_i`` term cancels in
        the row normalisation, per-row bandwidths included); with 2 it does
        not factorise.
    max_omega_norm : float, optional
        The given drift is globally rescaled (preserving relative
        magnitude across points) so its largest norm equals this value,
        enforcing the Randers admissibility bound ``||omega|| < 1``.
        Default 0.95.
    omega_scale : float, optional
        Use ``omega_scale * omega`` as is instead of rescaling by this
        call's own maximum -- for a sequence of affinities that must share
        one scale (e.g. snapshots over time, where the drift's magnitude
        changes and that change should stay visible). Every scaled norm
        must then be below ``max_omega_norm`` (checked). Default None.
        With ``drift_mode="potential"`` there is no automatic rescaling (a
        potential's admissibility, 1-Lipschitz, is a property of pairs, not
        of points): ``omega_scale * psi`` is used as given (default scale 1),
        and making it 1-Lipschitz over the pairs that matter is the caller's
        job; inadmissible pairs get distance 0 (clamped).
    doubly_stochastic : bool, optional
        Sinkhorn-balance to uniform marginals (default True) or leave as a
        plain row-stochastic softmax (False). See above.
    sinkhorn_iters, sinkhorn_tol : optional
        Sinkhorn iteration budget / early-stop tolerance on the log-domain
        potentials (only used if ``doubly_stochastic=True``). Default 5000:
        a tight perplexity bandwidth makes the kernel badly conditioned, and
        200 iterations left the marginals of every single-cell dataset here
        0.6-3.6% off. ``marginal_error_`` holds the achieved max relative
        marginal error, and a warning is raised above 1e-4.
    perplexity_iters : int, optional
        Bisection steps for the per-row bandwidth search. Default 50
        (more than enough for float64 precision).
    omega : Tensor of shape (n_samples, n_features), optional
        Default drift, used when ``omega`` isn't passed explicitly to
        ``__call__`` -- lets this affinity be used as a plain
        ``Affinity(X)`` callable (e.g. as ``affinity_in`` of a TorchDR
        ``AffinityMatcher``, which always calls ``affinity_in(X)`` with no
        extra kwargs).
    sparsity : bool, optional
        If True, each row keeps only its ``n_neighbors`` nearest points
        under the directed cost ``d_F(x_i -> .)``; the bandwidth is
        calibrated on those entries (same bisection as the dense path) and
        the call returns ``(values (n, k), indices (n, k))``, TorchDR's
        sparse-affinity format. Default False (dense (n, n) output, as
        before). **Recommended with ``doubly_stochastic=False``**, where it
        is exact up to the truncation. With ``doubly_stochastic=True`` the
        Sinkhorn balancing runs on the kNN support PLUS its reverse edges (a
        point may be nobody's nearest neighbour, and a column with no
        entries cannot receive its 1/n of mass); rows then have different
        lengths and are padded to the longest one with ~zero entries
        pointing at the row itself. This is a LOCAL variant of balancing,
        not an approximation of the dense one: the dense balanced plan can
        send a row's mass entirely outside its neighbourhood (once its
        neighbours' columns are full), which no local support represents --
        on full pancreas the two differ by a median row TV of 8% at 5x
        perplexity, and on a handful of rows completely, whatever k.
    n_neighbors : int, optional
        Neighbours kept per row when ``sparsity=True``. Default
        ``min(n - 1, ceil(5 * perplexity))``: t-SNE's usual 3x loses up to a
        quarter of a row's mass on heterogeneous data (full pancreas, 30-D
        PCA; max row gap 23% at 3x vs 12% at 5x, mean 2.4% vs 0.5%).
    backend : {None, "keops"}, optional
        How the sparse path finds the neighbours: a dense (n, n) cost
        (None, default) or a KeOps lazy cost with O(n) memory ("keops").
        The returned affinity is identical either way. The dense path
        (``sparsity=False``) only supports ``backend=None`` for now.
    """

    def __init__(
        self,
        perplexity: Union[float, str] = "auto",
        drift_mode: DriftMode = "averaged",
        max_omega_norm: float = 0.95,
        doubly_stochastic: bool = True,
        cost_power: float = 2.0,
        sinkhorn_iters: int = 5000,
        sinkhorn_tol: float = 1e-9,
        perplexity_iters: int = 50,
        omega: Optional[torch.Tensor] = None,
        sparsity: bool = False,
        n_neighbors: Optional[int] = None,
        omega_scale: Optional[float] = None,
        metric: str = "sqeuclidean",
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        if backend not in (None, "keops"):
            raise ValueError(
                f"[EntropicRandersAffinity] backend must be None or 'keops', got {backend!r}."
            )
        if backend == "keops" and not sparsity:
            raise NotImplementedError(
                "[EntropicRandersAffinity] backend='keops' currently requires "
                "sparsity=True (a lazy dense affinity is not implemented yet)."
            )
        if metric != "sqeuclidean":
            raise NotImplementedError(
                "[EntropicRandersAffinity] only metric='sqeuclidean' is "
                "supported (the Randers cost is built directly on Euclidean "
                "coordinates)."
            )
        super().__init__(
            metric=metric, zero_diag=False, device=device, backend=backend,
            verbose=verbose, compile=compile, _pre_processed=_pre_processed,
        )
        if not 0.0 < max_omega_norm < 1.0:
            raise ValueError(
                "[EntropicRandersAffinity] max_omega_norm must be in (0, 1) "
                f"(the Randers admissibility bound), got {max_omega_norm}."
            )
        self.perplexity = perplexity
        self.drift_mode = drift_mode
        self.max_omega_norm = max_omega_norm
        self.doubly_stochastic = doubly_stochastic
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_tol = sinkhorn_tol
        self.perplexity_iters = perplexity_iters
        self._omega_default = omega
        self.sparsity = sparsity
        self.n_neighbors = n_neighbors
        self.omega_scale = omega_scale
        self.cost_power = cost_power

    def _calibrate_bandwidth(self, C: torch.Tensor, target_perplexity: float) -> torch.Tensor:
        """Per-row bisection for tau such that softmax(-C/tau) has the
        target perplexity, vectorised across all rows at once. ``C`` must
        already have +inf on the diagonal (self excluded)."""
        n = C.shape[0]
        c_min = C.min(dim=1, keepdim=True).values
        c_shifted = C - c_min  # numerical stability only; doesn't change softmax
        lo = torch.full((n, 1), 1e-6, dtype=C.dtype, device=C.device)
        hi = torch.full((n, 1), 1e6, dtype=C.dtype, device=C.device)
        for _ in range(self.perplexity_iters):
            tau = 0.5 * (lo + hi)
            log_p = torch.log_softmax(-c_shifted / tau, dim=1)
            p = log_p.exp()
            H = -torch.where(p > 0, p * log_p, 0.0).sum(dim=1, keepdim=True)
            too_low = H.exp() < target_perplexity
            lo = torch.where(too_low, tau, lo)
            hi = torch.where(too_low, hi, tau)
        return (0.5 * (lo + hi)).squeeze(1)

    def _check_marginals(self, row_sums: torch.Tensor, col_sums: torch.Tensor, n: int):
        err = max((row_sums * n - 1).abs().max().item(), (col_sums * n - 1).abs().max().item())
        self.marginal_error_ = err
        if not err <= 1e-4:
            warnings.warn(
                f"[EntropicRandersAffinity] Sinkhorn did not reach the "
                f"marginals (max relative error {err:.2e} after "
                f"{self.sinkhorn_iters} iterations); increase sinkhorn_iters"
                f"{' or n_neighbors' if self.sparsity else ''}.", stacklevel=4)

    def _sinkhorn_log(self, log_K: torch.Tensor) -> torch.Tensor:
        """Dense log-domain Sinkhorn to uniform marginals 1/n (``log_K`` has
        -inf where the kernel is zero, e.g. the diagonal). Returns log P."""
        n = log_K.shape[0]
        log_a = -math.log(n)
        f = log_K.new_zeros(n)
        g = log_K.new_zeros(n)
        for _ in range(self.sinkhorn_iters):
            f_new = log_a - torch.logsumexp(log_K + g[None, :], dim=1)
            g = log_a - torch.logsumexp(log_K + f_new[:, None], dim=0)
            done = (f_new - f).abs().max() < self.sinkhorn_tol
            f = f_new
            if done:
                break
        log_P = log_K + f[:, None] + g[None, :]
        P = log_P.exp()
        self._check_marginals(P.sum(1), P.sum(0), n)
        return log_P

    def __call__(self, X, log: bool = False, return_indices: Optional[bool] = None,
                 **kwargs):
        """Dense mode (default): the (n, n) affinity, as before (or
        ``(affinity, None)`` if ``return_indices=True``). Sparse mode:
        ``(values (n, k), indices (n, k))`` (just ``values`` if
        ``return_indices=False``)."""
        if not self.sparsity:
            out = super().__call__(X, log=log, **kwargs)
            return (out, None) if return_indices else out
        if not self._pre_processed:
            X = to_torch(X)
        log_P, indices = self._compute_sparse_log_affinity(X, **kwargs)
        values = log_P if log else log_P.exp()
        return values if return_indices is False else (values, indices)

    def _prepare(self, X: torch.Tensor, omega: Optional[torch.Tensor]):
        """Move X to the compute device/dtype and return it with the
        admissible, sign-flipped drift used in the cost."""
        device = self._get_compute_device(X)
        dtype = self._get_dtype(X)
        X = X.to(device=device, dtype=dtype)
        if omega is None:
            omega = self._omega_default
        if self.drift_mode == "potential":
            n = X.shape[0]
            psi = (X.new_zeros(n) if omega is None else
                   torch.as_tensor(omega, dtype=dtype, device=device).reshape(-1))
            if psi.shape[0] != n:
                raise ValueError(
                    f"[EntropicRandersAffinity] a potential needs one value per point "
                    f"({n}), got {psi.shape[0]}."
                )
            # same sign convention as below: the cost uses -psi, so moving up
            # the forward potential is cheaper
            scale = 1.0 if self.omega_scale is None else self.omega_scale
            return X, -scale * psi
        if omega is None:
            omega = torch.zeros_like(X)
        else:
            omega = torch.as_tensor(omega, dtype=dtype, device=device)
            if omega.shape != X.shape:
                raise ValueError(
                    f"[EntropicRandersAffinity] omega has shape "
                    f"{tuple(omega.shape)}, expected {tuple(X.shape)}."
                )
        # Sign convention: moving WITH the given drift should be CHEAPER
        # (favoured), but d_F(x->y) = ||y-x|| + omega(x)^T(y-x) makes moving
        # ALIGNED with omega more expensive, not less -- so the admissible
        # field actually used in the cost is -omega, not omega.
        if self.omega_scale is None:
            return X, rescale_admissible(-omega, self.max_omega_norm)
        scaled = -self.omega_scale * omega
        top = scaled.norm(dim=-1).max().item() if scaled.numel() else 0.0
        if top > self.max_omega_norm:
            raise ValueError(
                f"[EntropicRandersAffinity] omega_scale * ||omega|| reaches {top:.3f} > "
                f"max_omega_norm={self.max_omega_norm} (Randers admissibility)."
            )
        return X, scaled

    def _target_perplexity(self, n: int) -> float:
        return math.sqrt(n) if self.perplexity == "auto" else float(self.perplexity)

    def _compute_sparse_log_affinity(self, X: torch.Tensor,
                                     omega: Optional[torch.Tensor] = None):
        """Row-stochastic affinity restricted to each row's k nearest points
        under d_F(x_i -> .); returns (log P (n, k), indices (n, k)), with
        the same overall scale as the dense path (entries sum to 1)."""
        n = self._get_n_samples(X)
        X, omega_scaled = self._prepare(X, omega)
        target_perp = self._target_perplexity(n)
        k = self.n_neighbors or min(n - 1, math.ceil(5 * target_perp))
        if not target_perp < k:
            raise ValueError(
                f"[EntropicRandersAffinity] perplexity {target_perp:.1f} must be "
                f"smaller than n_neighbors={k}."
            )
        d, indices = randers_knn(X, omega_scaled, k, mode=self.drift_mode,
                                 backend=self.backend)
        C = d ** self.cost_power
        tau = self._calibrate_bandwidth(C, target_perp)
        c_min = C.min(dim=1).values
        if not self.doubly_stochastic:
            log_K = -(C - c_min[:, None]) / tau.unsqueeze(1)
            log_P = log_K - torch.logsumexp(log_K, dim=1, keepdim=True) - math.log(n)
            return log_P.clamp_min(math.log(1e-300)), indices

        # Balanced: kNN support + reverse edges (symmetric support, every
        # point has >= k entries in its row and in its column).
        rows_k = torch.arange(n, device=X.device).repeat_interleave(k)
        cols_k = indices.reshape(-1)
        keys = torch.unique(torch.cat([rows_k * n + cols_k, cols_k * n + rows_k]))  # row-major
        rows, cols = keys // n, keys % n
        C_e = randers_distance_pairs(X, omega_scaled, rows, cols,
                                     mode=self.drift_mode) ** self.cost_power
        log_k = -(C_e - c_min[rows]) / tau[rows]
        log_p = self._sparse_sinkhorn(rows, cols, log_k, n)

        counts = torch.bincount(rows, minlength=n)
        width = int(counts.max())
        pos = torch.arange(rows.shape[0], device=X.device) - (counts.cumsum(0) - counts)[rows]
        log_P = log_p.new_full((n, width), math.log(1e-300))
        log_P[rows, pos] = log_p.clamp_min(math.log(1e-300))
        idx = torch.arange(n, device=X.device)[:, None].repeat(1, width)
        idx[rows, pos] = cols
        return log_P, idx

    def _sparse_sinkhorn(self, rows: torch.Tensor, cols: torch.Tensor, log_k: torch.Tensor,
                         n: int) -> torch.Tensor:
        """Log-domain Sinkhorn on an edge list, to uniform marginals 1/n.
        Returns log P on the edges; warns if the marginals are not met."""
        log_a = -math.log(n)
        f = log_k.new_zeros(n)
        g = log_k.new_zeros(n)
        for _ in range(self.sinkhorn_iters):
            f_new = log_a - _segment_logsumexp(log_k + g[cols], rows, n)
            g = log_a - _segment_logsumexp(log_k + f_new[rows], cols, n)
            done = (f_new - f).abs().max() < self.sinkhorn_tol
            f = f_new
            if done:
                break
        log_p = log_k + f[rows] + g[cols]
        p = log_p.exp()
        self._check_marginals(p.new_zeros(n).scatter_add(0, rows, p),
                              p.new_zeros(n).scatter_add(0, cols, p), n)
        return log_p

    def _compute_log_affinity(self, X: torch.Tensor, omega: Optional[torch.Tensor] = None):
        n = self._get_n_samples(X)
        X, omega_scaled = self._prepare(X, omega)
        device = X.device

        C = randers_distance(X, omega_scaled, mode=self.drift_mode) ** self.cost_power
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        C = C.masked_fill(diag_mask, float("inf"))

        tau = self._calibrate_bandwidth(C, self._target_perplexity(n))
        c_min = C.min(dim=1, keepdim=True).values
        log_K = (-(C - c_min) / tau.unsqueeze(1)).masked_fill(diag_mask, -math.inf)

        if self.doubly_stochastic:
            log_P = self._sinkhorn_log(log_K)
        else:
            log_P = log_K - torch.logsumexp(log_K, dim=1, keepdim=True) - math.log(n)

        return log_P.clamp_min(math.log(1e-300))


# ===========================================================================
# Output side: embedding-space Randers kernel with a pluggable flow network
# ===========================================================================

class RandersAffinity(NormalizedStudentAffinity):
    r"""Asymmetric Student-t-like output kernel for an embedding, built
    from the closed-form Randers distance with drift supplied by a
    pluggable ``flow`` network (``flowsne.flows``):

    .. math::
        d_F(z_i \to z_j) = \lVert z_j - z_i \rVert
        + \omega_\theta(z_i, z_j)^\top (z_j - z_i)

    where the drift term uses either endpoint's ``flow(z)`` alone
    (``drift_mode="frozen"``) or their average (``drift_mode="averaged"``,
    default -- see ``distances.py``). A plain symmetric output kernel is
    provably blind to any asymmetry in the input affinity (a cross-entropy
    loss against it only ever depends on the symmetric part
    :math:`P+P^\top`); this is the fix.

    ``flow`` can be anything: ``flowsne.flows.ConstantFlow`` recovers the
    "canonical Randers" global-drift construction of Finsler MDS (Dagès et
    al., CVPR 2025); ``mlp_flow`` / ``input_std_norm_flow`` / ``rff_flow``
    give a position-varying drift of increasing architectural
    sophistication. Any custom ``nn.Module`` mapping
    ``(n, n_components) -> (n, n_components)`` works -- ``RandersAffinity``
    applies the admissibility reparametrisation itself
    (``distances.bounded_direction``), so the network's own output is raw
    and unconstrained.

    ``drift_mode="potential"``: the drift is a gradient, omega = grad(h), and
    the distance is the exact ``||z_j - z_i|| + h(z_j) - h(z_i)``. ``flow``
    then maps ``(n, n_components) -> (n, 1)`` and must be 1-Lipschitz
    (``flowsne.flows.potential_flow`` is, by construction); the potential is
    ``h = max_omega_norm * flow(z)`` (``potential_at``), so it is
    ``max_omega_norm``-Lipschitz everywhere, not only at the points.
    ``omega_at`` returns grad(h) (autograd), with the same sign convention:
    the favoured direction is ``-omega_at(Z)``, i.e. down h; ``-h`` is the
    forward landscape.

    Fit ``(Z, flow.parameters())`` jointly via
    ``flowsne.embedding.JointAffinityMatcher`` -- an ordinary, single-loop
    optimisation (``flow(Z)`` is just an ordinary differentiable function
    composed into the forward pass, no inner loop or implicit-function
    backward needed).

    Parameters
    ----------
    n_components : int
        Embedding dimension -- must match the embedding method's own
        ``n_components``, and the ``flow`` network's input/output size.
    flow : nn.Module, optional
        The drift network. Default ``mlp_flow(n_components)`` (a small
        zero-initialised Tanh MLP), or ``potential_flow(n_components)`` with
        ``drift_mode="potential"``.
    drift_mode : {"frozen", "averaged", "potential"}, optional
        Default "averaged".
    max_omega_norm : float, optional
        Admissibility bound :math:`\rho` on :math:`\lVert\omega\rVert`.
        Default 0.95.
    backend : {None, "keops"}, optional
        Dense (n, n) cost (None, default) or a KeOps LazyTensor with O(n)
        memory ("keops"); ``self(Z)`` then returns a LazyTensor, and
        ``cross_entropy`` computes its normaliser lazily. Same values and
        gradients either way.
    degrees_of_freedom, zero_diag, normalization_dim, device, verbose,
    compile, _pre_processed : see ``torchdr.NormalizedStudentAffinity``.
    """

    n_components: int

    def __init__(
        self,
        n_components: int,
        flow: Optional[nn.Module] = None,
        drift_mode: DriftMode = "averaged",
        max_omega_norm: float = 0.95,
        degrees_of_freedom: float = 1.0,
        zero_diag: bool = True,
        normalization_dim: Union[int, Tuple[int, ...]] = (0, 1),
        device: str = "auto",
        backend: Optional[str] = None,
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        if backend not in (None, "keops"):
            raise ValueError(
                f"[RandersAffinity] backend must be None or 'keops', got {backend!r}."
            )
        super().__init__(
            degrees_of_freedom=degrees_of_freedom, metric="sqeuclidean",
            zero_diag=zero_diag, device=device, backend=backend, verbose=verbose,
            normalization_dim=normalization_dim, compile=compile,
            _pre_processed=_pre_processed,
        )
        if not 0.0 < max_omega_norm < 1.0:
            raise ValueError(
                "[RandersAffinity] max_omega_norm must be in (0, 1) (the "
                f"Randers admissibility bound), got {max_omega_norm}."
            )
        self.n_components = n_components
        self.drift_mode = drift_mode
        self.max_omega_norm = max_omega_norm
        if flow is None:
            flow = potential_flow(n_components) if drift_mode == "potential" else mlp_flow(n_components)
        self.flow = flow

    def potential_at(self, Z: torch.Tensor) -> torch.Tensor:
        """The fitted potential h at every row of ``Z``, shape (n,)
        (``drift_mode="potential"`` only)."""
        if self.drift_mode != "potential":
            raise ValueError("[RandersAffinity] potential_at needs drift_mode='potential'.")
        return self.max_omega_norm * self.flow(Z).reshape(Z.shape[0])

    def _field(self, Z: torch.Tensor) -> torch.Tensor:
        """What the distance functions take: the potential (n,) or the
        admissible drift (n, d)."""
        field = self.potential_at(Z) if self.drift_mode == "potential" else self.omega_at(Z)
        return field.to(dtype=Z.dtype, device=Z.device)

    def omega_at(self, Z: torch.Tensor) -> torch.Tensor:
        """The fitted drift field at every row of ``Z`` -- for inspecting
        or plotting (e.g. as a quiver plot over the embedding). NOTE the
        sign convention: under d_F(z_i->z_j) = ||dz|| + omega(z_i)^T dz, a
        correctly-fit omega points AWAY from the direction the affinity
        favours (moving with omega makes the distance larger, not
        smaller) -- plot ``-omega_at(Z)`` for the "flow direction", not the
        raw value. With ``drift_mode="potential"``: grad(h) by autograd
        (detached, also under ``torch.no_grad``)."""
        if self.drift_mode == "potential":
            with torch.enable_grad():
                Zg = Z.detach().requires_grad_(True)
                return torch.autograd.grad(self.potential_at(Zg).sum(), Zg)[0]
        raw = self.flow(Z)
        return bounded_direction(raw, self.max_omega_norm, dim=-1)

    def _distance_matrix(self, Z: torch.Tensor, k=None, return_indices: bool = False):
        if k is not None or return_indices:
            raise NotImplementedError(
                f"[{type(self).__name__}] sparsity/backend kNN is not "
                "supported; this affinity is dense-only."
            )
        n, m = Z.shape
        if m != self.n_components:
            raise ValueError(
                f"[{type(self).__name__}] embedding has {m} dimensions, "
                f"expected n_components={self.n_components}."
            )
        return self._cost(Z, self._field(Z))

    def _cost(self, Z: torch.Tensor, omega_field: torch.Tensor):
        """Squared Randers cost, dense or lazy, diagonal excluded with a
        large finite sentinel if ``zero_diag``."""
        n = Z.shape[0]
        if self.backend == "keops":
            from torchdr.utils import identity_matrix

            C_F = randers_distance_lazy(Z, omega_field, mode=self.drift_mode) ** 2
            if self.zero_diag:
                C_F = C_F + identity_matrix(n, keops=True, device=Z.device, dtype=Z.dtype) * 1e12
            return C_F
        d_F = randers_distance(Z, omega_field, mode=self.drift_mode)
        C_F = d_F ** 2

        if self.zero_diag:
            # A large *finite* sentinel, not literal inf: matches TorchDR's
            # own `pairwise_distances(..., exclude_diag=True)` convention --
            # `0 * (-inf)` is `nan` in a cross-entropy loss, `0 * (a large
            # finite number)` is `0`.
            diag_mask = torch.eye(n, dtype=torch.bool, device=Z.device)
            C_F = C_F.masked_fill(diag_mask, 1e12)

        return C_F

    def _log_kernel(self, C):
        """Unnormalised Student log-kernel, as in NormalizedStudentAffinity."""
        nu = self.degrees_of_freedom
        return -0.5 * (nu + 1) * (C / nu + 1).log()

    def cross_entropy(self, Z: torch.Tensor, P: torch.Tensor,
                      indices: Optional[torch.Tensor] = None,
                      exaggeration: float = 1.0) -> torch.Tensor:
        r"""Cross-entropy :math:`-\sum_{ij} P_{ij} \log Q_{ij}` against this
        kernel, with ``Q`` normalised over all pairs.

        Dense ``P`` (n, n), ``indices=None``: the plain formula. Sparse ``P``
        (n, k) with ``indices`` (n, k) (``EntropicRandersAffinity(sparsity=
        True)`` output): attraction on the kept entries plus
        :math:`(\sum P)\,\log Z`, the normaliser over all pairs computed
        densely or lazily (``backend="keops"``) -- never an (n, n) tensor in
        the KeOps case. Exactly equal to the dense loss on the densified P.
        ``exaggeration`` multiplies the attraction term only (early
        exaggeration); 1 gives the cross-entropy itself.
        """
        if self.normalization_dim not in ((0, 1), (1, 0)):
            raise NotImplementedError(
                "[RandersAffinity] cross_entropy assumes normalization_dim=(0, 1)."
            )
        omega = self._field(Z)
        log_kernel = self._log_kernel(self._cost(Z, omega))
        if self.backend == "keops":
            log_Z = torch.logsumexp(log_kernel.logsumexp(dim=1).reshape(-1), dim=0)
        else:
            log_Z = torch.logsumexp(log_kernel.reshape(-1), dim=0)
        if indices is None:
            if self.backend == "keops":
                raise NotImplementedError(
                    "[RandersAffinity] a dense P with backend='keops' would need "
                    "the (n, n) kernel; pass a sparse P with its indices."
                )
            return -exaggeration * (P * log_kernel).sum() + P.sum() * log_Z
        d = randers_distance_indexed(Z, omega, indices, mode=self.drift_mode)
        return -exaggeration * (P * self._log_kernel(d ** 2)).sum() + P.sum() * log_Z
