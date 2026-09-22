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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torchdr.affinity.base import LogAffinity
from torchdr.affinity.entropic import NormalizedStudentAffinity

from .distances import DriftMode, bounded_direction, randers_distance, rescale_admissible
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
    drift_mode : {"frozen", "averaged"}, optional
        Which closed-form Randers distance approximation to use (see
        ``distances.py``). Default "averaged" (one order more accurate,
        same cost).
    max_omega_norm : float, optional
        The given drift is globally rescaled (preserving relative
        magnitude across points) so its largest norm equals this value,
        enforcing the Randers admissibility bound ``||omega|| < 1``.
        Default 0.95.
    doubly_stochastic : bool, optional
        Sinkhorn-balance to uniform marginals (default True) or leave as a
        plain row-stochastic softmax (False). See above.
    sinkhorn_iters, sinkhorn_tol : optional
        Sinkhorn iteration budget / early-stop tolerance (only used if
        ``doubly_stochastic=True``).
    perplexity_iters : int, optional
        Bisection steps for the per-row bandwidth search. Default 50
        (more than enough for float64 precision).
    omega : Tensor of shape (n_samples, n_features), optional
        Default drift, used when ``omega`` isn't passed explicitly to
        ``__call__`` -- lets this affinity be used as a plain
        ``Affinity(X)`` callable (e.g. as ``affinity_in`` of a TorchDR
        ``AffinityMatcher``, which always calls ``affinity_in(X)`` with no
        extra kwargs).
    """

    def __init__(
        self,
        perplexity: Union[float, str] = "auto",
        drift_mode: DriftMode = "averaged",
        max_omega_norm: float = 0.95,
        doubly_stochastic: bool = True,
        sinkhorn_iters: int = 200,
        sinkhorn_tol: float = 1e-9,
        perplexity_iters: int = 50,
        omega: Optional[torch.Tensor] = None,
        metric: str = "sqeuclidean",
        device: str = "auto",
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        if metric != "sqeuclidean":
            raise NotImplementedError(
                "[EntropicRandersAffinity] only metric='sqeuclidean' is "
                "supported (the Randers cost is built directly on Euclidean "
                "coordinates)."
            )
        super().__init__(
            metric=metric, zero_diag=False, device=device, backend=None,
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
            p = torch.softmax(-c_shifted / tau, dim=1)
            H = -(p * p.clamp_min(1e-300).log()).sum(dim=1, keepdim=True)
            too_low = H.exp() < target_perplexity
            lo = torch.where(too_low, tau, lo)
            hi = torch.where(too_low, hi, tau)
        return (0.5 * (lo + hi)).squeeze(1)

    def _sinkhorn(self, K: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        a = torch.full((n,), 1.0 / n, dtype=K.dtype, device=K.device)
        u = torch.ones(n, dtype=K.dtype, device=K.device)
        v = torch.ones(n, dtype=K.dtype, device=K.device)
        for _ in range(self.sinkhorn_iters):
            u_new = a / (K @ v).clamp_min(1e-300)
            v_new = a / (K.T @ u_new).clamp_min(1e-300)
            if (u_new - u).abs().max() < self.sinkhorn_tol:
                u, v = u_new, v_new
                break
            u, v = u_new, v_new
        return (u.unsqueeze(1) * K) * v.unsqueeze(0)

    def _compute_log_affinity(self, X: torch.Tensor, omega: Optional[torch.Tensor] = None):
        n = self._get_n_samples(X)
        device = self._get_compute_device(X)
        dtype = self._get_dtype(X)
        X = X.to(device=device, dtype=dtype)

        if omega is None:
            omega = self._omega_default
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
        omega_scaled = rescale_admissible(-omega, self.max_omega_norm)

        C = randers_distance(X, omega_scaled, mode=self.drift_mode) ** 2
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        C = C.masked_fill(diag_mask, float("inf"))

        target_perp = math.sqrt(n) if self.perplexity == "auto" else float(self.perplexity)
        tau = self._calibrate_bandwidth(C, target_perp)
        c_min = C.min(dim=1, keepdim=True).values
        K = torch.exp(-(C - c_min) / tau.unsqueeze(1))
        K = K.masked_fill(diag_mask, 0.0)

        if self.doubly_stochastic:
            P = self._sinkhorn(K)
        else:
            P = (K / K.sum(dim=1, keepdim=True).clamp_min(1e-300)) / n

        return P.clamp_min(1e-300).log()


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
        zero-initialised Tanh MLP).
    drift_mode : {"frozen", "averaged"}, optional
        Default "averaged".
    max_omega_norm : float, optional
        Admissibility bound :math:`\rho` on :math:`\lVert\omega\rVert`.
        Default 0.95.
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
        verbose: bool = False,
        compile: bool = False,
        _pre_processed: bool = False,
    ):
        super().__init__(
            degrees_of_freedom=degrees_of_freedom, metric="sqeuclidean",
            zero_diag=zero_diag, device=device, backend=None, verbose=verbose,
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
        self.flow = flow if flow is not None else mlp_flow(n_components)

    def omega_at(self, Z: torch.Tensor) -> torch.Tensor:
        """The fitted drift field at every row of ``Z`` -- for inspecting
        or plotting (e.g. as a quiver plot over the embedding). NOTE the
        sign convention: under d_F(z_i->z_j) = ||dz|| + omega(z_i)^T dz, a
        correctly-fit omega points AWAY from the direction the affinity
        favours (moving with omega makes the distance larger, not
        smaller) -- plot ``-omega_at(Z)`` for the "flow direction", not the
        raw value."""
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
        omega_field = self.omega_at(Z).to(dtype=Z.dtype, device=Z.device)
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
