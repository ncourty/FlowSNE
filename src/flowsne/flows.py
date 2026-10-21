"""Flow networks: the drift field omega_theta(z) used by RandersAffinity's
output kernel (flowsne.affinity). Any ``nn.Module`` mapping (n, n_components)
-> (n, n_components) can be passed as ``flow`` to ``RandersAffinity`` --
``RandersAffinity`` itself applies the admissibility reparametrisation
(``distances.bounded_direction``), so a flow network's output is a RAW,
unconstrained vector, not something the network needs to bound itself.

Every builder below zero-initialises its last layer, so the resulting
affinity starts as an exact no-op (omega_theta == 0 everywhere, i.e. the
plain symmetric kernel) -- a deliberate, load-bearing convention: it means
fitting can never do *worse* than the symmetric baseline by construction
at initialisation, and any directional structure the optimiser finds is
attributable to the data, not to network init noise.

Three architectures are provided:

- ``mlp_flow``: a plain Tanh MLP. The simplest choice, and a reasonable
  default for most data.
- ``input_std_norm_flow``: prepends a global std-rescaling of z before the
  MLP. Decouples the network's effective input range from wherever the
  embedding's own optimiser happens to put z's absolute scale, which can
  otherwise vary by an order of magnitude across problems and drift during
  joint optimisation, silently saturating Tanh's non-linearity.
- ``rff_flow``: prepends a random Fourier feature encoding (fixed, not
  learned) before the MLP -- gives the network a high-frequency inductive
  bias, useful when the true drift field varies quickly with position
  (e.g. a rotational field), at the cost of being more sensitive to the
  ``sigma`` bandwidth matching the data's own scale.

For a gradient drift (``RandersAffinity(drift_mode="potential")``) the
network is a scalar potential instead: ``potential_flow`` maps
(n, n_components) -> (n, 1) and is 1-Lipschitz by construction (every
linear layer's spectral norm clipped to <= 1, 1-Lipschitz activations), which
is the admissibility condition for the exact distance
||z_j - z_i|| + h(z_j) - h(z_i). Its last layer is zero-initialised too.
``FixedPotential`` is the non-parametric alternative: a given per-point
potential (e.g. the input one -- a scalar per point needs no translation
from the data space to the embedding) times one learned scale; admissibility
then constrains the layout instead (pairs with |c dpsi| > ||dz|| get distance 0).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class InputStdNorm(nn.Module):
    """Divides z by its own current global std. Detached (not backpropped
    through): a slowly-adapting external rescaling re-estimated fresh at
    every forward pass, not a differentiable coupling of every point's
    drift to the whole configuration's statistics."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale = z.detach().std().clamp_min(1e-6)
        return z / scale


class RandomFourierFeatures(nn.Module):
    """Fixed (not learned) random Fourier feature encoding:
    gamma(z) = [cos(2*pi*B*z), sin(2*pi*B*z)], B ~ N(0, sigma^2). ``sigma``
    controls the encoding's characteristic frequency relative to z's own
    scale -- must be calibrated to it (see ``input_std_norm_flow`` /
    ``rff_flow``'s own docstrings for how each handles this)."""

    def __init__(self, in_dim: int, n_features: int, sigma: float = 1.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(n_features, in_dim, generator=g) * sigma
        self.register_buffer("B", B)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (z @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def mlp_flow(n_components: int, hidden_dim: int = 16, n_hidden_layers: int = 1) -> nn.Module:
    """Plain Tanh MLP, zero-initialised last layer."""
    dims = [n_components] + [hidden_dim] * n_hidden_layers
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    last = nn.Linear(dims[-1], n_components)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


def input_std_norm_flow(n_components: int, hidden_dim: int = 16,
                        n_hidden_layers: int = 1) -> nn.Module:
    """InputStdNorm -> Tanh MLP, zero-initialised last layer. The most
    broadly robust choice found across this project's own real-data
    testing: fixes Tanh saturation without any per-dataset recalibration."""
    return nn.Sequential(InputStdNorm(), mlp_flow(n_components, hidden_dim, n_hidden_layers))


def rff_flow(n_components: int, hidden_dim: int = 16, n_hidden_layers: int = 1,
            n_fourier: int = 128, sigma: float = 1.0, seed: int = 0) -> nn.Module:
    """RandomFourierFeatures -> Tanh MLP, zero-initialised last layer.
    ``sigma`` must be calibrated to z's own scale (the number of
    oscillations the encoding produces across z's typical range) -- if z
    is expected to stay near unit scale (e.g. after PCA-init with
    ``init_scaling=1.0``), start with ``sigma`` around 0.01-1.0 and sweep;
    there is no universal default, unlike ``input_std_norm_flow``."""
    rff = RandomFourierFeatures(n_components, n_fourier, sigma=sigma, seed=seed)
    dims = [2 * n_fourier] + [hidden_dim] * n_hidden_layers
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.Tanh()]
    last = nn.Linear(dims[-1], n_components)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(rff, *layers)


class ConstantFlow(nn.Module):
    """omega_theta(z) == a single learnable constant vector, independent of
    z -- the "global drift" special case (Finsler MDS's own canonical
    Randers metric), expressed as just another flow network rather than a
    separate affinity class. Exact and closed-form regardless of
    ``drift_mode`` in ``RandersAffinity`` (frozen and averaged coincide for
    a constant field -- see ``distances.py``'s module docstring)."""

    def __init__(self, n_components: int, init: torch.Tensor | None = None):
        super().__init__()
        self.raw = nn.Parameter(
            torch.zeros(n_components) if init is None else torch.as_tensor(init).clone()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.raw.expand(z.shape[0], -1)


class LipschitzLinear(nn.Linear):
    """``nn.Linear`` whose weight is divided by ``max(1, ||W||_2)`` (exact
    spectral norm) in the forward pass: 1-Lipschitz, and unlike
    ``torch.nn.utils.spectral_norm`` it allows a zero weight (no 0/0)."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sigma = torch.linalg.matrix_norm(self.weight, ord=2)
        return nn.functional.linear(z, self.weight / sigma.clamp_min(1.0), self.bias)


def potential_flow(n_components: int, hidden_dim: int = 64, n_hidden_layers: int = 2) -> nn.Module:
    """Scalar potential network, 1-Lipschitz in z: spectrally clipped linear
    layers with Tanh, zero-initialised last layer (h == 0 at start). Takes
    raw z (no input rescaling: that would change the Lipschitz constant)."""
    dims = [n_components] + [hidden_dim] * n_hidden_layers
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [LipschitzLinear(d_in, d_out), nn.Tanh()]
    last = LipschitzLinear(dims[-1], 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class FixedPotential(nn.Module):
    """A given per-point potential times a scale: ``h_i = scale * psi_i``,
    ``scale`` learned (``learn_scale=True``, default) and initialised at
    ``init_scale`` (default 0: the symmetric kernel at start). ``forward(z)``
    ignores z's values: row i of z must be point i (as in an embedding fit),
    so the potential is only defined at the points -- interpolate per-point
    values to draw it -- and ``RandersAffinity.omega_at`` is meaningless (0).
    Not Lipschitz by construction: see the module docstring."""

    def __init__(self, psi: torch.Tensor, learn_scale: bool = True, init_scale: float = 0.0):
        super().__init__()
        self.register_buffer("psi", torch.as_tensor(psi).reshape(-1).clone())
        scale = torch.tensor(float(init_scale), dtype=self.psi.dtype)
        if learn_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[0] != self.psi.shape[0]:
            raise ValueError(f"FixedPotential: {z.shape[0]} rows, but the potential has "
                             f"{self.psi.shape[0]} points (row i must be point i).")
        return (self.scale * self.psi).to(z.dtype).unsqueeze(-1)
