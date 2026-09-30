"""Toy: points on the unit sphere S^2, with a rotation drift field
omega(x) = e_z x x -- automatically tangent to the sphere, tracing circles
of latitude with speed sin(colatitude) (zero at the poles, maximal at the
equator). A position-varying field, so it specifically tests whether the
embedding's own drift network can recover direction that changes across
space, not just a single global preference.

Fits four output kernels against the same input affinity: symmetric
(blind to any asymmetry by construction), global (a single learnable
drift, ConstantFlow -- provably cannot represent a direction that varies
with position), and two neural ones that can: input_std_norm_flow, and
rff_flow (sigma=0.03, 128 Fourier features -- FinslerAffinity's
sphere-toy calibration, where it recovered the latitude-dependent speed
better than the plain Tanh network).

Run: conda run -n dev python experiments/toy/sphere_vortex.py [--n 1500]
Writes: experiments/toy/sphere_vortex.png
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher,
    ConstantFlow, input_std_norm_flow, rff_flow,
)
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "sphere_vortex.png")


def make_sphere_vortex(n=800, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    X = v / np.linalg.norm(v, axis=1, keepdims=True)
    omega = np.stack([-X[:, 1], X[:, 0], np.zeros(n)], axis=1)
    phi = np.arctan2(X[:, 1], X[:, 0])
    speed = np.linalg.norm(omega, axis=1)  # = sin(colatitude)
    return X, omega, phi, speed


def pca_init(X_np, n_components=2, seed=0):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=seed).fit_transform(X_np)


def fit(affinity_out, P, init_np, matcher_cls, seed=0, max_iter=1500, lr=0.1,
       verbose=False, check_interval=250):
    torch.manual_seed(seed)
    dr = matcher_cls(
        affinity_in="precomputed", affinity_out=affinity_out, n_components=2,
        loss_fn="cross_entropy_loss", optimizer="Adam", lr=lr, max_iter=max_iter,
        init=init_np, init_scaling=1.0, random_state=seed, verbose=verbose,
        check_interval=check_interval,
    )
    return dr.fit_transform(P)


def main(n=800):
    X_np, omega_np, phi, speed = make_sphere_vortex(n=n)
    X = torch.tensor(X_np, dtype=torch.float64)
    omega = torch.tensor(omega_np, dtype=torch.float64)

    P = EntropicRandersAffinity(perplexity="auto", omega=omega)(X).detach()
    init_np = pca_init(X_np)

    out_sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True,
                                        normalization_dim=(0, 1))
    Z_sym = fit(out_sym, P, init_np, AffinityMatcher)

    out_glob = RandersAffinity(n_components=2, flow=ConstantFlow(2), degrees_of_freedom=1.0).double()
    Z_glob = fit(out_glob, P, init_np, JointAffinityMatcher)

    out_neu = RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                              degrees_of_freedom=1.0).double()
    Z_neu = fit(out_neu, P, init_np, JointAffinityMatcher, max_iter=3000, lr=0.03, verbose=True)

    out_rff = RandersAffinity(n_components=2, flow=rff_flow(2, hidden_dim=16, n_fourier=128,
                                                           sigma=0.03),
                              degrees_of_freedom=1.0).double()
    Z_rff = fit(out_rff, P, init_np, JointAffinityMatcher, max_iter=2000, lr=0.1, verbose=True)

    losses, fields = {}, {}
    for name, out, Z in [("symmetric", out_sym, Z_sym), ("global", out_glob, Z_glob),
                         ("neural", out_neu, Z_neu), ("rff", out_rff, Z_rff)]:
        with torch.no_grad():
            losses[name] = cross_entropy_loss(P, out(Z, log=True), log=True).item()
        with torch.no_grad():
            fields[name] = -out.omega_at(Z).numpy() if hasattr(out, "omega_at") else None
    print("losses:", {k: round(v, 4) for k, v in losses.items()})

    corrs = {k: np.corrcoef(np.linalg.norm(fields[k], axis=1), speed)[0, 1]
             for k in ("neural", "rff")}
    for k, c in corrs.items():
        print(f"corr(||fitted field||, true speed) [{k}] = {c:+.3f}")

    fig = plt.figure(figsize=(24, 5.4))
    ax0 = fig.add_subplot(1, 5, 1, projection="3d")
    ax0.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], c=phi, cmap="twilight", s=10)
    step = max(1, n // 130)
    ax0.quiver(X_np[::step, 0], X_np[::step, 1], X_np[::step, 2],
              omega_np[::step, 0], omega_np[::step, 1], omega_np[::step, 2],
              length=0.25, color="dimgray", linewidth=0.8, alpha=0.8)
    ax0.set_title("Ground truth: S^2 + rotation field", fontsize=10)
    ax0.set_box_aspect([1, 1, 1])
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_zticks([])

    panels = [
        (Z_sym.detach().numpy(), None, f"symmetric\nloss={losses['symmetric']:.3f}"),
        (Z_glob.detach().numpy(), None, f"global\nloss={losses['global']:.3f}"),
        (Z_neu.detach().numpy(), fields["neural"],
         f"neural (input std-norm)\nloss={losses['neural']:.3f}, "
         f"corr(||field||,speed)={corrs['neural']:+.2f}"),
        (Z_rff.detach().numpy(), fields["rff"],
         f"neural (RFF, sigma=0.03)\nloss={losses['rff']:.3f}, "
         f"corr(||field||,speed)={corrs['rff']:+.2f}"),
    ]
    # Same fixed subset of arrows in every panel (all points still scattered).
    arrow_idx = np.random.default_rng(0).choice(n, size=min(n, 200), replace=False)
    sca = None
    for j, (Z, om, title) in enumerate(panels):
        ax = fig.add_subplot(1, 5, j + 2)
        sca = ax.scatter(Z[:, 0], Z[:, 1], c=phi, cmap="twilight", s=18, zorder=2)
        if om is not None:
            scale = 0.12 * (Z.max(0) - Z.min(0)).max() / max(np.abs(om).max(), 1e-9)
            Za, oma = Z[arrow_idx], om[arrow_idx]
            ax.quiver(Za[:, 0], Za[:, 1], oma[:, 0], oma[:, 1], color="crimson",
                     scale=1.0 / scale, scale_units="xy", angles="xy", width=0.004, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(sca, ax=fig.get_axes()[1:], shrink=0.7, pad=0.02, label="azimuth")
    fig.suptitle(f"Sphere vortex (n={n}): symmetric / global / neural output kernels",
                 fontsize=12, y=1.04)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=800)
    main(n=parser.parse_args().n)
