"""Divergence-free (stream-function) drift vs the free flow network, on the
sphere vortex (S^2, rotation drift e_z x x, n = 1500).

Theory (article/ICML): under a BALANCED input affinity the loss does not
reward the gradient part of the output drift to first order, so a
divergence-free drift omega = J grad h (``RandersAffinity(drift_mode=
"stream")``) should lose nothing; under a ROW-STOCHASTIC input it drops the
gradient channel. The sphere's drift is solenoidal on S^2, so even there the
stream mode is not expected to lose much -- this is the easy case first.

Same data, input affinities (rho = 0.9), PCA init, fit schedules and metrics
as experiments/baselines/finsler_embedding_comparison.py:
  sym              symmetric Student kernel
  {flow,stream}_joint      joint (Z, flow) fit, Adam lr 0.03, 3000 its
  {flow,stream}_driftonly  flow fitted on the frozen symmetric layout,
                           lr 0.05, 1500 its
  flow64_*         free flow with the stream networks' trunk (InputStdNorm, Tanh 64 x 2)
  {flow,stream}_rff<c>_*  RFF first layer after InputStdNorm (128 features, sigma = c
                   oscillations per std of z), Tanh 64 x 2; the stream with the global bound
  stream32[_rff<c>]_*  global-bound stream with the free flow's small trunk (Tanh 32 x 1)
  stream_global_*  stream function an unconstrained MLP, the field rescaled by one
                   global factor (stream_bound="global")
  stream_density_* exploratory: omega = c J grad h / rho (rho a KDE of the embedding),
                   so the mass flux rho omega is divergence-free (seed 0 only)
  stream_raw_*     the same with the stream function a 1-Lipschitz network on
                   raw z (bounded range, about 2 sqrt(64)): too weak a circulation
                   across a large embedding -- the default is now scale-equivariant
Metrics: loss; cos_ref / speed_spearman against the push-forward of the true
drift; kNN preservation; and the fitted field's RMS divergence and curl over
the points (autograd), the free flow's share of divergence.

Run:    conda run -n dev python experiments/toy/stream_sphere.py run --seed 0 --norm ds
        conda run -n dev python experiments/toy/stream_sphere.py report
Writes: experiments/toy/results/stream_sphere/{sphere_<norm>_seed<s>.{json,npz}, STREAM_SPHERE.md,
        sphere_vortex_stream.png}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "baselines"))

from flowsne import (  # noqa: E402
    EntropicRandersAffinity, RandersAffinity, JointAffinityMatcher, bounded_direction,
    fit_drift_only, input_std_norm_flow, rff_flow, scalar_flow,
)
from flowsne.flows import InputStdNorm  # noqa: E402
from torchdr import AffinityMatcher, NormalizedStudentAffinity  # noqa: E402
from torchdr.utils import cross_entropy_loss  # noqa: E402
from finsler_embedding_comparison import RHO, evaluate, fit_matcher, load_dataset  # noqa: E402

RES = os.path.join(HERE, "results", "stream_sphere")
DT = torch.float64
SEEDS = [0, 1, 2]
RFF_C = ["0.3", "1"]
METHODS = ["sym", "flow_joint", "flow64_joint", "stream_joint", "stream_global_joint", "stream_density_joint",
           "stream_raw_joint", "flow_driftonly", "flow64_driftonly", "stream_driftonly", "stream_global_driftonly",
           "stream_density_driftonly", "stream_raw_driftonly"] + [
    f"{k}_rff{c}_{fit}" for fit in ("joint", "driftonly") for k in ("flow", "stream") for c in RFF_C] + [
    f"stream32{r}_{fit}" for fit in ("joint", "driftonly") for r in ("", "_rff0.3", "_rff1")]


class DensityStream(RandersAffinity):
    """Exploratory: omega = c J grad h / rho(z), rho a Gaussian KDE of the
    embedding (detached, mean 1, Scott bandwidth), c one global factor. The
    MASS FLUX rho omega is then divergence-free -- the density-weighted
    (Leray) cyclic part of the theory -- while omega itself is not."""

    @staticmethod
    def _kde(Z):
        Zd = Z.detach()
        b = Zd.std() * len(Zd) ** (-1 / 6)
        rho = torch.exp(-torch.cdist(Zd, Zd) ** 2 / (2 * b ** 2)).mean(1)
        return rho / rho.mean()

    def _parts(self, Z, detach_scale=False):
        outer = torch.is_grad_enabled()
        with torch.enable_grad():
            Zg = Z if Z.requires_grad else Z.detach().requires_grad_(True)
            g = torch.autograd.grad(self.flow(Zg).sum(), Zg, create_graph=outer)[0]
        g = g / self._kde(Z)[:, None]
        c = self.max_omega_norm / g.norm(dim=1).max().clamp_min(1.0)
        return g, (c.detach() if detach_scale else c), outer

    def _stream_field(self, Z, detach_scale=False):
        g, c, outer = self._parts(Z, detach_scale)
        omega = c * torch.stack([-g[:, 1], g[:, 0]], dim=1)
        return omega if outer else omega.detach()

    def stream_function_at(self, Z):
        """Stream function of the mass flux rho omega."""
        _, c, _ = self._parts(Z)
        with torch.no_grad():
            return (c * self.flow(Z).reshape(Z.shape[0])).detach()


def make(kind, seed):
    torch.manual_seed(seed)
    if "_rff" in kind:  # {flow,stream}_rff<c>: InputStdNorm -> RFF (sigma = c per std) -> Tanh 64 x 2
        base, c = kind.split("_rff")
        if base == "flow":
            net = torch.nn.Sequential(InputStdNorm(), rff_flow(2, hidden_dim=64, n_hidden_layers=2,
                                                               n_fourier=128, sigma=float(c)))
            return RandersAffinity(n_components=2, flow=net, degrees_of_freedom=1.0).double()
        hid, nl = (32, 1) if base == "stream32" else (64, 2)
        return RandersAffinity(n_components=2, drift_mode="stream", degrees_of_freedom=1.0,
                               flow=scalar_flow(2, hid, nl, n_fourier=128, sigma=float(c)),
                               stream_bound="global").double()
    if kind == "stream32":  # global-bound stream, small trunk (Tanh 32 x 1) as the free flow
        return RandersAffinity(n_components=2, drift_mode="stream", degrees_of_freedom=1.0,
                               flow=scalar_flow(2, 32, 1), stream_bound="global").double()
    if kind == "stream_density":
        return DensityStream(n_components=2, drift_mode="stream", degrees_of_freedom=1.0).double()
    if kind == "flow":
        return RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                               degrees_of_freedom=1.0).double()
    if kind == "flow64":  # the stream networks' trunk (64 x 2), vector output
        return RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=64,
                               n_hidden_layers=2), degrees_of_freedom=1.0).double()
    return RandersAffinity(n_components=2, drift_mode="stream", degrees_of_freedom=1.0,
                           stream_bound="global" if kind == "stream_global" else "lipschitz").double()


def div_curl(aff, Z):
    """RMS over the points of div omega and curl omega (2-D scalar curl)."""
    with torch.enable_grad():
        Zg = Z.detach().clone().requires_grad_(True)
        om = aff._stream_field(Zg, detach_scale=True) if aff.drift_mode == "stream" else \
            bounded_direction(aff.flow(Zg), aff.max_omega_norm, dim=-1)
        J0 = torch.autograd.grad(om[:, 0].sum(), Zg, retain_graph=True)[0]
        J1 = torch.autograd.grad(om[:, 1].sum(), Zg)[0]
    div, curl = J0[:, 0] + J1[:, 1], J1[:, 0] - J0[:, 1]
    rms = lambda v: float(v.pow(2).mean().sqrt())
    return rms(div), rms(curl)


def run(seed, norm, only=None):
    D = load_dataset("sphere", seed)
    X, om = torch.tensor(D["X"], dtype=DT), torch.tensor(D["omega"], dtype=DT)
    from sklearn.decomposition import PCA
    init_np = PCA(n_components=2, random_state=seed).fit_transform(D["X"])
    P = EntropicRandersAffinity(omega=om, max_omega_norm=RHO, doubly_stochastic=(norm == "ds"))(X).detach()
    out = dict(seed=seed, norm=norm, n=len(X), runs={})
    saved = dict(color=D["color"])
    path = os.path.join(RES, f"sphere_{norm}_seed{seed}")
    os.makedirs(RES, exist_ok=True)
    if only and only != "stream":
        out = json.load(open(path + ".json"))
        saved = dict(np.load(path + ".npz"))
    if only == "stream":
        # refit the stream fits only; keep the earlier ones (1-Lipschitz network on
        # raw z, bounded range) as stream_raw_*
        out = json.load(open(path + ".json"))
        saved = dict(np.load(path + ".npz"))
        for m in ("stream_joint", "stream_driftonly"):
            if m in out["runs"] and "stream_raw_" + m[7:] not in out["runs"]:
                out["runs"]["stream_raw_" + m[7:]] = out["runs"].pop(m)
                for suf in ("", "_drift", "_h"):
                    saved["stream_raw_" + m[7:] + suf] = saved.pop(m + suf)

    def record(name, Z, aff, secs):
        Z = Z.detach()
        with torch.no_grad():
            loss = cross_entropy_loss(P, aff(Z, log=True), log=True).item()
        r = dict(loss=loss, secs=secs)
        if name != "sym":
            with torch.no_grad():
                F = -aff.omega_at(Z).numpy()
            r.update(evaluate(D, Z.numpy(), F))
            r["div_rms"], r["curl_rms"] = div_curl(aff, Z)
            saved[name + "_drift"] = F
            if aff.drift_mode == "stream":
                with torch.no_grad():
                    saved[name + "_h"] = aff.stream_function_at(Z).numpy()
        else:
            from finsler_embedding_comparison import knn_preservation
            r["knn"] = knn_preservation(D["X"], Z.numpy())
        saved[name] = Z.numpy()
        out["runs"][name] = r
        print(f"[{norm} s{seed}] {name:17s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
        # merge into what is on disk (parallel --only jobs share the files)
        import fcntl
        with open(path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(path + ".json"):
                disk = json.load(open(path + ".json"))
                disk["runs"].update(out["runs"])
                out["runs"] = disk["runs"]
                disk_npz = dict(np.load(path + ".npz"))
                saved.update({k: v for k, v in disk_npz.items() if k not in saved})
            json.dump(out, open(path + ".json", "w"), indent=1)
            np.savez_compressed(path + ".npz", **saved)

    if only:
        Z_sym = torch.tensor(saved["sym"], dtype=DT)
    else:
        t0 = time.time()
        sym = NormalizedStudentAffinity(degrees_of_freedom=1, zero_diag=True, normalization_dim=(0, 1))
        Z_sym = fit_matcher(sym, P, init_np, AffinityMatcher, seed, 1500, 0.1).detach()
        record("sym", Z_sym, sym, time.time() - t0)
    for kind in ((only,) if only else ("flow", "stream")):
        aff = make(kind, seed)
        t0 = time.time()
        Z = fit_matcher(aff, P, init_np, JointAffinityMatcher, seed, 3000, 0.03)
        record(f"{kind}_joint", Z, aff, time.time() - t0)
        aff = make(kind, seed)
        t0 = time.time()
        Z = fit_drift_only(aff, P, Z_sym, seed=seed, max_iter=1500, lr=0.05)
        record(f"{kind}_driftonly", Z, aff, time.time() - t0)


def report():
    rows = {}
    for f in sorted(glob.glob(os.path.join(RES, "sphere_*_seed*.json"))):
        d = json.load(open(f))
        for m, r in d["runs"].items():
            r = dict(r, dloss=r["loss"] - d["runs"]["sym"]["loss"])
            rows.setdefault((d["norm"], m), []).append(r)
    keys = ["loss", "dloss", "cos_ref", "speed_spearman", "knn", "div_rms", "curl_rms"]
    L = ["# Stream-function (divergence-free) drift vs free flow: sphere vortex", "",
         "See `stream_sphere.py`. Mean ± sd over seeds "
         f"({', '.join(str(s) for s in SEEDS)}); n = 1500. `dloss` = loss minus the symmetric "
         "fit's (same seed and input). `cos_ref`, `speed_spearman`: fitted -omega against the "
         "push-forward of the true drift. `div_rms` / `curl_rms`: RMS divergence / curl of the "
         "fitted field over the points.", ""]
    for norm in ("ds", "rs"):
        L += [f"## {'balanced' if norm == 'ds' else 'row-stochastic'} input", "",
              "| method | seeds | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
        for m in METHODS:
            rs = rows.get((norm, m))
            if not rs:
                continue
            cells = []
            for k in keys:
                v = [r[k] for r in rs if k in r]
                cells.append(f"{np.mean(v):.4f} ± {np.std(v):.4f}" if v else "")
            L.append(f"| {m} | {len(rs)} | " + " | ".join(cells) + " |")
        L.append("")
    open(os.path.join(RES, "STREAM_SPHERE.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    figure()


def figure(seed=0):
    """sphere_vortex.png-style figure, seed 0: 3-D ground truth, then for each
    input (rows) the symmetric, free-flow and stream joint fits, arrows = -omega
    on a fixed subset of points, the stream fit over its stream function h."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = load_dataset("sphere", seed)
    X, om, c = D["X"], D["omega"], D["color"]
    fig = plt.figure(figsize=(18.5, 7.2))
    gs = fig.add_gridspec(2, 6, width_ratios=[1.15, 1, 1, 1, 1, 0.05], wspace=0.08, hspace=0.18,
                          left=0.01, right=0.96, top=0.86, bottom=0.03)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=c, cmap="twilight", s=6, linewidths=0)
    st = max(1, len(X) // 150)
    ax3.quiver(X[::st, 0], X[::st, 1], X[::st, 2], om[::st, 0], om[::st, 1], om[::st, 2],
               length=0.25, color="dimgray", linewidth=0.7)
    ax3.set_title("ground truth: $S^2$ + rotation $e_z\\times x$", fontsize=10)
    ax3.set_box_aspect([1, 1, 1]); ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([])
    idx = np.random.default_rng(0).choice(len(X), 200, replace=False)
    sca = None
    for row, norm in enumerate(("ds", "rs")):
        f = os.path.join(RES, f"sphere_{norm}_seed{seed}")
        d = np.load(f + ".npz") if os.path.exists(f + ".npz") else {}
        runs = json.load(open(f + ".json"))["runs"] if os.path.exists(f + ".json") else {}
        for col, (m, label) in enumerate((("sym", "symmetric"), ("flow_joint", "free flow"),
                                          ("stream_joint", "stream, Lipschitz h"),
                                          ("stream_global_joint", "stream, global rescale"))):
            ax = fig.add_subplot(gs[row, col + 1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if col == 0:
                ax.set_ylabel("balanced input" if norm == "ds" else "row-stochastic input", fontsize=10)
            if m not in runs:
                ax.text(0.5, 0.5, f"{label}\n(still running)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
                continue
            Z, r = d[m], runs[m]
            if m + "_h" in d:
                ax.tricontourf(Z[:, 0], Z[:, 1], d[m + "_h"], levels=16, cmap="Greys", alpha=0.3)
                ax.tricontour(Z[:, 0], Z[:, 1], d[m + "_h"], levels=16, colors="gray",
                              linewidths=0.3, alpha=0.6)
            sca = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="twilight", s=8, linewidths=0, zorder=2)
            title = f"{label}\nloss={r['loss']:.3f}"
            if m + "_drift" in d:
                F = d[m + "_drift"]
                span = (Z.max(0) - Z.min(0)).max()
                ax.quiver(Z[idx, 0], Z[idx, 1], F[idx, 0], F[idx, 1], color="crimson", angles="xy",
                          scale_units="xy", scale=np.abs(F).max() / (0.12 * span), width=0.004, zorder=3)
                title += (f", cos={r['cos_ref']:.2f}, speed $\\rho$={r['speed_spearman']:.2f}\n"
                          f"rms div={r['div_rms']:.3f}, curl={r['curl_rms']:.3f}")
            ax.set_title(title, fontsize=8.5)
    if sca is not None:
        fig.colorbar(sca, cax=fig.add_subplot(gs[:, 5]), label="azimuth")
    fig.suptitle("Sphere vortex (n = 1500, seed 0): free flow vs divergence-free (stream) drift",
                 fontsize=12, y=0.985)
    out = os.path.join(RES, "sphere_vortex_stream.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Where do the vortex centres land? (the true drift's centres are the poles)
# ---------------------------------------------------------------------------

def _smooth(Z, vals, G, b):
    """Nadaraya-Watson regression of per-point ``vals`` onto grid nodes ``G``,
    and the kernel mass at each node (support mask)."""
    w = np.exp(-((G[:, None, :] - Z[None]) ** 2).sum(-1) / (2 * b ** 2))
    m = w.sum(1)
    return (w @ vals) / m.clip(1e-300)[:, None], m


def critical_points(Z, F, z, res=60, bw=1.5, support=0.25):
    """Critical points of the field F (n, 2) sampled on the embedding Z.

    F and the sphere height z are kernel-smoothed onto a res x res grid
    (bandwidth ``bw`` x the median 5-NN distance); grid nodes with kernel mass
    below ``support`` x the median mass at the points are outside the data.
    The Poincare index of each cell is the winding of F around its four
    corners; a +1 cell is a CENTRE if the local curl dominates the divergence
    (else a source/sink node), a -1 cell a SADDLE. Returns a list of dicts
    (kind, position, |z| there, speed there / median speed)."""
    from sklearn.neighbors import NearestNeighbors
    d5 = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)[0][:, -1]
    b = bw * np.median(d5)
    lo, hi = Z.min(0) - b, Z.max(0) + b
    xs, ys = np.linspace(lo[0], hi[0], res), np.linspace(lo[1], hi[1], res)
    GX, GY = np.meshgrid(xs, ys)
    G = np.stack([GX.ravel(), GY.ravel()], 1)
    V, m = _smooth(Z, np.column_stack([F, z]), G, b)
    m_pts = _smooth(Z, np.zeros((len(Z), 1)), Z, b)[1]
    inside = (m >= support * np.median(m_pts)).reshape(res, res)
    U, W, H = (V[:, k].reshape(res, res) for k in range(3))
    th = np.arctan2(W, U)
    speed_med = np.median(np.linalg.norm(F, axis=1))

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    out = []
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    for i in range(res - 1):
        for j in range(res - 1):
            if not inside[i:i + 2, j:j + 2].all():
                continue
            loop = [th[i, j], th[i, j + 1], th[i + 1, j + 1], th[i + 1, j], th[i, j]]
            idx = int(round(sum(wrap(b_ - a_) for a_, b_ in zip(loop[:-1], loop[1:])) / (2 * np.pi)))
            if idx == 0:
                continue
            ux = (U[i, j + 1] + U[i + 1, j + 1] - U[i, j] - U[i + 1, j]) / (2 * dx)
            uy = (U[i + 1, j] + U[i + 1, j + 1] - U[i, j] - U[i, j + 1]) / (2 * dy)
            wx = (W[i, j + 1] + W[i + 1, j + 1] - W[i, j] - W[i + 1, j]) / (2 * dx)
            wy = (W[i + 1, j] + W[i + 1, j + 1] - W[i, j] - W[i, j + 1]) / (2 * dy)
            curl, div = wx - uy, ux + wy
            kind = "saddle" if idx < 0 else ("centre" if abs(curl) > abs(div) else "node")
            c = np.array([(xs[j] + xs[j + 1]) / 2, (ys[i] + ys[i + 1]) / 2])
            sp = np.hypot(U[i:i + 2, j:j + 2], W[i:i + 2, j:j + 2]).mean()
            out.append(dict(kind=kind, pos=c, absz=float(abs(H[i:i + 2, j:j + 2].mean())),
                            speed=float(sp / speed_med), sense=int(np.sign(curl))))
    return out


POLE_METHODS = (("flow_joint", "free flow"), ("flow64_joint", "free flow, 64x2"), ("stream_joint", "stream, Lipschitz h"),
                ("stream_global_joint", "stream, global rescale"))
POLE_METHODS += tuple((f"{k}_rff{c}_joint", f"{'free flow' if k == 'flow' else 'stream'}, RFF c={c}")
                      for k in ("flow", "stream") for c in RFF_C)
POLE_METHODS += tuple((f"stream32{r}_joint", f"stream 32x1{', RFF c=' + r[4:] if r else ''}")
                      for r in ("", "_rff0.3", "_rff1"))


def poles():
    """Table (all seeds) + figure (seed 0): vortex centres of the fitted drift
    -omega and of the push-forward of the true drift, with |z| at each."""
    from finsler_embedding_comparison import pushforward
    L = ["# Sphere vortex: where the vortex centres land", "",
         "See `stream_sphere.py poles`. The true drift e_z x x has exactly two centres, the poles "
         "(|z| = 1), and its speed sin(colatitude) vanishes there. Critical points of the fitted "
         "field -omega (`fit`) and of the push-forward of the true drift into the same layout "
         "(`ref`) are found by the Poincare index on a kernel-smoothed grid (`critical_points`). "
         "`centres`: count; `|z|`: kernel-smoothed sphere height at each centre (slowest first; "
         "smoothing caps it near 0.95 even exactly on a pole); `speed`: field speed there / "
         "median speed. Joint fits.", "",
         "| input | seed | method | ref centres | ref |z| | fit centres | fit |z| | fit speed | fit saddles+nodes |",
         "|---|---|---|---|---|---|---|---|---|"]
    figdata = {}
    for norm in ("ds", "rs"):
        for seed in SEEDS:
            f = os.path.join(RES, f"sphere_{norm}_seed{seed}.npz")
            if not os.path.exists(f):
                continue
            d = np.load(f)
            D = load_dataset("sphere", seed)
            z = D["X"][:, 2]
            for m, label in POLE_METHODS:
                if m not in d:
                    continue
                Z, F = d[m], d[m + "_drift"]
                ref = pushforward(D["X"], Z, D["omega"], D["tan_dim"])
                cps = {"ref": critical_points(Z, ref, z), "fit": critical_points(Z, F, z)}
                cen = {k: sorted([c for c in v if c["kind"] == "centre"], key=lambda c: c["speed"])
                       for k, v in cps.items()}
                fmt = lambda cs, key: ", ".join(f"{c[key]:.2f}" for c in cs) or "-"
                other = sum(c["kind"] != "centre" for c in cps["fit"])
                L.append(f"| {norm} | {seed} | {label} | {len(cen['ref'])} | {fmt(cen['ref'], 'absz')} | "
                         f"{len(cen['fit'])} | {fmt(cen['fit'], 'absz')} | {fmt(cen['fit'], 'speed')} | {other} |")
                if seed == 0:
                    figdata[(norm, m)] = (Z, F, ref, z, cps)
    out = os.path.join(RES, "POLES.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    _poles_figure(figdata)


def _poles_figure(figdata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(POLE_METHODS), figsize=(4.7 * len(POLE_METHODS), 9.2), gridspec_kw=dict(wspace=0.05, hspace=0.16))
    idx = np.random.default_rng(0).choice(1500, 220, replace=False)
    sca = None
    mk = {"centre": ("o", 70), "saddle": ("X", 55), "node": ("s", 45)}
    for row, norm in enumerate(("ds", "rs")):
        for col, (m, label) in enumerate(POLE_METHODS):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if (norm, m) not in figdata:
                continue
            Z, F, ref, z, cps = figdata[(norm, m)]
            sca = ax.scatter(Z[:, 0], Z[:, 1], c=z, cmap="coolwarm", vmin=-1, vmax=1, s=7,
                             linewidths=0, zorder=1)
            span = (Z.max(0) - Z.min(0)).max()
            ax.quiver(Z[idx, 0], Z[idx, 1], F[idx, 0], F[idx, 1], color="k", angles="xy",
                      scale_units="xy", scale=np.abs(F).max() / (0.10 * span), width=0.003,
                      alpha=0.75, zorder=2)
            for who, ec in (("ref", "limegreen"), ("fit", "k")):
                for c in cps[who]:
                    s_, sz = mk[c["kind"]]
                    ax.scatter(*c["pos"], marker=s_, s=sz if who == "fit" else sz * 2.2,
                               facecolors="none" if who == "ref" else "gold",
                               edgecolors=ec, linewidths=1.6 if who == "ref" else 0.8, zorder=4)
            nref = sum(c["kind"] == "centre" for c in cps["ref"])
            nfit = sum(c["kind"] == "centre" for c in cps["fit"])
            ax.set_title(f"{label}\ncentres: fit {nfit}, true push-forward {nref}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("balanced input" if norm == "ds" else "row-stochastic input")
    if sca is not None:
        fig.colorbar(sca, ax=axes, shrink=0.6, pad=0.01, label="height z on the sphere (poles = ±1)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc="gold", mec="k", label="fit: centre"),
         Line2D([], [], marker="X", ls="", mfc="gold", mec="k", label="fit: saddle"),
         Line2D([], [], marker="s", ls="", mfc="gold", mec="k", label="fit: node"),
         Line2D([], [], marker="o", ls="", mfc="none", mec="limegreen", mew=1.6, ms=10,
                label="true push-forward: critical point")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle("Sphere vortex, seed 0: joint fits coloured by sphere height; arrows = fitted "
                 "$-\\omega$", fontsize=11.5, y=0.97)
    out = os.path.join(RES, "sphere_poles.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report", "figure", "poles"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--norm", choices=["ds", "rs"], default="ds")
    ap.add_argument("--only", default=None,
                    help="stream | stream_global | stream_density | flow64 | {flow,stream}_rff<c>")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.seed, a.norm, a.only)
    elif a.cmd == "figure":
        figure()
    elif a.cmd == "poles":
        poles()
    else:
        report()
