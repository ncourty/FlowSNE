"""FlowSNE on MNIST learning dynamics: one embedding per training checkpoint
(see dynamics_data.py), drift = -grad of each sample's own loss w.r.t. its
penultimate activation, played as an animation.

Consistency across frames: frame 0 is a cold start (PCA init); every later
frame is warm-started from the previous one, both the layout Z and the flow
network (the same RandersAffinity is reused), optionally with dynamic t-SNE's
temporal coherence term ``lambda * mean_i ||z_i - z_i^prev||^2``. Several
lambdas run in one call (--anchor-weights), advancing together frame by frame
so each frame's input affinity is computed once and shared.

All frames share ONE drift scale (rho / the largest gradient norm over all
checkpoints), so the arrows fade as training settles instead of being
renormalised frame by frame.

Per-frame metrics (metrics.json):
  stability        rms(Z_t - Z_{t-1}) / rms spread of Z_t  (0 = frozen layout)
  flow_vs_disp     cos(fitted flow at z_i^t, z_i^{t+1} - z_i^t): does the
                   fitted field predict where points move next in the embedding?
  grad_vs_disp_in  cos(-grad_a L_i at t, a_i^{t+1} - a_i^t) in the 64-d input
                   space: how predictive the raw gradient itself is (it is not
                   how activations move -- they move through weight updates)
  acc, mean_loss, mean_grad_norm on the sample set; fit loss.
Long-horizon (long_horizon(), stored in metrics.json): does the field at
frame t point to where points END UP (final frame, unit-spread layouts), or
to their digit's final cluster -- against an outward-from-centre control?

Run:    conda run -n dev python experiments/mnist/dynamics.py --n 1000
Writes: experiments/mnist/dynamics_results/n<n>_lam<lambda>/{frames.npz,
        metrics.json, dynamics.gif|mp4}; frames.npz also holds each frame's
        flow-network weights (for animate_streamlines.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)

from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding  # noqa: E402
from flowsne.flows import input_std_norm_flow  # noqa: E402
from dynamics_data import checkpoints, features, sample_set  # noqa: E402

RESULTS = os.path.join(HERE, "dynamics_results")
RHO = 0.9


def _cos(A, B):
    den = A.norm(dim=1) * B.norm(dim=1)
    return torch.where(den > 0, (A * B).sum(1) / den.clamp_min(1e-30), torch.zeros_like(den))


class _Run:
    """State of one lambda's sequence: the reused output affinity (its flow
    is warm-started), the previous layout, and everything recorded."""

    def __init__(self, lam):
        self.lam = lam
        self.aff_out = None
        self.Z_prev = None
        self.Z, self.F, self.theta, self.zstd, self.metrics = [], [], [], [], []

    def arrays(self):
        return (torch.stack(self.Z).cpu().double().numpy(),
                torch.stack(self.F).cpu().double().numpy(),
                torch.stack(self.theta).numpy(), np.array(self.zstd))


def save_run(run, out, labels, idx, steps, args, omega_scale, extra=None):
    Z, F, theta, zstd = run.arrays()
    np.savez_compressed(os.path.join(out, "frames.npz"), Z=Z.astype(np.float32),
                        F=F.astype(np.float32), labels=labels.numpy(), sample_indices=idx,
                        steps=np.array(steps[:len(run.Z)]), flow_params=theta, z_std=zstd,
                        flow_hidden_dim=32)
    json.dump(dict(args=vars(args), omega_scale=omega_scale, metrics=run.metrics,
                   complete=len(run.Z) == len(steps), **(extra or {})),
              open(os.path.join(out, "metrics.json"), "w"), indent=1)


def run_sequences(ckpts, images, labels, lams, args, device, dtype, omega_scale, outs, idx):
    """All lambdas advance together, frame by frame: each frame's input
    affinity (the expensive part at scale: kNN search) is computed once and
    shared. Results are saved every --save-every frames."""
    sparse = args.backend == "keops"
    perp = args.perplexity if args.perplexity == "auto" else float(args.perplexity)
    runs = [_Run(lam) for lam in lams]
    steps = [s for s, _ in ckpts]
    a_prev = g_prev = None
    for t, (step, path) in enumerate(ckpts):
        t0 = time.time()
        f = features(path, images, labels, device=device)
        X = f["activations"].to(dtype)
        V = -f["grad"].to(dtype)
        aff_in = EntropicRandersAffinity(omega=V, omega_scale=omega_scale, perplexity=perp,
                                         doubly_stochastic=False, sparsity=sparse,
                                         backend="keops" if sparse else None)
        P, idx_P = aff_in(X) if sparse else (aff_in(X), None)
        t_aff = time.time() - t0
        common = dict(step=step, acc=(f["pred"] == labels.to(device)).float().mean().item(),
                      mean_loss=f["loss"].mean().item(),
                      mean_grad_norm=f["grad"].norm(dim=1).mean().item(),
                      affinity_secs=round(t_aff, 2))
        grad_vs_disp_in = _cos(g_prev, X - a_prev).mean().item() if a_prev is not None else None
        for run in runs:
            t1 = time.time()
            if t == 0:
                torch.manual_seed(args.seed)
                run.aff_out = RandersAffinity(
                    n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                    backend="keops" if sparse else None).to(device=device, dtype=dtype)
                Xc = X - X.mean(0)
                init = Xc @ torch.linalg.svd(Xc, full_matrices=False).Vh[:2].T     # PCA
                Z = fit_embedding(run.aff_out, P, init, indices=idx_P, max_iter=args.iters_first,
                                  lr=args.lr, seed=args.seed)
            else:
                Z = fit_embedding(run.aff_out, P, run.Z_prev, indices=idx_P, max_iter=args.iters,
                                  lr=args.lr, seed=args.seed, rescale_init=False,
                                  anchor=run.Z_prev if run.lam > 0 else None,
                                  anchor_weight=run.lam)
            with torch.no_grad():
                loss = run.aff_out.cross_entropy(Z, P, idx_P).item()
                F = -run.aff_out.omega_at(Z)
            m = dict(common, fit_loss=loss, fit_secs=round(time.time() - t1, 2))
            if run.Z_prev is not None:
                spread = (Z - Z.mean(0)).pow(2).sum(1).mean().sqrt()
                m["stability"] = ((Z - run.Z_prev).pow(2).sum(1).mean().sqrt() / spread).item()
                run.metrics[-1]["flow_vs_disp"] = _cos(run.F[-1], Z - run.Z_prev).mean().item()
                run.metrics[-1]["grad_vs_disp_in"] = grad_vs_disp_in
            run.metrics.append(m)
            run.Z.append(Z.detach())
            run.F.append(F.detach())
            # the flow network itself, to evaluate the field anywhere (streamlines);
            # InputStdNorm divides by the std of its input, so keep Z's std too
            run.theta.append(torch.nn.utils.parameters_to_vector(run.aff_out.flow.parameters())
                             .detach().cpu().double())
            run.zstd.append(Z.detach().std().item())
            run.Z_prev = Z.detach()
            print(f"   step {step:4d} lam={run.lam:<6}: acc {m['acc']:.3f}  |grad| "
                  f"{m['mean_grad_norm']:.2e}  fit loss {loss:.3f}"
                  + (f"  stability {m['stability']:.3f}" if "stability" in m else "")
                  + f"  (affinity {t_aff:.1f}s, fit {m['fit_secs']:.1f}s)", flush=True)
        a_prev, g_prev = X.detach(), V.detach()
        if (t + 1) % args.save_every == 0 or t + 1 == len(ckpts):
            for run, out in zip(runs, outs):
                save_run(run, out, labels, idx, steps, args, omega_scale)
    return runs


def long_horizon(ckpts, images, labels, Z, F, device):
    """Does the field at frame t point to where things END UP?

    Frames are first centred and scaled to unit rms spread, so the layout's
    global expansion does not count as "prediction". For each t, mean over
    samples of:
      final_disp      cos(F_t(z_i), z~_i^T - z~_i^t)          (where the point ends up)
      class_target    cos(F_t(z_i), c~_{y_i}^T - z~_i^t)      (its digit's final cluster)
    and the same with an OUTWARD field (z~_i^t, i.e. away from the centre) as
    control -- a separating layout makes outward arrows look predictive by
    default. Input space: cos(-grad_t, a_i^T - a_i^t) (raw gradient vs final
    displacement of the 64-d activation)."""
    Zn = (Z - Z.mean(1, keepdims=True))
    Zn = Zn / np.sqrt((Zn ** 2).sum(2).mean(1))[:, None, None]
    T = Z.shape[0]
    lab = labels.numpy()
    cT = np.stack([Zn[-1, lab == c].mean(0) for c in range(10)])[lab]       # final class centre

    def cos(A, B):
        den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)
        return np.where(den > 0, (A * B).sum(1) / np.where(den > 0, den, 1), 0.0)

    rows = []
    a_T = features(ckpts[-1][1], images, labels, device=device)["activations"].double().cpu()
    for t in range(T - 1):
        f = features(ckpts[t][1], images, labels, device=device)
        g, a = -f["grad"].double().cpu(), f["activations"].double().cpu()
        rows.append(dict(
            step=ckpts[t][0],
            final_disp=float(cos(F[t], Zn[-1] - Zn[t]).mean()),
            class_target=float(cos(F[t], cT - Zn[t]).mean()),
            outward_final_disp=float(cos(Zn[t], Zn[-1] - Zn[t]).mean()),
            outward_class_target=float(cos(Zn[t], cT - Zn[t]).mean()),
            grad_vs_final_disp_in=float(_cos(g, a_T - a).mean())))
    return rows

def animate(Z, F, labels, metrics, path, fps=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    T, n, _ = Z.shape
    lo, hi = np.percentile(Z.reshape(-1, 2), [0.5, 99.5], axis=0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    sub = np.random.default_rng(0).choice(n, size=min(n, 200), replace=False)
    fmax = max(np.linalg.norm(F, axis=2).max(), 1e-12)
    arrow = 0.08 * (hi - lo).max() / fmax            # ONE scale for all frames: arrows fade
    colors = np.array(matplotlib.colormaps["tab10"].colors)[labels]

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    size, alpha = (6, 1.0) if n <= 2000 else (max(0.3, 6000 / n), 0.5)   # 60k points: no blob
    sc = ax.scatter(Z[0, :, 0], Z[0, :, 1], c=colors, s=size, alpha=alpha, linewidths=0)
    q = ax.quiver(Z[0, sub, 0], Z[0, sub, 1], F[0, sub, 0] * arrow, F[0, sub, 1] * arrow,
                  angles="xy", scale_units="xy", scale=1, width=0.003, color="black", alpha=0.7)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    title = ax.set_title("")

    def update(t):
        sc.set_offsets(Z[t])
        q.set_offsets(Z[t, sub])
        q.set_UVC(F[t, sub, 0] * arrow, F[t, sub, 1] * arrow)
        m = metrics[t]
        title.set_text(f"iteration {m['step']}   accuracy {m['acc']:.2f}   "
                       f"mean |grad| {m['mean_grad_norm']:.1e}")
        return sc, q, title

    anim = animation.FuncAnimation(fig, update, frames=T, blit=False)
    if shutil.which("ffmpeg") and path.endswith(".mp4"):
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=130)
    else:
        path = os.path.splitext(path)[0] + ".gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="samples (fixed training subset)")
    ap.add_argument("--anchor-weights", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1],
                    help="temporal coherence lambdas (all run together, sharing affinities)")
    ap.add_argument("--iters-first", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=300, help="iterations per warm-started frame")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--backend", choices=["auto", "dense", "keops"], default="auto",
                    help="auto = keops + sparse input affinity on CUDA, dense otherwise")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all checkpoints")
    ap.add_argument("--save-every", type=int, default=4, help="save results every k frames")
    ap.add_argument("--no-animation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    if args.backend == "auto":
        args.backend = "keops" if device.type == "cuda" else "dense"
    ckpts = checkpoints()
    if not ckpts:
        sys.exit("no checkpoints: run experiments/mnist/dynamics_data.py first")
    if args.max_frames:
        ckpts = ckpts[:args.max_frames]
    images, labels, idx = sample_set(args.n, seed=args.seed)

    # one drift scale for the whole sequence
    gmax = max(features(p, images, labels, device=device)["grad"].norm(dim=1).max().item()
               for _, p in ckpts)
    omega_scale = RHO / gmax
    print(f"{len(ckpts)} frames, n={args.n}, lambdas {args.anchor_weights}, backend "
          f"{args.backend}, {dtype}, global max |grad| {gmax:.3e} -> omega_scale "
          f"{omega_scale:.3e}", flush=True)

    outs = [os.path.join(RESULTS, f"n{args.n}_lam{lam:g}") for lam in args.anchor_weights]
    for out in outs:
        os.makedirs(out, exist_ok=True)
    t0 = time.time()
    runs = run_sequences(ckpts, images, labels, args.anchor_weights, args, device, dtype,
                         omega_scale, outs, idx)
    minutes = round((time.time() - t0) / 60, 1)

    summary = {}
    steps = [s for s, _ in ckpts]
    for run, out in zip(runs, outs):
        Z, F, _, _ = run.arrays()
        lh = long_horizon(ckpts, images, labels, Z, F, device)
        save_run(run, out, labels, idx, steps, args, omega_scale, extra=dict(long_horizon=lh))
        get = lambda k: [m[k] for m in run.metrics if m.get(k) is not None]
        early = lh[:10]
        summary[run.lam] = dict(
            stability=float(np.mean(get("stability"))),
            flow_vs_disp=float(np.mean(get("flow_vs_disp"))),
            grad_vs_disp_in=float(np.mean(get("grad_vs_disp_in"))),
            early_final_disp=float(np.mean([r["final_disp"] for r in early])),
            early_class_target=float(np.mean([r["class_target"] for r in early])),
            early_outward_class_target=float(np.mean([r["outward_class_target"] for r in early])),
            early_grad_vs_final_disp_in=float(np.mean([r["grad_vs_final_disp_in"] for r in early])),
            final_fit_loss=run.metrics[-1]["fit_loss"])
        if not args.no_animation:
            summary[run.lam]["animation"] = animate(Z, F, labels.numpy(), run.metrics,
                                                    os.path.join(out, "dynamics.mp4"))
    json.dump(dict(minutes=minutes, summary={str(k): v for k, v in summary.items()}),
              open(os.path.join(RESULTS, f"summary_n{args.n}.json"), "w"), indent=1)

    print(f"\nsummary ({minutes} min). Next-frame metrics are means over frames; "
          "'early' = first 10 frames vs the FINAL frame (unit-spread layouts):")
    print("  lambda   stability  flow_vs_next  grad_vs_next_in | final_disp  class_target "
          "(outward ctrl)  grad_vs_final_in | final fit loss")
    for lam, s_ in summary.items():
        print(f"  {lam:<7}  {s_['stability']:.3f}      {s_['flow_vs_disp']:+.3f}        "
              f"{s_['grad_vs_disp_in']:+.3f}        | {s_['early_final_disp']:+.3f}      "
              f"{s_['early_class_target']:+.3f}        ({s_['early_outward_class_target']:+.3f})"
              f"          {s_['early_grad_vs_final_disp_in']:+.3f}        | {s_['final_fit_loss']:.3f}")


if __name__ == "__main__":
    main()
