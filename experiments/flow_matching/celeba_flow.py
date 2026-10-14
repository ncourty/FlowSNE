"""Visualising a pretrained CelebA generative model's flow with FlowSNE (GPU
script; also runs on CPU/MPS at toy sizes for smoke tests).

Model: ``google/ddpm-celebahq-256`` (DDPM UNet2DModel, 113.7M parameters,
CelebA-HQ 256x256, Apache-2.0), the model the FinslerAffinity CelebA
experiment used, sampled deterministically with DDIM (eta = 0). Conventions
taken from the model's own files, not assumed (that project's lesson): the
scheduler is loaded from its ``scheduler_config.json`` (linear betas
1e-4..0.02, clip_sample true), so pixels live in [-1, 1] and are displayed as
(x + 1) / 2.

Time and flow. With N DDIM steps, the sampler visits x_0 (pure noise), x_1,
..., x_N (a face); time is s = k / N in [0, 1] (0 = noise, 1 = face). The
drift is, by default (--drift step), the sampler's own motion
    v(x_k) = (x_{k+1} - x_k) * N,
i.e. the deterministic DDIM displacement per unit of s -- the flow the
generator actually follows, read off diffusers' own scheduler step rather
than re-derived from the noise schedule by hand. --drift x0 instead uses
x0_hat(x_k) - x_k (the predicted clean image minus the current one), the
FinslerAffinity experiment's definition.

Sampling. By default every noise sample is integrated once and recorded at
all snapshot times (shared trajectories: N model calls per sample).
--independent instead draws fresh noise for each snapshot time, as the
FinslerAffinity experiment did (no trajectory links between times), at about
5x the cost. Images and velocities are stored downsampled by average pooling
to --store-res (128 by default): pooling is linear, so the stored velocity is
exactly the pooled image's velocity.

Pipeline (each stage saves before the next; generation resumes per batch):
  generate  trajectories, snapshots at --times points, sample grid and one
            trajectory strip to LOOK AT before trusting anything.
  embed     PCA of all snapshots (--n-pcs), velocities projected linearly into
            the same basis (exact); sparse row-stochastic Randers input
            affinity (KeOps on CUDA); FlowSNE (flow fitted jointly) and a
            symmetric baseline (same input affinity, zero flow); diagnostics.
  plot      embeddings coloured by s; on FlowSNE's, the divergence of the
            fitted field (red = source, blue = sink) and the mean drift per
            grid cell (flowsne.viz); and a version with image thumbnails.
  all       the three in order.

Diagnostics: forward-time bias of the input affinity (affinity-weighted mean
s of where a point sends mass, minus its own s; Randers vs symmetric), and
the cosine between the fitted field and the local gradient of s in the
embedding (does the flow point towards later s?).

Run:  python experiments/flow_matching/celeba_flow.py all --n-traj 1000 --times 11
      python experiments/flow_matching/celeba_flow.py all --n-traj 8 --times 5 --steps 10 \
          --batch 4 --fit-iters 200 --tag smoke      # quick CPU/MPS check
Writes: experiments/flow_matching/data/celeba/samples_<tag>/  (snapshots, sanity images)
        experiments/flow_matching/celeba_results/<tag>/      (embedding, figures, metrics)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

DATA = os.path.join(HERE, "data", "celeba")
MODEL_ID = "google/ddpm-celebahq-256"
IMG = 256


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32       # generation only; FlowSNE runs on CPU
    return torch.device("cpu"), torch.float32


def empty_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


# ===========================================================================
# Model
# ===========================================================================

def load_model(device, steps):
    from diffusers import DDIMScheduler, UNet2DModel
    model = UNet2DModel.from_pretrained(MODEL_ID).to(device).eval()
    scheduler = DDIMScheduler.from_pretrained(MODEL_ID)      # the model's own config
    scheduler.set_timesteps(steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {MODEL_ID}, {n_params / 1e6:.1f}M parameters, on {device}; DDIM {steps} "
          f"steps, clip_sample={scheduler.config.clip_sample}", flush=True)
    return model, scheduler


@torch.no_grad()
def ddim_step(model, scheduler, x, k):
    """One deterministic DDIM step from x_k (k = 0 is pure noise). Returns
    (x_{k+1}, x0_hat(x_k))."""
    ts = scheduler.timesteps[k]
    eps = model(x, ts).sample
    out = scheduler.step(eps, ts, x, eta=0.0)
    return out.prev_sample, out.pred_original_sample


def pool(x, res):
    return x if res == x.shape[-1] else torch.nn.functional.adaptive_avg_pool2d(x, res)


def snapshot_indices(times, steps):
    """Snapshot k's in [0, steps - 1] (the velocity at k needs x_{k+1})."""
    return np.unique(np.round(np.linspace(0, steps - 1, times)).astype(int))


# ===========================================================================
# 1. Generate
# ===========================================================================

@torch.no_grad()
def generate(args):
    device, dtype = device_and_dtype()
    model, scheduler = load_model(device, args.steps)
    out = os.path.join(DATA, f"samples_{args.tag}")
    os.makedirs(out, exist_ok=True)
    ks = snapshot_indices(args.times, args.steps)
    json.dump(dict(model=MODEL_ID, steps=args.steps, snapshot_k=ks.tolist(),
                   s=(ks / args.steps).tolist(), drift=args.drift, independent=args.independent,
                   store_res=args.store_res), open(os.path.join(out, "info.json"), "w"), indent=1)

    def record(x, k):
        x_next, x0 = ddim_step(model, scheduler, x, k)
        v = (x_next - x) * args.steps if args.drift == "step" else x0 - x
        return x_next, pool(x, args.store_res).half().cpu(), pool(v, args.store_res).half().cpu()

    t_start = time.time()
    n_batches = (args.n_traj + args.batch - 1) // args.batch
    for b in range(n_batches):
        path = os.path.join(out, f"batch{b:04d}.npz")
        if os.path.exists(path):
            continue
        t0 = time.time()
        bs = min(args.batch, args.n_traj - b * args.batch)
        xs, vs = [], []
        if not args.independent:              # one trajectory, recorded at every snapshot
            g = torch.Generator().manual_seed(10_000 + b)
            x = torch.randn(bs, 3, IMG, IMG, generator=g).to(device=device, dtype=dtype)
            for k in range(args.steps):
                if k in ks:
                    x, xp, vp = record(x, k)
                    xs.append(xp); vs.append(vp)
                else:
                    x, _ = ddim_step(model, scheduler, x, k)
            final = x
        else:                                 # fresh noise for every snapshot
            for j, kk in enumerate(ks):
                g = torch.Generator().manual_seed(10_000 + 1000 * b + j)
                x = torch.randn(bs, 3, IMG, IMG, generator=g).to(device=device, dtype=dtype)
                for k in range(kk):
                    x, _ = ddim_step(model, scheduler, x, k)
                x_next, xp, vp = record(x, kk)
                xs.append(xp); vs.append(vp)
                final = x_next
        np.savez(path, x=torch.stack(xs).numpy(), v=torch.stack(vs).numpy(),
                 s=(ks / args.steps).astype(np.float32))
        if b == 0:
            np.save(os.path.join(out, "final_batch0.npy"), final[:16].half().cpu().numpy())
        empty_cache(device)
        print(f"  batch {b + 1}/{n_batches}: {bs} trajectories, {time.time() - t0:.0f}s "
              f"(total {time.time() - t_start:.0f}s); snapshot pixel range "
              f"[{float(min(t.float().min() for t in xs)):.2f}, "
              f"{float(max(t.float().max() for t in xs)):.2f}]", flush=True)
    sanity_images(model, scheduler, args, device, dtype, out)


@torch.no_grad()
def sanity_images(model, scheduler, args, device, dtype, out):
    """A grid of finished samples and one trajectory across s, to look at
    BEFORE trusting the pipeline (the FinslerAffinity CelebA experiment's bugs
    -- pixel range, layout -- were only caught by looking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = lambda x: ((torch.as_tensor(x).float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    final = np.load(os.path.join(out, "final_batch0.npy"))
    m = min(16, len(final))
    cols = min(4, m)
    rows = (m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img in zip(axes.flat, final[:m]):
        ax.imshow(show(img))
    fig.suptitle(f"finished samples (DDIM {args.steps} steps, {IMG}x{IMG})", fontsize=10)
    fig.savefig(os.path.join(out, "samples_final.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 3, IMG, IMG, generator=g).to(device=device, dtype=dtype)
    snaps = []
    every = max(1, args.steps // 8)
    for k in range(args.steps):
        if k % every == 0:
            snaps.append((k / args.steps, x[0].cpu()))
        x, _ = ddim_step(model, scheduler, x, k)
    snaps.append((1.0, x[0].cpu()))
    fig, axes = plt.subplots(1, len(snaps), figsize=(2 * len(snaps), 2.2))
    for ax, (s, img) in zip(axes, snaps):
        ax.imshow(show(img)); ax.set_title(f"s={s:.2f}", fontsize=8); ax.axis("off")
    fig.savefig(os.path.join(out, "trajectory.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  sanity images: {out}/samples_final.png, trajectory.png  -- look at them", flush=True)


# ===========================================================================
# 2. Embed
# ===========================================================================

def load_samples(tag):
    """All snapshots: x, v (n, 3, R, R) float16, s (n,), trajectory id (n,)."""
    out = os.path.join(DATA, f"samples_{tag}")
    files = sorted(f for f in os.listdir(out) if f.startswith("batch") and f.endswith(".npz"))
    xs, vs, ss, traj = [], [], [], []
    offset = 0
    for f in files:
        d = np.load(os.path.join(out, f))
        K, B = d["x"].shape[:2]
        xs.append(d["x"].reshape(K * B, *d["x"].shape[2:]))
        vs.append(d["v"].reshape(K * B, *d["v"].shape[2:]))
        ss.append(np.repeat(d["s"], B))
        traj.append(np.tile(np.arange(offset, offset + B), K))
        offset += B
    return (np.concatenate(xs), np.concatenate(vs), np.concatenate(ss).astype(np.float32),
            np.concatenate(traj))


def forward_time_bias(P, idx, t):
    """Affinity-weighted mean t of where each point sends mass, minus its own t."""
    if idx is None:
        return ((P / P.sum(1, keepdim=True)) @ t - t)
    w = P / P.sum(1, keepdim=True)
    return (w * t[idx]).sum(1) - t


def embed(args):
    from flowsne import EntropicRandersAffinity, RandersAffinity, fit_embedding
    from flowsne.flows import input_std_norm_flow, mlp_flow

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")          # KeOps/FlowSNE: CUDA or CPU
    dtype = torch.float32 if cuda else torch.float64
    backend = "keops" if cuda else None
    res = os.path.join(HERE, "celeba_results", args.tag)
    os.makedirs(res, exist_ok=True)

    x, v, t_np, _ = load_samples(args.tag)
    n = len(x)
    X = torch.tensor(x.reshape(n, -1), dtype=torch.float32, device=device)
    V = torch.tensor(v.reshape(n, -1), dtype=torch.float32, device=device)
    t0 = time.time()
    mean = X.mean(0)
    q = min(args.n_pcs, n - 1)
    _, S, Vh = torch.pca_lowrank(X - mean, q=q, center=False, niter=4)
    Xp = ((X - mean) @ Vh).to(dtype)
    Vp = (V @ Vh).to(dtype)                                   # linear map: exact projection
    total_var = (X - mean).pow(2).sum()
    explained = (S ** 2).sum() / total_var
    vel_kept = (Vp.double().norm(dim=1) / V.double().norm(dim=1)).mean().item()
    del X, V
    print(f"PCA: {n} points x {x[0].size} dims -> {q} PCs, {explained:.1%} of variance, "
          f"mean fraction of velocity norm kept {vel_kept:.1%}  ({time.time() - t0:.0f}s)", flush=True)

    t = torch.tensor(t_np, dtype=dtype, device=device)
    perp = args.perplexity if args.perplexity == "auto" else float(args.perplexity)
    kw = dict(doubly_stochastic=False, sparsity=True, perplexity=perp, backend=backend)
    P, idx = EntropicRandersAffinity(omega=Vp, **kw)(Xp)
    P0, idx0 = EntropicRandersAffinity(omega=None, **kw)(Xp)
    fb, fb0 = forward_time_bias(P, idx, t), forward_time_bias(P0, idx0, t)
    print(f"input forward-time bias: Randers {fb.mean():+.4f} vs symmetric {fb0.mean():+.4f} "
          f"(win rate {(fb > fb0).float().mean():.3f})", flush=True)

    init = Xp[:, :2]
    results = dict(n=n, n_pcs=q, pca_explained=float(explained), velocity_norm_kept=vel_kept,
                   input_ftb_randers=float(fb.mean()), input_ftb_symmetric=float(fb0.mean()),
                   input_ftb_win_rate=float((fb > fb0).float().mean()))
    fits = {}
    for name, flow, train_flow in (("symmetric", mlp_flow(2), False),
                                   ("flowsne", input_std_norm_flow(2, hidden_dim=32), True)):
        torch.manual_seed(args.seed)
        aff = RandersAffinity(n_components=2, flow=flow, backend=backend).to(device=device,
                                                                            dtype=dtype)
        t1 = time.time()
        Z = fit_embedding(aff, P, init, indices=idx, max_iter=args.fit_iters, lr=args.lr,
                          seed=args.seed, train_flow=train_flow)
        with torch.no_grad():
            loss = aff.cross_entropy(Z, P, idx).item()
            F = -aff.omega_at(Z)
        fits[name] = (Z, F, aff)
        results[f"{name}_loss"] = loss
        results[f"{name}_secs"] = round(time.time() - t1, 1)
        print(f"  {name:9s} loss {loss:.4f}  ({results[f'{name}_secs']:.0f}s)", flush=True)

    Z, F, aff = fits["flowsne"]
    g = local_gradient(Z.cpu().double().numpy(), t_np)
    Fn = F.cpu().double().numpy()
    den = np.linalg.norm(Fn, axis=1) * np.linalg.norm(g, axis=1)
    cos = np.where(den > 0, (Fn * g).sum(1) / np.where(den > 0, den, 1), 0)
    results["flow_vs_time_gradient"] = float(cos.mean())
    results["flow_vs_time_gradient_by_t"] = {f"{tt:.2f}": float(cos[t_np == tt].mean())
                                             for tt in np.unique(t_np)}
    print(f"fitted field vs local gradient of t in the embedding: cos {cos.mean():+.3f}", flush=True)

    np.savez_compressed(
        os.path.join(res, "embedding.npz"), t=t_np,
        Z_symmetric=fits["symmetric"][0].cpu().float().numpy(),
        Z_flowsne=Z.cpu().float().numpy(), F_flowsne=Fn.astype(np.float32),
        flow_params=torch.nn.utils.parameters_to_vector(aff.flow.parameters()).detach().cpu().double().numpy(),
        z_std=np.float64(Z.std().item()), flow_hidden_dim=32)
    json.dump(dict(args=vars(args), **results), open(os.path.join(res, "metrics.json"), "w"),
              indent=1)
    print(f"saved {res}/embedding.npz, metrics.json", flush=True)


def local_gradient(Z, f, k=15):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
    g = np.zeros_like(Z)
    for i in range(len(Z)):
        dZ = Z[idx[i]] - Z[i]
        g[i] = np.linalg.lstsq(dZ, f[idx[i]] - f[i], rcond=None)[0]
    return g


# ===========================================================================
# 3. Plot
# ===========================================================================

def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from flowsne.viz import default_bins, divergence_map, field_from_flow, grid_arrows

    res = os.path.join(HERE, "celeba_results", args.tag)
    E = np.load(os.path.join(res, "embedding.npz"))
    M = json.load(open(os.path.join(res, "metrics.json")))
    t = E["t"]
    n = len(t)
    size = 6 if n <= 2000 else max(0.3, 8000 / n)

    def axes_of(Z):
        lo, hi = np.percentile(Z, [0.5, 99.5], axis=0)
        pad = 0.06 * (hi - lo)
        return lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, key, title in ((axes[0], "Z_symmetric", f"symmetric  (loss {M['symmetric_loss']:.3f})"),
                           (axes[1], "Z_flowsne", f"FlowSNE  (loss {M['flowsne_loss']:.3f}; field vs "
                                                  f"time gradient {M['flow_vs_time_gradient']:+.2f})")):
        Z = E[key]
        lo, hi = axes_of(Z)
        if key == "Z_flowsne":
            # divergence of the fitted field (red = source, blue = sink) behind
            # the points, and the mean drift per grid cell on top
            field = field_from_flow(E["flow_params"], E["z_std"], int(E["flow_hidden_dim"]))
            divergence_map(ax, field, Z, lo=lo, hi=hi)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=size, linewidths=0, alpha=0.7,
                        zorder=1)
        if key == "Z_flowsne":
            grid_arrows(ax, Z, E["F_flowsne"], bins=default_bins(n), lo=lo, hi=hi, zorder=3)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=axes, shrink=0.75, label="s  (0 = noise, 1 = face)")
    fig.suptitle(f"CelebA-HQ DDPM ({MODEL_ID}, DDIM flow), n = {n}, input forward-time bias "
                 f"Randers {M['input_ftb_randers']:+.3f} vs symmetric {M['input_ftb_symmetric']:+.3f}",
                 fontsize=11)
    fig.savefig(os.path.join(res, "embedding.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # thumbnails on the FlowSNE embedding
    x, _, _, _ = load_samples(args.tag)
    Z = E["Z_flowsne"]
    lo, hi = axes_of(Z)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=size, linewidths=0, alpha=0.5)
    rng = np.random.default_rng(0)
    for i in rng.choice(n, size=min(n, args.thumbnails), replace=False):
        img = ((x[i].astype(np.float32).clip(-1, 1) + 1) / 2).transpose(1, 2, 0)
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.28 * 128 / img.shape[0]), Z[i],
                                     frameon=False))
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title("FlowSNE embedding with sample thumbnails", fontsize=11)
    fig.savefig(os.path.join(res, "embedding_thumbnails.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"figures: {res}/embedding.png, embedding_thumbnails.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "embed", "plot", "all"])
    ap.add_argument("--tag", default="main", help="names the sample/result directories")
    ap.add_argument("--n-traj", type=int, default=1000,
                    help="trajectories (shared mode) or samples per snapshot (--independent)")
    ap.add_argument("--times", type=int, default=11, help="snapshot times in [0, 1)")
    ap.add_argument("--steps", type=int, default=50, help="DDIM steps")
    ap.add_argument("--drift", choices=["step", "x0"], default="step",
                    help="step: the DDIM displacement per unit time; x0: x0_hat - x")
    ap.add_argument("--independent", action="store_true",
                    help="fresh noise per snapshot time (no trajectory links), ~5x the cost")
    ap.add_argument("--store-res", type=int, default=128, help="stored resolution (avg pooling)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-pcs", type=int, default=50)
    ap.add_argument("--perplexity", default="auto")
    ap.add_argument("--fit-iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--thumbnails", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd in ("generate", "all"):
        generate(a)
    if a.cmd in ("embed", "all"):
        embed(a)
    if a.cmd in ("plot", "all"):
        plot(a)
