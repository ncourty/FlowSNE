"""Visualising a pretrained CelebA flow-matching model's velocity field with
FlowSNE (GPU script; also runs on CPU/MPS at toy sizes for smoke tests).

Model: the OT flow-matching UNet on CelebA 128x128 from PnP-Flow (Martin et
al., ICLR 2025, github.com/annegnx/PnP-Flow), checked in their source rather
than assumed:
  * architecture: their ``pnpflow.models.UNet(input_channels=3,
    input_height=128, ch=32, ch_mult=(1,2,4,8), num_res_blocks=6,
    attn_resolutions=(16,8), resamp_with_conv=True)``, called ``model(x, t)``
    with t of shape (B,);
  * convention: x_t = t * x1 + (1 - t) * x0 with x0 ~ N(0, I) (noise) and x1
    data, and the network predicts x1 - x0: t = 0 is NOISE, t = 1 is DATA,
    and the velocity points towards data;
  * pixels in [-1, 1] (CenterCrop 178 -> Resize 128 -> Normalize(0.5, 0.5));
    displayed as (x + 1) / 2.
Their repository has no licence file, so their model code is NOT copied here:
``models.py`` is downloaded at run time from a pinned commit, and the
checkpoint from their Google Drive link (via gdown), into data/celeba/.

Pipeline (each stage saves before the next, and generation resumes per time):
  generate  for each of --times evenly spaced t in [0, 1], --n-per-t fresh
            noise samples integrated with Euler (--steps over [0, 1]) up to t;
            stores x_t and v(x_t, t) per time (float16), plus a sample grid
            and one trajectory strip to LOOK AT before trusting anything.
  embed     PCA of all x_t (--n-pcs), velocities projected linearly into the
            same basis (exact); sparse row-stochastic Randers input affinity
            (KeOps on CUDA); FlowSNE (flow fitted jointly) and a symmetric
            baseline (same input affinity, zero flow); diagnostics.
  plot      embeddings coloured by t, with streamlines of the fitted field,
            and a version with image thumbnails.
  all       the three in order.

Diagnostics: forward-time bias of the input affinity (affinity-weighted mean
t of where a point sends mass, minus its own t; Randers vs symmetric), and
the cosine between the fitted field and the local gradient of t in the
embedding (does the flow point towards later t?).

Run:  python experiments/flow_matching/celeba_flow.py all --n-per-t 1000 --times 11
      python experiments/flow_matching/celeba_flow.py all --n-per-t 16 --times 5 --steps 20 \
          --fit-iters 200 --tag smoke        # quick CPU/MPS check
Writes: experiments/flow_matching/data/celeba/  (model code, checkpoint, samples)
        experiments/flow_matching/celeba_results/<tag>/  (embedding, figures, metrics)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

DATA = os.path.join(HERE, "data", "celeba")
PNPFLOW_COMMIT = "ee5b159e6edff4d585ebf36c9a5333ffa85e066c"
PNPFLOW_MODELS = (f"https://raw.githubusercontent.com/annegnx/PnP-Flow/{PNPFLOW_COMMIT}/"
                  "pnpflow/models.py")
CKPT_DRIVE_ID = "1ZZ6S-PGRx-tOPkr4Gt3A6RN-PChabnD6"
IMG = 128


def device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32       # generation only; FlowSNE runs on CPU
    return torch.device("cpu"), torch.float32


# ===========================================================================
# Model
# ===========================================================================

def load_model(device):
    os.makedirs(DATA, exist_ok=True)
    code = os.path.join(DATA, f"pnpflow_models_{PNPFLOW_COMMIT[:8]}.py")
    if not os.path.exists(code):
        print(f"downloading PnP-Flow model code (commit {PNPFLOW_COMMIT[:8]})", flush=True)
        urllib.request.urlretrieve(PNPFLOW_MODELS, code)
    spec = importlib.util.spec_from_file_location("pnpflow_models", code)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpt = os.path.join(DATA, "pnpflow_celeba_ot_model_final.pt")
    if not os.path.exists(ckpt):
        import gdown
        print("downloading PnP-Flow CelebA checkpoint", flush=True)
        gdown.download(f"https://drive.google.com/uc?id={CKPT_DRIVE_ID}", ckpt, quiet=False)
    model = mod.UNet(input_channels=3, input_height=IMG, ch=32, ch_mult=(1, 2, 4, 8),
                     num_res_blocks=6, attn_resolutions=(16, 8), resamp_with_conv=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: PnP-Flow OT UNet, {n_params / 1e6:.1f}M parameters, on {device}", flush=True)
    return model


@torch.no_grad()
def velocity(model, x, t):
    return model(x, torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype))


# ===========================================================================
# 1. Generate
# ===========================================================================

@torch.no_grad()
def generate(args):
    device, dtype = device_and_dtype()
    model = load_model(device)
    out = os.path.join(DATA, f"samples_{args.tag}")
    os.makedirs(out, exist_ok=True)
    times = np.linspace(0.0, 1.0, args.times)
    for k, t in enumerate(times):
        path = os.path.join(out, f"t{k:02d}.npz")
        if os.path.exists(path):
            print(f"  t={t:.2f}: exists, skipping", flush=True)
            continue
        t0 = time.time()
        g = torch.Generator().manual_seed(1000 + k)               # fresh noise per time
        n_steps = int(round(t * args.steps))
        xs, vs = [], []
        for s in range(0, args.n_per_t, args.batch):
            b = min(args.batch, args.n_per_t - s)
            x = torch.randn(b, 3, IMG, IMG, generator=g).to(device=device, dtype=dtype)
            for i in range(n_steps):                               # Euler from 0 to t
                x = x + (t / n_steps) * velocity(model, x, i * t / n_steps)
            xs.append(x.half().cpu())
            vs.append(velocity(model, x, t).half().cpu())
        x, v = torch.cat(xs), torch.cat(vs)
        np.savez(path, x=x.numpy(), v=v.numpy(), t=np.float32(t))
        print(f"  t={t:.2f}: {args.n_per_t} samples, {n_steps} Euler steps, pixel range "
              f"[{x.float().min():.2f}, {x.float().max():.2f}], mean |v| "
              f"{v.float().flatten(1).norm(dim=1).mean():.1f}  ({time.time() - t0:.0f}s)",
              flush=True)
    sanity_images(model, args, device, dtype, out)


@torch.no_grad()
def sanity_images(model, args, device, dtype, out):
    """A 4x4 grid of finished samples and one trajectory across t, to look at
    BEFORE trusting the pipeline (the last CelebA attempt's bugs -- pixel range,
    layout -- were only caught by looking)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = lambda x: ((x.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    last = np.load(os.path.join(out, f"t{args.times - 1:02d}.npz"))
    x1 = torch.tensor(last["x"][:16])
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img in zip(axes.flat, x1):
        ax.imshow(show(img)); ax.axis("off")
    fig.suptitle(f"t = 1 samples ({args.steps} Euler steps)", fontsize=10)
    fig.savefig(os.path.join(out, "samples_t1.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 3, IMG, IMG, generator=g).to(device=device, dtype=dtype)
    snaps, n = [], args.steps
    for i in range(n + 1):
        if i % max(1, n // 8) == 0 or i == n:
            snaps.append((i / n, x[0].clone()))
        if i < n:
            x = x + (1.0 / n) * velocity(model, x, i / n)
    fig, axes = plt.subplots(1, len(snaps), figsize=(2 * len(snaps), 2.2))
    for ax, (t, img) in zip(axes, snaps):
        ax.imshow(show(img)); ax.set_title(f"t={t:.2f}", fontsize=8); ax.axis("off")
    fig.savefig(os.path.join(out, "trajectory.png"), dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"  sanity images: {out}/samples_t1.png, trajectory.png  -- look at them", flush=True)


# ===========================================================================
# 2. Embed
# ===========================================================================

def load_samples(tag):
    out = os.path.join(DATA, f"samples_{tag}")
    files = sorted(f for f in os.listdir(out) if f.startswith("t") and f.endswith(".npz"))
    xs, vs, ts = [], [], []
    for f in files:
        d = np.load(os.path.join(out, f))
        xs.append(d["x"]); vs.append(d["v"]); ts.append(np.full(len(d["x"]), float(d["t"])))
    return np.concatenate(xs), np.concatenate(vs), np.concatenate(ts).astype(np.float32)


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

    x, v, t_np = load_samples(args.tag)
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
    from matplotlib.colors import Normalize
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from scipy.ndimage import maximum_filter
    from flowsne.distances import bounded_direction
    from flowsne.flows import input_std_norm_flow

    res = os.path.join(HERE, "celeba_results", args.tag)
    E = np.load(os.path.join(res, "embedding.npz"))
    M = json.load(open(os.path.join(res, "metrics.json")))
    t = E["t"]
    n = len(t)
    size = 6 if n <= 2000 else max(0.3, 8000 / n)

    def field_on_grid(Z, lo, hi, res_=70):
        flow = input_std_norm_flow(2, hidden_dim=int(E["flow_hidden_dim"])).double()
        torch.nn.utils.vector_to_parameters(torch.tensor(E["flow_params"]), flow.parameters())
        xs, ys = np.linspace(lo[0], hi[0], res_), np.linspace(lo[1], hi[1], res_)
        GX, GY = np.meshgrid(xs, ys)
        G = torch.tensor(np.stack([GX.ravel(), GY.ravel()], 1))
        with torch.no_grad():                              # InputStdNorm uses Z's std
            Fg = -bounded_direction(flow[1](G / float(E["z_std"])), 0.95).numpy()
        cnt, _, _ = np.histogram2d(Z[:, 1], Z[:, 0], bins=res_, range=[[lo[1], hi[1]], [lo[0], hi[0]]])
        mask = ~(maximum_filter(cnt, size=5) > 0)
        U = np.ma.array(Fg[:, 0].reshape(res_, res_), mask=mask)
        V = np.ma.array(Fg[:, 1].reshape(res_, res_), mask=mask)
        return xs, ys, U, V

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
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=size, linewidths=0, alpha=0.8)
        if key == "Z_flowsne":
            xs, ys, U, V = field_on_grid(Z, lo, hi)
            sp = np.ma.array(np.hypot(U.data, V.data), mask=U.mask)
            rel = (sp / max(sp.max(), 1e-12)).filled(0.0)
            ax.streamplot(xs, ys, U, V, density=1.3, color=rel, cmap="Greys",
                          norm=Normalize(-0.15, 1.0, clip=True), linewidth=2.2 * rel, arrowsize=0.7)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_title(title, fontsize=10)
    fig.colorbar(sc, ax=axes, shrink=0.75, label="t  (0 = noise, 1 = face)")
    fig.suptitle(f"CelebA flow matching (PnP-Flow OT model), n = {n}, input forward-time bias "
                 f"Randers {M['input_ftb_randers']:+.3f} vs symmetric {M['input_ftb_symmetric']:+.3f}",
                 fontsize=11)
    fig.savefig(os.path.join(res, "embedding.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    # thumbnails on the FlowSNE embedding
    x, _, t_s = load_samples(args.tag)
    Z = E["Z_flowsne"]
    lo, hi = axes_of(Z)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(Z[:, 0], Z[:, 1], c=t, cmap="viridis", s=size, linewidths=0, alpha=0.5)
    rng = np.random.default_rng(0)
    for i in rng.choice(n, size=min(n, args.thumbnails), replace=False):
        img = ((x[i].astype(np.float32).clip(-1, 1) + 1) / 2).transpose(1, 2, 0)
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.28), Z[i], frameon=False))
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
    ap.add_argument("--n-per-t", type=int, default=1000)
    ap.add_argument("--times", type=int, default=11, help="evenly spaced t in [0, 1]")
    ap.add_argument("--steps", type=int, default=100, help="Euler steps over [0, 1]")
    ap.add_argument("--batch", type=int, default=64)
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
