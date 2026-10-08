"""GPU benchmark for FlowSNE's dense vs KeOps backends.

What it does, in order:
  1. Environment report: torch / CUDA / GPU, pykeops (and whether it can use
     the GPU), TorchDR. Stops if CUDA is unavailable (unless --allow-cpu).
  2. Correctness on this device: dense vs KeOps loss and gradients on a small
     problem, float32 and float64 (must agree before timings mean anything).
  3. Scaling: for each n, build the sparse input affinity (KeOps kNN), then
     time fit steps (loss + backward + Adam) for each output backend, with
     peak GPU memory. Dense runs are skipped when they cannot fit in GPU
     memory; out-of-memory errors are caught and recorded per configuration.
  4. Optional end-to-end fit at one size (--fit-n): loss and how well the
     fitted field recovers the true drift speed.

Data: points on the unit sphere with the rotation drift e_z x x (speed =
sin(colatitude)), embedded isometrically in --dim dimensions (random
orthonormal frame) plus small isotropic noise, so the input-side kNN search
runs in a realistic dimension.

Setup on the GPU machine (from the repository root):
    pip install torch torchdr==0.4 pykeops numpy   # torch built for your CUDA
    python benchmarks/gpu_benchmark.py                       # defaults below
    python benchmarks/gpu_benchmark.py --sizes 10000 50000 100000 --fit-n 50000

Writes benchmarks/results/gpu_benchmark_<hostname>_<timestamp>.json and
prints a summary table. Send back the JSON.

Notes:
  * Default --perplexity 50 (k = 5 x 50 = 250 neighbours). FlowSNE's own
    default is "auto" = sqrt(n), i.e. k = 5 sqrt(n) (1581 at n = 100k);
    KeOps' argKmin gets slow for very large k, so "auto" is available but
    not the default here.
  * float32 by default: what GPUs are fast at. --dtype float64 also works.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


# ===========================================================================
# Environment
# ===========================================================================

def keops_gpu_check():
    """Run a tiny KeOps reduction on the GPU and compare it with torch.
    ``pykeops.config.gpu_available`` only says the CUDA libraries were found;
    this catches a KeOps engine that failed to build (e.g. compiled against
    an older CUDA toolkit's nvrtc.h than the runtime it detected)."""
    from pykeops.torch import LazyTensor
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1000, 3, device="cuda", generator=g)
    ref = torch.exp(-torch.cdist(x, x) ** 2).sum(1)
    xi, xj = LazyTensor(x[:, None, :]), LazyTensor(x[None, :, :])
    got = (-((xi - xj) ** 2).sum(-1)).exp().sum(dim=1).squeeze(1)
    if got.device.type != "cuda":
        raise RuntimeError(f"KeOps returned a result on {got.device}, not on the GPU")
    return ((got - ref).abs().max() / ref.abs().max()).item()


def environment(allow_cpu: bool) -> dict:
    env = dict(host=socket.gethostname(), platform=platform.platform(),
               python=sys.version.split()[0], torch=torch.__version__,
               cuda_available=torch.cuda.is_available(),
               torch_cuda=torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 1e9, 2),
                   gpu_count=torch.cuda.device_count())
    try:
        import pykeops
        pykeops.set_verbose(False)            # silence per-formula compile logs
        env["pykeops"] = pykeops.__version__
        try:
            import pykeops.config as kc
            env["pykeops_gpu_available"] = bool(getattr(kc, "gpu_available", False))
        except Exception as e:  # noqa: BLE001
            env["pykeops_gpu_available"] = f"unknown ({e!r})"
    except ImportError:
        env["pykeops"] = None
    try:
        import torchdr
        env["torchdr"] = torchdr.__version__
    except ImportError:
        env["torchdr"] = None

    print("== Environment")
    for k, v in env.items():
        print(f"   {k:22s} {v}")
    problems = []
    if not env["cuda_available"] and not allow_cpu:
        problems.append("CUDA is not available to torch (use --allow-cpu to run anyway).")
    if env["pykeops"] is None:
        problems.append("pykeops is not installed (pip install pykeops).")
    if env["torchdr"] is None:
        problems.append("torchdr is not installed (pip install torchdr==0.4).")
    if env["cuda_available"] and env.get("pykeops_gpu_available") is False:
        problems.append("pykeops does not see the GPU (CUDA libraries / nvrtc not found): KeOps "
                        "would silently run on CPU. Check your CUDA toolkit install.")
    if env["cuda_available"] and env["pykeops"] and env.get("pykeops_gpu_available") is True:
        try:
            env["keops_gpu_check_rel_err"] = keops_gpu_check()
            if not env["keops_gpu_check_rel_err"] < 1e-4:
                raise RuntimeError(f"relative error {env['keops_gpu_check_rel_err']:.1e}")
        except Exception as e:  # noqa: BLE001
            env["keops_gpu_check_rel_err"] = None
            problems.append(
                f"KeOps failed a test reduction on the GPU ({e!r}). Most likely its engine was "
                "compiled against CUDA headers older than the runtime it detected (e.g. errors "
                "about nvrtcGetCUBIN). Point it at headers matching torch's CUDA, e.g. "
                "`conda install -c nvidia cuda-nvrtc-dev=<ver> cuda-cudart-dev=<ver>` and "
                "`export CUDA_PATH=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX`, then "
                "`python -c \"import pykeops; pykeops.clean_pykeops()\"` and re-run.")
    if problems:
        print("\n".join("!! " + p for p in problems))
        sys.exit(1)
    return env


# ===========================================================================
# Data
# ===========================================================================

def sphere_vortex(n: int, dim: int, seed: int, device, dtype):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    x3 = v / v.norm(dim=1, keepdim=True)
    w3 = torch.stack([-x3[:, 1], x3[:, 0], torch.zeros(n, dtype=torch.float64)], 1)
    speed = w3.norm(dim=1)
    if dim > 3:
        Q, _ = torch.linalg.qr(torch.randn(dim, 3, generator=g, dtype=torch.float64))
        X = x3 @ Q.T + 0.01 * torch.randn(n, dim, generator=g, dtype=torch.float64)
        W = w3 @ Q.T
    else:
        X, W = x3, w3
    return (X.to(device=device, dtype=dtype), W.to(device=device, dtype=dtype),
            speed.numpy(), x3[:, :2].numpy())


# ===========================================================================
# Helpers
# ===========================================================================

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb(device):
    return round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def make_output(backend, dtype, device, seed=0):
    from flowsne import RandersAffinity, input_std_norm_flow
    torch.manual_seed(seed)
    return RandersAffinity(n_components=2, flow=input_std_norm_flow(2, hidden_dim=32),
                           backend=backend).to(device=device, dtype=dtype)


# ===========================================================================
# 2. Correctness on this device
# ===========================================================================

def correctness(device, n=2000, dim=10) -> list:
    from flowsne import EntropicRandersAffinity
    print(f"\n== Correctness on {device} (n={n}): dense vs KeOps, loss and gradients")
    rows = []
    for dtype in (torch.float32, torch.float64):
        X, W, _, _ = sphere_vortex(n, dim, 0, device, dtype)
        aff = EntropicRandersAffinity(omega=W, doubly_stochastic=False, sparsity=True,
                                      perplexity=30, backend="keops")
        P, idx = aff(X)
        P0, idx0 = EntropicRandersAffinity(omega=W, doubly_stochastic=False, sparsity=True,
                                           perplexity=30)(X)
        # fraction of shared neighbours (float32 near-ties at the k-th
        # position may legitimately differ between backends)
        knn_agree = (idx.sort(1).values == idx0.sort(1).values).double().mean().item()
        Z = torch.randn(n, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
        Z = Z.to(device=device, dtype=dtype)
        out = {}
        for backend in (None, "keops"):
            a = make_output(backend, dtype, device, seed=0)
            # non-zero, deterministic last layer so the drift terms are exercised
            last = [m for m in a.flow.modules() if isinstance(m, torch.nn.Linear)][-1]
            with torch.no_grad():
                last.weight.copy_(torch.linspace(-0.3, 0.3, last.weight.numel(),
                                                 device=device, dtype=dtype).reshape(last.weight.shape))
            Zg = Z.clone().requires_grad_()
            loss = a.cross_entropy(Zg, P, idx)
            loss.backward()
            out[backend] = (loss.item(), Zg.grad.detach().clone(),
                            torch.cat([q.grad.reshape(-1) for q in a.parameters()]))
        (l0, gz0, gp0), (l1, gz1, gp1) = out[None], out["keops"]
        r = dict(dtype=str(dtype).split(".")[-1], knn_agreement=knn_agree,
                 loss_rel_diff=abs(l1 - l0) / abs(l0),
                 grad_z_rel_diff=((gz1 - gz0).norm() / gz0.norm()).item(),
                 grad_theta_rel_diff=((gp1 - gp0).norm() / gp0.norm()).item())
        tol = 1e-4 if dtype == torch.float32 else 1e-9
        r["ok"] = bool(r["loss_rel_diff"] < tol and r["grad_z_rel_diff"] < tol * 10
                       and r["grad_theta_rel_diff"] < tol * 10)
        rows.append(r)
        print(f"   {r['dtype']:8s} kNN agreement {r['knn_agreement']:.4f}  loss {r['loss_rel_diff']:.1e}  "
              f"grad Z {r['grad_z_rel_diff']:.1e}  grad θ {r['grad_theta_rel_diff']:.1e}  "
              f"-> {'OK' if r['ok'] else 'MISMATCH'}")
    return rows


# ===========================================================================
# 3. Scaling
# ===========================================================================

def bench_size(n, args, device, dtype, gpu_mem_gb) -> list:
    from flowsne import EntropicRandersAffinity
    rows = []
    X, W, _, init2 = sphere_vortex(n, args.dim, 0, device, dtype)
    perp = args.perplexity if args.perplexity == "auto" else float(args.perplexity)

    # input affinity (sparse, KeOps kNN)
    reset_peak(device)
    try:
        sync(device); t = time.perf_counter()
        aff = EntropicRandersAffinity(omega=W, doubly_stochastic=args.balanced, sparsity=True,
                                      perplexity=perp, backend="keops")
        P, idx = aff(X)
        sync(device); t_aff = time.perf_counter() - t
        aff_row = dict(n=n, stage="input_affinity", backend="keops",
                       balanced=args.balanced, k=int(idx.shape[1]), seconds=round(t_aff, 3),
                       peak_gb=peak_gb(device),
                       marginal_error=getattr(aff, "marginal_error_", None))
    except Exception as e:  # noqa: BLE001
        aff_row = dict(n=n, stage="input_affinity", backend="keops",
                       error="OOM" if is_oom(e) else repr(e)[:300])
        rows.append(aff_row)
        print(f"   n={n}: input affinity FAILED ({aff_row['error']})")
        return rows
    rows.append(aff_row)
    print(f"   n={n:7d}  input affinity (k={aff_row['k']}): {aff_row['seconds']:.2f}s  "
          f"peak {aff_row['peak_gb']} GB")

    Z0 = torch.tensor(init2, dtype=dtype, device=device)
    Z0 = Z0 / Z0[:, 0].std()
    bytes_per = torch.finfo(dtype).bits // 8
    for backend in ("dense", "keops"):
        est_gb = 6 * n * n * bytes_per / 1e9     # cost, kernel, autograd buffers
        if backend == "dense" and gpu_mem_gb is not None and est_gb > 0.8 * gpu_mem_gb:
            rows.append(dict(n=n, stage="fit_step", backend=backend,
                             skipped=f"needs ~{est_gb:.0f} GB > 80% of {gpu_mem_gb} GB"))
            print(f"   n={n:7d}  {backend:6s} skipped (needs ~{est_gb:.0f} GB)")
            continue
        reset_peak(device)
        try:
            a = make_output(None if backend == "dense" else "keops", dtype, device)
            Z = Z0.clone().requires_grad_()
            opt = torch.optim.Adam([{"params": [Z]}, {"params": list(a.parameters())}], lr=0.03)
            times = []
            for s in range(args.warmup + args.steps):
                sync(device); t = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = a.cross_entropy(Z, P, idx)
                loss.backward()
                opt.step()
                sync(device)
                if s >= args.warmup:
                    times.append(time.perf_counter() - t)
            row = dict(n=n, stage="fit_step", backend=backend, k=int(idx.shape[1]),
                       seconds_per_step=round(float(np.median(times)), 4),
                       seconds_per_step_min=round(float(np.min(times)), 4),
                       peak_gb=peak_gb(device), loss=float(loss.item()))
            print(f"   n={n:7d}  {backend:6s} {row['seconds_per_step']:.4f} s/step  "
                  f"peak {row['peak_gb']} GB")
        except Exception as e:  # noqa: BLE001
            row = dict(n=n, stage="fit_step", backend=backend,
                       error="OOM" if is_oom(e) else repr(e)[:300])
            print(f"   n={n:7d}  {backend:6s} FAILED ({row['error']})")
        rows.append(row)
        del Z
        reset_peak(device)
    return rows


# ===========================================================================
# 4. End-to-end fit
# ===========================================================================

def end_to_end(n, args, device, dtype) -> dict:
    from flowsne import EntropicRandersAffinity, fit_embedding
    print(f"\n== End-to-end fit, n={n}, {args.fit_iters} iterations, KeOps")
    X, W, speed, init2 = sphere_vortex(n, args.dim, 0, device, dtype)
    perp = args.perplexity if args.perplexity == "auto" else float(args.perplexity)
    sync(device); t = time.perf_counter()
    P, idx = EntropicRandersAffinity(omega=W, doubly_stochastic=args.balanced, sparsity=True,
                                     perplexity=perp, backend="keops")(X)
    a = make_output("keops", dtype, device)
    Z = fit_embedding(a, P, torch.tensor(init2, dtype=dtype, device=device), indices=idx,
                      max_iter=args.fit_iters, lr=0.03)
    sync(device); secs = time.perf_counter() - t
    with torch.no_grad():
        loss = a.cross_entropy(Z, P, idx).item()
        field = -a.omega_at(Z)
    corr = float(np.corrcoef(field.norm(dim=1).cpu().double().numpy(), speed)[0, 1])
    r = dict(n=n, iters=args.fit_iters, seconds=round(secs, 1), loss=loss,
             corr_field_norm_vs_true_speed=corr)
    print(f"   {secs:.1f}s total, loss {loss:.4f}, corr(|fitted field|, true speed) {corr:+.3f}")
    return r


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 20000, 50000, 100000])
    ap.add_argument("--dim", type=int, default=50, help="input dimension (default 50)")
    ap.add_argument("--perplexity", default="50", help='number or "auto" (= sqrt(n))')
    ap.add_argument("--balanced", action="store_true",
                    help="balanced (Sinkhorn) input affinity; default row-stochastic")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--steps", type=int, default=10, help="timed fit steps per config")
    ap.add_argument("--warmup", type=int, default=3, help="untimed steps (KeOps compiles here)")
    ap.add_argument("--fit-n", type=int, default=0, help="end-to-end fit at this n (0 = skip)")
    ap.add_argument("--fit-iters", type=int, default=1000)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="run without CUDA (smoke test)")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    env = environment(args.allow_cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    gpu_mem = env.get("gpu_memory_gb")
    report = dict(env=env, args=vars(args), started=datetime.now().isoformat(timespec="seconds"))

    out = args.out or os.path.join(
        HERE, "results",
        f"gpu_benchmark_{env['host']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def save():
        json.dump(report, open(out, "w"), indent=1)

    try:
        if not args.skip_correctness:
            report["correctness"] = correctness(device)
            save()
            if not all(r["ok"] for r in report["correctness"]):
                print("!! dense and KeOps disagree on this device; timings below are suspect.")
        print(f"\n== Scaling ({args.dtype}, perplexity {args.perplexity}, "
              f"{'balanced' if args.balanced else 'row-stochastic'} input, dim {args.dim})")
        report["scaling"] = []
        for n in args.sizes:
            report["scaling"] += bench_size(n, args, device, dtype, gpu_mem)
            save()
        if args.fit_n:
            report["end_to_end"] = end_to_end(args.fit_n, args, device, dtype)
    except Exception:  # noqa: BLE001
        report["fatal_error"] = traceback.format_exc()
        print(report["fatal_error"])
    finally:
        report["finished"] = datetime.now().isoformat(timespec="seconds")
        save()
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
