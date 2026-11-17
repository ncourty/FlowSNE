"""Embedding/optimisation-pipeline glue for TorchDR's ``AffinityMatcher``.

``AffinityMatcher`` (and every embedding method built on it) only ever
optimises the embedding coordinates -- see its ``_set_params``, which
hardcodes ``[{"params": self.embedding_}]``. That's fine as long as
``affinity_out`` has no learnable parameters of its own, but
``RandersAffinity``'s flow network is exactly the kind of parameter that
should be fit jointly with the embedding, not fixed beforehand.
``JointAffinityMatcher`` is the minimal fix: same class, same lifecycle,
just also hands ``affinity_out``'s parameters (if any) to the optimiser.
"""

from __future__ import annotations

import math

import torch

from torchdr import AffinityMatcher


class JointAffinityMatcher(AffinityMatcher):
    r"""``AffinityMatcher`` that also optimises ``affinity_out``'s own
    learnable parameters (if it has any) jointly with the embedding.

    Identical to :class:`~torchdr.AffinityMatcher` in every other respect
    -- only ``_set_params`` is overridden.
    """

    def _set_params(self):
        params = [{"params": self.embedding_}]
        if isinstance(self.affinity_out, torch.nn.Module):
            extra = list(self.affinity_out.parameters())
            if extra:
                params.append({"params": extra})
        self.params_ = params
        return self.params_


def fit_embedding(affinity_out, P, init, indices=None, max_iter: int = 1500, lr: float = 0.1,
                  init_scaling: float = 1.0, seed: int = 0, train_embedding: bool = True,
                  train_flow: bool = True, rescale_init: bool = True, anchor=None,
                  anchor_weight: float = 0.0, history: list | None = None, log_every: int = 100,
                  flow_lr: float | None = None, lr_schedule: str = "constant",
                  early_exaggeration: float = 1.0, exaggeration_iters: int = 0,
                  warmup_iters: int = 0, tol: float = 0.0, field_tol: float = 1e-2,
                  diagnostics: list | None = None):
    """Fit an embedding (and/or ``affinity_out``'s flow) by Adam on
    ``affinity_out.cross_entropy(Z, P, indices)`` -- dense ``P`` (n, n) or
    sparse ``P`` (n, k) + ``indices`` (``EntropicRandersAffinity(sparsity=
    True)``), with either backend of ``RandersAffinity``.

    Reproduces TorchDR's ``AffinityMatcher`` loop step for step (init scaled
    to ``init_scaling`` std on the first coordinate, one Adam optimiser with
    the embedding and the flow parameters as two groups at the same ``lr``),
    so with a dense ``P`` it matches ``JointAffinityMatcher`` -- the point of
    this function is the sparse / KeOps path, which TorchDR's matcher does not
    accept for a precomputed affinity.

    ``train_embedding=False`` keeps ``init`` as the embedding (no rescaling)
    and fits the flow only (``fit_drift_only``); ``train_flow=False`` with a
    zero-initialised flow gives the symmetric Student baseline.

    Warm starts (e.g. a sequence of snapshots): ``rescale_init=False`` keeps
    ``init`` as is, and ``anchor`` (n, d) adds dynamic t-SNE's temporal
    coherence term ``anchor_weight * mean_i ||z_i - anchor_i||^2`` (Rauber et
    al. 2016) to the loss. The flow is warm-started simply by passing the
    same ``affinity_out`` again.

    Optional training schedule (defaults reproduce the plain loop above):
    ``flow_lr`` gives the flow's parameters their own learning rate (default:
    ``lr``); ``lr_schedule="cosine"`` decays every group's rate as
    ``0.5 (1 + cos(pi t / max_iter))`` (floored at 1% of the initial rate);
    ``early_exaggeration`` multiplies the attraction term for the first
    ``exaggeration_iters`` iterations (t-SNE's early exaggeration, which
    mainly shapes the global layout).

    ``warmup_iters`` ramps every rate linearly from 0 over the first
    iterations (before the cosine decay, if any).

    Convergence test (every ``log_every`` iterations, after exaggeration and
    warm-up): stop when, over the last window, the relative loss decrease is
    below ``tol`` AND the layout and the field have both stopped moving --
    ``||Z_t - Z_{t-w}|| / ||Z_t||`` and ``||w_t - w_{t-w}|| / ||w_t||`` (the
    drift at the points, ``affinity_out.omega_at``) below ``field_tol``. The
    loss alone can plateau while the layout still drifts, or the network still
    reshapes the field. All three are divided by the current schedule factor
    (Adam's step, hence the progress per window, scales with the rate): a
    decayed rate would otherwise pass for convergence. ``tol=0`` (default) runs
    all ``max_iter`` iterations.

    ``history``: a list to which ``(iteration, loss)`` is appended every
    ``log_every`` iterations and after the last one (loss before that step).
    ``diagnostics``: a list to which a dict per check is appended (iteration,
    loss, relative loss decrease, Z and field change over the window, gradient
    norm of each parameter group, current learning rate).
    """
    torch.manual_seed(seed)
    Z = torch.as_tensor(init).to(dtype=P.dtype, device=P.device).detach().clone()
    if train_embedding and rescale_init:
        Z = init_scaling * Z / Z[:, 0].std()
    if anchor is not None:
        anchor = torch.as_tensor(anchor).to(dtype=P.dtype, device=P.device).detach()
    Z.requires_grad_(train_embedding)
    groups = []
    if train_embedding:
        groups.append({"params": [Z]})
    if train_flow:
        flow_params = [q for q in affinity_out.parameters() if q.requires_grad]
        if flow_params:
            groups.append({"params": flow_params, "lr": lr if flow_lr is None else flow_lr})
    if not groups:
        raise ValueError("fit_embedding: nothing to train.")
    opt = torch.optim.Adam(groups, lr=lr)
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {lr_schedule!r}")

    def factor(t):
        f = 1.0
        if lr_schedule == "cosine":
            f = max(0.01, 0.5 * (1 + math.cos(math.pi * t / max_iter)))
        if t < warmup_iters:
            f *= (t + 1) / warmup_iters
        return f

    sched = (torch.optim.lr_scheduler.LambdaLR(opt, factor)
             if lr_schedule != "constant" or warmup_iters else None)
    has_field = hasattr(affinity_out, "omega_at")
    check = tol > 0 or diagnostics is not None
    prev = None  # (loss, Z, field) at the previous check
    for it in range(max_iter):
        opt.zero_grad(set_to_none=True)
        if it < exaggeration_iters and early_exaggeration != 1.0:
            loss = affinity_out.cross_entropy(Z, P, indices, exaggeration=early_exaggeration)
        else:
            loss = affinity_out.cross_entropy(Z, P, indices)
        if history is not None and (it % log_every == 0 or it == max_iter - 1):
            history.append((it, float(loss.detach())))
        if anchor is not None and anchor_weight:
            loss = loss + anchor_weight * ((Z - anchor) ** 2).sum(-1).mean()
        loss.backward()
        if check and it % log_every == 0:
            with torch.no_grad():
                Zc = Z.detach().clone()
                Fc = affinity_out.omega_at(Zc).detach() if has_field else None
                cur = float(loss.detach())
                f = factor(it) if sched is not None else 1.0
                rec = {"it": it, "loss": cur, "lr_factor": f,
                       "lr": [g["lr"] for g in opt.param_groups],
                       "grad_norm": [float(torch.sqrt(sum((q.grad ** 2).sum() for q in g["params"]
                                                          if q.grad is not None)))
                                     for g in opt.param_groups]}
                if prev is not None:
                    rec["rel_loss_drop"] = (prev[0] - cur) / abs(cur)
                    rec["z_change"] = float((Zc - prev[1]).norm() / Zc.norm().clamp_min(1e-12))
                    rec["field_change"] = (float((Fc - prev[2]).norm() / Fc.norm().clamp_min(1e-12))
                                           if has_field else 0.0)
                prev = (cur, Zc, Fc)
            if diagnostics is not None:
                diagnostics.append(rec)
            if (tol > 0 and "rel_loss_drop" in rec and it >= max(exaggeration_iters, warmup_iters)
                    + log_every
                    and abs(rec["rel_loss_drop"]) < tol * f
                    and rec["z_change"] < field_tol * f and rec["field_change"] < field_tol * f):
                if history is not None and history[-1][0] != it:
                    history.append((it, float(loss.detach())))
                break
        opt.step()
        if sched is not None:
            sched.step()
    return Z.detach()


def fit_drift_only(affinity_out, P, Z_fixed, seed: int = 0, max_iter: int = 1500,
                   lr: float = 0.05, indices=None):
    """Freeze the embedding at ``Z_fixed`` and fit only ``affinity_out``'s
    own parameters (e.g. a flow network) against it, via plain Adam.
    ``P`` dense (n, n), or sparse (n, k) with its ``indices``.

    The correct fallback when a joint (Z, flow) fit is unstable: fit the
    symmetric embedding to convergence first, then call this with its
    result as ``Z_fixed``. See individual experiment scripts for when this
    is actually needed (MNIST) versus when a plain joint fit works fine
    (the toy examples, single-cell data).
    """
    return fit_embedding(affinity_out, P, Z_fixed, indices=indices, max_iter=max_iter, lr=lr,
                         seed=seed, train_embedding=False)
