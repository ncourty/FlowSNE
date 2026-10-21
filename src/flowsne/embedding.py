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
                  anchor_weight: float = 0.0, history: list | None = None, log_every: int = 100):
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

    ``history``: a list to which ``(iteration, loss)`` is appended every
    ``log_every`` iterations and after the last one (loss before that step).
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
            groups.append({"params": flow_params})
    if not groups:
        raise ValueError("fit_embedding: nothing to train.")
    opt = torch.optim.Adam(groups, lr=lr)
    for it in range(max_iter):
        opt.zero_grad(set_to_none=True)
        loss = affinity_out.cross_entropy(Z, P, indices)
        if history is not None and (it % log_every == 0 or it == max_iter - 1):
            history.append((it, float(loss.detach())))
        if anchor is not None and anchor_weight:
            loss = loss + anchor_weight * ((Z - anchor) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
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
