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


def fit_drift_only(affinity_out, P, Z_fixed, seed: int = 0, max_iter: int = 1500,
                   lr: float = 0.05):
    """Freeze the embedding at ``Z_fixed`` and fit only ``affinity_out``'s
    own parameters (e.g. a flow network) against it, via plain Adam --
    no ``AffinityMatcher`` machinery needed since nothing about the
    embedding coordinates is being optimised.

    The correct fallback when a joint (Z, flow) fit is unstable: fit the
    symmetric embedding to convergence first, then call this with its
    result as ``Z_fixed``. See individual experiment scripts for when this
    is actually needed (MNIST) versus when a plain joint fit works fine
    (the toy examples, single-cell data).
    """
    from torchdr.utils import cross_entropy_loss

    torch.manual_seed(seed)
    Z = Z_fixed.detach().clone()
    opt = torch.optim.Adam(affinity_out.parameters(), lr=lr)
    for _ in range(max_iter):
        opt.zero_grad()
        log_Q = affinity_out(Z, log=True)
        loss = cross_entropy_loss(P, log_Q, log=True)
        loss.backward()
        opt.step()
    return Z
