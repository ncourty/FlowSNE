# FlowSNE — project context

This project is a **stabilized spin-off of `FinslerAffinity`**
(`../FinslerAffinity`), which is itself a spin-off of `FinslerOT`
(`../FinslerOT`). `FinslerAffinity` was a long exploratory investigation —
architecture search, an approximation-error analysis of the Randers
distance, several construction variants compared across five real/toy
domains. This repo is the **cleaned-up result**: only the constructions
that investigation actually validated, none of the exploratory dead ends,
none of the narrated history. If you need the "why" behind a design choice
that isn't explained here, `FinslerAffinity/CLAUDE.md` has the full record.

## The idea, precisely

Standard dimensionality-reduction affinities (t-SNE, UMAP, SNEkhorn) use a
**symmetric** ground cost between points, so self-transport under them has
a built-in pull toward "don't move." Swap in a **Randers** (direction-
dependent) ground cost — cheaper to move with a given drift than against
it — and that degeneracy goes away: the resulting entropic self-affinity is
shaped by local directional structure (e.g. RNA velocity, a gradient
field, a generative model's own velocity field), not just local density.
This is a direct extension of SNEkhorn (Van Assel, Vayer, Flamary, Courty,
NeurIPS 2023 — the user's own prior work) to an asymmetric ground cost.

## What changed from `FinslerAffinity`

**Only two closed-form Randers distance approximations are kept.** The
true Finsler distance `d_F(x->y)` is an infimum over paths with no closed
form when the drift `omega` varies with position. `FinslerAffinity`
explored three ways to approximate it: (1) a graph heat-kernel diffusion
proxy (`FinslerSinkhornAffinity`, forward-backward Sinkhorn over a Randers
generator — the project's original, most expensive construction), (2) an
eikonal PDE solver (grid-only, never generalized to point clouds), (3) two
closed-form, path-straight-line approximations. Extensive testing (see
`FinslerAffinity/CLAUDE.md`'s later sections) found the closed-form
approximations, once given a properly calibrated per-row bandwidth
(perplexity, not a fixed neighbour count), match or beat the much more
expensive heat-kernel proxy — cheaper, simpler, and with a rigorously
derived error bound. **This repo keeps only those two**:

- `dhat_F(x->y) = ||y-x|| + omega(x)^T(y-x)` — freezes the drift at the
  source. Exact when `omega` is constant; first-order accurate otherwise,
  error `O(L * r^2)` (`L` a Lipschitz bound on `omega`, `r = ||y-x||`).
- `bdF(x->y) = ||y-x|| + ((omega(x)+omega(y))/2)^T(y-x)` — averages the
  drift at both endpoints. One order more accurate, `O((L^2+L2) * r^3)`,
  at the same cost (both endpoint values are already available: given
  per-point on the input side, cheap to evaluate anywhere for a network on
  the output side). Default in `src/flowsne/`, since it matched or beat
  the frozen version everywhere tested, including on joint-fit stability
  (same probe verdict as the frozen version on every domain checked).

No graph/heat-kernel/eikonal machinery is implemented here. If a
grid-based or graph-diffusion Finsler distance is ever needed again,
that's `FinslerAffinity`'s `graph.py` / `operators.py` / `fbfs.py` /
`randers_eikonal.py`, not this repo.

**The input affinity is now a perplexity-calibrated entropic affinity**
(`EntropicRandersAffinity`), not a heat-kernel construction: per-row
bandwidth calibrated by bisection to a target perplexity (default
`sqrt(n)`, a scale-adaptive choice — a fixed target becomes an
ever-shrinking fraction of the dataset as `n` grows, the actual bug behind
an early, discarded k-nearest-neighbour-restricted attempt), then either
Sinkhorn-balanced to uniform marginals (`doubly_stochastic=True`, default
— this is SNEkhorn's own actual contribution, controlled marginals fixing
the density-heterogeneity/hub problem) or left as a plain row-stochastic
softmax (`doubly_stochastic=False`, cheaper, no marginal control).

**The output/embedding kernel now takes a pluggable flow network.**
`FinslerAffinity` had two separate classes, `CanonicalRandersAffinity`
(single global drift) and `NeuralRandersAffinity` (fixed MLP
architecture), plus several MLP variants tried only as local experiment
files (`InputStdNorm`-prepended, RFF-encoded, BatchNorm, ...). Here there
is **one** class, `RandersAffinity(n_components, flow=...)`, and the
architecture is just an `nn.Module` you pass in — a global drift is
`ConstantFlow`, a plain MLP is `mlp_flow`, and so on
(`src/flowsne/flows.py`). Write your own for anything else; the affinity
class handles admissibility (`||omega|| < 1`) and the sign convention
itself, so a custom flow only needs to output a raw, unconstrained vector.

## Things worth knowing before touching either affinity class

1. **Sign convention**: under `d_F(z_i->z_j) = ||dz|| + omega(z_i)^T dz`,
   moving *with* `omega` makes the distance *larger*, not smaller — the
   opposite of the intuitive reading. A correctly-fit `omega` points AWAY
   from the direction the affinity favours. `EntropicRandersAffinity`
   negates the given drift internally (so passing the true physical drift,
   e.g. RNA velocity, does the right thing); `RandersAffinity.omega_at(Z)`
   returns the raw fitted field, so plot `-omega_at(Z)` for the "flow
   direction" when visualising a fitted embedding, not the raw value.
2. **Joint-fit stability is dataset-dependent, not architecture-dependent.**
   A plain joint `(Z, flow.parameters())` fit works fine on most data, but
   was found unstable on MNIST specifically (every learning rate gave a
   worse-than-symmetric loss) regardless of which closed-form distance was
   used. The fix, `flowsne.embedding.fit_drift_only`: fit the symmetric
   embedding to convergence first, freeze it, fit only the flow network on
   top. Probe with a short joint fit (a few hundred iterations) against
   the symmetric baseline's own loss before committing to a full run on a
   new dataset, and fall back to `fit_drift_only` if the probe looks
   worse — don't assume either strategy transfers to a new domain.
3. **Sinkhorn needs thousands of iterations here.** A tight perplexity
   bandwidth makes the balanced kernel badly conditioned: the old default of
   200 iterations left every single-cell dataset's marginals 0.6–3.6% off
   (results before Oct 2026 that used `doubly_stochastic=True` on real data
   were not truly balanced). Sinkhorn is now log-domain (float32-safe),
   defaults to 5000 iterations with early stopping, stores
   `marginal_error_` and warns above 1e-4. Full pancreas (3696 cells) takes
   ~130 s on CPU; ε-scaling would speed this up.
4. **Backends and sparsity** (`EntropicRandersAffinity(sparsity=True,
   backend=None|"keops")`). Sparse mode keeps each row's `n_neighbors`
   (default 5× perplexity) nearest points under the *directed* cost and
   returns TorchDR's `(values, indices)` format; KeOps only changes how
   neighbours are found (O(n) memory), the result is bitwise identical.
   Row-stochastic sparse is exact up to truncation and is the recommended
   sparse mode. Balanced sparse runs Sinkhorn on kNN + reverse edges and is a
   *local* variant of balancing, not an approximation of the dense one (on
   full pancreas: median row TV 8%, a few rows completely different, for
   any k). The output kernel and loss do not have a KeOps path yet.

## Layout

- `src/flowsne/` — the library. `distances.py` (the two closed-form
  approximations + the admissibility reparametrisation), `flows.py`
  (pluggable drift-network architectures + three example builders),
  `affinity.py` (`EntropicRandersAffinity`, `RandersAffinity`),
  `embedding.py` (`JointAffinityMatcher`, `fit_drift_only`).
- `experiments/` — `toy/` (synthetic vortex + rotating-sphere), `single_cell/`
  (pancreas + dentate gyrus RNA velocity), `mnist/` (gradient-as-drift),
  `flow_matching/` (a pretrained flow-matching model's own velocity field),
  `transformer/` (GPT-2 residual-stream updates as drift). Every script
  here uses only `flowsne`'s public API — no heat-kernel code paths, no
  exploratory narration; see `FinslerAffinity/CLAUDE.md` for the full
  history of what was tried and why each domain's current setup won out.
- `tests/` — `test_backends.py`: dense vs KeOps equivalence, sparse
  affinities, log-domain Sinkhorn vs the previous implementation, float32.
  Run with `conda run -n dev python -m pytest tests/ -q`.
- `article/draft/` — working status document (`draft.md`) and its figures.

## Conventions (carried over from `FinslerAffinity` / `FinslerOT`)

- Run project code inside the conda env named `dev`
  (`conda run -n dev ...`), same as the sibling projects.
- Derive new affinities from TorchDR's `Affinity` hierarchy
  (`torchdr.affinity.base`) and follow its constructor conventions
  (`metric`, `zero_diag`, `device`, `backend`, `verbose`, `compile`,
  `_pre_processed`) — both classes in `src/flowsne/affinity.py` already do
  this; keep doing it for anything new.
- Multi-panel comparison figures should share one colorbar/scale, not be
  normalized per-panel.
- Before any destructive git operation, check status and stash/back up
  first.
