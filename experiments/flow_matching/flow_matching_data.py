"""A fourth kind of drift, and the cleanest ground truth in this project: a
*pretrained* flow-matching generative model's own velocity field.

`Dinghuai/flow-matching-cifar10` (HuggingFace Hub) is a small (~38M param),
ungated UNet2DModel trained with a flow-matching objective on CIFAR-10:
integrating dx/dt = v_theta(x,t) from t=0 (x0 ~ N(0,I), pure noise) to t=1
along this model's own velocity field generates a CIFAR-10-like image. Unlike
every other drift used in this project (RNA velocity is independently
measured, the MNIST gradient is a first-order proxy, the transformer's
residual update is a byproduct of a next-token objective), v_theta(x,t) here
*is*, by the model's own training objective, exactly the instantaneous
transport direction along the noise-to-data path -- as unambiguous as a
drift gets.

A known diffusers bug, worked around below: `UNet2DModel.forward` ends with
`if self.config.time_embedding_type == "fourier": sample = sample /
timesteps` -- a leftover from a variance-exploding-SDE parameterisation that
does not apply to a flow-matching velocity field, and divides by exactly
zero at t=0. The model card's fix (renaming the *registered* time embedding
type so this branch is skipped, while still using a Fourier time
embedding for `self.time_proj`) must be re-applied *after* `from_pretrained`,
not only in `__init__`: `from_pretrained` re-registers config from the
checkpoint's `config.json` after instantiation, silently reverting an
in-`__init__` override. Confirmed by direct inspection: integrating without
the post-load re-application gives `nan` from the very first step; with it,
40-step Euler integration yields recognisable CIFAR-10 images (boats, a
deer, a car, a plane -- see `flow_matching_data.py`'s own sanity check in
CLAUDE.md).

Run: conda run -n dev python experiments/flow_matching/flow_matching_data.py
Writes: experiments/flow_matching/data/flow_matching_activations.npz
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(DATA_DIR, "flow_matching_activations.npz")

MODEL_ID = "Dinghuai/flow-matching-cifar10"


def load_model(device=None):
    from diffusers.models.unets import UNet2DModel
    from diffusers.models.embeddings import GaussianFourierProjection

    class MyUNet2DModel(UNet2DModel):
        def __init__(self):
            super().__init__(
                in_channels=3, out_channels=3, time_embedding_type="fourier",
                down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D",
                                  "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D",
                                "UpBlock2D"),
                block_out_channels=(128, 256, 256, 256), layers_per_block=2,
                norm_num_groups=32, norm_eps=1e-6,
            )
            self.time_proj = GaussianFourierProjection(
                embedding_size=128, scale=16, set_W_to_weight=False, log=False,
            )
            self.config.time_embedding_type = "my_fourier"

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = MyUNet2DModel.from_pretrained(MODEL_ID)
    model.config.time_embedding_type = "my_fourier"  # re-apply: from_pretrained clobbers it
    return model.to(device).eval(), device


def preprocess(n_traj=80, n_times=10, n_substeps=5, seed=0, device=None):
    """Integrate `n_traj` independent trajectories from Gaussian noise (t=0)
    to a CIFAR-10-like image (t=1) via Euler steps, recording the exact
    state x(t) *and* the model's own velocity v_theta(x(t), t) at each of
    `n_times` evenly-spaced checkpoints (`n_substeps` Euler sub-steps are
    taken between consecutive checkpoints for integration accuracy; only
    the checkpoint states/velocities are kept)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    model, device = load_model(device)
    torch.manual_seed(seed)

    x = torch.randn(n_traj, 3, 32, 32, device=device, dtype=torch.float32)
    t_anchors = torch.linspace(0, 1, n_times, device=device)

    acts, drifts, traj_id, t_idx, t_val = [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, t_anchor in enumerate(t_anchors):
            t_batch = t_anchor.expand(n_traj)
            v_anchor = model(x, t_batch).sample  # velocity AT this checkpoint, before stepping on
            acts.append(x.reshape(n_traj, -1).cpu().numpy())
            drifts.append(v_anchor.reshape(n_traj, -1).cpu().numpy())
            traj_id.append(np.arange(n_traj))
            t_idx.append(np.full(n_traj, k))
            t_val.append(np.full(n_traj, t_anchor.item(), dtype=np.float32))

            if k == n_times - 1:
                break
            t_next = t_anchors[k + 1]
            sub_ts = torch.linspace(t_anchor.item(), t_next.item(), n_substeps + 1,
                                    device=device)
            v = v_anchor
            for s in range(n_substeps):
                dt = sub_ts[s + 1] - sub_ts[s]
                x = x + v * dt
                if s < n_substeps - 1:
                    v = model(x, sub_ts[s + 1].expand(n_traj)).sample
            print(f"checkpoint {k} (t={t_anchor.item():.3f}) done, "
                 f"{time.time()-t0:.1f}s elapsed", flush=True)

    activations = np.concatenate(acts).astype(np.float32)
    drift = np.concatenate(drifts).astype(np.float32)
    traj_id = np.concatenate(traj_id).astype(np.int64)
    t_idx = np.concatenate(t_idx).astype(np.int64)
    t_val = np.concatenate(t_val).astype(np.float32)

    print(f"finite: activations={np.isfinite(activations).all()} "
         f"drift={np.isfinite(drift).all()}", flush=True)

    np.savez_compressed(
        CACHE_PATH,
        activations=activations, drift=drift,
        traj_id=traj_id, t_idx=t_idx, t_val=t_val,
        n_traj=n_traj, n_times=n_times, image_shape=(3, 32, 32),
    )
    print(f"n_points={len(activations)}  saved {CACHE_PATH}", flush=True)
    return CACHE_PATH


def load():
    if not os.path.exists(CACHE_PATH):
        preprocess()
    d = np.load(CACHE_PATH, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_pca(n_components=50, seed=0):
    """Raw flattened pixels are 3072-dimensional -- far higher than any
    other domain in this project (MNIST activations, GPT-2's 768-dim
    hidden state, pancreas's own 30-dim scVelo PCA representation) -- and
    building the kNN graph directly on them gave a badly-conditioned
    resolvent solve in practice (a `FinslerSinkhornAffinity` call that had
    not converged after 30+ minutes of BiCGStab iteration, vs. ~150s for
    every other domain here). Reduce to `n_components` PCs first, exactly
    the same move `pancreas_data.py` already makes for the same reason
    (raw gene expression is even higher-dimensional than this), and project
    the velocity through the *same* linear map (no mean-centering --
    velocity is a difference, not a position), matching `scVelo`'s own
    `velocity_embedding(..., basis="pca")` convention."""
    from sklearn.decomposition import PCA

    d = load()
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(d["activations"])
    V_pca = d["drift"] @ pca.components_.T
    out = dict(d)
    out["activations_pca"] = X_pca.astype(np.float32)
    out["drift_pca"] = V_pca.astype(np.float32)
    out["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
    return out


INDEPENDENT_CACHE_PATH = os.path.join(DATA_DIR, "flow_matching_activations_independent.npz")
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                   "horse", "ship", "truck"]


def load_real_samples(n_per_class=10, seed=0):
    """`n_per_class` real CIFAR-10 training images per class, normalised to
    the same [0, 1] pixel convention as the model's own generated samples
    (confirmed empirically, not assumed -- see the [-1,1]-normalisation bug
    note in CLAUDE.md: generated t=1 samples have min~=-0.06, max~=1.04,
    mean~=0.48, matching [0,1]-native data almost exactly, not the [-1,1]
    this function originally (wrongly) assumed; visually, un-shifted display
    recovers correctly saturated blacks/colour that the old `(x+1)/2` shift
    washed out to gray). These are not part of the generative process at all
    -- they are the actual target
    distribution the flow-matching model was trained to reproduce -- so
    including them alongside generated t=1 samples in the same affinity/
    embedding is a direct mode-collapse check: do generated samples actually
    reach the neighbourhood of every real class, or do some classes have no
    generated samples nearby?"""
    # torchvision's own CIFAR10(download=True) pulls from a slow/flaky
    # university server (cave.cs.toronto.edu) that stalled mid-download in
    # practice; the HuggingFace mirror uses HF's own fast CDN instead.
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", split="train")
    targets = np.array(ds["label"])
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(10):
        ci = np.flatnonzero(targets == c)
        idx.append(rng.choice(ci, size=n_per_class, replace=False))
    idx = np.sort(np.concatenate(idx))

    imgs = np.stack([np.asarray(ds[int(i)]["img"]) for i in idx])  # (N, 32, 32, 3) uint8
    imgs = imgs.astype(np.float32) / 255.0                    # -> [0, 1], matches model space
    imgs = imgs.transpose(0, 3, 1, 2).reshape(len(idx), -1)  # -> (N, 3*32*32), channel-first
    class_label = targets[idx]
    return imgs.astype(np.float32), class_label.astype(np.int64)


def preprocess_independent(n_per_t=300, n_times=10, steps_per_unit=45,
                           n_real_per_class=10, seed=0, device=None):
    """The methodologically cleaner version of `preprocess`: instead of one
    set of trajectories followed end-to-end (so every checkpoint's points
    are literally the deterministic future of the SAME earlier points --
    any coherence the embedding shows across time could just be reflecting
    shared trajectory identity, not genuine structure in how the marginal
    distribution p_t(x) itself moves), draw a *fresh, independent* batch of
    Gaussian noise for every checkpoint and integrate each batch only as far
    as its own t -- so every timestep's point cloud is a genuine i.i.d.
    sample from p_t, with no cross-time correspondence at all (there is
    deliberately no `traj_id`-like field here).

    Also appends `n_real_per_class` real CIFAR-10 images per class (see
    `load_real_samples`), placed at the final checkpoint (t=1, alongside the
    generated "data" samples) with their own v_theta(x, 1) evaluated the
    same way every other t=1 point's drift is -- a direct mode-collapse
    check: `is_real` and `class_label` mark them for downstream plotting."""
    os.makedirs(DATA_DIR, exist_ok=True)
    model, device = load_model(device)
    torch.manual_seed(seed)

    t_anchors = torch.linspace(0, 1, n_times)
    acts, drifts, t_idx_list, t_val_list = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k, t_anchor in enumerate(t_anchors):
            t_k = float(t_anchor)
            n_steps = int(round(t_k * steps_per_unit))
            x = torch.randn(n_per_t, 3, 32, 32, device=device, dtype=torch.float32)
            if n_steps > 0:
                sub_ts = torch.linspace(0.0, t_k, n_steps + 1, device=device)
                for s in range(n_steps):
                    v = model(x, sub_ts[s].expand(n_per_t)).sample
                    x = x + v * (sub_ts[s + 1] - sub_ts[s])
            v_final = model(x, torch.full((n_per_t,), t_k, device=device)).sample
            acts.append(x.reshape(n_per_t, -1).cpu().numpy())
            drifts.append(v_final.reshape(n_per_t, -1).cpu().numpy())
            t_idx_list.append(np.full(n_per_t, k))
            t_val_list.append(np.full(n_per_t, t_k, dtype=np.float32))
            print(f"checkpoint {k} (t={t_k:.3f}, {n_steps} fresh Euler steps) done, "
                 f"{time.time()-t0:.1f}s elapsed", flush=True)

    activations = np.concatenate(acts).astype(np.float32)
    drift = np.concatenate(drifts).astype(np.float32)
    t_idx = np.concatenate(t_idx_list).astype(np.int64)
    t_val = np.concatenate(t_val_list).astype(np.float32)
    is_real = np.zeros(len(activations), dtype=bool)
    class_label = np.full(len(activations), -1, dtype=np.int64)

    print("loading real CIFAR-10 reference samples...", flush=True)
    real_imgs, real_labels = load_real_samples(n_per_class=n_real_per_class, seed=seed)
    with torch.no_grad():
        x_real = torch.tensor(real_imgs, device=device).reshape(-1, 3, 32, 32)
        v_real = model(x_real, torch.ones(len(real_imgs), device=device)).sample
    activations = np.concatenate([activations, real_imgs])
    drift = np.concatenate([drift, v_real.reshape(len(real_imgs), -1).cpu().numpy().astype(np.float32)])
    t_idx = np.concatenate([t_idx, np.full(len(real_imgs), n_times - 1, dtype=np.int64)])
    t_val = np.concatenate([t_val, np.full(len(real_imgs), 1.0, dtype=np.float32)])
    is_real = np.concatenate([is_real, np.ones(len(real_imgs), dtype=bool)])
    class_label = np.concatenate([class_label, real_labels])

    print(f"finite: activations={np.isfinite(activations).all()} "
         f"drift={np.isfinite(drift).all()}", flush=True)

    np.savez_compressed(
        INDEPENDENT_CACHE_PATH,
        activations=activations, drift=drift, t_idx=t_idx, t_val=t_val,
        is_real=is_real, class_label=class_label,
        class_names=np.array(CIFAR10_CLASSES),
        n_per_t=n_per_t, n_times=n_times, n_real_per_class=n_real_per_class,
        image_shape=(3, 32, 32),
    )
    print(f"n_points={len(activations)} ({is_real.sum()} real)  saved {INDEPENDENT_CACHE_PATH}",
         flush=True)
    return INDEPENDENT_CACHE_PATH


def load_independent():
    if not os.path.exists(INDEPENDENT_CACHE_PATH):
        preprocess_independent()
    d = np.load(INDEPENDENT_CACHE_PATH, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_pca_independent(n_components=50, seed=0):
    """Same rationale as `load_pca`, applied to the independent-sampling
    dataset (real CIFAR-10 reference samples included, so they land in the
    exact same PCA space as every generated point)."""
    from sklearn.decomposition import PCA

    d = load_independent()
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(d["activations"])
    V_pca = d["drift"] @ pca.components_.T
    out = dict(d)
    out["activations_pca"] = X_pca.astype(np.float32)
    out["drift_pca"] = V_pca.astype(np.float32)
    out["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
    return out


if __name__ == "__main__":
    preprocess()
