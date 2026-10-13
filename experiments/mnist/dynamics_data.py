"""Learning-dynamics snapshots for MNIST: train the small CNN, save a
checkpoint every ``--every`` SGD iterations, and recompute on demand, for a
FIXED set of training samples, each sample's penultimate (fully connected)
activation and the gradient of its own loss with respect to it.

Checkpoints instead of features: the CNN has ~106k parameters (~0.4 MB per
checkpoint, ~20 MB per epoch at every=10), whereas storing activations and
gradients for 60k samples would take ~31 MB per snapshot (~1.5 GB per
epoch) and fix the sample set and layer in advance. Recomputing them from a
checkpoint is one forward/backward pass.

The drift is -grad_a loss(a_i, y_i) with the TRAINING label: the direction
the training signal pushes that sample's representation (see mnist_data.py
for why per-sample gradients come from one backward pass: no BatchNorm).

Run:    conda run -n dev python experiments/mnist/dynamics_data.py [--every 10 --epochs 1]
Writes: experiments/mnist/data/dynamics/ckpt_*.pt, train_log.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from mnist_data import DATA, SmallCNN  # noqa: E402

OUT = os.path.join(DATA, "dynamics")


def train_and_checkpoint(every=10, epochs=1, batch_size=128, lr=1e-3, seed=0, out=OUT):
    import torchvision
    torch.manual_seed(seed)
    train = torchvision.datasets.MNIST(root=DATA, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "ckpt_*.pt")):
        os.remove(f)
    log, step = [], 0

    def save():
        torch.save(model.state_dict(), os.path.join(out, f"ckpt_{step:06d}.pt"))

    save()
    for _ in range(epochs):
        for images, labels in loader:
            model.train()
            opt.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            step += 1
            log.append(dict(step=step, loss=loss.item(),
                            acc=(logits.argmax(1) == labels).float().mean().item()))
            if step % every == 0:
                save()
    if step % every:
        save()
    json.dump(dict(every=every, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                   n_steps=step, log=log), open(os.path.join(out, "train_log.json"), "w"))
    print(f"trained {step} iterations, {len(checkpoints(out))} checkpoints in {out}")


def checkpoints(out=OUT):
    """Sorted list of (step, path)."""
    files = sorted(glob.glob(os.path.join(out, "ckpt_*.pt")))
    return [(int(os.path.basename(f)[5:-3]), f) for f in files]


def sample_set(n, seed=0):
    """A fixed, uniformly random subset of the TRAINING set (seeded):
    returns (images (n, 1, 28, 28), labels (n,), indices (n,))."""
    import torchvision
    train = torchvision.datasets.MNIST(root=DATA, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
    idx = np.sort(np.random.default_rng(seed).choice(len(train), size=n, replace=False))
    images = torch.stack([train[i][0] for i in idx])
    labels = torch.tensor([train[i][1] for i in idx])
    return images, labels, idx


@torch.no_grad()
def _forward_acts(model, images, batch=4096):
    return torch.cat([model.activations(images[s:s + batch]) for s in range(0, len(images), batch)])


def features(ckpt_path, images, labels, device="cpu"):
    """Penultimate activations, per-sample gradient of each sample's own loss
    w.r.t. its activation, predictions and losses, at one checkpoint."""
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    a = _forward_acts(model, images.to(device)).detach().clone().requires_grad_(True)
    logits = model.fc2(a)
    per_sample = F.cross_entropy(logits, labels.to(device), reduction="none")
    per_sample.sum().backward()          # a.grad[i] depends only on per_sample[i]
    return dict(activations=a.detach(), grad=a.grad.detach(),
                pred=logits.argmax(1).detach(), loss=per_sample.detach())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train_and_checkpoint(every=a.every, epochs=a.epochs, batch_size=a.batch_size, seed=a.seed)
