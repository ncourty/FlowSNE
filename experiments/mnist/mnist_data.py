"""A third domain for this project's core construction, after the toy 2D
demos and real RNA velocity: neural-network activations, with the "velocity"
coming from gradient information rather than biology.

The analogy to RNA velocity is structural, not just decorative: a point is
one sample's activation vector at some layer (high-dimensional -- the
"genes ~ neurons" correspondence), and the drift is
omega_i = -grad_{a_i}(loss_i), the per-sample gradient of that sample's own
loss with respect to its own activation. This is not a metaphor for
velocity -- under gradient-flow dynamics on the activation itself (not just
the weights), -grad_a(loss) literally *is* the instantaneous direction that
activation would move to locally reduce its own loss. Unlike RNA velocity
(which needs the spliced/unspliced inference machinery) this "velocity" is
exact and free: one backward pass.

A fully-converged network has ~zero gradient almost everywhere (every
training example already correctly, confidently classified), which would
make the drift field degenerate -- so this deliberately checkpoints *during*
training, not after convergence, when a meaningful fraction of examples
still have real gradient signal (the hard / boundary / not-yet-resolved
ones), which are exactly the interesting ones for a directional-affinity
story.

No BatchNorm in the CNN below: BatchNorm mixes statistics across the batch
during a training-mode forward pass, which would break the property this
whole construction relies on -- that sample i's loss depends only on sample
i's own activation, so summing per-sample losses and taking one backward
pass gives the correct *per-sample* gradient in `activation.grad`, no
vmap/functorch or per-sample loop needed.

Run: conda run -n dev python experiments/mnist/mnist_data.py
Writes: experiments/mnist/data/mnist_activations.npz
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(DATA, "mnist_activations.npz")


class SmallCNN(nn.Module):
    """No BatchNorm -- see module docstring for why that matters here."""

    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def activations(self, x):
        h = F.max_pool2d(F.relu(self.conv1(x)), 2)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)
        h = h.flatten(1)
        return F.relu(self.fc1(h))  # the penultimate activation -- our "point"

    def forward(self, x):
        return self.fc2(self.activations(x))


def train_to_checkpoint(target_train_acc=0.88, max_steps=2000, batch_size=128,
                        seed=0, device="cpu"):
    """Trains only until a *moderate* train accuracy is reached -- deliberately
    not to convergence, so a real mix of resolved/unresolved examples remains
    (see module docstring)."""
    import torchvision

    torch.manual_seed(seed)
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(root=DATA, train=True, download=True,
                                           transform=tfm)
    test_set = torchvision.datasets.MNIST(root=DATA, train=False, download=True,
                                          transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    step = 0
    running_correct, running_n = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        model.train()
        opt.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()

        running_correct += (logits.argmax(1) == labels).sum().item()
        running_n += labels.shape[0]
        step += 1

        if step % 20 == 0:
            acc = running_correct / running_n
            print(f"  step {step:4d}  running train acc {acc:.3f}", flush=True)
            if acc >= target_train_acc:
                print(f"  reached target train acc {target_train_acc} at step {step}",
                     flush=True)
                break
            running_correct, running_n = 0, 0
        if step >= max_steps:
            print("  hit max_steps before reaching target accuracy", flush=True)
            break

    return model, test_set


def extract(model, dataset, n_samples=2000, seed=0, device="cpu"):
    """Per-sample activations + true/predicted labels + loss + the per-sample
    gradient of that sample's own loss w.r.t. its own activation -- one
    backward pass over the whole extraction batch (see module docstring for
    why this gives exact per-sample gradients, not an approximation)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n_samples, replace=False)
    images = torch.stack([dataset[i][0] for i in idx]).to(device)
    labels = torch.tensor([dataset[i][1] for i in idx], dtype=torch.long, device=device)

    model.eval()  # no dropout/BatchNorm here, kept for clarity/future-proofing
    a = model.activations(images).detach().clone().requires_grad_(True)
    logits = model.fc2(a)
    per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
    per_sample_loss.sum().backward()  # a.grad[i] depends only on per_sample_loss[i]

    grad = a.grad.detach()
    with torch.no_grad():
        pred = logits.argmax(1)
        probs = F.softmax(logits, dim=1)
        confidence = probs.gather(1, pred[:, None]).squeeze(1)

    return dict(
        activations=a.detach().numpy().astype(np.float32),
        grad=grad.numpy().astype(np.float32),
        labels=labels.numpy(),
        pred=pred.numpy(),
        loss=per_sample_loss.detach().numpy().astype(np.float32),
        confidence=confidence.numpy().astype(np.float32),
    )


def preprocess():
    os.makedirs(DATA, exist_ok=True)
    print("training SmallCNN on MNIST to a moderate (not converged) checkpoint ...",
         flush=True)
    model, test_set = train_to_checkpoint()
    print("extracting activations + per-sample gradients on a test-set batch ...",
         flush=True)
    d = extract(model, test_set)
    acc = (d["pred"] == d["labels"]).mean()
    grad_norm = np.linalg.norm(d["grad"], axis=1)
    print(f"extraction batch: n={len(d['labels'])}  accuracy={acc:.3f}  "
         f"grad norm: mean={grad_norm.mean():.4f} "
         f"frac(near-zero, <1e-4)={ (grad_norm < 1e-4).mean():.3f}")
    np.savez_compressed(CACHE, **d)
    print(f"cached {CACHE}", flush=True)


def load():
    if not os.path.exists(CACHE):
        preprocess()
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    d = load()
    acc = (d["pred"] == d["labels"]).mean()
    grad_norm = np.linalg.norm(d["grad"], axis=1)
    print(f"n={len(d['labels'])}  accuracy={acc:.3f}")
    print(f"grad norm: mean={grad_norm.mean():.4f}  median={np.median(grad_norm):.4f}  "
         f"frac(near-zero, <1e-4)={(grad_norm < 1e-4).mean():.3f}")
    print("\nper-digit accuracy:")
    for c in range(10):
        m = d["labels"] == c
        print(f"  digit {c}: n={m.sum():4d}  acc={ (d['pred'][m]==d['labels'][m]).mean():.3f}  "
             f"mean grad norm={grad_norm[m].mean():.4f}")
