"""A fourth domain, after toy 2D, real RNA velocity, and neural-network
gradients: a transformer's own residual stream, where the "velocity" needs
no gradient and no training loop at all.

A transformer's hidden state updates by addition at every layer,
h_{l+1} = h_l + F_l(h_l) (the attention+MLP sublayer output added to the
residual stream). F_l(h_l) = h_{l+1} - h_l is therefore *literally* the
instantaneous displacement of a representation as it moves through depth --
not a metaphor for velocity the way gradient descent's -grad_a(loss) is
(mnist_data.py), and not an inferred quantity the way RNA velocity is
(pancreas_data.py): it is read directly off a single forward pass of any
pretrained model, no training required.

Point = one sentence's mean-pooled hidden state at layer L (over its own,
non-padding tokens). Drift = that same pooling applied to h_{L+1} - h_L,
layer L's own contribution to the residual stream. Model: GPT-2 small (the
standard subject of residual-stream interpretability work: logit lens,
TransformerLens, etc.), no fine-tuning, inference only.

Validation needs reframing relative to the pancreas/MNIST cases: there is no
training loss or prediction here to check "correction" against, only a
frozen pretrained model's own forward pass. But sentences still carry real,
externally-known topic labels (AG News), so the same same_class_mass
diagnostic from mnist_affinity.py still applies directly: does a sentence's
early/mid-layer residual-stream drift already point toward its own topic's
region, before the representation has fully separated by topic (which it is
known to do by the final layers of a reasonable pretrained LM) -- i.e. does
the model's own internal processing "anticipate" the correct topic the way
RNA velocity anticipates a cell's future differentiated state.

Run: conda run -n dev python experiments/transformer/transformer_data.py
Writes: experiments/transformer/data/transformer_activations.npz
"""

from __future__ import annotations

import os

import numpy as np
import torch

DATA = os.path.join(os.path.dirname(__file__), "data")
CACHE = os.path.join(DATA, "transformer_activations.npz")

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def stratified_subset(dataset, n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset["label"])
    idx = []
    for c in range(4):
        ci = np.flatnonzero(labels == c)
        ci = rng.choice(ci, size=min(n_per_class, len(ci)), replace=False)
        idx.append(ci)
    idx = np.sort(np.concatenate(idx))
    return dataset.select(idx)


def extract(model, tokenizer, texts, layer, batch_size=32, device="cpu"):
    model.eval()
    all_pooled_L, all_pooled_L1 = [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=64).to(device)
            out = model(**enc, output_hidden_states=True)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
            denom = mask.sum(1).clamp(min=1.0)  # (B, 1)

            hL = out.hidden_states[layer]        # (B, T, D)
            hL1 = out.hidden_states[layer + 1]

            pooled_L = (hL * mask).sum(1) / denom
            pooled_L1 = (hL1 * mask).sum(1) / denom

            all_pooled_L.append(pooled_L)
            all_pooled_L1.append(pooled_L1)

    pooled_L = torch.cat(all_pooled_L, 0).numpy().astype(np.float32)
    pooled_L1 = torch.cat(all_pooled_L1, 0).numpy().astype(np.float32)
    return pooled_L, pooled_L1 - pooled_L


def preprocess(layer=4, n_per_class=400, model_name="gpt2"):
    from datasets import load_dataset
    from transformers import GPT2Model, GPT2TokenizerFast

    os.makedirs(DATA, exist_ok=True)
    print("loading AG News ...", flush=True)
    ds = load_dataset("fancyzhx/ag_news", split="test")
    ds = stratified_subset(ds, n_per_class)
    print(f"  {len(ds)} sentences, {n_per_class}/class", flush=True)

    print(f"loading {model_name} ...", flush=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained(model_name)
    print(f"  {model.config.n_layer} layers, hidden dim {model.config.n_embd}, "
         f"using layer {layer} -> {layer+1}", flush=True)

    activations, drift = extract(model, tokenizer, ds["text"], layer)
    labels = np.array(ds["label"])

    np.savez_compressed(
        CACHE, activations=activations, drift=drift, labels=labels,
        text=np.array(ds["text"]), layer=layer,
    )
    print(f"cached {CACHE}", flush=True)
    print(f"drift norm: mean={np.linalg.norm(drift,axis=1).mean():.3f}  "
         f"activation norm: mean={np.linalg.norm(activations,axis=1).mean():.3f}")


def load():
    if not os.path.exists(CACHE):
        preprocess()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    d = load()
    print(f"n={len(d['labels'])}  layer={int(d['layer'])}")
    for c, name in enumerate(LABEL_NAMES):
        m = d["labels"] == c
        print(f"  {name:10s} n={m.sum():4d}")
    print(f"\nexample: {d['text'][0]!r}  label={LABEL_NAMES[d['labels'][0]]}")
