"""GPT-2 small residual stream at a cyclic concept token (days of the week,
months), many contexts per concept, every layer. All names are single tokens
with a leading space; the concept is always the last token.

Writes data/cyclic_gpt2_{days,months}.npz: H (L+1, n_contexts, n_concepts, 768),
concepts, contexts.

    conda run -n dev python experiments/transformer/numbers/cyclic_data.py
"""
import itertools
import os

import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ["We", "They", "My sister", "The team", "Our neighbours", "The doctor"]
CONCEPTS = {
    "days": ("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
             ["will meet on", "arrived on", "left last", "always rest on", "went shopping on",
              "have a test on", "called me on", "play football every", "will be back by",
              "were busy all day"]),
    "months": ("January February March April May June July August September October "
               "November December".split(),
               ["will travel in", "was born in", "moved here in", "started work in",
                "got married in", "went skiing in", "finished school in", "plant seeds every",
                "will be back by", "were away all of"]),
}


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()
    for name, (concepts, verbs) in CONCEPTS.items():
        contexts = [f"{s} {v}" for s, v in itertools.product(SUBJECTS, verbs)]
        H = np.zeros((model.config.n_layer + 1, len(contexts), len(concepts),
                      model.config.n_embd), np.float32)
        with torch.no_grad():
            for i, ctx in enumerate(contexts):
                for j, c in enumerate(concepts):
                    ids = tok.encode(f"{ctx} {c}")
                    assert ids[-1] == tok.encode(" " + c)[0] and len(tok.encode(" " + c)) == 1
                    hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
                    H[:, i, j] = torch.stack([h[0, -1] for h in hs]).numpy()
        out = os.path.join(HERE, "data", f"cyclic_gpt2_{name}.npz")
        np.savez_compressed(out, H=H, concepts=np.array(concepts), contexts=np.array(contexts))
        print("wrote", out, H.shape, flush=True)


if __name__ == "__main__":
    main()
