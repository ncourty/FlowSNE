"""GPT-2 small residual stream at the number token, integers 0..360 (all single
tokens with a leading space), several templates, every layer.

Writes data/numbers_gpt2.npz: H (L+1, n_templates, 361, 768) float32 --
hidden_states[l] at the number token (l = 0 is the embedding output),
numbers, templates.

    conda run -n dev python experiments/transformer/numbers/numbers_data.py
"""
import os

import numpy as np
import torch
from transformers import GPT2Model, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "numbers_gpt2.npz")
NUMBERS = np.arange(361)
TEMPLATES = [
    "I counted the boxes and there were {}",
    "The answer to the question is {}",
    "She bought {}",
    "The page number is {}",
    "We waited for {}",
    "The temperature reached {}",
    "My locker number was {}",
    "In total, the team scored {}",
    "He lives at number {}",
    "The final count came to {}",
]


def main():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2").eval()
    for n in NUMBERS:
        assert len(tok.encode(" " + str(n))) == 1, n
    H = np.zeros((model.config.n_layer + 1, len(TEMPLATES), len(NUMBERS), model.config.n_embd),
                 np.float32)
    with torch.no_grad():
        for t, tpl in enumerate(TEMPLATES):
            texts = [tpl.format(n) for n in NUMBERS]
            for i, text in enumerate(texts):
                ids = torch.tensor([tok.encode(text)])
                assert ids[0, -1].item() == tok.encode(" " + str(NUMBERS[i]))[0]
                hs = model(ids, output_hidden_states=True).hidden_states
                H[:, t, i] = torch.stack([h[0, -1] for h in hs]).numpy()
            print(f"template {t} done", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, H=H, numbers=NUMBERS, templates=np.array(TEMPLATES))
    print("wrote", OUT, H.shape)


if __name__ == "__main__":
    main()
