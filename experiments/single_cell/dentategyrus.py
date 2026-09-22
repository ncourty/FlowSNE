"""Dentate gyrus neurogenesis: real RNA velocity as drift. 14 clusters,
only some on the neurogenic lineage (Radial Glia-like -> nIPC ->
Neuroblast -> Granule immature -> Granule mature); cluster order is
derived from mean pseudotime rather than hardcoded, since there is no
single natural total order across the non-neurogenic populations too.

A naive joint (Z, flow) fit is unstable on this dataset for the neural
kernel (same failure signature as MNIST) -- uses the "auto" probe-then-
fallback strategy, not a plain joint fit.

Run: conda run -n dev python experiments/single_cell/dentategyrus.py [--subsample N] [--full]
Writes: experiments/single_cell/dentategyrus.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from dentategyrus_data import load  # noqa: E402
from _common import run_domain  # noqa: E402


def cluster_order(clusters, pt):
    return sorted(set(clusters), key=lambda c: pt[clusters == c].mean())


def main(subsample=80):
    d = load(subsample=subsample, seed=0)
    run_domain("dentategyrus", d, os.path.dirname(__file__), cluster_order,
              neural_strategy="auto")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=80, help="max cells per cluster")
    p.add_argument("--full", action="store_true", help="use the full dataset")
    args = p.parse_args()
    main(subsample=(None if args.full else args.subsample))
