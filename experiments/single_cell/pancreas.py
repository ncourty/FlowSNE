"""Pancreatic endocrinogenesis: real RNA velocity as drift. Cluster order
follows the known differentiation hierarchy (Ductal -> Ngn3 low -> Ngn3
high -> Pre-endocrine -> {Alpha, Beta, Delta, Epsilon}).

Run: conda run -n dev python experiments/single_cell/pancreas.py [--subsample N] [--full]
Writes: experiments/single_cell/pancreas.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pancreas_data import load  # noqa: E402
from _common import run_domain  # noqa: E402

CLUSTER_ORDER = ["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine",
                 "Alpha", "Beta", "Delta", "Epsilon"]


def cluster_order(clusters, pt):
    return [c for c in CLUSTER_ORDER if c in set(clusters)]


def main(subsample=80):
    d = load(subsample=subsample, seed=0)
    run_domain("pancreas", d, os.path.dirname(__file__), cluster_order,
              neural_strategy="joint")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=80, help="max cells per cluster")
    p.add_argument("--full", action="store_true", help="use the full dataset")
    args = p.parse_args()
    main(subsample=(None if args.full else args.subsample))
