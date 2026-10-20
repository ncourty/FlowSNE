#!/usr/bin/env bash
# Gastrulation representation comparison on a CUDA machine: given the same
# independent RNA velocity (scVelo's gene-space velocity projected linearly
# onto our own PCA), which d-dimensional representation -- PCA, a symmetric
# neighbour embedding with scVelo-style projected velocity, or FlowSNE --
# best preserves direction, scored against the real embryonic stages?
# See experiments/single_cell/gastrulation_repr.py (replaces the circular
# scVelo comparison of gastrulation_markov.py).
#
# Usage, from anywhere in the repository:
#     bash experiments/single_cell/run_gastrulation_repr_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N_CELLS   cells embedded (uniform random subset)   [30000]
#     DIMS      embedding dimensions                     ["2 10"]
#     SEEDS     seeds (FlowSNE flow initialisation)      ["0 1 2"]
#     K_CHAIN   neighbours per row, same for all chains  ["30 100"]
#     ITERS     fit iterations                           [3000]
#     TAG       run name                                 [repr]
# Each fit is saved, so an interrupted run resumes (same TAG).
# Needs (besides FlowSNE's own requirements): pip install scvelo scanpy
#
# Cost: the cached preprocessing from earlier runs is outdated (it used the
# file's own PCA, without loadings); it is redone automatically once (CPU,
# same cost as the first time). Then 2 symmetric + 6 FlowSNE fits on 30k
# cells, roughly 1.5-2 h on an RTX 2080 Ti class GPU (estimate).
#
# Send back: experiments/single_cell/gastrulation_results/<TAG>/RESULTS.md,
# repr.json and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_CELLS=${N_CELLS:-30000}
DIMS=${DIMS:-"2 10"}
SEEDS=${SEEDS:-"0 1 2"}
K_CHAIN=${K_CHAIN:-"30 100"}
ITERS=${ITERS:-3000}
TAG=${TAG:-repr}
OUT=experiments/single_cell/gastrulation_results
mkdir -p "$OUT"
LOG="$OUT/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N_CELLS=$N_CELLS   DIMS=$DIMS   SEEDS=$SEEDS   K_CHAIN=$K_CHAIN   ITERS=$ITERS   TAG=$TAG"

python -c "import scvelo, scanpy" 2>/dev/null || { echo "!! pip install scvelo scanpy"; exit 1; }

echo "== 1/2 preprocessing (redone once if the cache is outdated) ($(date))"
PYTHONUTF8=1 python -u experiments/single_cell/gastrulation_data.py --subset full

echo "== 2/2 representations vs stage ($(date))"
PYTHONUTF8=1 python -u experiments/single_cell/gastrulation_repr.py --subset full --n-cells "$N_CELLS" \
    --dims $DIMS --seeds $SEEDS --k-chain $K_CHAIN --iters "$ITERS" --tag "$TAG"

echo "== done ($(date)). Results in $OUT/$TAG"
