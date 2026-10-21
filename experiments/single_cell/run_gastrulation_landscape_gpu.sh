#!/usr/bin/env bash
# Gradient (landscape) FlowSNE on mouse gastrulation, on a CUDA machine:
# the velocity projected onto gradients (Hodge potential) as an exact Randers
# input cost, a 1-Lipschitz scalar potential network on the output side,
# compared with the vector-field FlowSNE, PCA and a symmetric embedding
# against the real embryonic stages (walk scores + pseudotime correlation).
# See experiments/single_cell/gastrulation_landscape.py.
#
# Usage, from anywhere in the repository:
#     bash experiments/single_cell/run_gastrulation_landscape_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N_CELLS      cells embedded (uniform random subset)   [30000]
#     DIMS         embedding dimensions                     ["2 10"]
#     SEEDS        seeds (flow initialisation)              ["0 1 2"]
#     K_CHAIN      neighbours per row, same for all chains  ["30 100"]
#     ITERS        fit iterations                           [6000]
#     INPUT_POWER  input cost exponent of the potential     [1]
#     TAG          run name                                 [landscape]
# Each fit is saved, so an interrupted run resumes (same TAG).
# Needs the cached full preprocessing (PREP_VERSION 2, from the repr run).
#
# Cost (estimate): 2 symmetric + 6 vector + 6 potential fits of 6000
# iterations on 30k cells, ~15-20 min each on an RTX 2080 Ti: ~4 h. SEEDS=0
# alone: ~1.5 h.
#
# Send back: experiments/single_cell/gastrulation_results/<TAG>/LANDSCAPE.md,
# landscape.json and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_CELLS=${N_CELLS:-30000}
DIMS=${DIMS:-"2 10"}
SEEDS=${SEEDS:-"0 1 2"}
K_CHAIN=${K_CHAIN:-"30 100"}
ITERS=${ITERS:-6000}
INPUT_POWER=${INPUT_POWER:-1}
TAG=${TAG:-landscape}
OUT=experiments/single_cell/gastrulation_results
mkdir -p "$OUT"
LOG="$OUT/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N_CELLS=$N_CELLS   DIMS=$DIMS   SEEDS=$SEEDS   K_CHAIN=$K_CHAIN   ITERS=$ITERS   INPUT_POWER=$INPUT_POWER   TAG=$TAG"

echo "== tests (potential mode)"
python -m pytest tests/test_potential.py -q

echo "== landscape comparison ($(date))"
PYTHONUTF8=1 python -u experiments/single_cell/gastrulation_landscape.py --subset full --n-cells "$N_CELLS" \
    --dims $DIMS --seeds $SEEDS --k-chain $K_CHAIN --iters "$ITERS" --input-power "$INPUT_POWER" --tag "$TAG"

echo "== done ($(date)). Results in $OUT/$TAG"
