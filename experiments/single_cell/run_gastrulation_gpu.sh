#!/usr/bin/env bash
# Gastrulation Markov-chain test on a CUDA machine: preprocess the full mouse
# gastrulation dataset with scVelo (once), then compare Markov chains from
# scVelo's velocity kernel, FlowSNE's input affinity and FlowSNE embeddings
# (d = 2 and 10) against the real embryonic stages.
# See experiments/single_cell/gastrulation_markov.py.
#
# Usage, from anywhere in the repository:
#     bash experiments/single_cell/run_gastrulation_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N_CELLS   cells embedded (uniform random subset)   [30000]
#     CONFIGS   embedding dim:degrees of freedom         ["2:1 10:1 10:9"]
#     SEEDS     seeds (flow initialisation)              ["0 1 2"]
#     K_CHAIN   neighbours per row, same for all chains  ["30 100 300"]
#     ITERS     fit iterations                           [3000]
#     TAG       run name                                 [fair]
# Each fit is saved, so an interrupted run resumes (same TAG).
# Needs (besides FlowSNE's own requirements): pip install scvelo scanpy
#
# Cost (estimate, not measured): the one-off preprocessing downloads 4.7 GB
# and runs scVelo on 116k cells on the CPU (tens of minutes to ~1.5 h; needs
# a few tens of GB of RAM); it is cached in experiments/single_cell/data/.
# Then ~2 h on an RTX 2080 Ti class GPU for 30k cells and 3 seeds (one seed of
# the four fits took ~36 min in the first run).
#
# Send back: experiments/single_cell/gastrulation_results/<TAG>/RESULTS.md,
# markov.json and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_CELLS=${N_CELLS:-30000}
CONFIGS=${CONFIGS:-"2:1 10:1 10:9"}
SEEDS=${SEEDS:-"0 1 2"}
K_CHAIN=${K_CHAIN:-"30 100 300"}
ITERS=${ITERS:-3000}
TAG=${TAG:-fair}
OUT=experiments/single_cell/gastrulation_results
mkdir -p "$OUT"
LOG="$OUT/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N_CELLS=$N_CELLS   CONFIGS=$CONFIGS   SEEDS=$SEEDS   K_CHAIN=$K_CHAIN   ITERS=$ITERS   TAG=$TAG"

python -c "import scvelo, scanpy" 2>/dev/null || { echo "!! pip install scvelo scanpy"; exit 1; }

echo "== preflight: environment, KeOps-on-GPU check, dense vs KeOps correctness"
python benchmarks/gpu_benchmark.py --sizes 10000 --steps 2 --warmup 1 \
    --out "$OUT/preflight_${TAG}.json"

echo "== 1/2 preprocessing (cached after the first run) ($(date))"
python -u experiments/single_cell/gastrulation_data.py --subset full

echo "== 2/2 Markov chains vs stage ($(date))"
python -u experiments/single_cell/gastrulation_markov.py --subset full --n-cells "$N_CELLS" \
    --configs $CONFIGS --seeds $SEEDS --k-chain $K_CHAIN --iters "$ITERS" --tag "$TAG"

echo "== done ($(date)). Results in $OUT/$TAG"
