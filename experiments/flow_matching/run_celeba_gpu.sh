#!/usr/bin/env bash
# CelebA flow-matching visualisation (PnP-Flow OT model, 128x128) on a CUDA
# machine: preflight, sample generation along the flow, FlowSNE embedding,
# figures. See experiments/flow_matching/celeba_flow.py for details.
#
# Usage, from anywhere in the repository:
#     bash experiments/flow_matching/run_celeba_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N_PER_T   samples per time point       [1000]
#     TIMES     time points in [0, 1]        [11]   -> n = N_PER_T * TIMES points
#     STEPS     Euler steps over [0, 1]      [100]
#     TAG       run name                     [main]
# Needs (besides FlowSNE's own requirements): pip install gdown
#
# Rough cost on an RTX 2080 Ti class GPU (estimate, not measured): generation
# ~10-20 min for the defaults (n = 11,000); embedding a few minutes.
# Disk: samples are stored as float16, ~200 MB per 1000 samples per time
# point (~2.2 GB for the defaults), under experiments/flow_matching/data/celeba/.
# Generation resumes per time point if interrupted.
#
# Look at data/celeba/samples_<TAG>/samples_t1.png and trajectory.png first:
# they show whether the model and conventions are right before any embedding.
# Send back: celeba_results/<TAG>/metrics.json, embedding.png,
# embedding_thumbnails.png and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_PER_T=${N_PER_T:-1000}
TIMES=${TIMES:-11}
STEPS=${STEPS:-100}
TAG=${TAG:-main}
OUT=experiments/flow_matching/celeba_results
mkdir -p "$OUT"
LOG="$OUT/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N_PER_T=$N_PER_T   TIMES=$TIMES   STEPS=$STEPS   TAG=$TAG"

python -c "import gdown" 2>/dev/null || { echo "!! gdown missing: pip install gdown"; exit 1; }

echo "== preflight: environment, KeOps-on-GPU check, dense vs KeOps correctness"
python benchmarks/gpu_benchmark.py --sizes 10000 --steps 2 --warmup 1 \
    --out "$OUT/preflight_${TAG}.json"

echo "== generate / embed / plot ($(date))"
python -u experiments/flow_matching/celeba_flow.py all --tag "$TAG" \
    --n-per-t "$N_PER_T" --times "$TIMES" --steps "$STEPS"

echo "== done ($(date)). Results in $OUT/$TAG"
