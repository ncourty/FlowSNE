#!/usr/bin/env bash
# CelebA-HQ flow visualisation (google/ddpm-celebahq-256, deterministic DDIM
# sampling) on a CUDA machine: preflight, trajectory generation, FlowSNE
# embedding, figures. See experiments/flow_matching/celeba_flow.py.
#
# Usage, from anywhere in the repository:
#     bash experiments/flow_matching/run_celeba_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N_TRAJ       samples per snapshot time     [1000]
#     TIMES        snapshot times in [0, 1)      [11]   -> n = N_TRAJ * TIMES points
#     STEPS        DDIM steps                    [50]
#     DRIFT        step | x0                     [step]
#     SAMPLING     independent | shared          [independent]
#                  independent: fresh noise for every snapshot time (different
#                  realisations); shared: one trajectory recorded at every time
#                  (~5-6x cheaper). A sample folder keeps its mode: use a new TAG.
#     PERPLEXITY   number or auto (= sqrt(n))    [auto]
#     TAG          run name                      [main]
# Needs (besides FlowSNE's own requirements): pip install diffusers
# The model (~455 MB, Apache-2.0) is downloaded from the Hugging Face Hub.
#
# Rough cost on an RTX 2080 Ti class GPU (estimate, not measured): generation
# ~2-3.5 h for the defaults with independent sampling (~280 model calls per
# sample index across the 11 times, 113.7M-parameter UNet at 256x256), or
# ~20-40 min with SAMPLING=shared; embedding a few minutes. Lower STEPS or
# N_TRAJ to go faster. Disk: snapshots are stored as
# float16 at 128x128, ~2.2 MB per sample index across all times (~2.2 GB for the
# defaults), under
# experiments/flow_matching/data/celeba/. Generation resumes per batch.
#
# Look at data/celeba/samples_<TAG>/samples_final.png and trajectory.png first.
# Send back: celeba_results/<TAG>/metrics.json, embedding.png,
# embedding_thumbnails.png and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_TRAJ=${N_TRAJ:-1000}
TIMES=${TIMES:-11}
STEPS=${STEPS:-50}
DRIFT=${DRIFT:-step}
SAMPLING=${SAMPLING:-independent}
PERPLEXITY=${PERPLEXITY:-auto}
TAG=${TAG:-main}
OUT=experiments/flow_matching/celeba_results
mkdir -p "$OUT"
LOG="$OUT/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N_TRAJ=$N_TRAJ   TIMES=$TIMES   STEPS=$STEPS   DRIFT=$DRIFT   SAMPLING=$SAMPLING   PERPLEXITY=$PERPLEXITY   TAG=$TAG"

python -c "import diffusers" 2>/dev/null || { echo "!! diffusers missing: pip install diffusers"; exit 1; }

echo "== preflight: environment, KeOps-on-GPU check, dense vs KeOps correctness"
python benchmarks/gpu_benchmark.py --sizes 10000 --steps 2 --warmup 1 \
    --out "$OUT/preflight_${TAG}.json"

echo "== generate / embed / plot ($(date))"
python -u experiments/flow_matching/celeba_flow.py all --tag "$TAG" \
    --n-traj "$N_TRAJ" --times "$TIMES" --steps "$STEPS" --drift "$DRIFT" \
    --sampling "$SAMPLING" --perplexity "$PERPLEXITY"

echo "== done ($(date)). Results in $OUT/$TAG"
