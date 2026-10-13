#!/usr/bin/env bash
# Full-scale MNIST learning-dynamics run (all 60k training samples) on a CUDA
# machine: preflight checks, checkpoints, one FlowSNE embedding per checkpoint
# for each temporal-coherence lambda, arrow and streamline animations.
#
# Usage, from anywhere in the repository:
#     bash experiments/mnist/run_dynamics_gpu.sh
# Settings (environment variables, defaults in brackets):
#     N           samples                           [60000]
#     LAMBDAS     temporal-coherence weights        ["0 0.01 0.03 0.1"]
#     ITERS       fit iterations per warm frame     [300]
#     PERPLEXITY  number or "auto" (= sqrt(n))      [auto]
#     RETRAIN=1   retrain the CNN even if checkpoints exist
# e.g.  LAMBDAS="0 0.03" bash experiments/mnist/run_dynamics_gpu.sh
#
# Time, estimated from the RTX 2080 Ti benchmarks (rough): input affinity
# ~2.5 min per frame at n=60k, perplexity sqrt(n) (shared by all lambdas);
# ~0.35 s per fit step, i.e. ~1.8 min per frame per lambda. 48 frames:
#     4 lambdas ~ 8-9 h,   2 lambdas ~ 5 h,   1 lambda ~ 3.5 h.
# Results are saved every 4 frames, so a crash keeps everything done so far.
#
# Send back: experiments/mnist/dynamics_results/summary_n60000.json, the run
# log (dynamics_results/run_n60000_*.log) and the contact sheet
# streamlines_vs_arrows_n60000.png; the MP4s are worth a look on your side.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N=${N:-60000}
LAMBDAS=${LAMBDAS:-"0 0.01 0.03 0.1"}
ITERS=${ITERS:-300}
PERPLEXITY=${PERPLEXITY:-auto}
OUT=experiments/mnist/dynamics_results
mkdir -p "$OUT"
LOG="$OUT/run_n${N}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   N=$N   LAMBDAS=$LAMBDAS   ITERS=$ITERS   PERPLEXITY=$PERPLEXITY"

echo "== 0/3 preflight: environment, KeOps-on-GPU check, dense vs KeOps correctness"
python benchmarks/gpu_benchmark.py --sizes 10000 --steps 2 --warmup 1 \
    --out "$OUT/preflight_n${N}.json"

echo "== 1/3 training checkpoints (one epoch, every 10 iterations)"
if [ -f experiments/mnist/data/dynamics/train_log.json ] && [ -z "${RETRAIN:-}" ]; then
    echo "checkpoints already present, skipping (RETRAIN=1 to redo)"
else
    python experiments/mnist/dynamics_data.py --every 10 --epochs 1
fi

echo "== 2/3 FlowSNE per checkpoint ($(date))"
python -u experiments/mnist/dynamics.py --n "$N" --anchor-weights $LAMBDAS \
    --iters "$ITERS" --perplexity "$PERPLEXITY" --backend keops

echo "== 3/3 streamline animations ($(date))"
DIRS=""
for l in $LAMBDAS; do
    DIRS="$DIRS n${N}_lam$(python -c "print(f'{float(\"$l\"):g}')")"
done
python experiments/mnist/animate_streamlines.py --dirs $DIRS

echo "== done ($(date)). Results in $OUT"
