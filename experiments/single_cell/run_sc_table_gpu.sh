#!/usr/bin/env bash
# Single-cell results table for the paper (pancreas + gastrulation) on a CUDA
# machine. Produces experiments/single_cell/SC_TABLE.md and the LaTeX rows
# article/ICML_FlowSNE/tables/single_cell_rows.tex (see sc_table.py).
#
#   1. preflight: environment + KeOps-on-GPU check (benchmarks/gpu_benchmark.py)
#   2. pancreas, full (n = 3696), 3 seeds: symmetric t-SNE on both inputs,
#      FlowSNE joint and frozen-layout on both inputs, Debaussart-Joniec Isomap
#      if FINSLER_EMBEDDING_SRC points at their code (sc_table.py pancreas)
#   3. gastrulation, 30k cells, 2-D and 10-D, 3 seeds, once per input velocity
#      (gastrulation_repr.py; the scVelo preprocessing is cached after the
#      first run, ~1 h on the CPU and a 4.7 GB download; needs scvelo + scanpy)
#   4. the table (sc_table.py table)
#
# Usage, from anywhere in the repository:
#     bash experiments/single_cell/run_sc_table_gpu.sh
# Settings (environment variables, defaults in brackets):
#     SEEDS       seeds                                          ["0 1 2"]
#     ITERS       joint-fit iterations                           [6000]
#                 (3000 left the 2-D gastrulation fit still improving: DIAG.md)
#     N_CELLS     gastrulation cells (uniform random subset)     [30000]
#     VELOCITIES  gastrulation input velocities, one run each    ["scvelo direct"]
#                 scvelo: scVelo's transition-matrix projection (smooth, 67% of
#                 cells forward); direct: gene-space velocity projected linearly
#                 on our PCA (noisier, 43% forward)
#     DIMS        gastrulation embedding dimensions              ["2 10"]
#     SKIP_PANCREAS / SKIP_GASTRULATION   set to 1 to skip a part
# Every fit is saved, so an interrupted run resumes.
#
# Cost (RTX 2080 Ti class, estimate): pancreas ~5 min per seed; gastrulation
# ~2-3 h per velocity at 6000 iterations.
#
# Send back: experiments/single_cell/SC_TABLE.md,
#            article/ICML_FlowSNE/tables/single_cell_rows.tex,
#            experiments/baselines/results/pancreas_full_seed*.{json,npz},
#            experiments/single_cell/gastrulation_results/repr_*/{repr.json,RESULTS.md}
#            and the run log.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

SEEDS=${SEEDS:-"0 1 2"}
ITERS=${ITERS:-6000}
N_CELLS=${N_CELLS:-30000}
VELOCITIES=${VELOCITIES:-"scvelo direct"}
DIMS=${DIMS:-"2 10"}
OUT=experiments/single_cell/gastrulation_results
mkdir -p "$OUT"
LOG="$OUT/run_sc_table_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG   SEEDS=$SEEDS   ITERS=$ITERS   N_CELLS=$N_CELLS   VELOCITIES=$VELOCITIES   DIMS=$DIMS"

echo "== preflight: environment, KeOps-on-GPU check ($(date))"
python benchmarks/gpu_benchmark.py --sizes 10000 --steps 2 --warmup 1 --out "$OUT/preflight_sc_table.json"

if [ "${SKIP_PANCREAS:-0}" != 1 ]; then
    echo "== pancreas, full, seeds $SEEDS ($(date))"
    [ -n "${FINSLER_EMBEDDING_SRC:-}" ] || echo "   (FINSLER_EMBEDDING_SRC not set: no Debaussart-Joniec row)"
    PYTHONUTF8=1 python -u experiments/single_cell/sc_table.py pancreas --seeds $SEEDS --iters "$ITERS"
fi

if [ "${SKIP_GASTRULATION:-0}" != 1 ]; then
    python -c "import scvelo, scanpy" 2>/dev/null || { echo "!! pip install scvelo scanpy"; exit 1; }
    echo "== gastrulation preprocessing (cached after the first run) ($(date))"
    PYTHONUTF8=1 python -u experiments/single_cell/gastrulation_data.py --subset full
    for vel in $VELOCITIES; do
        echo "== gastrulation, velocity $vel, seeds $SEEDS ($(date))"
        PYTHONUTF8=1 python -u experiments/single_cell/gastrulation_repr.py --subset full \
            --n-cells "$N_CELLS" --dims $DIMS --seeds $SEEDS --k-chain 30 100 --iters "$ITERS" \
            --velocity "$vel" --tag "repr_$vel"
    done
fi

echo "== table ($(date))"
TAGS=""; for vel in $VELOCITIES; do TAGS="$TAGS repr_$vel"; done
PYTHONUTF8=1 python experiments/single_cell/sc_table.py table --gastrulation-tags $TAGS

echo "== done ($(date))"
