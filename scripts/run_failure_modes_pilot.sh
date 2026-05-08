#!/bin/bash
# Run the 2026-05-08 mode-subset Stage-1 pilot (18 cells, ~4.5 hr total).
# 6 mode subsets x 3 pilot seeds {42, 123, 7} at fixed ff=0.5.
# Each cell: 200-iter fine-tune on slippery + eval on rough+slippery.
# Compared seed-paired against existing n=3 ff=0.0 multiseed_pilot baseline
# (results/multiseed_pilot_2026-05-07_failure_fraction=0p0_seed=*).
set -uo pipefail

ASHFALL_ROOT="/home/yusuf/Projects/ashfall"
LOGS_DIR="$ASHFALL_ROOT/results/_logs"
mkdir -p "$LOGS_DIR"

# Subset outer, seed inner — matches the run-doc spec. Subset names track
# the labels in src/ashfall/experiment/sweep.py::_FAILURE_MODES_LABELS.
SUBSETS=(
    "all_modes"
    "slip_only"
    "command_mismatch_only"
    "slip_plus_cm"
    "severe_only"
    "severe_plus_slip"
)
SEEDS=(42 123 7)

# All 18 cells share the same generation timestamp from prepare_run.
# Auto-detect by globbing one cell's stamp dir.
FIRST_CELL_GLOB="$ASHFALL_ROOT/results/failure_modes_pilot_2026-05-08_failure_modes=all_modes_seed=42"
SAMPLE_CELL_DIR=$(ls -d "$FIRST_CELL_GLOB"/*/ 2>/dev/null | head -1)
if [ -z "$SAMPLE_CELL_DIR" ]; then
    echo "ERROR: no generated cells under $FIRST_CELL_GLOB" >&2
    echo "Run scripts/run_ablation.sh configs/experiments/failure_modes_pilot_2026-05-08.yaml first." >&2
    exit 1
fi
STAMP=$(basename "${SAMPLE_CELL_DIR%/}")
echo "[failure-modes-pilot] using stamp $STAMP"

MANIFEST="$LOGS_DIR/failure_modes_pilot_manifest.txt"
echo "# Ashfall mode-subset Stage-1 pilot manifest" > "$MANIFEST"
echo "# started: $(date -Is)" >> "$MANIFEST"
echo "# stamp:   $STAMP" >> "$MANIFEST"

OVERALL_START=$SECONDS

for subset in "${SUBSETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        CELL_NAME="failure_modes_pilot_2026-05-08_failure_modes=${subset}_seed=${seed}"
        CELL_DIR="$ASHFALL_ROOT/results/${CELL_NAME}/${STAMP}"
        LOG="$LOGS_DIR/failure_modes_pilot_${subset}_seed_${seed}.log"
        echo "" | tee -a "$MANIFEST"
        echo "=== Cell ${subset} seed=${seed} === $(date -Is)" | tee -a "$MANIFEST"
        if [ ! -d "$CELL_DIR" ]; then
            echo "MISSING: $CELL_DIR" | tee -a "$MANIFEST"
            exit 2
        fi
        START=$SECONDS
        bash "$CELL_DIR/commands.sh" 2>&1 | tee "$LOG" | tail -3
        RC=${PIPESTATUS[0]}
        DUR=$((SECONDS - START))
        echo "Cell ${subset} seed=${seed} rc=$RC duration=${DUR}s" | tee -a "$MANIFEST"
        if [ $RC -ne 0 ]; then
            echo "Cell ${subset} seed=${seed} FAILED, stopping sweep." | tee -a "$MANIFEST"
            exit 1
        fi
    done
done

OVERALL_DUR=$((SECONDS - OVERALL_START))
echo "" | tee -a "$MANIFEST"
echo "# completed: $(date -Is)" | tee -a "$MANIFEST"
echo "# total duration: ${OVERALL_DUR}s" | tee -a "$MANIFEST"
