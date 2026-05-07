#!/bin/bash
# Run the 2026-05-07 multi-seed pilot (6 cells, ~2 hours total).
# 2 ff values (0.0, 0.5) x 3 seeds (42, 123, 7).
# Each cell: 200-iter fine-tune on slippery + eval on rough+slippery.
set -uo pipefail

ASHFALL_ROOT="/home/yusuf/Projects/ashfall"
LOGS_DIR="$ASHFALL_ROOT/results/_logs"
mkdir -p "$LOGS_DIR"

# Cells in priority order: ff=0.0 controls first (so we have a baseline
# distribution to compare against if the run is interrupted), then
# ff=0.5 trial cells in the same seed order.
CELLS=(
    "0p0_seed=42"
    "0p0_seed=123"
    "0p0_seed=7"
    "0p5_seed=42"
    "0p5_seed=123"
    "0p5_seed=7"
)

# All 6 cells share the same generation timestamp from prepare_run.
# Auto-detect by globbing one cell's stamp dir.
SAMPLE_CELL_DIR=$(ls -d "$ASHFALL_ROOT/results/multiseed_pilot_2026-05-07_failure_fraction=0p0_seed=42"/*/ 2>/dev/null | head -1)
if [ -z "$SAMPLE_CELL_DIR" ]; then
    echo "ERROR: no generated cells under $ASHFALL_ROOT/results/multiseed_pilot_2026-05-07_*" >&2
    echo "Run the pilot generator first." >&2
    exit 1
fi
STAMP=$(basename "${SAMPLE_CELL_DIR%/}")
echo "[multiseed] using stamp $STAMP"

MANIFEST="$LOGS_DIR/multiseed_pilot_manifest.txt"
echo "# Ashfall multi-seed pilot manifest" > "$MANIFEST"
echo "# started: $(date -Is)" >> "$MANIFEST"
echo "# stamp:   $STAMP" >> "$MANIFEST"

OVERALL_START=$SECONDS

for cell in "${CELLS[@]}"; do
    CELL_NAME="multiseed_pilot_2026-05-07_failure_fraction=${cell}"
    CELL_DIR="$ASHFALL_ROOT/results/${CELL_NAME}/${STAMP}"
    LOG="$LOGS_DIR/multiseed_${cell//=/_}.log"
    echo "" | tee -a "$MANIFEST"
    echo "=== Cell ${cell} === $(date -Is)" | tee -a "$MANIFEST"
    if [ ! -d "$CELL_DIR" ]; then
        echo "MISSING: $CELL_DIR" | tee -a "$MANIFEST"
        exit 2
    fi
    START=$SECONDS
    bash "$CELL_DIR/commands.sh" 2>&1 | tee "$LOG" | tail -3
    RC=${PIPESTATUS[0]}
    DUR=$((SECONDS - START))
    echo "Cell ${cell} rc=$RC duration=${DUR}s" | tee -a "$MANIFEST"
    if [ $RC -ne 0 ]; then
        echo "Cell ${cell} FAILED, stopping sweep." | tee -a "$MANIFEST"
        exit 1
    fi
done

OVERALL_DUR=$((SECONDS - OVERALL_START))
echo "" | tee -a "$MANIFEST"
echo "# completed: $(date -Is)" | tee -a "$MANIFEST"
echo "# total duration: ${OVERALL_DUR}s" | tee -a "$MANIFEST"
