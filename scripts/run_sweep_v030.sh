#!/bin/bash
# Run the v0.3.0 failure_fraction sweep (6 cells, ~80 min total).
# Each cell: 200-iter fine-tune on slippery + eval on rough+slippery.
set -uo pipefail

ASHFALL_ROOT="/home/yusuf/Projects/ashfall"
LOGS_DIR="$ASHFALL_ROOT/results/_logs"
mkdir -p "$LOGS_DIR"

CELLS=(0p0 0p1 0p25 0p5 0p75 1p0)
STAMP="2026-04-28_00-17-07"

# Capture timings into a manifest.
MANIFEST="$LOGS_DIR/sweep_manifest.txt"
echo "# Ashfall v0.3.0 sweep manifest" > "$MANIFEST"
echo "# started: $(date -Is)" >> "$MANIFEST"

for ff in "${CELLS[@]}"; do
    CELL_NAME="ablation_failure_fraction_failure_fraction=${ff}"
    CELL_DIR="$ASHFALL_ROOT/results/${CELL_NAME}/${STAMP}"
    LOG="$LOGS_DIR/cell_${ff}.log"
    echo "" | tee -a "$MANIFEST"
    echo "=== Cell ff=${ff} === $(date -Is)" | tee -a "$MANIFEST"
    if [ ! -d "$CELL_DIR" ]; then
        echo "MISSING: $CELL_DIR" | tee -a "$MANIFEST"
        continue
    fi
    START=$SECONDS
    bash "$CELL_DIR/commands.sh" 2>&1 | tee "$LOG" | tail -5
    RC=${PIPESTATUS[0]}
    DUR=$((SECONDS - START))
    echo "Cell ff=${ff} rc=$RC duration=${DUR}s" | tee -a "$MANIFEST"
    if [ $RC -ne 0 ]; then
        echo "Cell ff=${ff} FAILED — stopping sweep." | tee -a "$MANIFEST"
        exit 1
    fi
done

echo "" | tee -a "$MANIFEST"
echo "# completed: $(date -Is)" >> "$MANIFEST"
