#!/bin/bash
# Ashfall full A/B experiment: baseline train → adapted train → eval both.
# Run from the Ashfall repo root.
#
# Usage:
#   ./scripts/run_full_experiment.sh
#
# Prerequisites:
#   - Baseline already trained: checkpoints/ashfall-baseline/latest.pt in Phoenix
#   - Synth failures generated: data/failures/*.parquet in Ashfall

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASHFALL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PHOENIX_ROOT="${PHOENIX_ROOT:-$HOME/workspace/go2-phoenix}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/Sim/IsaacLab}"
ISAAC_VENV="${ISAAC_VENV:-$HOME/Sim/isaac-sim-venv}"

# Activate Isaac Sim venv
if [[ -z "${VIRTUAL_ENV:-}" && -f "$ISAAC_VENV/bin/activate" ]]; then
    source "$ISAAC_VENV/bin/activate"
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONPATH="${PHOENIX_ROOT}/src:${ASHFALL_ROOT}/src:${PYTHONPATH:-}"

RESULTS="$ASHFALL_ROOT/results"

echo "============================================"
echo " Ashfall Full A/B Experiment"
echo " Phoenix: $PHOENIX_ROOT"
echo " Ashfall: $ASHFALL_ROOT"
echo " Results: $RESULTS"
echo "============================================"

# --- Step 2: Adapted training (fine-tune baseline with failure curriculum) ---
echo ""
echo ">>> Step 2: Adapted training (failure_sample_fraction=0.25)"
cd "$PHOENIX_ROOT"
"$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.adaptation.fine_tune \
    --config configs/train/adaptation_ashfall.yaml \
    --trajectory-dir "$ASHFALL_ROOT/data/failures" \
    --num-envs 4096 \
    --max-iterations 200 \
    --device cuda:0

# --- Step 3: Evaluate both conditions on all 3 environments ---
echo ""
echo ">>> Step 3: Evaluating baseline on flat, rough, slippery"
for env in flat rough slippery; do
    echo "  -- baseline / $env --"
    "$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.training.evaluate \
        --checkpoint "checkpoints/ashfall-baseline/latest.pt" \
        --env-config "configs/env/${env}.yaml" \
        --num-envs 32 \
        --num-episodes 128 \
        --metrics-out "$RESULTS/ashfall-baseline/metrics/metrics_${env}.json" \
        --device cuda:0
done

echo ""
echo ">>> Step 3b: Evaluating adapted on flat, rough, slippery"
for env in flat rough slippery; do
    echo "  -- adapted / $env --"
    "$ISAACLAB_PATH/isaaclab.sh" -p -m phoenix.training.evaluate \
        --checkpoint "checkpoints/ashfall-adapted/latest.pt" \
        --env-config "configs/env/${env}.yaml" \
        --num-envs 32 \
        --num-episodes 128 \
        --metrics-out "$RESULTS/ashfall-adapted/metrics/metrics_${env}.json" \
        --device cuda:0
done

echo ""
echo "============================================"
echo " Experiment complete. Generating report..."
echo "============================================"

# --- Step 4: Generate comparison report ---
cd "$ASHFALL_ROOT"
python3 -m ashfall.analysis.report "$RESULTS"

echo ""
echo ">>> Report written to $RESULTS/REPORT.md"
echo ">>> Done."
