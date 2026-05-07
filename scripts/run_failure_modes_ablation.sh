#!/bin/bash
# Run the v0.4.0 failure-mode subset ablation (6 cells, fixed ff=0.5).
#
# Cells:
#   all_modes              : sanity reproducer of v0.3.0 ff=0.5
#   slip_only              : friction-axis only
#   command_mismatch_only  : tracking-error replay only
#   slip_plus_cm           : step-count-majority channels
#   severe_only            : attitude + collapse only
#   severe_plus_slip       : minimal severe + friction set
#
# Each cell warm-starts from the v0.2.0 ashfall-baseline checkpoint and
# fine-tunes for 200 iters on slippery with the cell's mode subset.
# Evaluation is 128 eps x 32 envs on rough + slippery.
#
# DO NOT run during a session where the GPU is reserved for another
# project; check `nvidia-smi` and confirm before invoking.
#
# Usage:
#   ./scripts/run_failure_modes_ablation.sh
#
# Output:
#   results/ablation_failure_modes_<cell>/<stamp>/{config.yaml,
#     adapt_override.yaml, commands.sh, metrics/}
#   results/_logs/failure_modes/<cell>.log
#   results/_logs/failure_modes/manifest.txt

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_activate.sh"

LOGS_DIR="$ASHFALL_ROOT/results/_logs/failure_modes"
mkdir -p "$LOGS_DIR"

CELLS=(
  all_modes
  slip_only
  command_mismatch_only
  slip_plus_cm
  severe_only
  severe_plus_slip
)

MANIFEST="$LOGS_DIR/manifest.txt"
{
  echo "# Ashfall v0.4.0 failure-mode subset ablation manifest"
  echo "# fixed ff=0.5, single seed=42"
  echo "# started: $(date -Is)"
} > "$MANIFEST"

for cell in "${CELLS[@]}"; do
  CONFIG="$ASHFALL_ROOT/configs/ablations/failure_modes/${cell}.yaml"
  LOG="$LOGS_DIR/${cell}.log"
  echo "" | tee -a "$MANIFEST"
  echo "=== Cell ${cell} === $(date -Is)" | tee -a "$MANIFEST"
  if [ ! -f "$CONFIG" ]; then
    echo "MISSING config: $CONFIG" | tee -a "$MANIFEST"
    continue
  fi

  # Prepare run dir.
  RUN_DIR=$(python3 -c "
from ashfall.experiment.runner import load_experiment_config, ExperimentRunner
from pathlib import Path
cfg = load_experiment_config('$CONFIG')
runner = ExperimentRunner(Path('$ASHFALL_ROOT'), Path('$PHOENIX_ROOT'))
print(runner.prepare_run(cfg))
")
  echo "  run_dir=$RUN_DIR" | tee -a "$MANIFEST"

  # Execute. Tee everything to a per-cell log.
  START=$SECONDS
  bash "$RUN_DIR/commands.sh" 2>&1 | tee "$LOG" | tail -5
  RC=${PIPESTATUS[0]}
  DUR=$((SECONDS - START))
  echo "Cell ${cell} rc=${RC} duration=${DUR}s" | tee -a "$MANIFEST"
  if [ $RC -ne 0 ]; then
    echo "Cell ${cell} FAILED. Stopping ablation." | tee -a "$MANIFEST"
    exit 1
  fi
done

{
  echo ""
  echo "# completed: $(date -Is)"
} >> "$MANIFEST"

echo ""
echo "[ashfall] Ablation complete. Generate report with:"
echo "  PYTHONPATH=src .venv/bin/python -m ashfall.analysis.sweep_report results"
