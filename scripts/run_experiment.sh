#!/bin/bash
# Run a single Ashfall experiment.
#
# Usage:
#   ./scripts/run_experiment.sh configs/experiments/baseline.yaml
#   ./scripts/run_experiment.sh configs/experiments/adapted.yaml
#
# This prepares the run directory, then executes each phase inside
# Isaac Lab's Python context.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_activate.sh"

CONFIG="${1:?Usage: $0 <experiment.yaml>}"

echo "[ashfall] Preparing experiment from $CONFIG"
RUN_DIR=$(python3 -c "
from ashfall.experiment.runner import load_experiment_config, ExperimentRunner
from pathlib import Path

cfg = load_experiment_config('$CONFIG')
runner = ExperimentRunner(Path('$ASHFALL_ROOT'), Path('$PHOENIX_ROOT'))
run_dir = runner.prepare_run(cfg)
print(run_dir)
")

echo "[ashfall] Run directory: $RUN_DIR"
echo "[ashfall] Executing phases..."

# Execute the generated commands
bash "$RUN_DIR/commands.sh"

echo "[ashfall] Experiment complete. Results in $RUN_DIR/metrics/"
