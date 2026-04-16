#!/bin/bash
# Run an ablation sweep.
#
# Usage:
#   ./scripts/run_ablation.sh configs/experiments/ablation_sweep.yaml
#
# Generates one experiment per sweep cell and runs them sequentially.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_activate.sh"

CONFIG="${1:?Usage: $0 <ablation_sweep.yaml>}"

echo "[ashfall] Generating ablation sweep from $CONFIG"
python3 -c "
import yaml
from pathlib import Path
from ashfall.experiment.runner import load_experiment_config, ExperimentRunner
from ashfall.experiment.sweep import generate_sweep, SweepConfig
from ashfall.experiment.schema import AblationAxis

cfg = load_experiment_config('$CONFIG')
axes = cfg.ablations
sweep = SweepConfig(base_experiment=cfg, axes=axes)
configs = generate_sweep(sweep)
print(f'Sweep has {len(configs)} cells')

runner = ExperimentRunner(Path('$ASHFALL_ROOT'), Path('$PHOENIX_ROOT'))
for i, cell_cfg in enumerate(configs):
    print(f'--- Cell {i+1}/{len(configs)}: {cell_cfg.name} ---')
    run_dir = runner.prepare_run(cell_cfg)
    print(f'  Run dir: {run_dir}')
    print(f'  Execute: bash {run_dir}/commands.sh')
"

echo "[ashfall] Ablation configs generated. Execute each cell's commands.sh in Isaac Lab context."
