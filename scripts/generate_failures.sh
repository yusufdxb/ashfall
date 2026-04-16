#!/bin/bash
# Generate synthetic failure trajectories for all 6 modes.
# These serve as curriculum training data until real hardware failures are collected.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_activate.sh"

OUTPUT_DIR="${1:-$ASHFALL_ROOT/data/failures}"
N_VARIANTS="${2:-3}"

echo "[ashfall] Generating synthetic failures to $OUTPUT_DIR ($N_VARIANTS variants per mode)"
python3 -c "
from ashfall.synth.generator import generate_all_failures
paths = generate_all_failures('$OUTPUT_DIR', n_variants=$N_VARIANTS)
print(f'Generated {len(paths)} failure files')
"
