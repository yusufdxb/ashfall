#!/bin/bash
# Generate analysis report and plots from experiment results.
#
# Usage:
#   ./scripts/analyze.sh [results_dir]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_activate.sh"

RESULTS_DIR="${1:-$ASHFALL_ROOT/results}"

echo "[ashfall] Generating report from $RESULTS_DIR"
python3 -m ashfall.analysis.report "$RESULTS_DIR"

echo "[ashfall] Report written to $RESULTS_DIR/REPORT.md"
