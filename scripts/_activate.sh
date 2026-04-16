#!/bin/bash
# Shared environment activation for Ashfall scripts.
# Sources Isaac Lab and adds Phoenix + Ashfall to PYTHONPATH.

set -euo pipefail

ASHFALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHOENIX_ROOT="${PHOENIX_ROOT:-$HOME/workspace/go2-phoenix}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"

if [ ! -d "$PHOENIX_ROOT" ]; then
    echo "ERROR: Phoenix not found at $PHOENIX_ROOT" >&2
    echo "Set PHOENIX_ROOT to the go2-phoenix directory." >&2
    exit 1
fi

export ASHFALL_ROOT PHOENIX_ROOT ISAACLAB_PATH
export PYTHONPATH="${ASHFALL_ROOT}/src:${PHOENIX_ROOT}/src:${PYTHONPATH:-}"

echo "[ashfall] ASHFALL_ROOT=$ASHFALL_ROOT"
echo "[ashfall] PHOENIX_ROOT=$PHOENIX_ROOT"
echo "[ashfall] ISAACLAB_PATH=$ISAACLAB_PATH"
