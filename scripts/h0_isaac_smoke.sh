#!/usr/bin/env bash
# H0 matched-pair smoke against Isaac Lab through the Phoenix backend.
#
# Exercises the simulator path end to end with a spec too small to reach
# alpha (see configs/h0/smoke_friction_slip.json). The evidence bundle it
# writes is an implementation check, not a research result, and its
# verdict.json says so through evidence_kind and spec_id.
#
# Usage:
#   PHOENIX_ROOT=/path/to/go2-phoenix ISAAC_PYTHON=/path/to/isaac-lab/python \
#     scripts/h0_isaac_smoke.sh [output_dir]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PHOENIX_ROOT:?set PHOENIX_ROOT to the go2-phoenix checkout}"
: "${ISAAC_PYTHON:?set ISAAC_PYTHON to the Isaac Lab interpreter}"
OUT="${1:-$REPO_ROOT/results/h0_smoke}"
[ "$#" -gt 0 ] && shift
export PHOENIX_ROOT OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$PHOENIX_ROOT/src:${PYTHONPATH:-}"
cd "$REPO_ROOT"
exec "$ISAAC_PYTHON" -m ashfall.cli h0 \
  --backend phoenix \
  --backend-config configs/h0/phoenix_flat_v3b.json \
  --spec configs/h0/smoke_friction_slip.json \
  --allow-config-block terrain \
  --allow-config-block termination \
  --allow-config-block observation.include \
  --allow-config-block robot.init_state \
  --allow-config-block robot.actuator \
  --allow-config-block perturbation.push_interval_s \
  --phoenix-repo "$PHOENIX_ROOT" \
  --output "$OUT" \
  "$@"
