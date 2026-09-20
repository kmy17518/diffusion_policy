#!/usr/bin/env bash
# Variant branch goal-image-late: the `image-late` condition of the two-task conditioning smoke matrix, trained by this
# checkout's code. Thin wrapper over scripts/b1k/run_navpickup_conditioning_smoke.sh (same overrides, e.g.
# DP_GPU_UUID, DP_CORES, DP_BATCH_SIZE, DP_MAX_STEPS; extra arguments go to train_b1k.py).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export DP_CONDITION=image-late DP_CHECKOUT="$HERE"
exec bash "$HERE/scripts/b1k/run_navpickup_conditioning_smoke.sh" "$@"
