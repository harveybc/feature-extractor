#!/usr/bin/env bash
# Pinned FS4 runner executable for tools/fs4_worker.py (predictor): stdin claim JSON -> stdout one JSON.
# Environment (set in the worker slot env file):
#   FS4_RUNNER_PYTHON   interpreter with tensorflow/keras/pyarrow (required)
#   FS4_INPUT_<ROLE>    governed input parquet per role (eurusd_ps1_batch_001..003, eth_train)
#   FS4_OUTPUT_ROOT     durable local directory for <task_id>/result.json and chosen weights
#   FS4_GPU_UUID        physical UUID, TRAINED_ENCODER slots only (the worker also sets CUDA_VISIBLE_DEVICES)
#   FS4_LD_LIBRARY_PATH_FILE  optional file whose single line becomes LD_LIBRARY_PATH (cu12 loader recipe)
set -euo pipefail
here="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
: "${FS4_RUNNER_PYTHON:?FS4_RUNNER_PYTHON is required}"
if [[ -n "${FS4_LD_LIBRARY_PATH_FILE:-}" ]]; then
  export LD_LIBRARY_PATH="$(< "$FS4_LD_LIBRARY_PATH_FILE")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
cd "$here"
exec "$FS4_RUNNER_PYTHON" -m app.fs4_task_runner "$@"
