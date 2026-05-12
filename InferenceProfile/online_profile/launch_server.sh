#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/data/public_models/Qwen3-30B-A3B}"

echo "Model: ${MODEL_PATH}"
nsys profile \
  -o vllm_server_profile \
  --force-overwrite=true \
  --trace-fork-before-exec=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end repeat \
  vllm serve "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port 23334 \
    --trust-remote-code \
    --enforce-eager \
    --profiler-config.profiler cuda
