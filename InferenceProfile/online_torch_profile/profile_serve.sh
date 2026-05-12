#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/data/public_models/Qwen3-30B-A3B}"

echo "Model: ${MODEL_PATH}"
vllm bench serve \
  --backend vllm \
  --model "${MODEL_PATH}" \
  --base-url http://127.0.0.1:23334 \
  --dataset-name random \
  --random-input-len 10240 \
  --random-output-len 256 \
  --num-prompts 8 \
  --temperature 0 \
  --profile
