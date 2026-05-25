#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/data/public_models/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-2}"
SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-./sglang_trace}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SGLANG_TORCH_PROFILER_DIR

mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Torch profiler trace dir: ${SGLANG_TORCH_PROFILER_DIR}"
echo "Layerwise NVTX tracing/marker: enabled"
echo "CUDA graph: disabled"

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --trust-remote-code \
  --enable-layerwise-nvtx-marker \
  --disable-cuda-graph
