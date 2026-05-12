#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_PATH="${MODEL_PATH:-/data/public_models/Qwen3-30B-A3B}"
TORCH_PROFILER_DIR="${TORCH_PROFILER_DIR:-${SCRIPT_DIR}/torch_profiler_traces}"

mkdir -p "${TORCH_PROFILER_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Torch profiler output: ${TORCH_PROFILER_DIR}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 23334 \
  --trust-remote-code \
  --enforce-eager \
  --enable-layerwise-nvtx-tracing \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":true,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":true,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"active_iterations\":5,\"warmup_iterations\":0,\"wait_iterations\":0}"
