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
    --enable-layerwise-nvtx-tracing \
    --enforce-eager \
    --profiler-config.profiler cuda

# nsys profile: 启动 NVIDIA Nsight Systems 采集，并把后面的 vllm serve 作为被采集进程。
# -o vllm_server_profile: nsys 输出文件名前缀，最终生成 vllm_server_profile*.nsys-rep。
# --force-overwrite=true: 允许覆盖同名 nsys 输出文件。
# --trace-fork-before-exec=true: 跟踪 fork/exec 出来的子进程，避免漏掉 vLLM EngineCore/worker 的 CUDA kernel。
# --capture-range=cudaProfilerApi: 只在程序触发 CUDA profiler start/stop API 的区间内采集。
# --capture-range-end repeat: 每次 stop 后结束当前采集区间，但继续等待下一次 start，适合 server 常驻场景。
# --enable-layerwise-nvtx-tracing: 添加 layer/module 级别 NVTX 标记，方便在 nsys trace 中看模型内部阶段。
# --profiler-config.profiler cuda: 让 vLLM 使用 CUDA profiler API 配合 bench 的 --profile 控制 nsys 采集区间。
