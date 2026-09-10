#!/usr/bin/env bash
# nsys 抓取:单进程(EAGER + 无 multiprocessing)下单卡 DS-V4 的时间线,
# 用于看清"每个 step 的 1s 到底花在哪"(CPU host 回调 / D2H / H2D / GPU kernel)。
# 用法: bash scripts/tune_nsys.sh <TAG> [EAGER=1]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-dsv4_nsys}"
EAGER="${EAGER:-1}"
PORT="${PORT:-8081}"
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH
export CUDA_VISIBLE_DEVICES="${GPUS:-2}" VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_SINGLECOPY=1
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=7200
export OMP_NUM_THREADS="${OMP:-48}" XIAOTU_MOE_THREADS="${THREADS:-96}"
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${PREFILL_MIN:-384}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p "$ROOT/report/tuning/nsys"
MODEL_DIR="${MODEL:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
ARGS=(
  "$MODEL_DIR"
  --host 0.0.0.0 --port "$PORT" --tensor-parallel-size 1
  --max-model-len "${MAXLEN:-262144}" --max-num-seqs "${SEQS:-64}"
  --max-num-batched-tokens "${MAX_NBT:-8192}" --gpu-memory-utilization "${GPU_UTIL:-0.90}"
  --no-enable-prefix-caching --kernel-config.enable_jit_warmup=false
  --kv-cache-dtype "${KV_DTYPE:-fp8_ds_mla}"
  --kv-cache-memory-bytes "${KV_MEM_BYTES:-12884901888}"
  --served-model-name DeepSeek-V4-Flash-xiaotu
)
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
exec /usr/local/cuda/bin/nsys profile -t cuda,nvtx --cuda-memory-usage=false \
  --force-overwrite true -o "$ROOT/report/tuning/nsys/$TAG" \
  vllm serve "${ARGS[@]}"
