#!/usr/bin/env bash
# 参考实现(LvLLM 2.5 + lk_moe)在 **DeepSeek-V4.1-Flash** 上的常驻内存足迹测量。
#
# 参数逐字来自 `Lvllm/RELEASE_NOTES.md` 指向的 `commands/dsv41_serve_tp2_3090_dspark.sh`,
# 只改两处以适配本机:GPU 0,1(本机是 3×A100-40G)、`LK_THREADS` 48→60
# (本机 192 物理核,我们自己的规则是每 CCD 4-5 核 ⇒ TP=2 每 rank 12 CCD × 5 = 60;
#  与 §500 的 A/B 保持同一线程口径)。
#
# 产物:report/tuning/logs/ref41.memfoot(逐次采样 + 末行 summary)
# 用法:TAG=ref41 PORT=8200 bash report/tuning/probes/ref_v41_mem.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/lvllm}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8200}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.95}"
MAXLEN="${MAXLEN:-65536}"
MBT="${MBT:-8192}"
SEQS="${SEQS:-2}"
THREADS="${THREADS:-60}"
TAG="${TAG:-ref41}"

export TAG PORT
RUN_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS=$THREADS OMP_NUM_THREADS=1 LK_THREAD_BINDING=CPU_CORE \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_POWER_SAVING=1 LVLLM_EMBEDDING_NUMA_ENABLED=1 \
VLLM_USE_V2_MODEL_RUNNER=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 LVLLM_GPU_PREFETCH_WINDOW=1 \
VLLM_ENGRAM_DROP_PAGE_CACHE=0 FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0"

RUN_ENV="$RUN_ENV" READY_TIMEOUT="${READY_TIMEOUT:-1800}" \
  bash "$ROOT/report/tuning/probes/mem_footprint.sh" \
  "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name dsv41 --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MBT" --max-num-seqs "$SEQS" \
  --dtype bfloat16 --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
