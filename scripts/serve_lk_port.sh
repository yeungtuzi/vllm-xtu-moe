#!/usr/bin/env bash
# serve_lk_port.sh — 用 lk 的全套编排链 + 我们的计算引擎(xiaotu_moe)启服务。
#
# 这不是新功能,而是把上游现成的东西拼起来:
#   * vLLM 侧 = `Lvllmds4-x`(guqiong96/Lvllmds4-x,SM80 分支)的 lk 编排,
#     只有两个文件与上游不同(`fused_moe/routed_experts.py` + `runner/moe_runner.py`),
#     且只做了 `lk_moe` -> `xiaotu_moe` 的机械改名(见该仓库 commit "port: rebind ...");
#   * 引擎   = 本仓的 `xiaotu_moe`(Apache-2.0),以 `pip install` 装进该 env;
#   * 启动参数 = lk 生产脚本 `process_data/scripts/dsv4.sh` 逐条照搬,
#     只把 max_model_len / max_num_seqs / gpu_util 改成我们本机测试用的值。
#
# 环境:conda env `lkxtu`(= `lvllmds4-x` 的干净克隆 + 上述两个文件 + 我们的引擎)。
# 注意:`lkxtu/bin/vllm` 是 cp -a 留下的**指向旧 env 的符号链接**,所以必须用
# `python -m vllm.entrypoints.openai.api_server` 启动,不能调 `vllm`。
#
# 用法:
#   bash scripts/serve_lk_port.sh                 # 无投机(干净基线)
#   SPEC=1 bash scripts/serve_lk_port.sh          # 带 dspark5(lk 生产同款)
#   TAG=xxx PORT=8070 THREADS=48 ... bash scripts/serve_lk_port.sh
#
# License: Apache-2.0
set -euo pipefail

ENV="${ENV:-/home/user/anaconda3/envs/lkxtu}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

TAG="${TAG:-lkport_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8070}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-8}"
MBT="${MBT:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
THREADS="${THREADS:-48}"          # lk 生产是每卡 48(LK_THREADS)
MINBATCH="${MINBATCH:-1024}"      # lk 生产同值:开 GPU prefill
SPEC_ON="${SPEC:-0}"
EAGER="${EAGER:-0}"   # 1 = 加 --enforce-eager(避开 lk 的 _cpu_prefill 在捕获期同步的问题)

OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/report/tuning/logs"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
rm -f /dev/shm/xiaotu_ep_*.bin 2>/dev/null || true

ARGS=(
  --model "$CKPT"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAXLEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --trust-remote-code
  --served-model-name DeepSeek-V4-Flash-xiaotu
  --compilation_config.cudagraph_mode FULL_DECODE_ONLY
  --enable-prefix-caching
  --enable-chunked-prefill
  --max-num-batched-tokens "$MBT"
  --dtype bfloat16
  --max-num-seqs "$SEQS"
  --enable-auto-tool-choice
  --kv-cache-dtype fp8_ds_mla
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"enable_thinking": true}'
  --disable-custom-all-reduce
)
if [ "$EAGER" = "1" ]; then
  ARGS+=(--enforce-eager)
fi
if [ "$SPEC_ON" = "1" ]; then
  ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}')
fi

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS mbt=$MBT"
  echo "gpu_util=$GPU_UTIL threads=$THREADS minbatch=$MINBATCH spec=$SPEC_ON eager=$EAGER"
  echo "env=$ENV"
  date -Is
} > "$OUTDIR/$TAG.env"

cd /tmp   # neutral CWD: see comment above
export PATH="$ENV/bin:$PATH"   # ninja/flashinfer JIT need to be visible to workers
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  LVLLM_MOE_NUMA_ENABLED=1 \
  LK_THREADS="$THREADS" \
  OMP_NUM_THREADS=1 \
  LK_THREAD_BINDING=CPU_CORE \
  LVLLM_GPU_PREFETCH_WINDOW=1 \
  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE="$MINBATCH" \
  LK_POWER_SAVING=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  HF_HUB_OFFLINE=1 \
  "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[lk_port] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

for _ in $(seq 1 900); do           # 最长等 75 分钟
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[lk_port] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[lk_port] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[lk_port] timeout; tail:"; tail -25 "$LOG"; exit 1
