#!/usr/bin/env bash
# 主线 vLLM + `vllm-xtu-moe` 插件(**零核心补丁**路径)。
#
# 与 `serve_lk_port.sh` 的区别:
#   * `serve_lk_port.sh` 走的是 **lk 编排 fork**(Lvllmds4-x + 我们的移植提交),
#     性能数字最好,但要跟着 fork 走;
#   * 本脚本走 **mainline vLLM + 插件**(`vllm.general_plugins` 入口),
#     不需要改任何主线源码 —— 见 `docs/UPSTREAM_DRIFT.md` 的"形态 0"。
#
# 用法:
#   ENV=/path/to/mainline-venv TAG=mymain PORT=8071 bash scripts/serve_mainline.sh
#   XIAOTU_OOT_OVERRIDE=0 ...   # 换成"主线 RoutedExperts + CPU backend"路径
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

TAG="${TAG:-mainline_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8071}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAXLEN="${MAXLEN:-8192}"
SEQS="${SEQS:-8}"
MBT="${MBT:-256}"                    # 与 fork 协议对齐(实测:不设时主线默认很大)
# chunked prefill:**建议保持主线默认(开)**。
#   * 关掉它时,主线要求 MBT >= max_model_len(否则直接报错),而且预填充会整段进
#     我们的 CPU 引擎(qlen=MBT)⇒ 一次 256 token 的预填充就要 26 ms/层 × 43 ≈ 1.1 s;
#   * 开着时,解码步里可能混进预填充块(qlen=257)⇒ 解码也会付这份钱。
# ⇒ 两种都不理想,根因是**形态 B 没有"预填充专用路径"**(fork 里有 `_cpu_prefill` /
#   `_gpu_prefill` 的分流)。见 NOTES §334s。默认先取主线默认(开)。
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
# 加载策略:实测插件路径读分片 2.5-4.8 s/片(fork 路径 0.6 s/片);
# 主线日志明确建议 EXT4 上用 prefetch 强制预取。
LOAD_STRATEGY="${LOAD_STRATEGY:-prefetch}"
# A100(SM80)没有 fp8e4nv:Triton 的 PackSeq 等内核在 JIT warmup 时会报
# "type fp8e4nv not supported in this architecture" ⇒ 直接关掉预热(实测 ml08modeB)
KERNEL_WARMUP="${KERNEL_WARMUP:-0}"
# 引擎线程:插件侧旋钮(每 rank 12 CCD ⇒ 5 核/CCD = 60)
THREADS="${THREADS:-60}"
# 额外常驻 GPU 的 MoE 层(与 lk 的 LVLLM_GPU_RESIDENT_MOE_LAYERS 同义)
RESIDENT="${RESIDENT-0-11}"
# 1 = OOT 模型覆盖(默认,零补丁);0 = 主线 RoutedExperts + 我们的 CPU 后端
OOT="${XIAOTU_OOT_OVERRIDE:-1}"
EXTRA_ENV="${EXTRA_ENV:-}"

OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
rm -f /dev/shm/xiaotu_ep_*.bin 2>/dev/null || true

ARGS=(
  --model "$CKPT"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --trust-remote-code
  --served-model-name DeepSeek-V4-Flash-xiaotu
)
if [ -n "$MBT" ]; then ARGS+=(--max-num-batched-tokens "$MBT"); fi
if [ "$CHUNKED_PREFILL" = "0" ]; then ARGS+=(--no-enable-chunked-prefill); fi
if [ "$CHUNKED_PREFILL" = "1" ]; then ARGS+=(--enable-chunked-prefill); fi
if [ -n "$LOAD_STRATEGY" ]; then ARGS+=(--safetensors-load-strategy "$LOAD_STRATEGY"); fi
if [ "$KERNEL_WARMUP" = "0" ]; then
  ARGS+=(--kernel-config '{"enable_jit_warmup": false}')
fi

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS gpu_util=$GPU_UTIL"
  echo "mbt='$MBT' chunked='$CHUNKED_PREFILL' threads=$THREADS resident='$RESIDENT' oot=$OOT load_strategy='$LOAD_STRATEGY' extra_env='$EXTRA_ENV'"
  echo "env=$ENV"; echo "ckpt=$CKPT"; date -Is
} > "$OUTDIR/$TAG.env"

# 等显存真正释放
for _g in $(echo "$GPUS" | tr ',' ' '); do
  for _i in $(seq 1 60); do
    _used=$(nvidia-smi --id="$_g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$_used" ] && [ "$_used" -lt 1024 ] && break
    sleep 5
  done
done

cd /tmp
export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-60}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}" \
  VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS="${VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS:-30}" \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_OOT_OVERRIDE="$OOT" \
  XIAOTU_MOE_THREADS="$THREADS" \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS="$RESIDENT" \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[mainline] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[mainline] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[mainline] server exited early; tail:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[mainline] timeout; tail:"; tail -30 "$LOG"; exit 1
