#!/usr/bin/env bash
# Qwen3.8-Flash-Next(官方 FP8)在单卡 SM80 上的调试服务 —— **默认 GPU2 / TP=1**。
#
# 纪律(AGENTS.md「自服务环境纪律」,用户 2026-09-29):
#   * 8070 是本 agent 自身推理后端 ⇒ **本脚本默认不碰 GPU0/1、不用 8070**;
#   * 调试需要 GPU 时用 **GPU2 + TP=1**;要动 GPU0/1 必须显式获得用户许可。
#   本脚本默认 `GPUS=2 TP=1 PORT=8140 MAXLEN=8192` —— 全部是"单卡调试"口径。
#
# 为什么官方 FP8 可行(依据 dev-docs/patches/probe_official_fp8_gate.py 的实测):
#   * 检查点在本地 HF 缓存(173 GiB,131 分片),`quant_method: fp8` ⇒ vLLM 原生 `Fp8Config`;
#     本插件按 `FusedMoEMethodBase` **泛化打 shim** ⇒ `Fp8MoEMethod` 自动走 CPU 专家;
#   * 权重分布(按张量头部精确统计):路由专家 **114.9 GiB**(留 CPU)、
#     **PLE/ngram 表 47.8 GiB**(必须 `XIAOTU_PLE_CPU=1` 走 UVA,否则单卡必 OOM,
#     见 vllm_xiaotu_moe/ple_offload.py 头部注释)、其余非专家仅 ~10 GiB;
#   * ⇒ PLE offload 后 GPU 侧非专家 ~10 GiB,40GB 单卡放得下。
#
# 用法:
#   bash scripts/serve_qwen38.sh                       # dummy? 不,默认真权重
#   LOAD=dummy bash scripts/serve_qwen38.sh            # 只验架构/显存,不读 173GiB(首选第一步)
#   MAXLEN=4096 GPUS=2 TP=1 bash scripts/serve_qwen38.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
# 官方 FP8 检查点(本地 HF 缓存快照;可用 CKPT= 覆盖)
CKPT="${CKPT:-$(ls -d /home/user/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/*/ 2>/dev/null | head -1)}"

TAG="${TAG:-qwen38_$(date +%m%d_%H%M%S)}"
PORT="${PORT:-8140}"        # ⚠️ 不能是 8070(生产)/8080(LMCache HTTP)/5555
GPUS="${GPUS:-2}"           # 【纪律】调试只用 GPU2
TP="${TP:-1}"               # 【纪律】TP=1
MAXLEN="${MAXLEN:-8192}"    # 【纪律】上下文尽量短
MBT="${MBT:-2048}"
SEQS="${SEQS:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
THREADS="${THREADS:-60}"
LOAD="${LOAD:-auto}"        # auto=读真权重;dummy=不读盘,只验架构与显存
EAGER="${EAGER:-1}"         # 单卡调试先从全 eager 起,稳了再上图
PLE_CPU="${XIAOTU_PLE_CPU:-1}"   # 必须有:PLE 47.8GiB 走主机内存(UVA)
GPU_PREFILL_MIN="${GPU_PREFILL_MIN:-0}"  # 单卡调试先关 GPU 预填
KV_DTYPE="${KV_DTYPE:-auto}"
TOOL_PARSER="${TOOL_PARSER:-}"
REASONING_PARSER="${REASONING_PARSER:-}"
OUTDIR="${OUTDIR:-$ROOT/dev-docs/report/tuning/logs}"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"
# env 桥写【仓库内】,绝不覆盖生产用的 /tmp/xiaotu_env
XTU_ENV_FILE="${XTU_ENV_FILE_OVERRIDE:-$OUTDIR/$TAG.envfile}"

if [ -z "$CKPT" ] || [ ! -f "$CKPT/config.json" ]; then
  echo "[qwen38] 检查点未找到: '$CKPT'(可显式 CKPT=<snapshot 目录>)" >&2; exit 2
fi
command -v numactl >/dev/null 2>&1 && NCTL=(numactl --interleave=all) || NCTL=()

{
  echo "XIAOTU_MOE_THREADS=$THREADS"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${SPIN_IDLE_US:-300}"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=$GPU_PREFILL_MIN"
  echo "XIAOTU_PLE_CPU=$PLE_CPU"
  echo "XIAOTU_RELEASE_SOURCE=${XIAOTU_RELEASE_SOURCE:-1}"
} > "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE

ARGS=(
  --model "$CKPT"
  --served-model-name Qwen3.8-Flash-Next
  --host 127.0.0.1 --port "$PORT"
  --tensor-parallel-size "$TP"
  --dtype bfloat16
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MBT"
  --max-num-seqs "$SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --load-format "$LOAD"
  --trust-remote-code
  --enable-chunked-prefill
  --disable-custom-all-reduce
)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ -n "$TOOL_PARSER" ] && ARGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ "${PROMPT_TOKENS_DETAILS:-1}" = "1" ] && ARGS+=(--enable-prompt-tokens-details)

echo "[qwen38] tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN mbt=$MBT load=$LOAD"
echo "[qwen38] PLE_CPU=$PLE_CPU(必须=1,否则单卡 OOM)  eager=$EAGER  gpu_prefill_min=$GPU_PREFILL_MIN"
echo "[qwen38] ckpt=$CKPT"
echo "[qwen38] log=$LOG   envfile=$XTU_ENV_FILE"

if [ "${DRYRUN:-0}" = "1" ]; then
  printf '[qwen38] DRYRUN gpus=%s tp=%s: ' "$GPUS" "$TP"
  printf '%q ' env CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
    VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_ENV_FILE="$XTU_ENV_FILE" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
  printf '\n'; exit 0
fi

cd "$OUTDIR"; mkdir -p run; cd run
export CUDA_VISIBLE_DEVICES="$GPUS"
nohup env \
  PYTHONPATH="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}" \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}" \
  VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-120}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_ENV_FILE="$XTU_ENV_FILE" \
  XIAOTU_PLE_CPU="$PLE_CPU" \
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GPU_PREFILL_MIN" \
  XIAOTU_MOE_THREADS="$THREADS" \
  OMP_NUM_THREADS=1 \
  "${NCTL[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[qwen38] pid=$(cat "$OUTDIR/$TAG.pid")"

DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[qwen38] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[qwen38] server exited early; tail:"; tail -40 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[qwen38] TIMEOUT; tail:"; tail -40 "$LOG"; exit 1
