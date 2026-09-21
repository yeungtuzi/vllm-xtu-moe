#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MoE 专家层 **GPU 常驻**的收益扫描(TP=2 / GLM-5.3-Flash)。
#
# 为什么这样设计(2026-09-21 用户要求"试验 moe 常驻若干层的实际收益"):
#
#   * **前提是"必须保证的上下文长度"**。所以 `MAXLEN` 与 `SEQS` 固定 ⇒ `KV_CACHE_BYTES`
#     固定,常驻层只能吃**剩下的**显存。随着常驻层数上升,总会在某一档放不下 ——
#     那一档就是"在这个上下文长度下的常驻上限",这正是最需要的数据。
#     反过来(为了多放常驻层而缩 KV)会把"延迟收益"和"并发损失"混在一起,测不出结论。
#
#   * **`XIAOTU_MOE_RESIDENT_BUDGET_GB=0`(不限)**,让 K 层真的被尝试;放不下就让它
#     失败/降级,而不是被我偷偷截断 —— 否则测到的是我设的上限,不是硬件的上限。
#
#   * **强制验证实际常驻层数**:日志里数 `GPU-resident` 行。serve 脚本会把调用者的
#     `XIAOTU_*` 透传进 env-bridge 文件(第 203-207 行),但"设了没生效"这个坑
#     本次会话已经踩过两次,所以**每次都必须回读日志确认**。
#
#   * 历史参考(`dev-docs/report/tuning/NOTES.md` §309/§320,旧代码):
#     每层常驻 ≈ **0.70 ms/token(C=1)**、**1.70 ms/token(C=4)**;TP=2/2×40GB 上限 **11 层**。
#     本轮是在**修掉"常驻层被重复分配"的 bug 之后**重测,所以数字可能与历史不同。
#
# 用法:
#   bash scripts/bench_resident_sweep.sh
#   KS="0 3 5 8 11" MAXLEN=32768 SEQS=8 CS="1 4" bash scripts/bench_resident_sweep.sh
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENVDIR="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENVDIR/bin/python"
export PATH="$ENVDIR/bin:$PATH" HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export XTU_TREE="${XTU_TREE:-/home/user/lvllm/vllm-up-133b71e0b}"

OUT="${OUT:-/tmp/resident}"; mkdir -p "$OUT"
LOGD="$ROOT/logs"; mkdir -p "$LOGD"
CKPT_G=/home/user/.cache/modelscope/models/ZhipuAI--GLM-5.3-Flash/snapshots/master
GIB=1073741824
RULER_GLM=48942                     # tok/GiB,scripts/serve_glm53_mainline.sh 的既有标尺

KS="${KS:-0 3 5 8 11}"              # 要扫的常驻层数(k=0 是基线)
MAXLEN="${MAXLEN:-32768}"           # **固定**:必须保证的上下文长度
SEQS="${SEQS:-8}"                   # **固定**:并发上限 ⇒ KV 需求固定
CS="${CS:-1 4}"                     # 测哪些并发
PROMPT="${PROMPT:-128}"             # 短 prompt:量的是**解码**(常驻层的作用域)
OUTLEN="${OUTLEN:-128}"
N="${N:-8}"
TAG_PREFIX="${TAG_PREFIX:-res}"

# KV cap:固定值。池必须装下 max(MAXLEN, SEQS × PROMPT) 个 token,再留 25% 余量。
KV_CAP=$("$PY" -c "
import sys
ruler, maxlen, seqs, prompt = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
need = max(maxlen, seqs * prompt)
print(int(need / ruler * 1024**3 * 1.25))" "$RULER_GLM" "$MAXLEN" "$SEQS" "$PROMPT")

srv_log() { echo "$LOGD/$1.log"; }

freegpu() { # 只在加载之前调用
  local f p
  for f in "$LOGD"/${TAG_PREFIX}*.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && { kill -9 -"$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" 2>/dev/null; kill -9 "$p" 2>/dev/null; }
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  local i=0
  while [ $i -lt 60 ]; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || return 0
    sleep 5; i=$((i+1))
  done
}

wait_ready() { # port pidfile log timeout_s
  local port=$1 pidf=$2 logf=$3 tmo=${4:-1800} i=0
  while [ $i -lt $(( tmo / 5 )) ]; do
    curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
    if [ -f "$pidf" ]; then
      local sp; sp=$(cat "$pidf" 2>/dev/null || true)
      if [ -n "$sp" ] && ! kill -0 "$sp" 2>/dev/null; then
        echo "      ✗ 服务进程($sp)已退出"; tail -6 "$logf" 2>/dev/null | sed 's/^/        /'; return 2
      fi
    fi
    if grep -qE "Engine core initialization failed|EngineDeadError|CUDA out of memory|larger than the available KV cache|No available memory for the cache blocks|server exited early" "$logf" "${logf}.wrapper" 2>/dev/null; then
      echo "      ✗ 启动即失败(命中致命错误)"
      grep -nE "Engine core initialization failed|ValueError|CUDA out of memory|No available memory" "$logf" "${logf}.wrapper" 2>/dev/null | tail -2 | sed 's/^/        /'
      return 3
    fi
    sleep 5; i=$((i+1))
  done
  echo "      ✗ 就绪超时"; return 4
}

echo "══ MoE 常驻层收益扫描 ══ ($(date -Is))"
echo "  固定上下文:MAXLEN=$MAXLEN SEQS=$SEQS ⇒ KV_CAP=$((KV_CAP/GIB)).$(printf '%02d' $(( (KV_CAP%GIB)*100/GIB ))) GiB ()$(( KV_CAP )) bytes)"
echo "  扫描 k=[$KS]  并发=[$CS]  prompt=$PROMPT N=$N"
echo "  历史参考(NOTES §309/§320,旧代码):0.70 ms/token/层 @C=1、1.70 @C=4;上限 11 层"
echo

printf "%-4s %-8s %-9s %-22s %s\n" "k" "resident" "VRAM" "TPOT/prefill" "ok"
for k in $KS; do
  tag="${TAG_PREFIX}k${k}"; port=$((8100 + k))
  spec=""
  if [ "$k" -gt 0 ]; then spec="0-$((k-1))"; fi
  freegpu
  rm -f "$LOGD/$tag.pid"
  setsid env PYTHONPATH="$XTU_TREE" TAG="$tag" PORT="$port" \
    GPU_UTIL=0.80 SPEC_K=0 SEQS=$SEQS MAXLEN=$MAXLEN MBT=4096 \
    KV_CACHE_BYTES="$KV_CAP" \
    XIAOTU_MOE_GPU_RESIDENT_LAYERS="$spec" \
    XIAOTU_MOE_RESIDENT_BUDGET_GB=0 \
    bash scripts/serve_glm53_mainline.sh > "$(srv_log $tag).wrapper" 2>&1 &
  if ! wait_ready "$port" "$LOGD/$tag.pid" "$(srv_log $tag)" 1800; then
    printf "%-4s %-8s %-9s %-22s %s\n" "$k" "-" "-" "-" "启动失败"
    continue
  fi
  # **强制验证实际常驻层数**(不要相信我设的值,只信日志)
  actual=$(grep -c "GPU-resident" "$(srv_log $tag)" 2>/dev/null || echo 0)
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  for C in $CS; do
    timeout 1800 "$PY" -m vllm.entrypoints.cli.main bench serve --backend openai-chat \
      --endpoint /v1/chat/completions --host 127.0.0.1 --port "$port" \
      --model "$CKPT_G" --served-model-name GLM-5.3-Flash \
      --dataset-name random --random-input-len "$PROMPT" --random-output-len "$OUTLEN" \
      --num-prompts "$N" --max-concurrency "$C" --seed $(( 70000 + k*97 + C )) \
      --save-result --result-dir "$OUT" --result-filename "${tag}_C${C}.json" \
      > "$OUT/${tag}_C${C}.log" 2>&1
    res=$("$PY" -c "
import json,sys
try:
    d=json.load(open('$OUT/${tag}_C${C}.json'))
    c=d.get('completed') or 0; n=d.get('num_prompts') or 0
    tot=d.get('total_input_tokens') or 0; t=d.get('mean_ttft_ms') or 0
    tp=d.get('mean_tpot_ms') or 0
    pre=(tot/(c*t/1000.0)) if (c and t) else 0
    print(f'C=$C pre={pre:.1f} tok/s tpot={tp:.2f} ms ok={c}/{n}')
except Exception as e: print(f'C=$C FAILED')" 2>/dev/null)
    printf "%-4s %-8s %-9s %-22s %s\n" "$k" "$actual" "${vram}MiB" "${res%% ok=*}" "${res##*ok=}"
  done
done
freegpu
echo "══ 完成 $(date -Is) ══"
echo "结果 JSON/日志在 $OUT/"
echo "RESIDENT_SWEEP_DONE"
