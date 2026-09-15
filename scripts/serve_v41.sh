#!/usr/bin/env bash
# DeepSeek-V4.1-Flash on 3xA100-40GB (SM80) — bring-up / smoke runner.
#
# V4.1 is a 475 GiB model on 120 GB of VRAM, so nothing fits without offload.
# The split vLLM already supports (verified 2026-09-14, NOTES §369):
#   * routed experts (269 GiB)  -> host, computed by the xiaotu CPU engine
#                                  (VLLM_EXPERTS_LOAD_DEVICE=cpu + this plugin)
#   * Engram tables (188.8 GiB) -> pinned host memory (engram_config.cpu_offload,
#                                  on by default)
#   * everything else (~23 GiB)  -> GPU
#
# XIAOTU_RELEASE_SOURCE=1 (default here) implements 切分一层/释放一层: once a
# layer's engine owns its NUMA-sharded copy, the vLLM-side host source tensor is
# dropped, so the expert bytes are not held twice (IRON_RULES R9). Tune with
# XIAOTU_RELEASE_SOURCE=0 to A/B it.
# The one thing that does NOT work out of the box on A100 is attention: V4.1
# ships only FlashMLA (SM90+) and FlashInfer (SM100/SM120) paths. This script is
# the end-to-end acceptance test for the SM80 fallback.
#
# Usage:
#   bash scripts/serve_v41.sh                 # dummy weights (fast bring-up)
#   LOAD=auto bash scripts/serve_v41.sh       # real weights (~long load)
#
# Env: TAG PORT GPUS TP MAXLEN LOAD GPU_UTIL EXTRA_ENV MAXSEQS
#
# 【性能默认已调】SPIN_IDLE_US 默认 5000(原 0):实测把 perf 里 ~42% 的 futex
#   (lll_lock_wait/wake)降到 ~0%,compute 0.43->0.37 ms/层,单流 25.7->27.5 tok/s。
#   见 NOTES §412。可用 SPIN 环境变量覆盖。
#
# EAGER(默认 **0** = 启用 CUDA graph):是否加 --enforce-eager。
#   **实测 +80%**:单流 14.31 -> 25.71 tok/s;每层 period 1.84 -> 0.95 ms,
#   其中 rest(GPU 侧算子)1.27 -> 0.52 ms —— 关掉 eager 省下的正是 batch=1 时
#   被 43 层放大的 kernel 启动开销。已在真实权重下验证启动完整、输出正确。
#   若遇到 CUDA graph 捕获问题(插件历史 TRIED_AND_REVERTED R4 提到
#   捕获期 cudaHostRegister 会作废捕获),用 EAGER=1 回退。见 NOTES §406。
#   实测(NOTES §405)每层 1.84 ms 里我们的 CPU MoE 只占 0.56 ms(31%),
#   69% 是 "rest"(GPU 侧算子 + 层间交接)。batch=1 时 GPU 算子极小,
#   kernel 启动开销被 43 层放大 ⇒ 关掉 eager(启用 CUDA graph)是头号候选优化。
#   默认仍为 1 以保持既有行为;测试用 EAGER=0。
#
# MAXSEQS(默认 1):并发批大小上限。**实测关键**:=1(原值)时 vLLM 同时只调度一个
#   序列,并发请求被完全串行化 —— 聚合吞吐恒定 ~13 tok/s、与并发无关,而单请求延迟
#   线性变差(C=8 时 64 token 要 21 s)。见 report/tuning/NOTES.md §400。
#   默认仍为 1 以保持既有行为不变;做吞吐测试时用 MAXSEQS=8/16 覆盖。
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
TAG="${TAG:-v41}"
PORT="${PORT:-8077}"
GPUS="${GPUS:-0}"
TP="${TP:-1}"
MAXLEN="${MAXLEN:-2048}"
LOAD="${LOAD:-dummy}"          # dummy = no disk read, exercises the kernels
GPU_UTIL="${GPU_UTIL:-0.85}"
EXTRA_ENV="${EXTRA_ENV:-}"

OUTDIR="$ROOT/report/tuning/logs"; mkdir -p "$OUTDIR"
LOG="$OUTDIR/$TAG.log"

if [ ! -f "$CKPT/config.json" ]; then
  echo "[v41] checkpoint not found: $CKPT" >&2; exit 2
fi

{
  echo "tag=$TAG port=$PORT gpus=$GPUS tp=$TP maxlen=$MAXLEN load=$LOAD util=$GPU_UTIL"
  echo "ckpt=$CKPT"; date -Is
} > "$OUTDIR/$TAG.env"

echo "[v41] starting (load=$LOAD maxlen=$MAXLEN tp=$TP gpus=$GPUS) log=$LOG"

# Optional memory sampler. V4.1 peaks near ~1.4 TB of host memory on a box whose
# 8 NUMA nodes are 193 GB each, so the failure mode is *one node* exhausting
# (kernel: "constraint=CONSTRAINT_MEMORY_POLICY, nodemask=N"), not total RAM.
# MEMTRACE=1 records per-node free memory plus the fattest process's
# Rss/Shmem/Pss so the next run says exactly which component grows where.
if [ "${MEMTRACE:-0}" = "1" ]; then
  MEMLOG="$OUTDIR/$TAG.mem"
  (
    while true; do
      echo "### $(date -Is)" >> "$MEMLOG"
      numactl --hardware 2>/dev/null | grep -E "^node [0-9]+ free" >> "$MEMLOG"
      fat=$(ps -eo pid,rss --sort=-rss --no-headers 2>/dev/null | head -1 | awk '{print $1}')
      if [ -n "$fat" ] && [ -r "/proc/$fat/smaps_rollup" ]; then
        echo "-- fattest pid=$fat" >> "$MEMLOG"
        grep -E "^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Dirty|Anonymous):" \
          "/proc/$fat/smaps_rollup" >> "$MEMLOG" 2>/dev/null
      fi
      sleep "${MEMTRACE_INTERVAL:-60}"
    done
  ) &
  echo $! > "$OUTDIR/$TAG.mem.pid"
  echo "[v41] memtrace -> $MEMLOG"
fi

# 【不要 cd /tmp】任何留在 /tmp 的 `*.py` 都会遮蔽同名模块 —— 2026-09-15 一个
# 遗留的 /tmp/attr.py 就命中了 aiohttp 的 `import attr`,让整个 cellA 起不来
# (`FileNotFoundError: '/proc/--model/status'`)。改用专用的空目录。
RUN_CWD="$OUTDIR/run"; mkdir -p "$RUN_CWD"; cd "$RUN_CWD"
export CUDA_VISIBLE_DEVICES="$GPUS"
# 【性能提示】下面四项是**为"先跑通"钉的保守值**,并不是引擎默认:
#   ASYNC        引擎默认开启,注释自带实测收益:每 token 6.53→1.80 ms(3.6×)、
#                C=4 聚合 71.8→107.3 t/s,且已验证"与 host-func 逐位相同"、
#                "greedy 文本 5/5 相同" ⇒ 关掉它等于白丢一个已验证的大收益;
#   SPIN_IDLE_US 插件默认 300(hybrid_model.py setdefault),这里钉成 0;
#   NSLICE_SMALL 引擎默认开启;=0 会**强制走 legacy 路径**;
#   THREADS      未设时插件按 n_ccd×5 自动调优,这里钉成 60。
# 一律写成 "${VAR:-<现值>}":**默认行为逐字不变**,但允许从外部覆盖做 A/B。
# 调参时配合 report/tuning/NOTES.md §389。
nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  VLLM_HANDSHAKE_TIMEOUT_MINS=120 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  XIAOTU_RELEASE_SOURCE="${XIAOTU_RELEASE_SOURCE:-1}" \
  XIAOTU_MOE_THREADS="${XIAOTU_MOE_THREADS:-60}" \
  XIAOTU_MOE_NSLICE_SMALL="${XIAOTU_MOE_NSLICE_SMALL:-0}" \
  XIAOTU_MOE_ASYNC="${XIAOTU_MOE_ASYNC:-0}" \
  XIAOTU_MOE_SPIN_IDLE_US="${XIAOTU_MOE_SPIN_IDLE_US:-5000}" \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  numactl --interleave=all "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name dsv41 \
    --load-format "$LOAD" \
    --max-model-len "$MAXLEN" --tensor-parallel-size "$TP" --max-num-seqs "${MAXSEQS:-1}" \
    --gpu-memory-utilization "$GPU_UTIL" $( [ "${EAGER:-0}" = "1" ] && echo --enforce-eager ) --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --kernel-config '{"enable_jit_warmup": false}' \
    --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[v41] pid=$(cat "$OUTDIR/$TAG.pid")"

# Wait for readiness. Model construction alone took ~23 min with dummy weights.
DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[v41] READY tag=$TAG"; exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[v41] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[v41] TIMEOUT waiting for readiness; tail:"; tail -25 "$LOG"; exit 1
