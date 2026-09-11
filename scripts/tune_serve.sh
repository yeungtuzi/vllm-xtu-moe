#!/usr/bin/env bash
# 调参统一服务端启动器(vllm-xtu-moe · 混合模式)
#
# 所有可变项都从环境变量进来,便于把"一次实验"记录成一行可复现的命令。
# 启动后会等待 /v1/models 就绪,并把关键日志行(加载耗时、KV 容量、后端/引擎形状)
# 写到 report/tuning/logs/<TAG>.meta。
#
# 用法(MODE=dsv4|qwen38):
#   MODE=dsv4 TAG=dsv4_tp1_fp8kv PORT=8081 TP=1 MAXLEN=262144 KV_DTYPE=fp8_ds_mla \
#     GPU_UTIL=0.90 SEQS=8 MAX_NBT=8192 scripts/tune_serve.sh
#
# 常用可选变量:
#   EP=1 启用专家并行;CPU_OFFLOAD_GB=N(仅 qwen38 需要);EAGER=0 关闭 enforce-eager;
#   PREFILL_MIN=N  → VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS;PREFILL_FILE=path 走运行期阈值文件;
#   THREADS=N → XIAOTU_MOE_THREADS;OMP=N → OMP_NUM_THREADS;SPEC='{...}' → --speculative-config;
#   GPUS=2 / GPUS=0,1 选择 GPU;KV_MEM_BYTES=N 显式限制 KV cache 字节数;
#   EXTRA='...' 追加任意参数。
#
# License: Apache-2.0
set -euo pipefail

MODE="${MODE:-dsv4}"
TAG="${TAG:-${MODE}_$(date +%m%d_%H%M%S)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/report/tuning/logs"
mkdir -p "$OUTDIR"

PORT="${PORT:-8081}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-8}"
MAX_NBT="${MAX_NBT:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
KV_DTYPE="${KV_DTYPE:-auto}"
EP="${EP:-0}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
EAGER="${EAGER:-1}"
TP="${TP:-1}"
PREFILL_MIN="${PREFILL_MIN:-384}"
PREFILL_FILE="${PREFILL_FILE:-}"
# 144 = 192 核 - 48:给调用线程(MoE host-fn 回调 / torch / CUDA 驱动 / 采样)留核。
# 实测(2026-09-11,微基准 2 轮复现):worker 数 == 核数时,调用线程只能抢某个 worker
# 的核 ⇒ 该 worker 成为屏障的拖后腿者,单次引擎调用 2.8-3.3 ms;留 16 核后 1.64-1.65 ms
# (1.7-2.0x)。服务端扫描(负载 31-50):192→8.25-9.98 tok/s、176→11.44、160→11.66、
# 144→12.57、128→12.66(每层 compute 4.4-5.7 → 1.61-1.67 ms);平台期在 128-144。
THREADS="${THREADS:-120}"   # 每 CCD 4-5 核(24 CCD ⇒ 96-120);见 NOTES §44,不要再加核
OMP="${OMP:-48}"
SPEC="${SPEC:-}"
EXTRA="${EXTRA:-}"
ENV_EXTRA="${ENV_EXTRA:-}"
KV_MEM_BYTES="${KV_MEM_BYTES:-}"
# 选择性地把参数段 offload 到 CPU(vLLM 的 UVA offloader):Qwen3.8 的 PLE n-gram
# 表(51.2 GB)在 CPU 内存里做查找,显存只留 dense/attention(~11 GB)→ 单卡可跑。
OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-}"

case "$MODE" in
  dsv4)
    MODEL="${MODEL:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
    SERVED="${SERVED:-DeepSeek-V4-Flash-xiaotu}"
    GPUS="${GPUS:-2}"
    ;;
  qwen38)
    MODEL="${MODEL:-/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13}"
    SERVED="${SERVED:-Qwen3.8-Flash-Next}"
    GPUS="${GPUS:-0,1}"
    ;;
  *) echo "MODE must be dsv4 or qwen38" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_EXPERTS_LOAD_DEVICE=cpu
# ---------------------------------------------------------------------------
# 【固定规则,不要再改】CPU 专家的并行模型:
#   1) 每个 CCD 开 4-5 个核(本机 24 CCD ⇒ THREADS=120),不要再加核;
#   2) 权重按 NUMA node 分片(nshard_ = numa_node_count() = 8),每个 node 的
#      worker 只读写本 node 绑定的那一份(MPOL_BIND) —— 全部 page-local;
#   3) node 之间只交换很小的数据(每层各 node 的激活切片/部分和),用池内
#      all-gather + 每 token 归约,量级远小于跨 node 读权重的开销。
#
# 引擎侧已把"单份连续拷贝"模式(XIAOTU_MOE_SINGLECOPY)**整段删除**,这里也
# 不再有任何开关 —— 分片是唯一布局(见 moe_v2.hpp "固定规则"注释)。

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-7200}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$OMP"
[ -n "$THREADS" ] && export XIAOTU_MOE_THREADS="$THREADS"
export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$PREFILL_MIN"
[ -n "$PREFILL_FILE" ] && export VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE="$PREFILL_FILE"
export PATH=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin:$PATH
# ENV_EXTRA 形如 "XIAOTU_MOE_PROFILE=1 XIAOTU_TIMING=1":临时打开埋点
for _kv in $ENV_EXTRA; do export "$_kv"; done

ARGS=(
  "$MODEL"
  --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$MAX_NBT"
  --gpu-memory-utilization "$GPU_UTIL"
  --no-enable-prefix-caching
  --kernel-config.enable_jit_warmup=false
  --served-model-name "$SERVED"
)
[ "$EP" = "1" ] && ARGS+=(--enable-expert-parallel)
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")
# 显式给 KV cache 设上限:256K 上下文只需要 ~8 GiB,其余显存要留给
# GPU prefill 的 ping-pong staging(每层 ~2 GiB)与激活,否则会 OOM。
[ -n "$KV_MEM_BYTES" ] && ARGS+=(--kv-cache-memory-bytes "$KV_MEM_BYTES")
[ "$CPU_OFFLOAD_GB" != "0" ] && ARGS+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
if [ -n "$OFFLOAD_PARAMS" ]; then
  # shellcheck disable=SC2206
  _opts=($OFFLOAD_PARAMS)
  ARGS+=(--cpu-offload-params "${_opts[@]}")
fi
[ "$EAGER" = "1" ] && ARGS+=(--enforce-eager)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

# 记录本次实验的"环境快照"(机器负载会显著影响数字)
{
  echo "tag=$TAG mode=$MODE port=$PORT tp=$TP ep=$EP maxlen=$MAXLEN seqs=$SEQS nbt=$MAX_NBT"
  echo "gpu_util=$GPU_UTIL kv_dtype=$KV_DTYPE cpu_offload_gb=$CPU_OFFLOAD_GB eager=$EAGER"
  echo "prefill_min=$PREFILL_MIN prefill_file=${PREFILL_FILE:-none} threads=$THREADS omp=$OMP"
  echo "spec=${SPEC:-none} extra=${EXTRA:-none} env_extra=${ENV_EXTRA:-none} gpus=$GPUS"
  date -Is
  uptime
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
} > "$OUTDIR/$TAG.env"

# 【必做】清理 EP 跨-rank 屏障的残留 shm 文件。
# 2026-09-11 定位:被 kill 掉的 TP>=2 进程会在 /dev/shm 留下
# `xiaotu_ep_L*_<hidden>_<tokens>_<world>.bin`,里面存着双 barrier 的世代计数;
# 新进程 attach 到状态错乱的旧文件后,两个 rank 的世代对不上 ⇒ **永久互等**,
# 表现为 vLLM 的 `shm_broadcast: No available shared memory broadcast block found
# in 60 seconds` 反复出现、引擎永远不就绪。清掉即可正常启动(实测:清理后 TP=2 250s 就绪)。
rm -f /dev/shm/xiaotu_ep_*.bin

# 等显存真正释放(上一轮进程可能还在退出中),避免 "Free memory ... less than desired"
for _g in $(echo "$GPUS" | tr ',' ' '); do
  for _i in $(seq 1 60); do
    _used=$(nvidia-smi --id="$_g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [ -n "$_used" ] && [ "$_used" -lt 1024 ] && break
    sleep 5
  done
done
LOG="$OUTDIR/$TAG.log"
nohup vllm serve "${ARGS[@]}" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[tune_serve] tag=$TAG pid=$(cat "$OUTDIR/$TAG.pid") log=$LOG"

READY=0
for _ in $(seq 1 720); do          # 最长等 60 分钟(超大模型加载)
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[tune_serve] server exited early; tail:"; tail -30 "$LOG"; exit 1
  fi
  sleep 5
done
if [ "$READY" != "1" ]; then
  echo "[tune_serve] timeout waiting for readiness; tail:"; tail -30 "$LOG"; exit 1
fi

{
  grep -E "init engine .* took|GPU KV cache size|Using CPU .*MoE backend|xiaotu MOE_" "$LOG" | tail -8
  echo "ready_at=$(date -Is)"
} > "$OUTDIR/$TAG.meta"
echo "[tune_serve] READY tag=$TAG"
cat "$OUTDIR/$TAG.meta"
