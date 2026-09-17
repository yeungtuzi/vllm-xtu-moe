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
# MBT(默认 **0 = 不传**,用 vLLM 默认 chunk 大小;如 8192):
#   **这是 GPU prefill 的真正前提**。vLLM 按 max_num_batched_tokens 把 prompt 切成
#   chunk,而"某层看到多少 token"= **chunk 大小**,不是 prompt 总长。所以
#   MBT=2048 时任何层都不可能超过 2048 ⇒ `VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS`
#   最多只能在 2048 这一档触发,而该档上**非重叠**的 GPU 流式(~400 ms/层)
#   仍慢于 CPU(实测 ~257 ms/层)⇒ 只会更慢(NOTES §463)。
#   要用 GPU prefill 必须同时把 MBT 提到 4096-8192(并相应留出 activation 显存)。
#   注意:vLLM 会读 `--max-num-batched-tokens` 反推 `max_num_seqs` 等预算,调大前先确认显存。
#
# SPEC(默认 **0 = 不启用**):开启 **DSpark** 投机解码(目标项 3)。
#   实测依据:参考实现 LvLLM-v2.5 在 V4.1 上报告 dspark 使 decode **+~48%**
#   (2x RTX3090 TP2:27 -> 26-40 t/s),而它用的就是**标准 vLLM 旗标**:
#     --speculative-config '{"method":"dspark","num_speculative_tokens":5,
#                            "draft_sample_method":"probabilistic"}'
#   我们的主线已内置 dspark(`vllm/config/speculative.py:68 DSparkModelTypes`,
#   以及 dspark_target_layer_ids / dspark_block_size 等校验),
#   且 V4.1 的 config 里 `dspark_block_size=5`、`dspark_target_layer_ids=[37,38,39]`
#   ⇒ **num_speculative_tokens 必须等于 5**。无需单独 draft model(旗标里没有 model 字段)。
#   可用 SPEC_CONFIG 覆盖;注意 draft 层要显存,配合 GPU_UTIL 一起调。见 NOTES §478。
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"draft_sample_method\":\"probabilistic\"}}"

# PREFIX_CACHE(默认 1 = 保持 vLLM 默认开启):设为 0 会加 --no-enable-prefix-caching。
#   **测量长上下文 prefill 时必须设 0** —— 否则不同 prompt 共享前缀会被整段命中,
#   量出"prefill 几千甚至上万 tok/s"的假象(NOTES §420 连续两次踩到)。
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
# 【R-VRAM/§507】**TP 默认 2**(用户 2026-09-16 指示:TP=1 不满足就明确说、并以 TP=2 为默认)。
# 按显存优先级算同一张 40 GB 卡:TP=2 每层常驻 3.36 GiB ⇒ 1M KV 之后还能放 3 层(20-22)+投机;
# TP=1 每层 6.72 GiB ⇒ 只能放 1 层。TP=1 只在"单卡/没有第二张卡"时才用。
TP="${TP:-2}"
MAXLEN="${MAXLEN:-2048}"
LOAD="${LOAD:-dummy}"          # dummy = no disk read, exercises the kernels
GPU_UTIL="${GPU_UTIL:-0.85}"
EXTRA_ENV="${EXTRA_ENV:-}"

# COMPILE(默认 0 = 与 cellV/参考实现一致的 vLLM 默认:mode=NONE + cudagraph
#   FULL_AND_PIECEWISE)。=1 时用参考实现 lk 的那组旗标
#   `{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}`(probe 里一直是这个)。
#   注意:一旦走 VLLM_COMPILE,vLLM 就会把 TRITON_CACHE_DIR 重定向进它自己按 hash
#   算出来的目录 ⇒ 换旗标/改我们一行代码都要**逐形状重新 JIT**。下面 source 的
#   lib_jitcache.sh 就是把那个目录钉死(JITCACHE=1,默认)。见 docs/RUNBOOK.md。
COMPILE="${COMPILE:-0}"

# ---- JIT 固定缓存目录(scripts/lib_jitcache.sh 有完整原理)--------------------
JITCACHE_MODEL="$(basename "$CKPT")"
JITCACHE_TP="$TP"
JITCACHE_MODE="$([ "$COMPILE" = "1" ] && echo vllm_compile || echo none)"
JITCACHE_COMPILING="$COMPILE"          # 只在编译路径上钉 Triton 目录(见 lib_jitcache.sh)
. "$ROOT/scripts/lib_jitcache.sh"
# 只有非空才往子进程传:`TRITON_CACHE_DIR=""` 与"未设"语义不同(Triton 会把空串当路径)。
JIT_ENV=""
if [ -n "${TRITON_CACHE_DIR:-}" ]; then
  JIT_ENV="TRITON_CACHE_DIR=$TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  [ -n "${TILELANG_CACHE_DIR:-}" ] && JIT_ENV="$JIT_ENV TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
fi
CC_JSON=""
if [ "$COMPILE" = "1" ]; then
  CC_JSON="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"mode\":\"VLLM_COMPILE\"${JITCACHE_CC_EXTRA}}"
fi

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
#   THREADS      未设时插件按 n_ccd×5 自动调优(本机 24 CCD ⇒ 120),这里钉成 **192**
#                (实测数据见下面 XIAOTU_MOE_THREADS 那一段;旧值 60 是错的)。
# 一律写成 "${VAR:-<现值>}":**默认行为逐字不变**,但允许从外部覆盖做 A/B。
# 调参时配合 report/tuning/NOTES.md §389。
# ---- 环境变量文件桥 ----------------------------------------------------------
# 实测:vLLM spawn EngineCore 时会**静默丢弃**部分 XIAOTU_* (THREADS/SPIN_IDLE_US/
# GPU_RESIDENT_LAYERS/RELEASE_SOURCE 都丢过),于是"设了开关却没生效、且无日志"。
# 插件在 __init__.py 里会从该文件补齐缺失项(真实环境优先)。见 NOTES §414。
# XIAOTU_MOE_THREADS(默认 **184** = 物理核 192 留 8 个余量(实测拐点);原来是 60,中途曾是 192):
#   【为什么留余量】SMT 关掉后,192 = **占满全部物理核**。单 token 解码的 B=1 延迟在
#   192 下是**双峰**的(静默机器上连测:0.58 / 3.81 / 11.17 ms),而 176/184 稳定 ~0.5 ms
#   —— 只要有一个 worker 被抢占(OS/vLLM/CUDA 线程),barrier 就要等它,而没有任何余核可去。
#   prefill 侧 176 只比 192 慢 ~4-5%(B=4096: 355 vs 341 ms),用 4% 换解码尾延迟的稳定很划算。
#   (SMT 开着时 192 曾是安全的,因为还有 192 个逻辑核当余量;现机器 SMT=off。)

#   **实测线程数是 prefill 的头号旋钮**(NOTES §424,真权重单层微基准,B=1400):
#     60  -> 386.8 ms/层(B=2048)   ← 旧默认,两头都亏
#     120 -> 219.2                  ← 引擎自身默认(为"解码带宽"调的,解码最优)
#     192 -> 170.9   ⇒ 比 60 快 **2.26×**,比 120 快 1.28×
#     384 -> 176.0   但**解码**从 0.45 掉到 8.88 ms(**17×**,SMT 在作业少时纯抢 L3/功耗)
#   解码在 192 下**没有退化**(真权重服务实测:串行 27.53/27.78 tok/s,
#   与历史最佳 27.5-28.5 一致;C=8 聚合 85.31→**92.34** tok/s,历史最好)。
#   ⇒ 60 既不是引擎默认也不是任何一档的最优,是当初"先跑通"钉下的保守值,故改 192。
#   引擎自身的默认(120)**没有动** —— 那是别的模型的解码最优点(NOTES §424)。
XTU_ENV_FILE="${XIAOTU_ENV_FILE:-/tmp/xiaotu_env}"
# ---- 【§504 修:内存+性能,都在我们自己的环境里实测出来的】-------------------
# TP=2 时 **每个 rank 的线程数**必须按"本 rank 的 CCD 数 × 4-5"取,而不是按整机核数:
#   184(/rank)× 2 rank = 368 线程挤 192 物理核(用户规则:至少给每个 worker 留 2 核,
#   且绝不用满)⇒ 实测 **TPOT 436.7 ms / C=1 1.93 tok/s**;
#   60/rank(= 12 CCD × 5)⇒ **TPOT 68.1 ms / C=1 9.62 tok/s**(解码快 6.4×)。
# 184 是**单进程**基准(world=1)的拐点,不能直接搬到 TP=2 服务上 —— 这是本轮踩到的坑。
# ---- 显存优先级策略(R-VRAM,用户 2026-09-16 定为固定规则)------------------
# 顺序固定:1) 1M 上下文(KV 先留够,不可降级) 2) GPU 预填充 3) GPU 投机解码 4) 专家层常驻;
# 任何一项不满足就 fallback(预填充退回 CPU / 投机关闭 / 常驻全放主机),**且不得额外多占系统内存**。
# 规划器把这条规则算成具体开关;显式设的 env 仍然优先(便于做对照实验)。
VRAM_POLICY="${VRAM_POLICY:-1}"
POLICY_GP_MIN=""; POLICY_RESIDENT=""; POLICY_DRAFT=""
if [ "$VRAM_POLICY" = "1" ]; then
  _PLAN="$(MAXLEN="$MAXLEN" "$PY" -m vllm_xiaotu_moe.vram_policy --maxlen "$MAXLEN" --tp "$TP" --emit-env 2>/dev/null \
            | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' || true)"
  if [ -n "$_PLAN" ]; then
    POLICY_GP_MIN="$(printf '%s\n' "$_PLAN" | sed -n 's/^VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=//p')"
    POLICY_RESIDENT="$(printf '%s\n' "$_PLAN" | sed -n 's/^XIAOTU_GPU_RESIDENT_LAYERS=//p')"
    POLICY_DRAFT="$(printf '%s\n' "$_PLAN" | sed -n 's/^XIAOTU_MOE_RESIDENT_DRAFT=//p')"
    POLICY_KV_BYTES="$(printf '%s\n' "$_PLAN" | sed -n 's/^XIAOTU_KV_CACHE_BYTES=//p')"
    echo "[v41] R-VRAM 规划(maxlen=$MAXLEN):gpu_prefill_min=${POLICY_GP_MIN:-?} resident='${POLICY_RESIDENT:-}' draft_on_gpu=${POLICY_DRAFT:-?}"
  fi
fi
THREADS_DEFAULT="${THREADS_DEFAULT:-60}"
# 自旋 5000 µs 是 §355 记录过的正反馈灾难(48 层 × 3 相 × 5 ms ⇒ 池几乎永不停转),
# 与上面的超订叠在一起会互相放大 ⇒ 默认 0(完全不自旋,worker 直接 futex 睡)。
SPIN_DEFAULT="${SPIN_DEFAULT:-0}"
XTU_ENV_FILE="${XIAOTU_ENV_FILE:-/tmp/xiaotu_env}"
{
  echo "XIAOTU_MOE_THREADS=${XIAOTU_MOE_THREADS:-$THREADS_DEFAULT}"
  echo "XIAOTU_MOE_NSLICE_SMALL=${XIAOTU_MOE_NSLICE_SMALL:-0}"
  echo "XIAOTU_MOE_ASYNC=${XIAOTU_MOE_ASYNC:-0}"
  echo "XIAOTU_MOE_SPIN_IDLE_US=${XIAOTU_MOE_SPIN_IDLE_US:-$SPIN_DEFAULT}"
  echo "XIAOTU_MOE_GPU_RESIDENT_LAYERS=${XIAOTU_MOE_GPU_RESIDENT_LAYERS:-$POLICY_RESIDENT}"
  echo "XIAOTU_MOE_RESIDENT_BUDGET_GB=${XIAOTU_MOE_RESIDENT_BUDGET_GB:-0}"
  echo "VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=${VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:-${POLICY_GP_MIN:-0}}"
  echo "XIAOTU_RELEASE_SOURCE=${XIAOTU_RELEASE_SOURCE:-1}"
  # 【§504 的结论,§519 更正】**多 rank 同机时"分片单位"与"核切分单位"必须一致**:
  #   * §504 遇到的问题是真的:NPS1(2 node)+TP=2 时按 node 分片会退化成 nshard_=1 ⇒
  #     分片路径失效、退回"每 socket 一份副本"⇒ **权重存 2 份**;
  #   * 但 §505 的解法(默认"按 socket 分片 + 核 CCD 交错")在 **NPS=4 上代价巨大**:整层权重
  #     只落在 2 个 NUMA node 上、worker 却铺满 8 个 ⇒ 引擎每层 0.44→0.96 ms,
  #     单流 24.94→14.17 t/s;微基准 B=8 上 node 分片快 **4.7×**(§518);
  #   * 现在由**引擎自适应**:node 数/world ≥ 2 ⇒ node 分片 + 核按 node 子集切
  #     (NPS4+TP2 ⇒ 每 rank 4 片,仍是**整机 1 份权重**且 node-local);
  #     会退化时才退回 socket。该组合实测 **18.22 t/s / 987 GiB**(§518)。
  #   ⇒ 所以这里**默认不设**这个键,交给 `resolve_shard_mode()`;显式设 2 才回到旧行为。
  [ -n "${XIAOTU_MOE_RANK_SPLIT:-}" ] && echo "XIAOTU_MOE_RANK_SPLIT=$XIAOTU_MOE_RANK_SPLIT"
  for kv in $EXTRA_ENV; do case "$kv" in *=*) echo "$kv";; esac; done
} > "$XTU_ENV_FILE"
export XIAOTU_ENV_FILE

nohup env \
  HF_HUB_OFFLINE=1 \
  VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  VLLM_HANDSHAKE_TIMEOUT_MINS=120 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_EXPERTS_LOAD_DEVICE=cpu \
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS:-${POLICY_GP_MIN:-0}}" \
  XIAOTU_RELEASE_SOURCE="${XIAOTU_RELEASE_SOURCE:-1}" \
  XIAOTU_MOE_THREADS="${XIAOTU_MOE_THREADS:-$THREADS_DEFAULT}" \
  XIAOTU_MOE_NSLICE_SMALL="${XIAOTU_MOE_NSLICE_SMALL:-0}" \
  XIAOTU_MOE_ASYNC="${XIAOTU_MOE_ASYNC:-0}" \
  XIAOTU_MOE_SPIN_IDLE_US="${XIAOTU_MOE_SPIN_IDLE_US:-$SPIN_DEFAULT}" \
  ${XIAOTU_MOE_RANK_SPLIT:+XIAOTU_MOE_RANK_SPLIT=$XIAOTU_MOE_RANK_SPLIT} \
  OMP_NUM_THREADS=1 \
  $JIT_ENV \
  $EXTRA_ENV \
  numactl --interleave=all "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name dsv41 \
    --load-format "$LOAD" \
    --max-model-len "$MAXLEN" --tensor-parallel-size "$TP" --max-num-seqs "${MAXSEQS:-1}" \
    $( [ "${MBT:-0}" -gt 0 ] 2>/dev/null && echo --max-num-batched-tokens "$MBT" ) \
    --gpu-memory-utilization "$GPU_UTIL" \
    $( [ -n "${KV_CACHE_BYTES:-${POLICY_KV_BYTES:-}}" ] && printf -- '--kv-cache-memory %s' "${KV_CACHE_BYTES:-${POLICY_KV_BYTES:-}}" ) \
    $( [ "${EAGER:-0}" = "1" ] && echo --enforce-eager ) \
    $( [ "${SPEC:-0}" = "1" ] && echo --speculative-config "$SPEC_CONFIG" ) \
    $( [ -n "$CC_JSON" ] && printf -- '--compilation-config %s' "$CC_JSON" ) \
    $( [ "${PREFIX_CACHE:-1}" = "1" ] || echo --no-enable-prefix-caching ) --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --kernel-config '{"enable_jit_warmup": false}' \
    --port "$PORT" > "$LOG" 2>&1 &
echo $! > "$OUTDIR/$TAG.pid"
echo "[v41] pid=$(cat "$OUTDIR/$TAG.pid")"

# Wait for readiness. Model construction alone took ~23 min with dummy weights.
DEADLINE=$(( SECONDS + ${READY_TIMEOUT:-3600} ))
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[v41] READY tag=$TAG"
  # 【§555】形状预热:把"首个长上下文请求付 ~237 s JIT"的成本挪到启动阶段
  if [ "${WARMUP:-1}" = "1" ]; then
    PORT="$PORT" LENS="${WARMUP_LENS:-8192 32768}" CKPT="$CKPT" MODEL=dsv41 \
      bash "$ROOT/scripts/warmup_shapes.sh" || echo "[v41] ⚠️ 预热失败(不致命,继续启动)"
  fi
  exit 0
  fi
  if ! kill -0 "$(cat "$OUTDIR/$TAG.pid")" 2>/dev/null; then
    echo "[v41] server exited early; tail:"; tail -25 "$LOG"; exit 1
  fi
  sleep 10
done
echo "[v41] TIMEOUT waiting for readiness; tail:"; tail -25 "$LOG"; exit 1
