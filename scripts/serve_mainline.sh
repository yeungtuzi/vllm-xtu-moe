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
# ---- GPU 预填充三开关(见 NOTES §340;缺一个就失效) ----
# 便捷预设(必须放在 MBT 解析之前):PREFILL=1 ⇒ 对齐参考 fork 的生产预填充协议
#   (serve_lk_port.sh: MBT=8192 / MINBATCH=1024 / 只捕获解码尺寸)。
#   实测端到端:2048 token TTFT 5.71 s / 366 t/s(CPU 侧要 92.6 s ⇒ 16.2×);
#   TP=2 16000 token ⇒ 1149 t/s(report/curve_thr.jsonl, report/curve_tp2.jsonl)。
if [ "${PREFILL:-0}" = "1" ]; then
  MBT="${MBT:-8192}"
  GP_MIN="${GP_MIN:-1024}"
  CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-1 2 4 8}"
fi
MBT="${MBT:-256}"                    # 与 fork 协议对齐(实测:不设时主线默认很大)
# GPU 预填充阈值(fork 里叫 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE,生产值 1024)。
#   0 = 关(默认,保持解码基准协议不变);N>0 时 qlen>=N 的 MoE 层走 GPU
#   (hybrid_model.py:982 的 _gp_min 分支)。**必须同时满足 MBT >= N**,
#   否则 batch 永远到不了阈值 —— 与 fork 把 MINBATCH 夹到 MBT 的语义一致。
GP_MIN="${GP_MIN:-0}"
# 只给这些 batch size 捕获 CUDA 图(**空格分隔**,对应 --cudagraph-capture-sizes 的 nargs='+')。**这是让 GPU 预填充生效的关键**:
#   预填充形状若不在捕获集里就不走图、直接 eager ⇒ hybrid_model 的 GPU 分支才会被选中;
#   同时解码(size<=SEQS)仍然享有 CUDA 图。留空 = 用主线默认(PIECEWISE 会把预填充
#   形状也捕获掉,重放永远走捕获时的 CPU 分支 ⇒ GPU 预填充形同虚设)。见 NOTES §340(c)。
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-}"
# NUMA 交错(**默认开,不要关**)。
# 为什么必须:vLLM 加载权重时每个 rank 装**全部 256 个专家**(≈138 GB/worker,
# EP 不在加载期切分存储),这部分是**未绑定**的 first-touch 分配,会顺着加载线程
# 所在的 node 堆下去。实测(2026-09-14,已加载 38/43 层时):
#     node 0 free:  7220 MB   ← 185 GB 已用      node 4 free: 168037 MB
#     node 2 free:  7951 MB   ← 185 GB 已用      node 7 free: 169973 MB
# **总空闲还有 863 GB,但 node 0/2 已 96% 满** ⇒ 下一次落上去的分配就:
#     oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0,
#               task=VLLM::Worker_TP, anon-rss:236289496kB
# 进程**静默死亡**(无 Traceback,日志"只加载到一半就没了")。
# 引擎自身的权重分片已由 mbind 正确铺开(见 moe_v2.hpp shard_region),但管不到
# vLLM 那 138 GB,所以必须在进程级交错。INTERLEAVE=0 仅在你自己已绑核/绑节点时用。
INTERLEAVE="${INTERLEAVE:-1}"
# ⚠️ 线程池自旋窗口(**默认不设 = 用引擎自己的 5 ms**;这是诊断旋钮,不是调优旋钮)
# 引擎 worker 默认 5 ms 没活就 park。实测主线模式 A(TP=2/MBT=256/RESIDENT=0-11,
# 512-token prompt,ignore_eos):
#     默认(5 ms)          : 1225 ms/token
#     SPIN_IDLE_US=600000 : 43.5 ms/token  ← 但**只是暂时**的
# 旋到 600000 后两个 worker 各烧 **3242% CPU(≈32 核)**、117 线程常驻自旋,
# `load average` 在几分钟内从 41 爬到 **119**(192 核机),随后同一个服务又退化回
# **1313 ms/token**。⇒ **空转只是把症状掩盖一会儿,同时把整机拖垮**,
# 所以**不设为默认**。真正的修法见 NOTES §353(针对 park/wake 本身,
# 以及解码只需 top_k 个线程却起 60 个线程的池同步开销)。
# 🎯 **必须设 0**。引擎默认 5000 µs:每次调用后**所有** worker 都要自旋 5 ms
# (wlimit=0 时门闸不生效)。43 层 × ~3 相 × 5 ms ⇒ 池几乎永不停转:
#   实测 worker 1433–1552% CPU(≈15 核/worker,两个 rank ≈30 核)常驻自旋,
#   load average 冲到 40+,而**自己制造的那 30 核争抢又反过来拖慢调用线程** ——
#   形成正反馈:同一服务会从 37 ms/token 一路漂到 1420 ms/token(NOTES §355)。
# 设 0 ⇒ 完全不自旋,worker 直接 futex 睡;CPU 145%、load 11、**37.7 ms/token 稳定**。
# 注意这与"旋到 600000"是两个方向:600000 是让所有人**永远**自旋(更糟)。
SPIN_IDLE_US="${SPIN_IDLE_US:-0}"
# 🎯 解码性能的**真正开关**(NOTES §354):关掉"小 batch N-slice"路径。
# 原因:`MOE_V2::small_batch_workers()` 按 `single_us = NASS*3*I*H/(8*3e3)` 估线程数,
# 假设 fp8 8 MAC/cycle;DS-V4 维度下 qlen=1/top_k=6 算出 **6292 µs**(实际单线程 ~百 µs 级)
# ⇒ `wlimit = 59 / nt=60` ⇒ `stride = 60/59 = 1` ⇒ **没有任何 worker 去 park,60 个全自旋**,
# 而且走的是 `parallel_for_limited` 那条**代码里明确警告过有边界竞态**的路径;
# 调用方每相还要先自旋 `spin_idle_us`(默认 5 ms)再退回 condvar。
# 实测(TP=2/MBT=256/RESIDENT=0-11,512-token prompt,ignore_eos):
#     默认            : **1225 ms/token**,worker 3242% CPU ×2 ⇒ load 119
#     NSLICE_SMALL=0  : **36–44 ms/token**,worker 146% CPU   ⇒ load 9
# 即 **28–33× 且 CPU 降到 1/22**。(对比:`SPIN_IDLE_US=600000` 只是把 park 换成
# 整机自旋,几分钟后反而退化 —— 那是掩盖,这才是修。)
NSLICE_SMALL="${NSLICE_SMALL:-0}"
# 🎯🎯 **主线解码性能的头号杀手**(NOTES §364):必须关掉引擎的**异步握手**路径。
# 引擎默认 `XIAOTU_MOE_ASYNC=1`:host 回调把活儿丢给 worker 线程就返回,GPU 靠
# mapped flag + 流内存操作等它。这条路在 fork 编排下正常(1.18 s/请求),
# **在主线下每层要多等 ~28 ms**(43 层 × 10 pass ≈ 11 s):
#     主线 async=1(默认) : 11.96 s/请求   ← 慢 8.4×
#     主线 async=0       : **1.42 s/请求**  ← 与 fork 的 1.18 s 同级
# 关掉后 host 回调**同步**做 CPU MoE,没有任何等待/握手。
# 代价:失去"投递与计算重叠",但在主线上重叠本来就是负的。
ASYNC="${ASYNC:-0}"
if [ "$GP_MIN" -gt 0 ] && [ "$GP_MIN" -gt "$MBT" ]; then
  echo "[mainline] GP_MIN=$GP_MIN > MBT=$MBT ⇒ 夹到 MBT(否则预填充永远够不到阈值;NOTES §319c)"
  GP_MIN="$MBT"
fi
# chunked prefill:**建议保持主线默认(开)**。
#   * 关掉它时,主线要求 MBT >= max_model_len(否则直接报错),而且预填充会整段进
#     我们的 CPU 引擎(qlen=MBT)⇒ 一次 256 token 的预填充就要 26 ms/层 × 43 ≈ 1.1 s;
#   * 开着时,解码步里可能混进预填充块(qlen=257)⇒ 解码也会付这份钱。
# ⇒ 两种都不理想,根因是**形态 B 没有"预填充专用路径"**(fork 里有 `_cpu_prefill` /
#   `_gpu_prefill` 的分流)。见 NOTES §334s。默认先取主线默认(开)。
# **补充(2026-09-14,NOTES §340)**:形态 A(OOT=1,本脚本默认)的 `hybrid_model.py`
#   里 **GPU 预填充路径早就写好了**(gpu_prefill.py:KV-Major 锁页缓存 + PrefetchSlot
#   重叠 + Triton MXFP4 分组 GEMM),只是本脚本一直**没导出阈值 + MBT=256 够不到 +
#   预填充形状被 CUDA 图捕获**,三个开关全缺 ⇒ 形同虚设。用 `PREFILL=1` 一次打开。
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
# 加载策略:实测插件路径读分片 2.5-4.8 s/片(fork 路径 0.6 s/片);
# 主线日志明确建议 EXT4 上用 prefetch 强制预取。
LOAD_STRATEGY="${LOAD_STRATEGY:-prefetch}"
# A100(SM80)没有 fp8e4nv:Triton 的 PackSeq 等内核在 JIT warmup 时会报
# "type fp8e4nv not supported in this architecture" ⇒ 直接关掉预热(实测 ml08modeB)
KERNEL_WARMUP="${KERNEL_WARMUP:-0}"
# EAGER=1 ⇒ --enforce-eager(关 CUDA 图)。诊断用:区分"图捕获契约"与"调度"问题。
EAGER="${EAGER:-0}"
# 引擎线程:插件侧旋钮(每 rank 12 CCD ⇒ 5 核/CCD = 60)
THREADS="${THREADS:-60}"
# 额外常驻 GPU 的 MoE 层(与 lk 的 LVLLM_GPU_RESIDENT_MOE_LAYERS 同义)
RESIDENT="${RESIDENT-0-11}"
# 1 = OOT 模型覆盖(默认,零补丁);0 = 主线 RoutedExperts + 我们的 CPU 后端
OOT="${XIAOTU_OOT_OVERRIDE:-1}"
EXTRA_ENV="${EXTRA_ENV:-}"

# ---- 启动前自检:把"宿主节奏决定引擎行为"的隐式契约变成显式断言 ----
# 这些开关缺失时**不报错、只静默变慢**(实测可达 30×),所以必须在起服务前拦住。
# CHECK=0 可跳过(仅供诊断);CHECK_STRICT=1 时自检不通过直接拒绝启动。
if [ "${CHECK:-1}" = "1" ]; then
  # 注意:TAG= 置空 ⇒ 自检读「当前 shell 的 env」而不是某个已启动实例的记录
  if ! TAG= MBT="$MBT" GP_MIN="$GP_MIN" CUDAGRAPH_SIZES="$CUDAGRAPH_SIZES" \
       INTERLEAVE="$INTERLEAVE" SPIN_IDLE_US="$SPIN_IDLE_US" \
       NSLICE_SMALL="$NSLICE_SMALL" THREADS="$THREADS" RESIDENT="$RESIDENT" \
       OOT="$OOT" ENV="$ENV" bash "$ROOT/scripts/check_mainline_env.sh"; then
    if [ "${CHECK_STRICT:-0}" = "1" ]; then
      echo "[mainline] 启动自检未通过,拒绝启动(CHECK_STRICT=1)"; exit 1
    fi
    echo "[mainline] ⚠️ 自检有未通过项,仍继续启动(CHECK_STRICT=1 可改为拒绝)"
  fi
fi
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
if [ "$EAGER" = "1" ]; then ARGS+=(--enforce-eager); fi
# 注意:`--cudagraph-capture-sizes` 是 nargs='+'(**空格分隔**,不是逗号)
if [ -n "$CUDAGRAPH_SIZES" ]; then ARGS+=(--cudagraph-capture-sizes $CUDAGRAPH_SIZES); fi
if [ -n "$LOAD_STRATEGY" ]; then ARGS+=(--safetensors-load-strategy "$LOAD_STRATEGY"); fi
if [ "$KERNEL_WARMUP" = "0" ]; then
  ARGS+=(--kernel-config '{"enable_jit_warmup": false}')
fi

{
  echo "tag=$TAG port=$PORT tp=$TP gpus=$GPUS maxlen=$MAXLEN seqs=$SEQS gpu_util=$GPU_UTIL"
  echo "mbt='$MBT' chunked='$CHUNKED_PREFILL' threads=$THREADS resident='$RESIDENT' oot=$OOT load_strategy='$LOAD_STRATEGY' extra_env='$EXTRA_ENV'"
echo "gp_min=$GP_MIN cudagraph_sizes='$CUDAGRAPH_SIZES' prefill_preset='${PREFILL:-0}' interleave='$INTERLEAVE' spin_idle_us='$SPIN_IDLE_US' nslice_small='$NSLICE_SMALL' async='$ASYNC'"
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
NCTL=()
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

if [ "$INTERLEAVE" = "1" ] && command -v numactl >/dev/null 2>&1; then
  NCTL=(numactl --interleave=all)
fi
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
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="$GP_MIN" \
  XIAOTU_MOE_THREADS="$THREADS" \
  XIAOTU_MOE_NSLICE_SMALL="$NSLICE_SMALL" \
  XIAOTU_MOE_ASYNC="$ASYNC" \
  XIAOTU_MOE_SPIN_IDLE_US="$SPIN_IDLE_US" \
  XIAOTU_MOE_GPU_RESIDENT_LAYERS="$RESIDENT" \
  OMP_NUM_THREADS=1 \
  $EXTRA_ENV \
  "${NCTL[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" > "$LOG" 2>&1 &
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
