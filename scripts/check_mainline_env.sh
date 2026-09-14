#!/usr/bin/env bash
# v0.2 主线插件**启动自检**:把"宿主节奏决定引擎行为"的那些隐式契约变成显式断言。
#
# 为什么需要:同一个 `xiaotu_moe.so`,在 fork 编排下 26.81 ms/token,在主线下曾达
# 1225 ms/token(47×)。差异**完全不在代码 diff 里**,而在宿主喂给引擎的节奏与配套开关上
# (见 docs/PLUGIN_INTERFACE.md)。这些开关缺失时**不会报错,只会静默变慢**,
# 所以必须在启动时断言,而不是等基准跑出来才发现。
#
# 用法:
#   bash scripts/check_mainline_env.sh                    # 检查当前 shell 的 env
#   TAG=ml_gp1 bash scripts/check_mainline_env.sh          # 检查某个已启动实例的 .env 记录
#   STRICT=1 bash scripts/check_mainline_env.sh            # 有 FAIL 就 exit 1(默认也 exit 非 0)
#
# 退出码 = FAIL 的条数(0 = 全通过)。
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- 取值来源:优先读取已启动实例的 .env 记录,否则用当前 env 的默认 ----
SRC="current shell"
if [ -n "${TAG:-}" ]; then
  ENVF="$ROOT/report/tuning/logs/$TAG.env"
  if [ -f "$ENVF" ]; then
    SRC="$ENVF"
    # 从 `k='v'` / `k=v` 记录里取回(serve_mainline.sh 写的格式)
    get() { sed -n "s/.*[ \t]${1}='\{0,1\}\([^' ]*\)'\{0,1\}.*/\1/p" "$ENVF" | head -1; }
    MBT="$(sed -n "s/^mbt='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    GP_MIN="$(sed -n 's/.*gp_min=\([0-9-]*\).*/\1/p' "$ENVF" | head -1)"
    CUDAGRAPH_SIZES="$(sed -n "s/.*cudagraph_sizes='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    INTERLEAVE="$(sed -n "s/.*interleave='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    SPIN_IDLE_US="$(sed -n "s/.*spin_idle_us='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    NSLICE_SMALL="$(sed -n "s/.*nslice_small='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    RESIDENT="$(sed -n "s/.*resident='\([^']*\)'.*/\1/p" "$ENVF" | head -1)"
    THREADS="$(sed -n "s/.*threads=\([0-9]*\).*/\1/p" "$ENVF" | head -1)"
    OOT="$(sed -n "s/.*oot=\([0-9]*\).*/\1/p" "$ENVF" | head -1)"
  else
    echo "[check] 找不到 $ENVF;改用当前 shell 的 env" >&2
  fi
fi
: "${MBT:=256}"; : "${GP_MIN:=0}"; : "${CUDAGRAPH_SIZES:=}"; : "${INTERLEAVE:=1}"
: "${SPIN_IDLE_US:=}"; : "${NSLICE_SMALL:=}"; : "${RESIDENT:=}"; : "${THREADS:=60}"; : "${OOT:=1}"

FAIL=0
ok()   { printf '  \033[32mPASS\033[0m  %-34s %s\n' "$1" "$2"; }
bad()  { printf '  \033[31mFAIL\033[0m  %-34s %s\n' "$1" "$2"; FAIL=$((FAIL+1)); }
warn() { printf '  \033[33mWARN\033[0m  %-34s %s\n' "$1" "$2"; }

echo "=============================================================="
echo " vllm-xtu-moe 主线插件启动自检"
echo " 取值来源: $SRC"
echo "=============================================================="

# --- 1) 线程池自旋窗口:必须 0 ------------------------------------------------
# 引擎默认 5000 µs ⇒ 每次调用后**所有** worker 自旋 5 ms;43 层×~3 相 ⇒ 池几乎不停转,
# worker 冲到 3242% CPU(≈32 核/worker)、load 119,而自造的争抢又反过来拖慢调用线程
# (NOTES 的 N4:park/wake 成本与层间隔耦合,是正反馈)。
# 注意反方向也对:旋到 600000(永不 park)更糟,几分钟后从 43 ms 退化到 1313 ms/token。
if [ "${SPIN_IDLE_US}" = "0" ]; then
  ok "XIAOTU_MOE_SPIN_IDLE_US=0" "worker 直接 futex 睡(实测 CPU 145%、load 11)"
else
  bad "XIAOTU_MOE_SPIN_IDLE_US=0" "当前='${SPIN_IDLE_US:-<未设=引擎默认 5000>}' ⇒ 解码可能从 37 ms 漂到 1225 ms/token"
fi

# --- 2) 小 batch N-slice:必须 0 ---------------------------------------------
# MOE_V2::small_batch_workers() 用标称 8 MAC/cycle 估线程数,DS-V4 解码维度算出 6292 µs
# (真实百 µs 级,高估约 50×)⇒ wlimit=59/nt=60 ⇒ stride=1 ⇒ worker_limit_ 门闸完全失效
# (没有任何 worker 去 park),且走 parallel_for_limited —— 那条路代码自己标注了会挂死。
if [ "${NSLICE_SMALL}" = "0" ]; then
  ok "XIAOTU_MOE_NSLICE_SMALL=0" "绕开 wlimit=59 的 limited 路径"
else
  bad "XIAOTU_MOE_NSLICE_SMALL=0" "当前='${NSLICE_SMALL:-<未设=默认开>}' ⇒ 60 个 worker 全部参与每一相"
fi

# --- 3) 线程数必须显式给 -----------------------------------------------------
# 引擎在 XIAOTU_MOE_THREADS 与 LK_THREADS 都缺省时退到 min(hw,120) = 120,
# 远超本机每 CCD 4-5 核(96-120 全核才跑满 DDR5 通道)的甜蜜点。
if [ -n "${THREADS}" ] && [ "${THREADS}" -gt 0 ] 2>/dev/null; then
  ok "XIAOTU_MOE_THREADS 已显式设置" "= $THREADS(本机推荐 60 = 12 CCD × 5 核/CCD)"
else
  bad "XIAOTU_MOE_THREADS 已显式设置" "未设 ⇒ 池退到 120,超出甜蜜点"
fi

# --- 4) GPU 预填充阈值 vs MBT -----------------------------------------------
# 阈值语义是 `hidden_states.size(0) >= T`;若 T > MBT,一个 chunk 永远到不了阈值 ⇒
# GPU 预填充**静默失效**(fork 的做法是把 MINBATCH 夹到 MBT)。
if [ "${GP_MIN}" -gt 0 ] 2>/dev/null; then
  if [ "${GP_MIN}" -le "${MBT}" ] 2>/dev/null; then
    ok "GP_MIN <= MBT" "GP_MIN=$GP_MIN MBT=$MBT ⇒ 阈值可达"
  else
    bad "GP_MIN <= MBT" "GP_MIN=$GP_MIN > MBT=$MBT ⇒ GPU 预填充永远不会触发"
  fi
  # --- 5) GPU 预填充需要预填充形状**不被 CUDA 图捕获** ---------------------
  # 主线默认 cudagraph_mode=PIECEWISE 会把预填充形状也捕获;捕获时走的是 CPU 分支,
  # 重放时永远重放那个分支 ⇒ GPU 路径形同虚设。上游原生解法:--cudagraph-capture-sizes
  # 只列解码尺寸(NOTES §340c)。
  if [ -n "${CUDAGRAPH_SIZES}" ]; then
    ok "cudagraph capture sizes 已限制" "='$CUDAGRAPH_SIZES' ⇒ 预填充形状走 eager"
  else
    bad "cudagraph capture sizes 已限制" "未限制 ⇒ PIECEWISE 会把预填充形状也捕获,GPU 预填充失效"
  fi
else
  ok "GPU 预填充未启用" "GP_MIN=0(解码基准协议口径)"
fi

# --- 6) NUMA 交错 -----------------------------------------------------------
# vLLM 加载时每个 rank 装**全部** 256 个专家(≈138 GB/worker,EP 不在加载期切分存储),
# 这是未绑定的 first-touch 分配;实测总空闲 863 GB 时 node 0/2 已用 185/193 GB ⇒
# `oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0`,进程**静默死亡**。
if [ "${INTERLEAVE}" = "1" ]; then
  if command -v numactl >/dev/null 2>&1; then
    ok "NUMA 交错" "numactl --interleave=all 已启用"
  else
    bad "NUMA 交错" "INTERLEAVE=1 但系统没有 numactl"
  fi
else
  bad "NUMA 交错" "INTERLEAVE=$INTERLEAVE ⇒ 极易单 node 耗尽被 OOM kill(日志会出现「只加载到一半就没了」)"
fi

# --- 7) 引擎可导入 + 变体 ---------------------------------------------------
PYBIN="${ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}/bin/python"
if [ -x "$PYBIN" ]; then
  V="$("$PYBIN" -c 'import xiaotu_moe;print(xiaotu_moe.__variant__, xiaotu_moe.__version__)' 2>/dev/null | tail -1)"
  if [ -n "$V" ]; then ok "引擎可导入" "$V (env=$PYBIN)"; else bad "引擎可导入" "$PYBIN 里 import xiaotu_moe 失败"; fi
  P="$("$PYBIN" -c 'import vllm_xiaotu_moe;print(vllm_xiaotu_moe.__file__)' 2>/dev/null | tail -1)"
  if [ -n "$P" ]; then ok "插件可导入" "$P"; else bad "插件可导入" "import vllm_xiaotu_moe 失败"; fi
else
  warn "引擎可导入" "找不到 $PYBIN(设 ENV=... 后重试)"
fi

# --- 8) 常驻层与 OOT 模式 ---------------------------------------------------
if [ -n "${RESIDENT}" ]; then ok "常驻 GPU 层" "RESIDENT='$RESIDENT'"; else warn "常驻 GPU 层" "0 层常驻(全部 MoE 走 CPU;对照基线用)"; fi
if [ "${OOT}" = "1" ]; then ok "插件模式" "A(OOT override ⇒ hybrid_model,含 GPU 预填充)"; else warn "插件模式" "B(主线 RoutedExperts + CPU backend;无 GPU 预填充路径)"; fi

echo "--------------------------------------------------------------"
if [ "$FAIL" -eq 0 ]; then
  echo " 结论:全部通过。"
else
  echo " 结论:$FAIL 项未通过 —— 这些**不会报错,只会静默变慢**(可达 30×)。"
fi
echo "=============================================================="
exit "$FAIL"
