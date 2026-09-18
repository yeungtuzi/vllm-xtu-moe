#!/usr/bin/env bash
# 一条命令装好:mainline vLLM + vllm-xtu-moe 补丁 + 插件。
#
#   bash scripts/install_mainline.sh            # 装到当前 python 环境
#   TREE=/path/to/vllm bash scripts/install_mainline.sh     # 指定 vLLM 源码树
#   LEVEL=0 bash scripts/install_mainline.sh    # **纯插件路径**:一个补丁都不打(受限,见下)
#   LEVEL=2 bash scripts/install_mainline.sh    # + PR2(A100/SM80 的 FP8 o_proj)
#   LEVEL=3 bash scripts/install_mainline.sh    # + PR3(SM80 DS-V4 移植,21 文件)
#
# 补丁级别(逐块"为什么主线做不到"的理由见 dev-docs/UPSTREAM_DRIFT.md):
#   L0  零补丁(纯插件):只走 `vllm.general_plugins` 入口 + OOT 注册表覆盖。
#       **本机(DS-V4 / 2x A100-40GB)不可用于真实服务**,缺两件上游还没有的能力:
#         - VLLM_EXPERTS_LOAD_DEVICE=cpu(pr1):没有它,138 GB 专家权重无处安放;
#         - 可配置的引擎握手超时(pr0):没有它,逐层构造 CPU 引擎会撞上硬编码 5 分钟。
#       它能证明的是:插件对主线可 import / 可注册 / 可覆盖(CI 冒烟级)。
#   L1(默认) pr0 + pr1 = "能用"的最小集(7 个文件;基线 dabc4362b)。
#   DRY=1 bash scripts/install_mainline.sh      # 只打印将要做什么
#
# 三种模式(自动选择):
#   A) --vllm-wheel <file>   已装好官方 vLLM ⇒ 只把补丁打到 site-packages(最快)
#   B) --tree <dir>          vLLM 源码树已存在 ⇒ 打补丁 + pip install -e .
#   C) 都没有                 clone 主线 → 打补丁 → pip install -e .(最慢)
#
# 详见 docs/INSTALL_MAINLINE.md(含失败回退表)。
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
LEVEL="${LEVEL:-1}"
DRY="${DRY:-0}"
TREE="${TREE:-}"
PLUGIN_WHEEL="${PLUGIN_WHEEL:-$(ls -t "$ROOT"/dist/vllm_xtu_moe-*.whl 2>/dev/null | head -1 || true)}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-}"
WORKDIR="${WORKDIR:-$HOME/xtu-moe-mainline}"

say() { echo "[install] $*"; }
run() { if [ "$DRY" = "1" ]; then echo "  DRY: $*"; else "$@"; fi; }

# 统一的"打补丁"入口:LEVEL<=0 时完全跳过(纯插件路径)
apply_patches() {
  local tree="$1"
  if [ "$LEVEL" -le 0 ]; then
    say "LEVEL=0 => 不打任何补丁(纯插件路径)"
    say "  警告:本机不可用于真实服务 —— 缺 pr1(VLLM_EXPERTS_LOAD_DEVICE=cpu),138GB 专家无处安放;"
    say "        缺 pr0(可配置握手超时),逐层构造 CPU 引擎会撞上硬编码 5 分钟。仅作 import/注册冒烟。"
    return 0
  fi
  run env LEVEL="$LEVEL" bash "$ROOT/scripts/apply_xtu_patches.sh" "$tree"
}

say "python = $($PY -c 'import sys;print(sys.executable)')"
say "level=$LEVEL tree='${TREE:-auto}' plugin_wheel='${PLUGIN_WHEEL:-<none>}' dry=$DRY"

# ---- 1) 定位 / 获取 vLLM 树 -------------------------------------------------
MODE=""
if [ -n "$TREE" ]; then
  MODE="B"
elif $PY -c "import vllm" 2>/dev/null; then
  MODE="A"
else
  MODE="C"
fi
say "mode=$MODE"

case "$MODE" in
  A)
    # 已安装:补丁直接打到 site-packages(补丁是纯 Python,不需要重编译)
    TREE="$($PY -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
    say "已安装 vLLM,补丁目标 = $TREE"
    apply_patches "$TREE"
    ;;
  B)
    say "使用已有 vLLM 源码树:$TREE"
    apply_patches "$TREE"
    run $PY -m pip install -e "$TREE" --no-build-isolation
    ;;
  C)
    say "clone vLLM 主线 → $WORKDIR/vllm"
    if [ ! -d "$WORKDIR/vllm" ]; then
      run mkdir -p "$WORKDIR"
      if [ -n "$VLLM_REF" ]; then
        run git clone --depth 1 --branch "$VLLM_REF" "$VLLM_REPO" "$WORKDIR/vllm"
      else
        run git clone --depth 1 "$VLLM_REPO" "$WORKDIR/vllm"
      fi
    fi
    TREE="$WORKDIR/vllm"
    apply_patches "$TREE"
    say "安装 vLLM(复用官方预编译二进制,避免数小时编译)"
    if [ "$DRY" = "1" ]; then
      echo "  DRY: VLLM_USE_PRECOMPILED=1 $PY -m pip install -e $TREE"
    else
      VLLM_USE_PRECOMPILED=1 LEVEL="$LEVEL" $PY -m pip install -e "$TREE" --no-build-isolation
    fi
    ;;
esac

# ---- 2) 装插件 --------------------------------------------------------------
if [ -n "$PLUGIN_WHEEL" ]; then
  say "安装插件 $PLUGIN_WHEEL"
  run $PY -m pip install --no-deps "$PLUGIN_WHEEL"
else
  say "⚠️ 没找到 dist/*.whl;改为从本仓库源码安装插件"
  run $PY -m pip install --no-deps "$ROOT"
fi

# ---- 3) 自检 ----------------------------------------------------------------
say "自检(不加载模型)"
if [ "$DRY" = "1" ]; then
  echo "  DRY: import xiaotu_moe / vllm_xiaotu_moe / entry-points"
else
  $PY - <<'PYEOF'
import importlib.metadata as md
import xiaotu_moe
m = xiaotu_moe.load()
print("  engine :", m.__  # 宿主契约自检:serve 时那些"缺失只静默变慢(可达 30x)"的开关(见 PLUGIN_INTERFACE.md)
  say "宿主契约自检(serve 时的必需 env)"
  if ! ENV="$($PY -c 'import sys;print(sys.prefix)')" TAG= bash "$ROOT/scripts/check_mainline_env.sh"; then
    say "上面有未通过项:不会让服务起不来,只会静默变慢 —— 起服务前请按提示设置。"
  fi
file__.rsplit("/", 1)[-1])
import vllm_xiaotu_moe  # noqa: F401
print("  plugin : ok")
eps = [e.value for e in md.entry_points().select(group="vllm.general_plugins")]
print("  entry  :", [e for e in eps if "xiaotu" in e] or "⚠️ 未注册")
PYEOF
fi

cat <<EOF

[install] 完成。起服务:
  export VLLM_EXPERTS_LOAD_DEVICE=cpu
  TAG=myrun PORT=8071 bash $ROOT/scripts/serve_mainline.sh

[install] 若启动失败,先看 $ROOT/docs/INSTALL_MAINLINE.md 的"失败回退"表
          (三个最常见的坑:prefetch 加载 / 关 JIT warmup / 关 FlashInfer sampler)。
[install] 失败回退(四个最常见的坑):
          1) 分片加载慢/卡住             => LOAD_STRATEGY=prefetch(EXT4 上主线明确建议)
          2) type fp8e4nv not supported  => KERNEL_WARMUP=0(关 JIT warmup;A100 无 FP8)
          3) FlashInfer 要 CUDA>=12.8     => VLLM_USE_FLASHINFER_SAMPLER=0(本机 CUDA 12.1)
          4) 启动只加载到一半就没了(无 Traceback) => 单 NUMA 节点耗尽,确认 INTERLEAVE=1

[install] 一次能用的完整配置(本机实测;同机对照见 scripts/ab_mainline_vs_fork.sh):
          ENV=<本环境> PREFILL=1 RESIDENT=0-11 THREADS=60 bash $ROOT/scripts/serve_mainline.sh
          (PREFILL=1 => MBT=8192/GP_MIN=1024/只捕获解码尺寸;serve_mainline.sh 已默认带上
           SPIN_IDLE_US=0 与 NSLICE_SMALL=0 —— 这两个缺失会让解码静默慢 30 倍)
EOF
