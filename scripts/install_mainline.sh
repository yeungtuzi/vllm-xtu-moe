#!/usr/bin/env bash
# 一条命令装好:mainline vLLM + vllm-xtu-moe 补丁 + 插件。
#
#   bash scripts/install_mainline.sh            # 装到当前 python 环境
#   TREE=/path/to/vllm bash scripts/install_mainline.sh     # 指定 vLLM 源码树
#   LEVEL=3 bash scripts/install_mainline.sh    # 附带 A100/SM80 移植补丁
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
    run bash "$ROOT/scripts/apply_xtu_patches.sh" "$TREE"
    ;;
  B)
    say "使用已有 vLLM 源码树:$TREE"
    run bash "$ROOT/scripts/apply_xtu_patches.sh" "$TREE"
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
    run bash "$ROOT/scripts/apply_xtu_patches.sh" "$TREE"
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
print("  engine :", m.__file__.rsplit("/", 1)[-1])
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
EOF
