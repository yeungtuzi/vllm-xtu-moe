#!/usr/bin/env bash
# 把 vLLM-XTU 的补丁打到一棵 vLLM 上(源码树或已安装的 site-packages 树)。
#
# ★ 开发模型(用户 2026-09-24 明确):
#   * vLLM 仓库**与上游保持一致** —— 定期 rebase,**不在 vLLM 侧留自有长命分支**;
#   * 我们的功能**全部以 patch 形式**保存在本仓(patches/xtu-series/),这是**唯一真源**。
#
# 补丁是**纯 Python**(不碰 csrc/rust)⇒ 既能打源码树,也能直接打 site-packages(无需编译)。
#
# 用法:
#   scripts/apply_xtu_patches.sh <含 vllm/ 的目录> [series_dir]    # 默认用 patches/xtu-series
#   DRY=1 ...                                                      # 只试不落盘
#   LEVEL=1|2|3 ...                                                # legacy:旧 topic patch(基线已过期,仅供追溯)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="${1:-}"
DRY="${DRY:-0}"
SERIES_DIR="${2:-$ROOT/patches/xtu-series}"

if [ -z "$TREE" ] || [ ! -d "$TREE/vllm" ]; then
  echo "用法: $0 <包含 vllm/ 的目录> [series_dir](当前:'$TREE')" >&2
  exit 2
fi

apply_one() {
  local p="$1"
  echo "[xtu-patch] $(basename "$p")"
  # 严格模式:**不做** patch(1) 回退 —— `patch -f` 会静默跳过打不上的 hunk ✗
  # 注意:系列必须**逐条顺序**应用(后一条依赖前一条改过的文件),故不支持"整体 dry-run"。
  if [ "$DRY" = "1" ]; then
    ( cd "$TREE" && git apply --check --whitespace=nowarn "$p" )
  else
    ( cd "$TREE" && git apply --whitespace=nowarn "$p" )
  fi
}

if [ "${LEVEL:-}" != "" ]; then
  echo "[xtu-patch] legacy LEVEL 模式(基线 dabc4362b,可能已过期)"
  PATCHES=("$ROOT/patches/upstream/pr0-handshake-timeout.patch"
           "$ROOT/patches/upstream/pr1-experts-load-device.patch")
  [ "$LEVEL" -ge 2 ] && PATCHES+=("$ROOT/patches/upstream/pr2-fp8-sm80-o-proj.patch")
  [ "$LEVEL" -ge 3 ] && PATCHES+=("$ROOT/patches/upstream/pr3-sm80-port.patch")
  echo "[xtu-patch] tree=$TREE level=$LEVEL dry=$DRY"
  for p in "${PATCHES[@]}"; do apply_one "$p"; done
else
  echo "[xtu-patch] series 模式:dir=$SERIES_DIR tree=$TREE dry=$DRY"
  [ -f "$SERIES_DIR/series" ] || { echo "缺少 $SERIES_DIR/series" >&2; exit 2; }
  while read -r line; do
    case "$line" in ''|'#'*) continue;; esac
    apply_one "$SERIES_DIR/$line"
  done < "$SERIES_DIR/series"
fi
echo "[xtu-patch] 完成 ✓"
