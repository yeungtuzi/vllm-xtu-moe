#!/usr/bin/env bash
# 把 vllm-xtu-moe 的补丁打到一棵 vLLM 主线上(源码树或已安装的 site-packages 树)。
#
# 补丁是**纯 Python**(不碰 csrc/rust),所以既能打在源码树上,也能直接打在
# `pip install vllm` 之后的 site-packages 里 —— 后者不需要编译任何东西。
#
# 基线:以下补丁针对 vLLM 主线 **dabc4362b**(2026-09-14)重新生成并实测全部干净应用。
# 规模:pr0=1 文件 / pr1=6 文件 / pr2=2 文件 / pr3=21 文件。
#
# 用法:
#   scripts/apply_xtu_patches.sh <vllm_tree>            # 只打 PR1(最小使能补丁)
#   LEVEL=1 scripts/apply_xtu_patches.sh <vllm_tree>    # 同上(默认)
#   LEVEL=2 scripts/apply_xtu_patches.sh <vllm_tree>    # + PR2(A100/SM80 的 FP8 o_proj)
#   LEVEL=3 scripts/apply_xtu_patches.sh <vllm_tree>    # + PR3(SM80 DS-V4 移植,含新内核文件)
#   DRY=1 ...                                           # 只 --dry-run
#
# <vllm_tree> 是**包含 `vllm/` 目录**的那一层,例如:
#   /path/to/vllm                       (源码树)
#   /path/to/venv/lib/python3.12/site-packages   (已安装)
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE="${1:-}"
LEVEL="${LEVEL:-1}"
DRY="${DRY:-0}"

if [ -z "$TREE" ] || [ ! -d "$TREE/vllm" ]; then
  echo "用法: $0 <包含 vllm/ 的目录>(当前:'$TREE')" >&2
  exit 2
fi

# pr0 必须最先打:它把"引擎握手超时"变成可配置的,否则 CPU 引擎逐层构造
# (>5 min)会在健康加载过程中被主线硬编码的 5 分钟握手超时掐掉。
PATCHES=("$ROOT/patches/upstream/pr0-handshake-timeout.patch"
         "$ROOT/patches/upstream/pr1-experts-load-device.patch")
[ "$LEVEL" -ge 2 ] && PATCHES+=("$ROOT/patches/upstream/pr2-fp8-sm80-o-proj.patch")
[ "$LEVEL" -ge 3 ] && PATCHES+=("$ROOT/patches/upstream/pr3-sm80-port.patch")

echo "[xtu-patch] tree=$TREE level=$LEVEL dry=$DRY"
for p in "${PATCHES[@]}"; do
  echo "[xtu-patch] $(basename "$p")"
  if [ "$DRY" = "1" ]; then
    ( cd "$TREE" && patch -p1 --dry-run < "$p" )
  else
    ( cd "$TREE" && patch -p1 < "$p" )
  fi
done
echo "[xtu-patch] done"
