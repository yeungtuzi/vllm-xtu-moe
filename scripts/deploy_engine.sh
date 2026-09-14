#!/usr/bin/env bash
# 把**本仓库**构建出来的引擎 `.so` 部署到服务用的 conda env(site-packages)。
#
# 为什么需要它:服务(`scripts/serve_lk_port.sh`)用的是 `$ENV` 里 **site-packages
# 下的一份 `xiaotu_moe` 拷贝**(不是本仓库!),而 harness(`scripts/bench_cd_plumbing.py`
# 等)通过 `PYTHONPATH` 用**本仓库**的 `xiaotu_moe`。
# ⇒ 只 `build_engine_variants.sh` **不会**改变服务行为(踩过:服务量出的 28.76 ms
#   其实还是旧的 host-func 路径,而 harness 已经跑 async)。
#
# 用法:
#   PYTHON=/path/to/env/bin/python [ENVS="/path/a /path/b"] scripts/deploy_engine.sh
#   仅复制(不重新编译):BUILD=0 PYTHON=... scripts/deploy_engine.sh
#
# License: Apache-2.0
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/xiaotu_moe"
PYTHON="${PYTHON:-/home/user/anaconda3/envs/lkxtu/bin/python}"
# 默认部署到两个 env(lkxtu = 我们的服务;vllm-xiaotu-moe = 对照/harness)
ENVS="${ENVS:-/home/user/anaconda3/envs/lkxtu /home/user/anaconda3/envs/vllm-xiaotu-moe}"
BUILD="${BUILD:-1}"

if [ "$BUILD" = "1" ]; then
  # pybind11 头文件:优先 pybind11 包,其次 torch 自带的 include(本机只有后者)。
  PYBIND11_INC="${PYBIND11_INC:-$(ls -d /home/user/anaconda3/lib/python3.*/site-packages/torch/include 2>/dev/null | head -1)}"
  echo "[deploy] build (PYBIND11_INC=$PYBIND11_INC)"
  PYTHON="$PYTHON" PYBIND11_INC="$PYBIND11_INC" bash "$ROOT/scripts/build_engine_variants.sh"
fi

for env in $ENVS; do
  dst="$env/lib/python3.12/site-packages/xiaotu_moe"
  [ -d "$dst" ] || { echo "[deploy] skip $env (no xiaotu_moe)"; continue; }
  echo "[deploy] -> $dst"
  cp -f "$SRC"/build/_xiaotu_moe_C_*.so "$dst/build/"
  # 头文件/源码也同步一份,方便在 env 里直接读代码(不参与编译)。
  cp -f "$SRC"/csrc/python_binding/binding.cpp "$dst/csrc/python_binding/binding.cpp" 2>/dev/null || true
  cp -f "$SRC"/csrc/moe/*.hpp "$dst/csrc/moe/" 2>/dev/null || true
  ls -l "$dst"/build/_xiaotu_moe_C_avx512_bf16*.so | awk '{print "    ", $NF, $5, $6, $7, $8}'
done
echo "[deploy] done"
