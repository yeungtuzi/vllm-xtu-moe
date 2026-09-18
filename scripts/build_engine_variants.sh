#!/usr/bin/env bash
# Build the bundled xiaotu engine in all ISA variants (one .so per ISA level).
#
# 本脚本构建的是 **本仓库内置的引擎**(`xiaotu_moe/csrc`),不依赖外部仓库。
# 同一份 `python_binding/binding.cpp` 编译多次,每次用不同的
# ISA 宏和一个独立的 pybind 模块名 `_xiaotu_moe_C_<suffix>`;`xiaotu_moe/loader.py`
# 在导入时按 /proc/cpuinfo 选最高可用变体。
#
# 变体阶梯(越高越新,优先选):
#   scalar          无 SIMD(兜底)
#   avx2            -mavx2 -mfma
#   avx512_base     -mavx512f -mavx512bw -mavx512vl -mavx512dq
#   avx512_vnni     base + -mavx512vnni
#   avx512_bf16     base + -mavx512bf16
#   avx512_amx      AMX(未接线,不构建)
#
# 用法:
#   PYTHON=/path/to/venv/bin/python scripts/build_engine_variants.sh
#   PYBIND11_INC=... CUDART_ROOT=... 可覆盖自动探测
#
# License: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../xiaotu_moe" && pwd)"
SRC="$ROOT/csrc/python_binding/binding.cpp"
OUT_DIR="$ROOT/build"
mkdir -p "$OUT_DIR"

PYTHON="${PYTHON:-python3}"
PY_INC="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
PY_EXT="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

# pybind11(header-only):优先环境变量,其次常见位置
if [[ -z "${PYBIND11_INC:-}" ]]; then
  # 【2026-09-17 修】本 env 里**没有独立的 pybind11 包**,但 **torch 自带一份**头文件
  # (`site-packages/torch/include/pybind11/`)。原来只试 `import pybind11` 和 /usr/include,
  # 于是构建立在 `exit 2` 上 —— 而 `BUILD_EXIT` 若被 `| tail` 吞掉,门禁就会**去验旧的 .so**。
  # 注意 `-I"$PYBIND11_INC"` 要求这里是**包含 `pybind11/` 的那个目录**(不是 pybind11 本身)。
  for cand in \
      "$("$PYTHON" -c 'import pybind11, os; print(os.path.join(pybind11.get_include()))' 2>/dev/null)" \
      "$("$PYTHON" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"include"))' 2>/dev/null)" \
      /usr/include /usr/local/include ; do
    [[ -n "$cand" && -d "$cand" ]] && PYBIND11_INC="$cand" && break
  done
fi
if [[ -z "${PYBIND11_INC:-}" || ! -d "$PYBIND11_INC" ]]; then
  echo "!! pybind11 include dir not found; set PYBIND11_INC" >&2
  exit 2
fi

# CUDA runtime:binding.cpp 用 cudaStream_t/cudaMemcpyAsync 等。
# 注意 pip 的 nvidia/cu13 包只有 libcudart.so.13(没有 .so 符号链接),
# 所以这里同时探测目录与真实库名,用 -l:<soname> 兜底。
_cudart_candidates() {
  [[ -n "${CUDART_ROOT:-}" ]] && echo "$CUDART_ROOT"
  "$PYTHON" -c 'import nvidia.cu13, os; print(os.path.dirname(nvidia.cu13.__file__))' 2>/dev/null || true
  echo /usr/local/cuda
}
CUDART_INC=""; CUDART_LIBDIR=""; CUDART_SONAME=""
while IFS= read -r cand; do
  [[ -n "$cand" && -d "$cand/include/crt" ]] || continue
  for libdir in "$cand/lib" "$cand/lib64"; do
    if [[ -e "$libdir/libcudart.so" ]]; then
      CUDART_INC="$cand/include"; CUDART_LIBDIR="$libdir"; CUDART_SONAME="libcudart.so"; break
    fi
    for f in "$libdir"/libcudart.so.*; do
      if [[ -e "$f" ]]; then
        CUDART_INC="$cand/include"; CUDART_LIBDIR="$libdir"
        CUDART_SONAME="$(basename "$f")"; break
      fi
    done
    [[ -n "$CUDART_SONAME" ]] && break
  done
  [[ -n "$CUDART_SONAME" ]] && break
done < <(_cudart_candidates | sort -u)

CUDA_FLAGS=()
if [[ -n "$CUDART_SONAME" ]]; then
  CUDA_FLAGS=(-I"$CUDART_INC" -L"$CUDART_LIBDIR"
              -Wl,--no-as-needed -l:"$CUDART_SONAME" -Wl,--as-needed
              -Wl,-rpath,"$CUDART_LIBDIR")
  echo ">> CUDA runtime: $CUDART_INC / $CUDART_LIBDIR ($CUDART_SONAME)"
else
  echo "!! CUDA runtime not found; set CUDART_ROOT (need include/crt + libcudart)" >&2
  exit 2
fi

# CUDA **驱动** API:item 1/3 的 `cuStreamWriteValue32`/`cuStreamWaitValue32` 在 libcuda。
# 不链接会在 `import` 时报 `undefined symbol: cuStreamWriteValue32_v2`(已踩)。
DRV_LIBDIR=""; DRV_SONAME=""
for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/cuda/lib64/stubs; do
  if [[ -e "$d/libcuda.so.1" ]]; then DRV_LIBDIR="$d"; DRV_SONAME="libcuda.so.1"; break; fi
  if [[ -e "$d/libcuda.so" ]];   then DRV_LIBDIR="$d"; DRV_SONAME="libcuda.so";   break; fi
done
if [[ -n "$DRV_SONAME" ]]; then
  CUDA_FLAGS+=(-L"$DRV_LIBDIR" -Wl,--no-as-needed -l:"$DRV_SONAME" -Wl,--as-needed)
  echo ">> CUDA driver: $DRV_LIBDIR ($DRV_SONAME)"
else
  echo "!! CUDA driver lib not found (libcuda.so.1 is required for cuStreamWriteValue32)" >&2
  exit 2
fi

COMMON="-std=c++17 -shared -fPIC -O3 -ffast-math -fno-finite-math-only"
# 【§629c】编译器可切(`CXX=g++-14` 或 conda 的
# `x86_64-conda-linux-gnu-g++`)。用来对照 lk_moe 的 GCC 13.3.1 世代与 Clang。
CXX_BIN="${CXX:-g++}"
# 【§630b】`MTUNE=<cpu>`:`-mtune` 只影响**调度**,不改 ISA ⇒ 不会破坏 ISA 变体阶梯
# (绝不能用 `-march=znver4`,那会把 AVX-512 塞进 scalar/avx2 兜底变体)。
MTUNE="${MTUNE:-}"
# 【§631】`EXTRA_FLAGS` 透传额外宏(如 `-DXIAOTU_MOE_FOLD_SCALE=1`),用于 A/B。
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
[[ -n "$MTUNE" ]] && COMMON="-mtune=$MTUNE $COMMON"
[[ -n "$EXTRA_FLAGS" ]] && COMMON="$COMMON $EXTRA_FLAGS"
echo ">> compiler: $($CXX_BIN --version | head -1)"
echo ">> COMMON=$COMMON"

VARIANTS=(
  "scalar|"
  "avx2|-mavx2 -mfma"
  "avx512_base|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma"
  "avx512_vnni|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma"
  "avx512_bf16|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mfma"
  "avx512_bf16_vbmi|-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vbmi -mfma"
)

for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  mod="_xiaotu_moe_C_${name}"
  OUT="$OUT_DIR/${mod}${PY_EXT}"
  echo ">> Building variant [$name] -> $OUT"
  # shellcheck disable=SC2086
  "$CXX_BIN" $COMMON $flags \
      -DXIAOTU_MOE_MODULE_NAME="$mod" \
      -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" \
      "${CUDA_FLAGS[@]}" \
      "$SRC" \
      -o "$OUT"
done

echo ">> Done. Variants in $OUT_DIR:"
ls -1 "$OUT_DIR"/_xiaotu_moe_C*"${PY_EXT}"
