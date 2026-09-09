#!/bin/bash
set -e
ROOT=/home/user/lvllm/xiaotu-moe
OUT=$ROOT/xiaotu_moe/build/_xiaotu_moe_C_avx512_bf16
PY=/home/user/anaconda3/envs/xiaotumoe-vllm/bin/python
PY_INC=$($PY -c "import sysconfig;print(sysconfig.get_paths()['include'])")
PY_EXT=$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11_INC=/home/user/lvllm/.search-venv/lib/python3.10/site-packages/pybind11/include
CUD=/home/user/anaconda3/envs/xiaotumoe-vllm/lib/python3.12/site-packages/nvidia/cu13
g++ -std=c++17 -shared -fPIC -O3 -ffast-math -fno-finite-math-only \
  -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mfma \
  -DXIAOTU_MOE_MODULE_NAME=_xiaotu_moe_C_avx512_bf16 \
  -I"$PY_INC" -I"$PYBIND11_INC" -I"$ROOT/csrc" -I"$CUD/include" \
  -L"$CUD/lib" -Wl,--no-as-needed -lcudart -Wl,--as-needed \
  -Wl,-rpath,"$CUD/lib" -lcudart \
  "$ROOT/csrc/python_binding/binding.cpp" -o "$OUT$PY_EXT" 2>&1 | grep -vE "%lu|%zu" || true
echo "BUILD_DONE exit=${PIPESTATUS[0]}"
