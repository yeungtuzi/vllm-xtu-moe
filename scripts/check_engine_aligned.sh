#!/usr/bin/env bash
# 第一条(把 CPU 解码引擎效率对齐 lk_moe)的**回归门禁** —— 一条命令跑完两项检查:
#
#   1) 数值门禁:`test_block23_equiv.py`(受控 me=2/3 路由 + numpy golden 逐元素对拍)。
#      R55 规定:**任何改动 `numa_pool.hpp` 同步结构或 `moe_v2_packed4.hpp` 内层循环的提交,
#      都必须先过这一项** —— 轮 68/70 的两次事故都是 bench 三次全过而对拍立刻挂/算错。
#   2) 性能门禁:`bench_engine_ab.py`(BS=6/K=6/THREADS=120/真实层权重/DEDUP=12),
#      验收 **≤0.70 ms/层**(NOTES §119/§73)。
#
# 用法:
#   scripts/check_engine_aligned.sh
#   NPZ=<真实模型目录> NPZ_EQ=<real_layer1_model.npz> THRESH=0.70 REP=60 scripts/check_engine_aligned.sh
#
# 当前基线(2026-xx,BS=6/DEDUP=12/THREADS=120):
#   xiaotu 0.66-0.68 ms/层(229 GB/s、1.9 GB/s·线程) vs lk_moe 0.57 ms/层(265 GB/s)
#   历史:1.22 ms/层(124 GB/s)—— 轮 67-73 的同步/布局重写后 1.8×。
#   服务端真实形状(na≈32):2.92 GB/s·线程,已超 lk 的 2.2(NOTES §124)。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
# 两项检查需要**不同的 fixture**:对拍要小 npz,bench 要真实模型目录。
NPZ="${NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
NPZ_EQ="${NPZ_EQ:-/home/user/lvllm/xiaotu-moe/scripts/real_layer1_model.npz}"
THRESH="${THRESH:-0.70}"
REP="${REP:-60}"
FAIL=0

echo "== [1/2] 数值门禁 test_block23_equiv.py =="
XIAOTU_LAYER1_NPZ="$NPZ_EQ" timeout 300 "$PY" "$ROOT/scripts/test_block23_equiv.py" > /tmp/eq_aligned.txt 2>&1
if [ ! -s /tmp/eq_aligned.txt ]; then echo "   !! 对拍无输出(检查 NPZ_EQ=$NPZ_EQ)"; tail -3 /tmp/eq_aligned.txt; fi
NOK=$(grep -acE '^\[OK' /tmp/eq_aligned.txt || true)
NBAD=$(grep -acE '^\[BAD' /tmp/eq_aligned.txt || true)
echo "   OK=$NOK BAD=$NBAD (me=1 的 BAD 是既有的 NR=8 fp32 重结合偏差,不计入)"
# me=2/3/混合/4..7 必须全 OK(共 7 项)
if [ "${NOK:-0}" -lt 7 ]; then echo "   !! 数值门禁未通过(需 7 OK)"; FAIL=1; else echo "   数值门禁通过"; fi

echo "== [2/2] 性能门禁 bench_engine_ab.py (DEDUP=12, 阈值 ${THRESH} ms/层) =="
MS=$(ENG=xiaotu XIAOTU_MOE_THREADS=120 XIAOTU_LAYER1_NPZ="$NPZ" BS=6 DEDUP=12 REP="$REP" \
     timeout 600 "$PY" "$ROOT/scripts/bench_engine_ab.py" 2>&1 \
     | grep -E '^ +6 ' | tail -1 | awk '{print $2}')
echo "   实测 ${MS:-?} ms/层"
if [ -z "${MS:-}" ]; then echo "   !! 未取到结果"; FAIL=1
elif awk "BEGIN{exit !($MS <= $THRESH)}"; then echo "   性能门禁通过(<= $THRESH)"
else echo "   !! 性能门禁未通过(> $THRESH)"; FAIL=1; fi

[ "$FAIL" = 0 ] && echo "== 全部门禁通过 ==" || echo "== 有门禁未通过 =="
exit "$FAIL"
