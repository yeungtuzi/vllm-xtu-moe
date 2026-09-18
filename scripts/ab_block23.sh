#!/usr/bin/env bash
# 同窗口交错 A/B:2/3 行分块内核(新)vs 4 行分块+单行兜底(旧)。
#
# 动机:本机是共享主机,负载在 30~130 之间波动,先后测的两次结果不可比
# (同一配置曾出现 2.4 / 10.1 / 7.4 ms/层)。必须**在同一时间窗口内交替**跑两个
# 二进制,再做配对比较。
#
# 前置:/tmp/old_vnni.so 与 /tmp/new_vnni.so(由 scripts/build_engine_variants.sh
#       分别用"回退后的源码"和"带 block_23 的源码"构建);构建目录里不能有
#       _avx512_bf16(loader 会优先选它,改变 ISA 会污染对比)。
#
# 用法:ROUNDS=4 scripts/ab_block23.sh
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD="$ROOT/xiaotu_moe/build"
SO="$BUILD/_xiaotu_moe_C_avx512_vnni.cpython-312-x86_64-linux-gnu.so"
ROUNDS="${ROUNDS:-4}"
OUT="${OUT:-$ROOT/dev-docs/report/tuning/ab_block23.txt}"
ENVV="XIAOTU_LAYER1_NPZ=${NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master} \
OMP_NUM_THREADS=1 BS=${BS:-6} REP=${REP:-79} DEDUP=${DEDUP:-12} XIAOTU_MOE_THREADS=${THREADS:-192} XIAOTU_MOE_PROFILE=1"

[ -f /tmp/old_vnni.so ] && [ -f /tmp/new_vnni.so ] || { echo "missing /tmp/{old,new}_vnni.so"; exit 2; }
[ -f "$BUILD/_xiaotu_moe_C_avx512_bf16.cpython-312-x86_64-linux-gnu.so" ] && { echo "!! 移除 build/ 里的 avx512_bf16 变体"; exit 2; }

: > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  for which in old new; do
    cp "/tmp/${which}_vnni.so" "$SO"
    load="$(awk '{print $1}' /proc/loadavg)"
    line="$(timeout 900 env $ENVV /home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python scripts/bench_cpu_engine.py 2>&1 \
            | grep -E "MOE-PROF\] calls=80" | sed 's/.*na=/na=/')"
    printf 'round=%s kernel=%-3s load=%s %s\n' "$r" "$which" "$load" "$line" | tee -a "$OUT"
  done
done
# 恢复新内核
cp /tmp/new_vnni.so "$SO"
python3 - "$OUT" <<'PY'
import re, sys
rows = {"old": [], "new": []}
for ln in open(sys.argv[1]):
    m = re.search(r"kernel=(\w+).*?A=([\d.]+)ms.*?B=([\d.]+)ms", ln)
    if m:
        rows[m.group(1)].append(float(m.group(2)) + float(m.group(3)))
for k, v in rows.items():
    if v:
        print(f"{k}: A+B per 40 calls = {[round(x,1) for x in v]}  中位数 {sorted(v)[len(v)//2]:.1f} ms")
if rows["old"] and rows["new"]:
    o = sorted(rows["old"])[len(rows["old"])//2]
    n = sorted(rows["new"])[len(rows["new"])//2]
    print(f"==> 新/旧 = {n/o:.3f}  ({(o/n):.2f}x 加速)")
PY
