#!/usr/bin/env bash
# 【§630】多编译器 / 多 -mtune 的引擎级对照。
#
# 协议(与 dev-docs/report/tuning/AB_LVLLM_VS_XIAOTU.md 的"引擎级"一节一致):
#   * 同一份真实 V4.1-Flash 第 3 层权重、同一个 python env(lvllm)
#   * 单引擎(等价服务内单 rank),THREADS=60
#   * 真实形状 DEDUP=226(§619 服务实测 na=226.2)
#   * 每个形状都跑同一份 lk 参照(同 session 分母)
#
# 用法: bash scripts/ab_compilers.sh gcc11 gcc14 gcc16 clang23
set -uo pipefail
REPO=/home/user/lvllm/vllm-xiaotu-moe
PY=/home/user/anaconda3/envs/lvllm/bin/python
EXT=".cpython-312-x86_64-linux-gnu.so"
BUILD="$REPO/xiaotu_moe/build"
OUT="${OUT:-/tmp/ab_compilers_results.txt}"
MODEL="${XIAOTU_LAYER1_NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
THREADS="${THREADS:-60}"
WARMUP="${WARMUP:-100}"
REP="${REP:-30}"
BS="${BS:-227,1893}"
LBS="${LBS:-1}"

cd "$REPO"
: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }

# ---------- lk 参照(与编译器无关,但同 session 量一次) ----------
say "################ lk_moe 参照 (THREADS=$THREADS WARMUP=$WARMUP REP=$REP) ################"
ENG=lk LK_THREADS="$THREADS" CUDA_VISIBLE_DEVICES=0 \
  XIAOTU_LAYER1_NPZ="$MODEL" BS="$BS" DEDUP=226 WARMUP="$WARMUP" REP="$REP" \
  "$PY" scripts/bench_engine_ab.py 2>&1 | tee -a "$OUT"
say "---- lk 解码(BS=$LBS DEDUP=6) ----"
ENG=lk LK_THREADS="$THREADS" CUDA_VISIBLE_DEVICES=0 \
  XIAOTU_LAYER1_NPZ="$MODEL" BS="$LBS" DEDUP=6 WARMUP="$WARMUP" REP="$REP" \
  "$PY" scripts/bench_engine_ab.py 2>&1 | tee -a "$OUT"

# ---------- 每个快照 ----------
for tag in "$@"; do
  src="/tmp/so_$tag"
  if [[ ! -d "$src" ]]; then say "!! $tag: $src 不存在,跳过"; continue; fi
  say ""
  say "################ arm=$tag ################"
  rm -f "$BUILD"/*"$EXT"
  cp -a "$src"/*"$EXT" "$BUILD/"
  say "# .so md5:"; (cd "$BUILD" && md5sum *"$EXT") | sed 's/^/#   /' | tee -a "$OUT"
  ENG=xiaotu XIAOTU_MOE_THREADS="$THREADS" \
    XIAOTU_LAYER1_NPZ="$MODEL" BS="$BS" DEDUP=226 WARMUP="$WARMUP" REP="$REP" \
    "$PY" scripts/bench_engine_ab.py 2>&1 | tee -a "$OUT"
  say "---- $tag 解码(BS=$LBS DEDUP=6) ----"
  ENG=xiaotu XIAOTU_MOE_THREADS="$THREADS" \
    XIAOTU_LAYER1_NPZ="$MODEL" BS="$LBS" DEDUP=6 WARMUP="$WARMUP" REP="$REP" \
    "$PY" scripts/bench_engine_ab.py 2>&1 | tee -a "$OUT"
done
say ""
say "################ DONE -> $OUT ################"
