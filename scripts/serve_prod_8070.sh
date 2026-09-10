#!/usr/bin/env bash
# 生产服务启动脚本(8070,1M 上下文,TP=2)—— 供实际使用/测试
#
# 为什么是 TP=2:1M 上下文的 KV 需要 1,048,576 × 29.5 KB ≈ 29.5 GiB(单张 40GB 卡
# 减去 ~19GB 非专家权重后放不下),TP=2 把 KV 按 rank 分片 ⇒ 每 rank 14.75 GiB。
#
# 与"最优单卡调参配置"的差异:
#   * TP=2 + `--enable-expert-parallel`:prefill 每 rank 只流一半专家(实测 TTFT 快 1.76×);
#     decode 目前每层多一次跨 rank all-reduce(实测每层 +4~6.5 ms),所以 decode 比单卡慢
#     —— 这是当前的主要优化方向(见 docs/PERFORMANCE_OPTIMIZATION.md §6/§7)。
#   * `XIAOTU_MOE_SINGLECOPY=0`:回到引擎默认的 NUMA 分片(单卡实测 +38%)。
#   * `EAGER=0`:CUDA graph(--max-num-seqs ≤128 才稳,实测 +5~7%)。
#   * `MAX_NBT=16384`:KV 上限放宽后允许更大的 prefill 分块(长上下文 TTFT 的关键)。
#
# 用法:  bash scripts/serve_prod_8070.sh          # 前台启动(日志同时写文件)
#        TAG=xxx PORT=8071 bash scripts/serve_prod_8070.sh   # 换端口/标签做灰度
#
# License: Apache-2.0
set -euo pipefail

TAG="${TAG:-dsv4_prod_8070}"
PORT="${PORT:-8070}"
GPUS="${GPUS:-0,1}"
TP="${TP:-2}"
SEQS="${SEQS:-64}"                 # CUDA graph 捕获尺寸上限受它约束(>128 会崩)
MAXLEN="${MAXLEN:-1048576}"        # 1M 上下文
MAX_NBT="${MAX_NBT:-16384}"
KV_MEM_BYTES="${KV_MEM_BYTES:-17179869184}"   # 16 GiB/rank ⇒ 1M×29.5KB/2 = 14.75 GiB 刚好装下
THREADS="${THREADS:-96}"           # 每 rank 96 线程(整机 192;96→192 无额外收益)
OMP="${OMP:-48}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec env \
  GPUS="$GPUS" TP="$TP" EP=1 MODE=dsv4 PORT="$PORT" TAG="$TAG" \
  MAXLEN="$MAXLEN" SEQS="$SEQS" MAX_NBT="$MAX_NBT" \
  KV_DTYPE=fp8_ds_mla GPU_UTIL=0.90 KV_MEM_BYTES="$KV_MEM_BYTES" \
  THREADS="$THREADS" OMP="$OMP" EAGER=0 PREFILL_MIN=384 \
  XIAOTU_MOE_SINGLECOPY=0 \
  scripts/tune_serve.sh
