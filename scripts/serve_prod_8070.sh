#!/usr/bin/env bash
# 生产服务启动脚本 —— 两种模式,按"能不能交互"取舍
#
#   MODE=fast (日常交互推荐):单卡、256K 上下文、DSpark 投机解码
#       实测 C=1 单路 85 ms/token(11.8 tok/s/路),TTFT 0.63 s;
#       关投机时吞吐口径 C=128 可达 106 tok/s(见 dev-docs/PERFORMANCE_OPTIMIZATION.md §15)。
#   MODE=1m  (默认,需要 1M 上下文):TP=2、1M 上下文、DSpark 投机解码
#       实测 C=1 单路 266 ms/token(3.7 tok/s/路),TTFT 2.2 s;KV 容量 1,876,112 tokens。
#
# 为什么 1M 必须 TP=2:KV 是 29.5 KB/token ⇒ 1M = 29.5 GiB;单卡扣掉 ~19.6 GB 非专家权重
# 后放不下,而 `--kv-cache-dtype nvfp4_ds_mla`(FP4 KV)在 A100/SM80 上被主线拒绝
# (只支持 fp8_ds_mla 布局);512K 单卡也会 OOM(2 GiB 预取槽放不下)。
#
# 用法:
#   bash scripts/serve_prod_8070.sh                       # 1M 模式(TP=2,端口 8070)
#   MODE=fast PORT=8090 bash scripts/serve_prod_8070.sh    # 快速模式(单卡)
#   SPEC_OFF=1 bash scripts/serve_prod_8070.sh             # 关投机(高并发吞吐优先)
#
# License: Apache-2.0
set -euo pipefail

TAG="${TAG:-dsv4_prod_8070}"
PORT="${PORT:-8070}"
MODE="${MODE:-1m}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"

if [ "${SPEC_OFF:-0}" = "1" ]; then
  SPEC=""
else
  # DSpark(与 lk-moe 生产同款,block=5)。注意:CUDA graph 下草稿模型捕获会崩,
  # 所以这里强制 eager;换来单路延迟 ~+43%(见 dev-docs/PERFORMANCE_OPTIMIZATION.md §16)。
  SPEC="{\"method\":\"dspark\",\"model\":\"$CKPT\",\"num_speculative_tokens\":4}"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$MODE" = "fast" ]; then
  exec env \
    GPUS="${GPUS:-2}" TP=1 EP=0 MODE=dsv4 PORT="$PORT" TAG="$TAG" \
    MAXLEN=262144 SEQS=128 MAX_NBT=8192 \
    KV_DTYPE=fp8_ds_mla GPU_UTIL=0.90 KV_MEM_BYTES=12884901888 \
    THREADS=192 OMP=96 EAGER=1 PREFILL_MIN=384 \
    SPEC="$SPEC" SERVED=DeepSeek-V4-Flash-xiaotu \
    scripts/tune_serve.sh
else
  exec env \
    GPUS="${GPUS:-0,1}" TP=2 EP=0 MODE=dsv4 PORT="$PORT" TAG="$TAG" \
    MAXLEN=1048576 SEQS=64 MAX_NBT=8192 \
    KV_DTYPE=fp8_ds_mla GPU_UTIL=0.90 KV_MEM_BYTES=19327352832 \
    THREADS=96 OMP=48 EAGER=1 PREFILL_MIN=384 \
    ENV_EXTRA="XIAOTU_MOE_EP=0" \
    SPEC="$SPEC" SERVED=DeepSeek-V4-Flash-xiaotu \
    scripts/tune_serve.sh
fi
