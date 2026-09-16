#!/usr/bin/env bash
# **同一把尺子**量我们自己的引擎:在 `lvllm` 环境里用我们的插件跑 DeepSeek-V4.1-Flash,
# 参数与 `ref_v41_mem.sh`(参考实现 lk_moe)**逐字相同** ⇒ 内存数字可直接对比。
#
# 为什么这样设计(§503):参考实现自己文档写"peak ≈ 590 GB",但那和我们的
# "peak Rss 851 GB" **不是同一把尺子**(进程 vs 进程树、含不含页缓存、不同 env)。
# 所以两边都用 `report/tuning/probes/mem_footprint.sh`(服务进程树总 RSS),同一模型、
# 同一 vLLM 基座(lvllm 2.5)、同一套参数,唯一变量 = CPU MoE 引擎。
#
# 插件如何进 lvllm 环境:受 `XTU_PLUGIN=1` 门控的 site-packages `.pth`
# (由 scripts/ab_lvllm_vs_xiaotu.sh 的 install_pth 装好)。**必须在仓库外启动**(§499)。
#
# 用法:TAG=xtu41 PORT=8201 bash report/tuning/probes/xtu_v41_mem.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV="${ENV:-/home/user/anaconda3/envs/lvllm}"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8201}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.95}"
MAXLEN="${MAXLEN:-65536}"
MBT="${MBT:-8192}"
SEQS="${SEQS:-2}"
THREADS="${THREADS:-60}"
TAG="${TAG:-xtu41}"

# 插件装载器(幂等写成同样一行;门控在 XTU_PLUGIN)
PTH="$ENV/lib/python3.12/site-packages/zz_xiaotu_plugin.pth"
printf 'import os,sys; (os.environ.get("XTU_PLUGIN")=="1") and (sys.path.insert(0,r"%s"), __import__("vllm_xiaotu_moe"))\n' "$ROOT" > "$PTH"
# 我们的 env 桥:显式指定 ⇒ 覆盖语义,且不污染别的脚本
XV="$ROOT/report/tuning/logs/$TAG.envfile"
printf 'XIAOTU_MOE_THREADS=%s\nXIAOTU_MOE_NSLICE_SMALL=0\nXIAOTU_MOE_ASYNC=0\nXIAOTU_MOE_SPIN_IDLE_US=0\nXIAOTU_MOE_GPU_RESIDENT_LAYERS=\n' "$THREADS" > "$XV"

export TAG PORT
RUN_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS \
XTU_PLUGIN=1 XIAOTU_ENV_FILE=$XV VLLM_EXPERTS_LOAD_DEVICE=cpu \
XIAOTU_MOE_THREADS=$THREADS XIAOTU_MOE_NSLICE_SMALL=0 XIAOTU_MOE_ASYNC=0 \
XIAOTU_MOE_SPIN_IDLE_US=0 XIAOTU_MOE_GPU_RESIDENT_LAYERS= \
LVLLM_MOE_NUMA_ENABLED=0 OMP_NUM_THREADS=1 \
LVLLM_ENABLE_NUMA_INTERLEAVE=1 LVLLM_EMBEDDING_NUMA_ENABLED=1 \
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGRAM_DROP_PAGE_CACHE=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0"

RUN_ENV="$RUN_ENV" READY_TIMEOUT="${READY_TIMEOUT:-1800}" \
  bash "$ROOT/report/tuning/probes/mem_footprint.sh" \
  "$ENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name dsv41-xtu --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$MBT" --max-num-seqs "$SEQS" \
  --dtype bfloat16 --kv-cache-dtype fp8_ds_mla \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}' \
  --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
