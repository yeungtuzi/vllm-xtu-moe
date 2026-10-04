#!/usr/bin/env bash
# LMCache MP 服务:L1=CPU 内存 + L2=SSD 持久化(服务重启后 prefix cache 仍在)
# 依据:https://docs.lmcache.ai/zh_CN/recipes/deepseek_v41_flash.html
#   --chunk-size 256 / --separate-object-groups 是 V4.1 多 KV 几何的必需参数
set -euo pipefail
LMCACHE_BIN="${LMCACHE_BIN:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/lmcache}"
L1_GB="${L1_GB:-100}"                       # CPU 内存层
L2_DIR="${L2_DIR:-/home/user/.cache/lmcache_l2}"   # SSD 持久层目录
L2_GB="${L2_GB:-100}"
TRANSFER_MODE="${TRANSFER_MODE:-lmcache_driven}"   # lmcache_driven|engine_driven|auto
# chunk 必须 ≥ 各模型 vLLM block 的最小公倍数:V4.1 需 64 的倍数 ✓;GLM 需 2176 的倍数 ✓
# ⇒ 2176 = 64 × 34 同时满足两者(见 EXPERIMENTS B172)✓
# ⭐ CHUNK 规则(实测 2026-09-23,见 EXPERIMENTS B198):
#   * chunk 必须是**该模型 vLLM block(按 DCP 缩放后)**的倍数,否则 connector 在启动时直接报错:
#       "LMCache chunk size 256 must be a multiple of 2176 (the vLLM block size scaled by
#        decode_context_parallel_size)"
#   * 实测:V4.1-Flash 需 **64** 的倍数 ✓;GLM-5.3 需 **2176** ✓
#   * **2176 = 64 × 34** ⇒ 同时满足两者 ⇒ **一个服务端即可服务 V4.1 与 GLM** ✓(默认值即此 ✓)
#   * 换 chunk 会让既有 L2 对象的键失效(需重建,无害 ✓)
CHUNK_SIZE="${CHUNK_SIZE:-2176}"                       # 跨模型公共值(见 B172)
# 实验传输模块:connector 侧开 transfer_intermediate_tensors 时,服务端必须 `--enable transfer_query`
# 与之配对(否则报 "Connector enables transfer_query but server does not",见 EXPERIMENTS B175)
ENABLE_MODULES="${ENABLE_MODULES:-}"
PORT="${PORT:-5555}"                        # connector 默认 tcp://localhost:5555
# ⭐ 2026-10-04【NUMA 修正】:L1 是**一次性大缓冲**,默认策略是"首次触碰的本地节点" ⇒
#   实测(python 2026-10-04)100 GB L1 里 **92.2 GiB 全落在 node7**、10.5 GiB 在 node5
#   ⇒ 8 个 NUMA node 严重不均(node7 空闲只剩 13.1 GiB),而引擎 TP=1 会按 8 node 分片权重
#   ⇒ 起第二个实例时 node7 直接 OOM 风险。
#   修法:用 `numactl --interleave=all` 让 L1 均摊(100 GB ⇒ ~12.5 GB/node)。
#   可用 NUMACTL="" 关掉(例如单节点机器)。
NUMACTL="${NUMACTL-numactl --interleave=all}"
mkdir -p "$L2_DIR"
echo "[lmcache] modules=${ENABLE_MODULES:-none}"
echo "[lmcache] L1=${L1_GB}GB L2=${L2_DIR}(${L2_GB}GB) port=${PORT} transfer=${TRANSFER_MODE} chunk=${CHUNK_SIZE}"
echo "[lmcache] numactl='${NUMACTL}'(空=不限制;interleave 避免 L1 全压单一 NUMA node ✓)"
exec ${NUMACTL} "$LMCACHE_BIN" server \
  --chunk-size "$CHUNK_SIZE" \
  --separate-object-groups \
  --l1-size-gb "$L1_GB" \
  --eviction-policy LRU \
  --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"$L2_DIR\",\"max_capacity_gb\":$L2_GB}" \
  --supported-transfer-mode "$TRANSFER_MODE" \
  ${ENABLE_MODULES:+--enable $ENABLE_MODULES} \
  --port "$PORT"
