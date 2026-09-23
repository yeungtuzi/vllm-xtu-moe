#!/usr/bin/env bash
# LMCache MP 服务:L1=CPU 内存 + L2=SSD 持久化(服务重启后 prefix cache 仍在)
# 依据:https://docs.lmcache.ai/zh_CN/recipes/deepseek_v41_flash.html
#   --chunk-size 256 / --separate-object-groups 是 V4.1 多 KV 几何的必需参数
set -euo pipefail
LMCACHE_BIN="${LMCACHE_BIN:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/lmcache}"
L1_GB="${L1_GB:-100}"                       # CPU 内存层
L2_DIR="${L2_DIR:-/home/user/.cache/lmcache_l2}"   # SSD 持久层目录
L2_GB="${L2_GB:-100}"                       # 磁盘容量上限
PORT="${PORT:-5555}"                        # connector 默认 tcp://localhost:5555
mkdir -p "$L2_DIR"
echo "[lmcache] L1=${L1_GB}GB L2=${L2_DIR}(${L2_GB}GB) port=${PORT}"
exec "$LMCACHE_BIN" server \
  --chunk-size 256 \
  --separate-object-groups \
  --l1-size-gb "$L1_GB" \
  --eviction-policy LRU \
  --l2-adapter "{\"type\":\"fs_native\",\"base_path\":\"$L2_DIR\",\"max_capacity_gb\":$L2_GB}" \
  --port "$PORT"
