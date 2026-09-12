#!/usr/bin/env bash
# 带重试的服务启动 —— 专治"启动阶段静默死掉白烧 13 分钟"。
#
# 背景(第 119-120 轮实测):TP=2 启动有约 1/3 概率失败,日志**没有任何 worker traceback**,
# 只在 EngineCore 侧留一行 `RuntimeError: cancelled`(这是 shm 读者被取消的**次生症状**,
# 不是超时,详见 NOTES §171)。旧做法:失败后人工发现 → 再花 13 分钟重启。
# 本脚本自动重试,并把每次尝试的 tag 递增,失败的日志保留以便事后比对。
#
# 用法:同 tune_serve.sh,额外支持 ATTEMPTS=<次数,默认 3>
#   MODE=dsv4 TAG=exp1 PORT=8070 TP=2 ... scripts/serve_retry.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATTEMPTS="${ATTEMPTS:-3}"
BASE_TAG="${TAG:-serve}"

for i in $(seq 1 "$ATTEMPTS"); do
  echo "[serve_retry] ===== 尝试 $i/$ATTEMPTS (tag=${BASE_TAG}_a$i) ====="
  # 每次尝试前都清干净:被杀掉的进程 + /dev/shm 残留(EP 屏障文件会让新的 rank 永久互等)
  WAIT=25 "$ROOT/scripts/kill_serve.sh" >/dev/null 2>&1
  if TAG="${BASE_TAG}_a$i" "$ROOT/scripts/tune_serve.sh"; then
    echo "[serve_retry] 第 $i 次成功,tag=${BASE_TAG}_a$i"
    # 把成功的那次 tag 记下来,方便调用方取日志
    echo "${BASE_TAG}_a$i" > "/tmp/serve_retry.last_tag"
    exit 0
  fi
  echo "[serve_retry] 第 $i 次失败,继续重试"
done
echo "[serve_retry] $ATTEMPTS 次全部失败"
exit 1
