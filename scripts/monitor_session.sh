#!/usr/bin/env bash
# 持续观察一个会话的退化指标(只读;不碰任何服务 ✓)
# 用法: bash scripts/monitor_session.sh <session-id> [间隔秒=120] [时长秒=14400]
set -u
SID="${1:?需要 session-id}"
IV="${2:-120}"
DUR="${3:-14400}"
R=/home/user/lvllm/vllm-xiaotu-moe
OUT="$R/dev-docs/report/tuning/logs/session_watch_${SID:8:12}.tsv"
[ -f "$OUT" ] || printf "时间\t轮\t调用\tmeta前\tmeta后\t损坏\t静默秒\t文件MB\n" > "$OUT"
END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  python3 "$R/scripts/_sess_sample.py" "$SID" "$OUT" || echo "[$(date +%H:%M:%S)] 采样失败" >> "$OUT"
  sleep "$IV"
done
echo "[$(date '+%F %T')] 观察结束(duration=${DUR}s)" >> "$OUT"
