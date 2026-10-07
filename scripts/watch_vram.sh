#!/usr/bin/env bash
# 持续采样空闲显存与并发(只读 ✓)。用法: watch_vram.sh [间隔秒=60] [时长秒=28800]
set -u
R=/home/user/lvllm/vllm-xiaotu-moe
IV="${1:-60}"; DUR="${2:-28800}"
OUT="$R/dev-docs/report/tuning/logs/vram_watch.tsv"
[ -f "$OUT" ] || printf "时间\t各卡free\t运行中\t排队\tKV使用\tRSS_GiB\n" > "$OUT"
END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  python3 "$R/scripts/_vram_sample.py" "$OUT" || true
  sleep "$IV"
done
