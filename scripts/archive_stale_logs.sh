#!/usr/bin/env bash
# 【规则·用户 2026-10-05 定】PID 对应的进程【已消失满 N 天】(默认 3)⇒ 打包归档其 .pid 与同批 .log
#
# 【判据·用户 2026-10-05 指出】"**活着的进程会不停更新它的 log**" ⇒
#   **日志 mtime 就是"它还活着"的直接证据** ✓ ⇒ 主判据 = 该实例最新日志的 mtime 距今是否 > DAYS 天 ✓
#   好处:不需要 liveness 探测,而且**天然躲开 PID 回收的坑** ✗(旧 PID 被无关进程复用会让 kill -0 误判"还活着")
#   兜底:`kill -0` 仍保留,仅用于"**活着但安静**(长时间无输出)"的实例 ⇒ **绝不归档活进程的 PID 文件** ✓
# 永不触碰:①活着进程的 PID 文件与日志 ②_launch_audit.log ③archive/ 本身
#
# 用法:bash scripts/archive_stale_logs.sh          # 执行(按 3 天)
#       DAYS=7 bash scripts/archive_stale_logs.sh  # 改阈值
#       DRY=1 bash scripts/archive_stale_logs.sh   # 只列不删
set -uo pipefail
DAYS="${DAYS:-3}"
DRY="${DRY:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$HERE/../dev-docs/report/tuning/logs"
cd "$L" || exit 0
[ -d archive ] || mkdir -p archive

# ---- 【R32】显式服务名单:即使 ppid=1 / 无日志更新,这些**服务**也绝不归档 ----
SERVICES_KEEP=" prometheus grafana node_exporter coremap coremap_png xtu_exporter "
# ---- 活着的一律保留(含其日志)----
KEEP=" _launch_audit.log "
for f in *.pid; do
  [ -f "$f" ] || continue
  p="$(tr -dc '0-9' <"$f" 2>/dev/null)"
  [ -n "$p" ] || continue
  if kill -0 "$p" 2>/dev/null; then
    n="${f%.pid}"; KEEP="$KEEP$f $n.log $n.mem.pid $n.mem.log "
  fi
done
for f in v41_8070.*.log; do            # 活实例的服务日志
  [ -f "$f" ] || continue
  p="$(echo "$f" | grep -oE '[0-9]{6,}' | head -1)"
  kill -0 "$p" 2>/dev/null && KEEP="$KEEP$f "
done

# ---- 候选:陈旧(按"日志 mtime 或 pid mtime 距今 > DAYS 天")且不活着 ----
: >/tmp/_asl_list.txt
for f in *.pid; do
  [ -f "$f" ] || continue
  case "$SERVICES_KEEP" in *" $n "*) continue;; esac     # 【R32】服务名显式豁免 ✓
  case "$KEEP" in *" $f "*) continue;; esac
  n="${f%.pid}"
  # 主判据:日志 mtime(活着的进程会持续写日志 ✓);无日志时才退回 pid 文件的 mtime
  newest=0
  for g in "$n.log" "$n.mem.log"; do
    [ -f "$g" ] || continue
    m=$(stat -c %Y "$g" 2>/dev/null || echo 0); [ "$m" -gt "$newest" ] && newest=$m
  done
  if [ "$newest" -eq 0 ] && [ -f "$f" ]; then
    newest=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  fi
  age=$(( ( $(date +%s) - newest ) / 86400 ))
  [ "$age" -ge "$DAYS" ] || continue
  p2="$(tr -dc '0-9' <"$f" 2>/dev/null)"
  if [ -n "$p2" ] && kill -0 "$p2" 2>/dev/null; then
    continue          # 活着但安静 ⇒ 保留(绝不归档活进程的 PID 文件)✓
  fi
  echo "$f" >>/tmp/_asl_list.txt
  for g in "$n.log" "$n.mem.pid" "$n.mem.log" "$n.pid.bak"; do
    [ -f "$g" ] && echo "$g" >>/tmp/_asl_list.txt
  done
done
N=$(wc -l </tmp/_asl_list.txt)
echo "[archive_stale_logs] 阈值 ${DAYS} 天 ⇒ 候选 ${N} 个文件"
if [ "$N" -eq 0 ]; then echo "[archive_stale_logs] 无陈旧件,跳过"; exit 0; fi
if [ "$DRY" != "0" ]; then echo "[archive_stale_logs] DRY=1 ⇒ 仅列出:"; sed 's/^/  /' /tmp/_asl_list.txt; exit 0; fi

TS=$(date +%Y%m%d-%H%M%S); DEST="archive/logs_stale_${TS}.tar.gz"
{ echo "# archived $TS  by archive_stale_logs.sh (DAYS=$DAYS)";
  echo "# kept(live): $KEEP"; echo "# --- archived ---"; cat /tmp/_asl_list.txt; } >/tmp/_asl_manifest.txt
tar czf "$DEST" -T /tmp/_asl_list.txt -C /tmp _asl_manifest.txt 2>/dev/null
IN=$(tar tzf "$DEST" 2>/dev/null | wc -l)
if [ "$IN" -ne "$((N+1))" ] || [ ! -s "$DEST" ]; then
  echo "[archive_stale_logs] ✗ 包内 ${IN} / 期望 $((N+1)) ⇒ 不删除任何原件"; rm -f "$DEST"; exit 1
fi
while IFS= read -r f; do [ -f "$f" ] && rm -f "$f"; done </tmp/_asl_list.txt
echo "[archive_stale_logs] ✅ 归档 ${N} 个 → $DEST(校验 $((N+1)) 通过,原件已删)"
