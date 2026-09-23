#!/usr/bin/env bash
# 进程管理助手 —— 本项目最高优先级纪律的强制执行工具。
#   启动:proc.sh spawn <name> <cmd...>   ⇒ 写 <logdir>/<name>.pid 与 <name>.log(日志持久化)
#   停止:proc.sh stop  <name>            ⇒ 只读 PID 文件,绝不匹配命令行 ✗
#   其它:proc.sh status|pid|adopt <name> [pid]
# 背景:按名字/命令行匹配杀进程会杀掉自己(命令行含同一模式),本会话已发生 40+ 次。
set -uo pipefail
LOGDIR="${LOGDIR:-/home/user/lvllm/vllm-xiaotu-moe/dev-docs/report/tuning/logs}"
mkdir -p "$LOGDIR"
cmd="${1:-}"; name="${2:-}"; shift 2 2>/dev/null || true
pf="$LOGDIR/${name}.pid"; lf="$LOGDIR/${name}.log"
case "$cmd" in
  spawn)
    [ -n "$name" ] && [ $# -gt 0 ] || { echo "usage: proc.sh spawn <name> <cmd...>"; exit 2; }
    # 由子进程自己写 PID(pid 精确、且 setsid 后自成进程组 ⇒ 可用 -PID 整组杀)
    setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pf" "$@" > "$lf" 2>&1 < /dev/null &
    sleep 1
    echo "spawned name=$name pid=$(cat "$pf" 2>/dev/null) log=$lf"
    ;;
  adopt)
    [ -n "${1:-}" ] || { echo "usage: proc.sh adopt <name> <pid>"; exit 2; }
    echo "$1" > "$pf"; echo "adopted name=$name pid=$1 log=$lf"
    ;;
  stop)
    [ -f "$pf" ] || { echo "REFUSE: 无 PID 文件 $pf ⇒ 请用端口派生 PID(ss -ltnp | grep :PORT)后 adopt,禁止按名字匹配"; exit 3; }
    pid="$(cat "$pf")"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null; sleep 3
      kill -9 -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
      echo "stopped name=$name pid=$pid(仅按 PID 文件)"
    else
      echo "not running: name=$name pid=$pid"
    fi
    ;;
  status)
    if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo "$name: RUNNING pid=$(cat "$pf") log=$lf"; else echo "$name: NOT RUNNING (pid file: $([ -f "$pf" ] && cat "$pf" || echo none))"; fi
    ;;
  pid) [ -f "$pf" ] && cat "$pf" || { echo "no pid file"; exit 3; } ;;
  *) echo "usage: proc.sh {spawn|stop|status|pid|adopt} <name> [args...]"; exit 2 ;;
esac
