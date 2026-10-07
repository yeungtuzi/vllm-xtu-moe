#!/usr/bin/env bash
# wait_ready.sh —— 【机械门】启动服务后必须同步确认就绪,才允许进入下一步
#
# 来由(2026-10-07): 我把"启动 + 评测"捆进一个后台作业 ⇒ vLLM 启动连失败 3 次
#   (ninja 不在 PATH → nvcc 是 CUDA 12.1,不认识 --compress-mode=size),
#   整条作业静默无所作为,我却"发脚本→等结果" ✗ —— 违反 AGENTS.md 明写纪律:
#   "启动服务后必须同步读启动日志确认,不许'发脚本→等结果'"
#
# ⭐ v2(2026-10-07 整改)。独立审计判定的 4 处硬伤及修法:
#   ① `curl` 没有 `-f` ⇒ HTTP 404/500 也算"就绪" ✗ ⇒ 改 `curl -sf`(只认 2xx)✓
#   ② **取不到端口属主【即放行】**(事故 F 未堵死)✗ ⇒ 取不到属主 ⇒ **判未就绪** ✓
#   ③ awk 正则 `$4 ~ ":814"` 未锚定 ⇒ `:8143` 会误命中 ✗ ⇒ 用 lib 的 `pi_port_listeners`
#      (按 `:` 切分后**整段相等**)✓
#   ④ `TIMEOUT` 非整数 ⇒ `[ -ge ]` 报错 ⇒ **死循环** ✗ ⇒ 入口先校验为正整数 ✓
#   另:不再用 `proc.sh status | grep "NOT RUNNING"` 子串判据(G13)⇒ 改读 PID 文件 + `kill -0` ✓
#
# 用法:
#   scripts/wait_ready.sh <实例名> <端口> [超时秒=180]
# 退出码:
#   0 = 就绪 · 1 = 进程在超时前死亡(会打印日志尾部) · 2 = 超时 · 3 = 参数错
#
# 判据(三条【全部】满足才叫就绪):
#   ① 该 pid 仍活着(PID 文件里的 PID)② 监听 <端口> 的属主 **就是**该 pid
#   ③ `/v1/models` 返回 2xx(且响应里含 API 的 JSON 特征)
set -uo pipefail
# ⭐ 参数校验:缺参也必须走"退出码 3"这条统一契约(不能用 `${1:?}` —— 那会 exit 1,契约就不唯一了)✗
if [ $# -lt 2 ]; then
  echo "用法: wait_ready.sh <实例名> <端口> [超时秒=180]" >&2; exit 3
fi
NAME="$1"
PORT="$2"
TIMEOUT="${3:-180}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
pf="$PROC_LOGDIR/$NAME.pid"
LOG="$PROC_LOGDIR/$NAME.log"

# ⭐ ④ 入口校验:端口/超时必须为正整数,否则会死循环或误判 ✓
case "$PORT" in ''|*[!0-9]*) echo "  [wait_ready] ⛔ 端口非法:'$PORT'" >&2; exit 3 ;; esac
case "$TIMEOUT" in ''|*[!0-9]*) echo "  [wait_ready] ⛔ 超时非法:'$TIMEOUT'(必须是正整数秒)" >&2; exit 3 ;; esac

echo "  [wait_ready] 等待 $NAME (:$PORT) 就绪, 超时 ${TIMEOUT}s"
t0=$(date +%s)
while :; do
  # ① 端口属主(PID 派生,绝不按名字/命令行匹配 ✓)
  pi_port_listeners "$PORT"
  OWNER="$(cat "$pf" 2>/dev/null || true)"
  owner_ok=0
  # ⭐ ② 取不到属主 ⇒ 判【未就绪】(绝不因为"读不到"就放行)✗
  if [ "$PI_SS_OK" = "1" ] && [ -n "$PI_PORT_PIDS" ] && pi_valid_pid "$OWNER"; then
    if printf '%s\n' "$PI_PORT_PIDS" | grep -qx "$OWNER"; then owner_ok=1; fi
  fi
  # ③ 只认 2xx,并要求响应像 OpenAI API 的 JSON
  http_ok=0
  if curl -sf --noproxy 127.0.0.1 --max-time 4 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"object"'; then
    http_ok=1
  fi
  if [ "$owner_ok" = "1" ] && [ "$http_ok" = "1" ]; then
    echo "  [wait_ready] ✅ 就绪(用时 $(( $(date +%s) - t0 ))s;端口 $PORT 属主 pid=$OWNER ✓)"
    exit 0
  fi
  # 进程死亡判定:只认 PID 文件 + kill -0(不是子串匹配 ✓)
  if pi_valid_pid "$OWNER" && ! pi_alive "$OWNER"; then
    echo "  [wait_ready] ❌ 进程已死亡(pid=$OWNER,来自 $pf)—— 启动失败! 日志尾部:" >&2
    tail -25 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /' >&2
    exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$TIMEOUT" ]; then
    echo "  [wait_ready] ⏰ 超时 ${TIMEOUT}s; 端口 $PORT 属主=${PI_PORT_PIDS:-读不到} PID文件=${OWNER:-无}" >&2
    echo "               日志尾部:" >&2
    tail -15 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/      /' >&2
    exit 2
  fi
  sleep 5
done
