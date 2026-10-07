#!/usr/bin/env bash
# G9 —— `scripts/wait_ready.sh` 的【行为测试】(2026-10-07)
#
# 为什么需要它(审计判定 wait_ready.sh 的 4 处硬伤):
#   ① `curl` 没有 `-f` ⇒ 404/500 也判"就绪" ✗
#   ② **取不到端口属主【即放行】**(事故 F 未堵死)✗ ⇒ 必须 fail-closed
#   ③ awk 正则未锚定 ⇒ `:814` 会匹配 `:8143` ✗
#   ④ `TIMEOUT` 非整数 ⇒ `[ -ge ]` 报错 ⇒ **死循环** ✗
#
# ⭐ 本测试用【假 endpoint + 临时 LOGDIR】,不碰生产、不碰 8070/8071 ✓
#   假服务通过 `proc.sh spawn` 起(顺带复用它的 PID 文件与存活保证)✓
#
# 用法: bash scripts/tests/test_wait_ready.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 用法/内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REAL_LOGDIR="$ROOT/dev-docs/report/tuning/logs"
# shellcheck source=../lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"
WAIT="$ROOT/scripts/wait_ready.sh"

TMP="$(mktemp -d)" || exit 2
export LOGDIR="$TMP"
case "$(readlink -f "$TMP")/" in
  "$(readlink -f "$REAL_LOGDIR")"/*) echo "⛔ 拒绝在真 LOGDIR 上运行" >&2; exit 2 ;;
esac

PASS=0; FAIL=0; SKIP=0
chk() {   # chk <描述> <期望 rc> <实得 rc>
  if [ "$2" = "$3" ]; then echo "  ✅ $1  (期望=$2 实得=$3)"; PASS=$((PASS+1))
  else echo "  ❌ $1  (期望=$2 实得=$3)"; FAIL=$((FAIL+1)); fi
}
skip() { echo "  ⏭  SKIP: $*"; SKIP=$((SKIP+1)); }
# ⭐ 2026-10-07 独立审计:核心用例"跳过"不得等于通过(除非显式 --allow-skip=<理由>)✗
ALLOW_SKIP_REASON=""
for _a in "$@"; do case "$_a" in --allow-skip=*) ALLOW_SKIP_REASON="${_a#--allow-skip=}" ;; esac; done
skipf() {
  echo "  ❌ 核心用例未执行:$*"
  if [ -n "$ALLOW_SKIP_REASON" ]; then
    echo "     (显式 --allow-skip=$ALLOW_SKIP_REASON ⇒ 降级为 SKIP)"; SKIP=$((SKIP+1))
  else FAIL=$((FAIL+1)); fi
}

# 假 endpoint:GET 任意路径都回指定状态码与 body(只为测 curl -f 与就绪判据)✓
cat > "$TMP/fakeapi.py" <<'PY'
import sys, http.server, socketserver
port = int(sys.argv[1]); code = int(sys.argv[2]); body = sys.argv[3].encode()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(code); self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('127.0.0.1', port), H) as s:
    s.serve_forever()
PY

_free_port() { python3 - <<'PY'
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()
PY
}

_cleanup() {
  for n in srvA srvB; do
    [ -f "$TMP/$n.pid" ] || continue
    # ⭐ G5 要求 stop 调用点必须"留下"退出码:这里用命令替换捕获(收尾清理,rc 只作记录)
    _o="$(bash "$ROOT/scripts/proc.sh" stop "$n" --force --reason=g9-cleanup 2>&1)"; : "$_o"
  done
  rm -rf "$TMP"
}
trap _cleanup EXIT

echo "G9 wait_ready.sh 行为测试(临时 LOGDIR=$TMP)"
echo

# ── 起两个假服务:A=200 + JSON; B=404 ─────────────────────────────────────────
PA="$(_free_port)"; PB="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvA python3 "$TMP/fakeapi.py" "$PA" 200 '{"object":"list"}' >/dev/null 2>&1
bash "$ROOT/scripts/proc.sh" spawn srvB python3 "$TMP/fakeapi.py" "$PB" 404 '{"error":"nope"}' >/dev/null 2>&1
sleep 1
PA_PID="$(cat "$TMP/srvA.pid" 2>/dev/null || true)"
PB_PID="$(cat "$TMP/srvB.pid" 2>/dev/null || true)"
if ! pi_valid_pid "$PA_PID" || ! pi_alive "$PA_PID"; then
  skip "假服务 A 没起来(port=$PA)⇒ 无法测试"; echo; echo "G9 结果:通过 $PASS / 失败 $FAIL / 跳过 $SKIP"; exit 1
fi

echo "  假服务 A: pid=$PA_PID port=$PA (200+JSON) · 假服务 B: pid=${PB_PID:-?} port=$PB (404)"
echo

# ── 1) 就绪:2xx + JSON + 端口属主==PID文件里的 pid ⇒ 0 ───────────────────────
note() { echo "  ── $*"; }
note "1) 就绪路径"
bash "$WAIT" srvA "$PA" 8 >/dev/null 2>&1; chk "2xx + 属主匹配 ⇒ 0" 0 "$?"

# ── 2) 404 不得判就绪(审计 fix #1)───────────────────────────────────────────
note "2) 非 2xx 不得判就绪"
bash "$WAIT" srvB "$PB" 4 >/dev/null 2>&1; chk "404 ⇒ 超时(2),不得放行" 2 "$?"

# ── 3) 端口属主不是本实例 ⇒ 必须拒绝(审计 fix #2,事故 F)───────────────────
note "3) 端口属主不匹配 ⇒ fail-closed"
# 3a) ⚠️ 必须是【另一个】活进程登记成 nC:端口 PA 的属主是 srvA,而 nC.pid 指向 srvB
#     (若把 nC.pid 指向 srvA 的同一个 pid,那就是"属主匹配",本来就该判就绪 —— 别写错用例 ✗)
if pi_valid_pid "$PB_PID" && pi_alive "$PB_PID"; then
  echo "$PB_PID" > "$TMP/nC.pid"
  bash "$WAIT" nC "$PA" 4 >/dev/null 2>&1; chk "PID 文件 pid(srvB) ≠ 端口属主(srvA)⇒ 超时(2)" 2 "$?"
  rm -f "$TMP/nC.pid"
else
  skipf "假服务 B 不在,无法构造属主不匹配用例"
fi
# 3b) 完全没有 PID 文件时,即使 2xx 也不得放行
bash "$WAIT" nosuchinst "$PA" 4 >/dev/null 2>&1; chk "无 PID 文件(取不到属主)⇒ 超时(2)" 2 "$?"

# ── 4) 端口子串误匹配(审计 fix #3)──────────────────────────────────────────
note "4) 端口必须整段比对(:814 ≠ :8143)"
PREFIX="${PA:0:${#PA}-1}"     # 例如 34567 ⇒ 3456
[ -n "$PREFIX" ] && [ "$PREFIX" != "$PA" ] || skip "端口太短,无法构造前缀用例"
if [ -n "$PREFIX" ] && [ "$PREFIX" != "$PA" ]; then
  bash "$WAIT" srvA "$PREFIX" 4 >/dev/null 2>&1; chk "查询 ${PREFIX}(服务其实在 ${PA})⇒ 超时(2)" 2 "$?"
fi

# ── 5) 进程已死 ⇒ 1(且不该等满超时)────────────────────────────────────────
note "5) 进程死亡判定"
if kill -0 999999 2>/dev/null; then skip "999999 竟然活着"; else
  echo 999999 > "$TMP/dead.pid"
  t0=$(date +%s); bash "$WAIT" dead 1 6 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "PID 文件里的进程已死 ⇒ 1" 1 "$rc"
  if [ "$dt" -le 5 ]; then chk "死亡判定是立即的(未等满超时,$dt s)" 0 0; else chk "死亡判定是立即的(未等满超时,$dt s)" 0 1; fi
  rm -f "$TMP/dead.pid"
fi

# ── 6) 参数校验:非整数超时/端口不得致死循环(审计 fix #4)────────────────────
note "6) 参数校验(防死循环)"
bash "$WAIT" srvA "$PA" abc >/dev/null 2>&1; chk "TIMEOUT 非整数 ⇒ 3(不死循环)" 3 "$?"
bash "$WAIT" srvA abc 5     >/dev/null 2>&1; chk "PORT 非整数 ⇒ 3" 3 "$?"
bash "$WAIT" srvA           >/dev/null 2>&1; chk "缺参数 ⇒ 3(统一契约)" 3 "$?"
bash "$WAIT"                >/dev/null 2>&1; chk "无参数 ⇒ 3" 3 "$?"

# ── 7) 服务中途死掉 ⇒ 1(而不是一直等到超时)───────────────────────────────
note "7) 启动中死亡 ⇒ 立即 1"
PF="$(_free_port)"
bash "$ROOT/scripts/proc.sh" spawn srvD python3 "$TMP/fakeapi.py" "$PF" 200 '{"object":"list"}' >/dev/null 2>&1
sleep 1
DPID="$(cat "$TMP/srvD.pid" 2>/dev/null || true)"
if pi_valid_pid "$DPID"; then
  # 直接杀掉它 ⇒ wait_ready 应在下一轮判"已死亡"
  kill -KILL "$DPID" 2>/dev/null || true
  sleep 0.5
  t0=$(date +%s); bash "$WAIT" srvD "$PF" 20 >/dev/null 2>&1; rc=$?; dt=$(( $(date +%s) - t0 ))
  chk "实例死亡 ⇒ 1" 1 "$rc"
  if [ "$dt" -le 12 ]; then chk "死亡后快速返回(未等满 20s,$dt s)" 0 0; else chk "死亡后快速返回(未等满 20s,$dt s)" 0 1; fi
else
  skipf "srvD 没起来(死亡判定未被验证)"
fi

echo
echo "G9 结果:通过 $PASS / 失败 $FAIL / 跳过 $SKIP"
[ "$FAIL" -eq 0 ] && echo "G9 ✅ 全绿" || echo "G9 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
