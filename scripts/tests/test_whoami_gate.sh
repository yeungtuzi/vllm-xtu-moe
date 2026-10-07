#!/usr/bin/env bash
# G3 —— `scripts/whoami_proc.sh`(归属门)的【行为矩阵】测试(2026-10-07)
#
# 为什么需要它(审计判定 v1/v2/v3 三版全不通过):
#   v3 的**判决性证据**:对真实生产跑门,`pid=114597 CV=1,2 端口=8070`
#     而 `PID文件= v41_8070.pid vllm_prod_8070.pid`(明明有值),**理由却打的是"端口"**
#     ⇒ 证明 `PID 文件分支从未触发`(because `PF_ALL` 带前导空格 vs `^…$` 锚定正则 ✗)
#   ⇒ 所以本测试的**核心断言**是:对生产,门的【理由】必须是 `PID文件…`,不是 `端口…` ✓
#
# 另一类必须堵死的 fail-open(审计 fix #2):
#   `LOGDIR` 不可读 / `ss` 失败 ⇒ v3 **静默放行** ✗ ⇒ 本测试断言此时必须 rc≠0 ✓
#
# ⭐ 本测试**全程只读**(门本身不杀任何进程),不会影响生产 ✓
# ⭐ 期望值**从产物生成**(生产 pid 从 PID 文件里读,不写死),避免"照错误记忆写用例" ✗
#
# 用法: bash scripts/tests/test_whoami_gate.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 用法/内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT/scripts/whoami_proc.sh"
REAL_LOGDIR="$ROOT/dev-docs/report/tuning/logs"
# shellcheck source=../lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"

# ⭐ 2026-10-07 独立审计:旁路不得来自环境变量 ⇒ 只认显式 --allow-skip=<理由>
ALLOW_SKIP_REASON=""
for _a in "$@"; do case "$_a" in --allow-skip=*) ALLOW_SKIP_REASON="${_a#--allow-skip=}" ;; esac; done
PASS=0; FAIL=0; SKIP=0
chk() {   # chk <描述> <期望 rc> <实得 rc> [<必须在输出里出现的子串>]
  local desc="$1" want="$2" got="$3" needle="${4:-}"
  if [ "$want" != "$got" ]; then
    echo "  ❌ $desc  (期望 rc=$want 实得 rc=$got)"; FAIL=$((FAIL+1)); return
  fi
  if [ -n "$needle" ] && ! printf '%s' "${LAST_OUT:-}" | grep -q -- "$needle"; then
    echo "  ❌ $desc  (rc 对但输出里找不到 '$needle')"; FAIL=$((FAIL+1)); return
  fi
  echo "  ✅ $desc  (rc=$got${needle:+ 且含 '$needle'})"; PASS=$((PASS+1))
}
# ⭐⭐⭐ 2026-10-07 独立审计(major):原来核心用例"缺条件就 skip"且 **skip 不影响退出码**
#   ⇒ 生产 rc=3 / LOGDIR 不可读 fail-closed / 唯一放行 rc=0 这些【核心断言】一次没跑
#     也可能全绿,而 run_all 会把它当验收证据 ✗
#   ⇒ 现在:核心用例 skip ⇒ 计 **FAIL**(除非显式 ALLOW_SKIP=1 承认"本次确实没验")✓
skip() { echo "  ⏭  SKIP: $*"; SKIP=$((SKIP+1)); }
skipf() {
  echo "  ❌ 核心用例未能执行(缺条件):$*"
  if [ -n "${ALLOW_SKIP_REASON:-}" ]; then SKIP=$((SKIP+1)); else FAIL=$((FAIL+1)); fi
}
note() { echo "  ── $*"; }

_run() { LAST_OUT="$(bash "$GATE" "$@" 2>&1)"; return $?; }   # 别名,便于阅读

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
SLEEP_PID=""
_spawn_benign() {   # 造一个"无身份证据"的进程(sleep),只为测门的默认拒绝 ✓
  [ -n "$SLEEP_PID" ] && return 0
  setsid sleep 300 >/dev/null 2>&1 &
  SLEEP_PID=$!
  sleep 0.3
}
trap 'kill -TERM "$SLEEP_PID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

echo "G3 whoami_proc.sh 归属门行为矩阵"

# ── 0) 从产物取真实身份(不写死 pid)────────────────────────────────────────
PROD_PID="$(cat "$REAL_LOGDIR/v41_8070.pid" 2>/dev/null || cat "$REAL_LOGDIR/vllm_prod_8070.pid" 2>/dev/null || true)"
ORPH_PID="$(cat "$REAL_LOGDIR/sm_17b.pid" 2>/dev/null || true)"
echo "  现场:生产 pid=${PROD_PID:-无}  孤儿 pid=${ORPH_PID:-无}"

# ── 1) 参数错 ────────────────────────────────────────────────────────────────
note "1) 用法与非法参数"
_run; chk "缺参数 ⇒ 2" 2 "$?"
for bad in 0 -1 abc "12 34"; do
  _run "$bad"; chk "非法 pid '$bad' ⇒ 3" 3 "$?"
done

# ── 2) 不存在 / 自身 / 祖先 ──────────────────────────────────────────────────
note "2) 不存在 · 自身 · 祖先链"
_run 999999; chk "不存在的 pid ⇒ 3" 3 "$?"
_run "$$";  chk "自身(测试 shell)⇒ 3" 3 "$?"
_run "$PPID"; chk "父进程 ⇒ 3" 3 "$?"

# ── 3) ⭐ 生产:rc≠0,且理由必须是【PID 文件】(v3 死代码的回归判据)───────────
note "3) 生产 8070(核心回归)"
if [ -n "$PROD_PID" ] && [ -d "/proc/$PROD_PID" ]; then
  _run "$PROD_PID"; rc=$?
  chk "生产 pid ⇒ 3" 3 "$rc"
  chk "生产 pid 的理由是 PID文件(v3 这里曾打成'端口')" 3 "$rc" "PID文件"
  chk "生产 pid 输出里能看到生产 PID 文件名" 3 "$rc" "v41_8070.pid"
else
  skipf "生产 pid 不可用(pid=${PROD_PID:-空})"
fi

# ── 4) 生产端口 / 生产 PID 文件 两条判据各自独立生效 ────────────────────────
note "4) 生产端口 与 生产 PID 文件 分别独立生效"
_run 114597 >/dev/null 2>&1 || true      # 预热(无副作用)
# 4a) 用【临时 LOGDIR】+ 生产名,验证"PID 文件判据"单独也能拦(CV 无所谓)
if [ -n "$PROD_PID" ] && [ -d "/proc/$PROD_PID" ]; then
  echo "$PROD_PID" > "$TMP/coremap.pid"
  LAST_OUT="$(LOGDIR="$TMP" bash "$GATE" "$PROD_PID" 2>&1)"
  chk "临时 LOGDIR 里挂生产名 coremap.pid ⇒ 3" 3 "$?" "PID文件"
  rm -f "$TMP/coremap.pid"
fi

# ── 5) ⭐ fail-closed:证据读不到 ⇒ 绝不放行 ─────────────────────────────────
note "5) 证据缺失必须 fail-closed(审计 fix #2)"
if [ -n "$ORPH_PID" ] && [ -d "/proc/$ORPH_PID" ]; then
  LAST_OUT="$(LOGDIR=/nonexistent-dir-for-g3 bash "$GATE" "$ORPH_PID" 2>&1)"
  chk "LOGDIR 不可读 ⇒ 即使 CV=0 vllm 也 ⇒ 3" 3 "$?"
  echo "$ORPH_PID" > "$TMP/v41_8070.pid"
  LAST_OUT="$(LOGDIR="$TMP" bash "$GATE" "$ORPH_PID" 2>&1)"
  chk "⭐ 回归#1:CV=0 的 vllm 却名为 v41_8070.pid ⇒ 必须 3" 3 "$?" "PID文件"
  rm -f "$TMP/v41_8070.pid"
else
  skipf "孤儿(唯一的真实 CV=0 vllm)不在,跳过 fail-closed / 回归#1"
fi

# ── 6) 无身份证据的普通进程 ⇒ 默认拒绝 ──────────────────────────────────────
note "6) 默认不可杀"
_spawn_benign
if [ -n "$SLEEP_PID" ] && [ -d "/proc/$SLEEP_PID" ]; then
  _run "$SLEEP_PID"; chk "无身份证据的 sleep ⇒ 3" 3 "$?"
else
  skip "未能造出 benign 进程"
fi

# ── 7) ⭐ 唯一放行路径:显式 CUDA_VISIBLE_DEVICES=0 的 vLLM ──────────────────
note "7) 唯一放行路径(用现场真实孤儿,若无则 skip)"
if [ -n "$ORPH_PID" ] && [ -d "/proc/$ORPH_PID" ]; then
  pi_cv "$ORPH_PID"; cv="$PI_CV"; rok="$PI_CV_READ_OK"
  if [ "$rok" = "1" ] && [ "$cv" = "0" ]; then
    _run "$ORPH_PID"; chk "显式 CV=0 的调试 vLLM ⇒ 0(允许)" 0 "$?"
  else
    skip "孤儿 CV 不是显式 0(实测 CV=${cv:-不可读})"
  fi
else
  skipf "孤儿不在(pid=${ORPH_PID:-空})⇒ 无法放行路径实测;构造合成 vllm 被审计明令禁止 ✗"
fi

# ── 8) 混合:只要有一个不可杀 ⇒ 整单 rc=3 ───────────────────────────────────
note "8) 多 pid 聚合"
if [ -n "$PROD_PID" ] && [ -d "/proc/$PROD_PID" ] && [ -n "$ORPH_PID" ] && [ -d "/proc/$ORPH_PID" ]; then
  _run "$PROD_PID" "$ORPH_PID"; chk "生产+孤儿混合 ⇒ 3" 3 "$?"
else
  skipf "现场进程不足,跳过混合用例"
fi

# ── 9) 名单唯一真源:门里不得重抄生产名单 ───────────────────────────────────
note "9) 名单唯一真源(G2 的运行时对照)"
n_def="$(grep -c 'PROD_PORTS=' "$GATE" 2>/dev/null)"; n_def="${n_def:-0}"
if [ "$n_def" = "0" ]; then chk "门里没有重复定义 PROD_PORTS(唯一真源在 lib)" 0 0
else chk "门里没有重复定义 PROD_PORTS(唯一真源在 lib)" 0 1; fi

echo
echo "G3 结果:通过 $PASS / 失败 $FAIL / 跳过 $SKIP"
[ "$FAIL" -eq 0 ] && echo "G3 ✅ 全绿" || echo "G3 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
