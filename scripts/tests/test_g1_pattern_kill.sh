#!/usr/bin/env bash
# G1 自测 —— 证明"禁止按模式杀进程"的静态门禁**不是恒绿的空门**(2026-10-07)
#
# 为什么必须有这一份(本项目刚踩过的教训):
#   ⭐ 门禁自己也会有洞:本轮实测发现 G1 的 `strip_comment` 用 `break` 截断整串
#      ⇒ 多行单元(heredoc / for 块)里【第一句注释之后】的 `os.kill` 全部看不见 ✗
#      ⇒ 对最危险的 `kill_serve.sh` **静默失明** ✗
#   ⇒ 所以每条门禁都必须有【反例转正】:故意写一个坏样本,门必须报出来 ✓
#
# 本测试 = 正例(必须被 G1 判 fail)+ 负例(必须被 G1 判 pass,即不误伤正当写法)✓
# ⭐ 全程只读:只为 fixture 建临时目录,不启动/不杀任何进程 ✓
#
# 用法: bash scripts/tests/test_g1_pattern_kill.sh
# 退出码:0 = 全绿 · 1 = 有用例不符 · 2 = 内部错
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
G1="$ROOT/scripts/gates/check_no_pattern_kill.sh"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

PASS=0; FAIL=0
run_case() {   # run_case <期望rc> <说明> ; 内容从 stdin
  local want="$1" desc="$2" d out rc
  d="$TMP/$(echo "$desc" | tr -c 'A-Za-z0-9' '_')"
  mkdir -p "$d"
  cat > "$d/case.sh"
  out="$(bash "$G1" "$d" 2>&1)"; rc=$?
  if [ "$rc" = "$want" ]; then
    echo "  ✅ $desc  (G1 rc=$rc)"; PASS=$((PASS+1))
  else
    echo "  ❌ $desc  (期望 G1 rc=$want,实得 rc=$rc)"; echo "$out" | sed 's/^/       /'; FAIL=$((FAIL+1))
  fi
}

echo "G1 自测:反例(必须 fail)+ 负例(必须 pass)"
echo
echo "── 反例:这些写法【必须】被 G1 判 fail ──"

run_case 1 "R1 pkill -f" <<'EOF'
#!/usr/bin/env bash
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
EOF

run_case 1 "R2 killall" <<'EOF'
#!/usr/bin/env bash
killall vllm 2>/dev/null || true
EOF

run_case 1 "R3 pgrep -f 选目标后 kill" <<'EOF'
#!/usr/bin/env bash
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server'); do kill -9 "$p"; done
EOF

run_case 1 "R3 ps|grep 选目标后 kill" <<'EOF'
#!/usr/bin/env bash
for p in $(ps -eo pid,args | grep "VLLM::Worke[r]" | awk '{print $1}'); do
  kill -TERM "$p" 2>/dev/null || true
done
EOF

run_case 1 "R3 nvidia-smi 全卡当目标" <<'EOF'
#!/usr/bin/env bash
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p"; done
EOF

run_case 1 "R3b 污染变量(pids=... 后 kill)" <<'EOF'
#!/usr/bin/env bash
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr '\n' ' ')
for p in $pids; do kill -9 "$p" 2>/dev/null; done
EOF

run_case 1 "R3/R4 python 扫 /proc 通配后 os.kill" <<'EOF'
#!/usr/bin/env bash
python3 - <<'PY'
import os, glob
for f in glob.glob('/proc/[0-9]*/cmdline'):
    pid = int(f.split('/')[2])
    os.kill(pid, 9)
PY
EOF

# ⭐ 回归用例:注释在前,真正的杀在后 —— 专测"strip_comment 不得截断整串"这个洞
run_case 1 "R4 注释在前 + os.kill 在后(截断洞回归)" <<'EOF'
#!/usr/bin/env bash
python3 - <<'PY'
# 这一行注释曾经让整个 heredoc 后续内容对 G1 不可见 ✗
import os, glob
for f in glob.glob('/proc/[0-9]*/cmdline'):
    os.kill(int(f.split('/')[2]), 9)
PY
EOF

echo
echo "── 负例:这些是【正当写法】,必须不被误伤(pass)──"

# ⭐ 回归用例(正例):2026-10-07 实测漏网的一类 —— `ps | awk '/名字/'` 在全机选目标再 kill
#    (dev-docs/w4a8-assets/phase_run.sh:48 就是这么把【生产的 EngineCore】一起 TERM 掉的 ✗✗)
run_case 1 "R3b ps|awk 名字匹配当 kill 目标(漏网回归)" <<'EOF'
#!/usr/bin/env bash
EC2=$(ps -eo pid,comm 2>/dev/null | awk '/VLLM::EngineCor/ {print $1}')
for q in $EC2; do kill -TERM $q 2>/dev/null; done
EOF

run_case 0 "N1 PID 文件派生" <<'EOF'
#!/usr/bin/env bash
for f in "$LOGD"/*.pid; do
  [ -f "$f" ] || continue
  p="$(cat "$f")"
  kill -TERM "$p" 2>/dev/null || true
done
EOF

run_case 0 "N2 pkill -P(PID 派生,不是名字)" <<'EOF'
#!/usr/bin/env bash
pkill -9 -P "$pid" 2>/dev/null || true
EOF

run_case 0 "N3 只读 nvidia-smi 等待(无 kill)" <<'EOF'
#!/usr/bin/env bash
while :; do
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || return 0
  sleep 5
done
EOF

run_case 0 "N4 只读 pgrep -af 诊断(无 kill)" <<'EOF'
#!/usr/bin/env bash
pgrep -af "vllm.entrypoints.openai.api_server" | sed 's/^/   /' | cut -c1-150
EOF

run_case 0 "N5 pgrep -P 递归找后代再 kill(PID 派生)" <<'EOF'
#!/usr/bin/env bash
for c in $(pgrep -P "$p"); do kill -9 "$c" 2>/dev/null || true; done
EOF

# ⭐ 负例:同一类"按名字选出 pid"但只用于【只读探活】⇒ 不是杀目标,不该误伤
#    (G1 必须把 `kill -0` 与真杀区分开;否则会把正当的监控写法判死 ⇒ 逼出 allow 标记 ✗)
run_case 0 "N9 ps|awk 选 pid 但只 kill -0 探活(只读)" <<'EOF'
#!/usr/bin/env bash
EC=$(ps -eo pid,comm 2>/dev/null | awk '/VLLM::EngineCor/ {print $1; exit}')
if kill -0 "$EC" 2>/dev/null; then echo "监控中 pid=$EC"; fi
EOF

run_case 0 "N6 仅 kill -0 探活" <<'EOF'
#!/usr/bin/env bash
pid="$(cat "$pf")"
if kill -0 "$pid" 2>/dev/null; then echo RUNNING; else echo NOT RUNNING; fi
EOF

run_case 0 "N7 注释里提到 pkill -f(不是代码)" <<'EOF'
#!/usr/bin/env bash
# 绝不要用 pkill -f "vllm" —— 会命中自己(本仓历史事故)
echo ok
EOF

run_case 0 "N8 端口派生 PID + 归属校验后 kill" <<'EOF'
#!/usr/bin/env bash
. scripts/lib_proc_identity.sh
pi_port_listeners "$PORT"
p="$(printf '%s\n' "$PI_PORT_PIDS" | head -1)"
if why="$(pi_prod_hit "$p")"; then echo "REFUSE $why"; exit 4; fi
kill -TERM "$p" 2>/dev/null || true
EOF

echo
echo "G1 自测结果:通过 $PASS / 失败 $FAIL"
[ "$FAIL" -eq 0 ] && echo "G1 自测 ✅ 全绿(门禁既可判坏、也不误伤)" || echo "G1 自测 ⛔ 不通过"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
