#!/usr/bin/env bash
# ⭐ 生产重启前的【强制前置检查】—— 目的:在【停生产之前】就发现脚本错误,
#    特别是 `set -u` 下的未定义变量(2026-10-06 事故:数组项被当成变量 ⇒ 启动即死,浪费 20 分钟 ✗)
# 用法:bash scripts/preflight_bringup.sh [脚本路径]   ⇒ 非 0 退出即【禁止重启】✓
set -uo pipefail
B="${1:-scripts/bringup_prod_8070.sh}"
R=/home/user/lvllm/vllm-xiaotu-moe
cd "$R" || exit 2
fail=0
note(){ printf '  %-46s %s\n' "$1" "$2"; }

# ① 语法
if bash -n "$B" 2>/dev/null; then note "① 语法(bash -n)" "✅"; else note "① 语法(bash -n)" "❌"; bash -n "$B"; fail=1; fi

# ② ⭐ 干跑:把脚本【从开头到 VLLM_ENV 数组闭合】那一段(参数与变量定义)在 set -u 下执行,
#    并把 port_up / proc.sh / say / wait_port 全部 stub 掉 ⇒ 不上服务也能暴露未定义变量 ✓
#    ⚠️ 2026-10-08 修:原来写死 `sed -n '1,130p'` ⇒ 而数组闭合**正好在第 130 行**
#       ⇒ 在数组内**加任何一行**都会把闭合推过 horizon ⇒ 干跑拿到【未闭合的数组】⇒ 前置检查假失败 ✗
#       (实测:我加 11 行注释 ⇒ 闭合到 141 ⇒ ❌)⇒ 改为**动态取闭合行** ✓
#    ⚠️ 仍未做(已登记):stub 没有 `set -e` ⇒ 段内"非最后一句"的失败仍可能被吞 ✗
STUB=$(mktemp)
{
  echo 'set -uo pipefail'
  echo 'say(){ :; }; port_up(){ return 0; }; wait_port(){ return 0; }; echo(){ :; }; curl(){ :; }; ss(){ :; }; nvidia-smi(){ :; }'
  echo 'mkdir -p /tmp/_pf 2>/dev/null'
  echo 'proc(){ :; }'
  # ⭐ 动态 horizon:VLLM_ENV=( … ) 的闭合行(取不到时退回 130,并打印告警)
  END=$(awk '/^VLLM_ENV=\(/ {f=1} f && /^\)[[:space:]]*$/ {print NR; exit}' "$B")
  if [ -z "$END" ]; then echo "⚠️ 取不到 VLLM_ENV 闭合行,退回 1,130" >&2; END=130; fi
  sed -n "1,${END}p" "$B"
} > "$STUB"
if bash "$STUB" >/tmp/_pf_out 2>/tmp/_pf_err; then
  note "② 干跑到数组闭合(set -u 未定义变量)" "✅"
else
  note "② 干跑到数组闭合(set -u 未定义变量)" "❌"
  grep -aE "unbound variable|command not found|syntax error" /tmp/_pf_err | head -4 | sed 's/^/      /'
  fail=1
fi
rm -f "$STUB"

# ③ 关键参数抽查(必须都能取到值且合法 ✓)
check(){ v=$(grep -oE "^[[:space:]]*$1=[^ #]+" "$B" | head -1 | cut -d= -f2 | tr -d '"'); \
  if [ -n "$v" ]; then note "③ $1" "✅ $v"; else note "③ $1" "❌ 取不到值"; fail=1; fi; }
check MAXLEN; check MBT; check SPEC; check TP
v=$(grep -oE 'XIAOTU_GP_ACT_RESERVE_GIB=[0-9.]+' "$B" | head -1 | cut -d= -f2)
if [ -n "$v" ]; then note "③ ACT_RESERVE" "✅ $v"; else note "③ ACT_RESERVE" "❌"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "  ✅ 前置检查通过 ⇒ 可以重启生产 ✓"; exit 0; else echo "  ❌ 前置检查未通过 ⇒ 【禁止重启生产】✗"; exit 1; fi
