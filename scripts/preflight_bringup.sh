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

# ② ⭐ 干跑:把脚本【前 120 行】(参数与变量定义 + LMCache 分支)在 set -u 下执行,
#    并把 port_up / proc.sh / say / wait_port 全部 stub 掉 ⇒ 不上服务也能暴露未定义变量 ✓
STUB=$(mktemp)
{
  echo 'set -uo pipefail'
  echo 'say(){ :; }; port_up(){ return 0; }; wait_port(){ return 0; }; echo(){ :; }; curl(){ :; }; ss(){ :; }; nvidia-smi(){ :; }'
  echo 'mkdir -p /tmp/_pf 2>/dev/null'
  echo 'proc(){ :; }'
  sed -n '1,130p' "$B" | sed 's#bash scripts/proc.sh spawn lmcache_server#: #'
} > "$STUB"
if bash "$STUB" >/tmp/_pf_out 2>/tmp/_pf_err; then
  note "② 干跑前 130 行(set -u 未定义变量)" "✅"
else
  note "② 干跑前 130 行(set -u 未定义变量)" "❌"
  grep -aE "unbound variable|command not found|syntax error" /tmp/_pf_err | head -4 | sed 's/^/      /'
  fail=1
fi
rm -f "$STUB"

# ③ 关键参数抽查(必须都能取到值且合法 ✓)
check(){ v=$(grep -oE "^[[:space:]]*$1=[^ #]+" "$B" | head -1 | cut -d= -f2 | tr -d '"'); \
  if [ -n "$v" ]; then note "③ $1" "✅ $v"; else note "③ $1" "❌ 取不到值"; fail=1; fi; }
check LMCACHE; check MAXLEN; check MBT; check SPEC; check TP
v=$(grep -oE 'XIAOTU_GP_ACT_RESERVE_GIB=[0-9.]+' "$B" | head -1 | cut -d= -f2)
if [ -n "$v" ]; then note "③ ACT_RESERVE" "✅ $v"; else note "③ ACT_RESERVE" "❌"; fail=1; fi
# ④ LMCache 分支一致性:LMCACHE 必须来自【真实变量或默认值】,不能裸用数组项 ✗
if grep -qE '\[ "\$LMCACHE" = "1" \]' "$B"; then note "④ LMCACHE 分支用裸变量" "❌(数组项不是变量)"; fail=1; else note "④ LMCache 分支" "✅"; fi

echo
if [ "$fail" -eq 0 ]; then echo "  ✅ 前置检查通过 ⇒ 可以重启生产 ✓"; exit 0; else echo "  ❌ 前置检查未通过 ⇒ 【禁止重启生产】✗"; exit 1; fi
