#!/usr/bin/env bash
# G11 —— 【生产名单漂移探测器】运行期反查门禁(2026-10-07)
#
# 来由:生产判据有两套 —— ① PID 文件名 ② 端口。二者都由
#   `scripts/lib_proc_identity.sh` 唯一真源定义,但**名单会随运维漂移**
#   (换实例名、换端口,却忘了更新真源)⇒ 归属门就会把生产判成"可杀" ✗✗
#   ⇒ 本门做**反向一致性检查**:在端口上监听的那个 pid,必须能在
#      【生产 PID 文件】里被认领;认领不到 ⇒ 名单漂移,报警 ✓
#
# 做法(全部**只读**,绝不 kill/start):
#   端口(默认只查 8070)──`pi_port_listeners`──▶ 属主 pid
#        └──`pi_prod_pidfiles_of`──▶ 命中生产 PID 文件?
#              命中  ⇒ ✅
#              不命中 ⇒ ⛔ 漂移(fail)
#   端口没人听 ⇒ ⏭ skip 且 exit 0(**不误报**)✓
#   证据读不到(ss 失败 / PID 文件目录不可读)⇒ fail-closed(按漂移处理)✓
#
# 名单/端口/解析逻辑**全部来自唯一真源** `scripts/lib_proc_identity.sh`,
#   本门**不重抄**任何名单(照抄正是本项目连错三版的根因)✓
#
# 用法: scripts/gates/check_prod_identity.sh [repo_root]
#   默认                查唯一真源里的【全部】生产端口 $PROD_PORTS
#   G11_PORTS="8070 5555"  只查指定端口
#   G11_PORTS=all          同默认(显式写法)
# ⚠️ 名单只能来自唯一真源;禁止在门里手写"默认只查 8070" —— 那会让默认覆盖 1/7
#    (审计 2026-10-07 证伪)✗
# 退出码:0 = 全绿(含"端口空闲"的 skip)· 1 = 漂移/证据缺失 · 2 = 用法或内部错
set -uo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -d "$ROOT" ] || { echo "G11: 找不到 repo root: $ROOT" >&2; exit 2; }
LIB="$ROOT/scripts/lib_proc_identity.sh"
[ -r "$LIB" ] || LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib_proc_identity.sh"
[ -r "$LIB" ] || { echo "G11: 唯一真源不可读: $LIB" >&2; exit 2; }

# shellcheck source=lib_proc_identity.sh
. "$LIB"

case "${G11_PORTS:-}" in
  ""|all|ALL) PORTS="$PROD_PORTS" ;;         # ⭐ 默认 = 真源全部生产端口(不再写死 8070)
  *)          PORTS="$G11_PORTS" ;;
esac

# ⭐ 防恒绿:名单为空(真源坏了)⇒ 内部错,绝不判通过 ✓
if [ -z "$PORTS" ]; then
  echo "G11 ⛔ 内部错:生产端口名单为空(真源 $LIB 的 PROD_PORTS 没读到)⇒ 拒绝判通过" >&2
  exit 2
fi

echo "G11 运行期反查端口: $PORTS"
echo "     真源(名单/端口/解析): $LIB"
echo "     PID 文件目录: $PROC_LOGDIR"

FAIL=0; CHECKED=0; SKIP=0
for port in $PORTS; do
  pi_port_listeners "$port"
  if [ "${PI_SS_OK:-0}" != "1" ]; then
    echo "   ⛔ 端口 $port:ss 读失败/超时 ⇒ 证据缺失(按漂移处理,不静默放过)"
    FAIL=1; continue
  fi
  if [ -z "$PI_PORT_PIDS" ]; then
    echo "   ⏭  端口 $port:没有监听 ⇒ skip(不算违规)"
    SKIP=$((SKIP+1)); continue
  fi
  for pid in $PI_PORT_PIDS; do
    CHECKED=$((CHECKED+1))
    if ! pi_prod_pidfiles_of "$pid"; then
      echo "   ⛔ 端口 $port 属主 pid=$pid:PID 文件目录不可读 ⇒ 无法反查(按漂移处理)"
      FAIL=1; continue
    fi
    if [ -n "$PI_PROD_PIDFILES" ]; then
      echo "   ✅ 端口 $port 属主 pid=$pid 命中生产 PID 文件:$(printf '%s' "$PI_PROD_PIDFILES" | paste -sd, -)"
    else
      echo "   ⛔ 端口 $port 属主 pid=$pid 【没有】命中任何生产 PID 文件 ⇒ 生产名单漂移!"
      if [ -n "$PI_PIDFILES" ]; then
        echo "        该 pid 的全部 pidfile:$(printf '%s' "$PI_PIDFILES" | paste -sd, -)"
      else
        echo "        该 pid 在 $PROC_LOGDIR 下没有任何 pidfile"
      fi
      echo "        ⇒ 修法:把真实 PID 文件名补进 lib 的生产名单,或把该端口从生产端口里去掉"
      FAIL=1
    fi
  done
done

echo
if [ "$FAIL" -ne 0 ]; then
  echo "G11 ⛔ 不通过:反查 $CHECKED 个监听者发现漂移/证据缺失(skip $SKIP 个空闲端口)"
  exit 1
fi
echo "G11 ✅ 通过:反查 $CHECKED 个监听者全部命中生产 PID 文件(skip $SKIP 个空闲端口)"
exit 0
