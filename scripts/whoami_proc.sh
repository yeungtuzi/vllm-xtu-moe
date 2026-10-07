#!/usr/bin/env bash
# whoami_proc.sh —— 【停进程前的机械门】:这个 PID 到底属于谁?能不能杀?
#
# ⭐ v4(2026-10-07 第三轮整改)。v1/v2/v3 都被独立审计判【不通过】,教训:
#   v1: ① `cut -c1-70` 切掉判别串 ② 生产判据写错 ③ 恒 exit 0 ④ 找错 PID 文件名
#   v2: ① `:67` 与 `:75` 的【中文字面量不一致】⇒ 恒拒(意外安全网,掩盖了②)
#       ② `:61` 有一条"cmdline 像 runner 就判可杀"的分支,【排在 CV 生产判据之前】⇒ 杀生产的洞 ✗✗
#       ③ `:45` 端口解析 `'[0-9]+$'` 对 `ss` 行尾 `))` 恒不匹配 ⇒ 死代码
#       ④ `:24` 名单漏 `v41_8070`(真生产的 PID 文件) ⑤ `:47` `head -1` 只取一个
#   v3(二审判决不通过,5 条必修)✗:
#       ① `PF_ALL="$PF_ALL $(basename…)"` 带前导空格 vs `^…$` 锚定正则 ⇒ **PID 文件判据是死代码**
#          (实测:对真生产 PID=114597,理由打的是"端口"而不是"PID文件" ⇒ 证明该分支从未触发)
#       ② LOGDIR 不可读 / `ss` 失败 ⇒ **静默放行**(fail-open)
#       ③ 端口 `break` 只取第一条 ⇒ 生死由 `ss` 输出顺序决定
#       ④ 祖先链用 `awk '{print $4}' /proc/<pid>/stat` ⇒ comm 含空格时静默截断
#       ⑤ 门按单 pid 判、proc.sh 却整组击杀(pgid≠pid 时退化)⇒ 该逻辑归 proc.sh,门报 PGID
#
# ⭐⭐ v4 的五条铁律:
#   1. **默认不可杀**:任何"读不到 / 不确定 / 异常"一律判【不可杀】 ✓
#   2. **绝不用进程名做放行判据** ✗ —— 放行只看【身份证据】:PID 文件 + 端口 + CUDA_VISIBLE_DEVICES
#   3. **用布尔 KILLABLE + 退出码**表达结论,**不做中文字面量比较** ✗
#   4. **放行条件只有一个**:该 vLLM 进程**显式声明** `CUDA_VISIBLE_DEVICES=0`,且
#      【不在】任何生产 PID 文件里、【不监听】任何生产端口、【命令行里没写】生产端口 ✓
#   5. **自身/祖先链保护**:gate 自己、它的父、以及从父往上的整条 ppid 链都不许杀 ✓
#
# ⭐ 名单/端口/解析逻辑【不在这里定义】—— 全部来自 scripts/lib_proc_identity.sh(唯一真源)
#    (来由:`scripts` 与 `proc.sh` 之间的分隔符被写成点号 vs 正确斜杠、"我的" vs "我自己的" ⇒ 分头手写必然对不上)
#
# 用法:  scripts/whoami_proc.sh <pid> [pid...]
# 退出码:0 = 全部可杀(仅"显式 GPU0 的调试实例") · 3 = 含不可杀 · 2 = 用法错
set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_proc_identity.sh
. "$_SELF_DIR/lib_proc_identity.sh"

if [ $# -eq 0 ]; then
  echo "用法: scripts/whoami_proc.sh <pid> [pid...]" >&2
  exit 2
fi

# ⭐ 唯一放行标签 —— 【定义一次、处处引用】✓(禁止在比较里手打中文 ✗)
LBL_ALLOW="✓ 显式 GPU0 的调试实例"

RC=0
for pid in "$@"; do
  KILLABLE=0; OWN=""; REASON=""
  CV_RAW=""; CV_READ_OK=0
  PORTS=""; SS_OK=0; PROD_PORT_HIT=""
  PF_ALL=""; PP_ALL=""
  PIDFILE_READ_OK=0
  CMD_FULL=""

  if ! pi_valid_pid "$pid"; then
    OWN="⛔ 非法 PID(拒绝 0/负数/多 token/非数字)"; REASON="参数非法"
  elif [ ! -d "/proc/$pid" ]; then
    OWN="⛔ 进程不存在(陈旧 PID 文件)"; REASON="无此进程"
  else
    CMD_FULL="$(pi_cmdline "$pid")"

    # ---- CUDA_VISIBLE_DEVICES(读不到 ⇒ READ_OK=0 ⇒ 不可杀)----
    pi_cv "$pid"; CV_RAW="$PI_CV"; CV_READ_OK="$PI_CV_READ_OK"

    # ---- 端口:收集【全部】,任一命中生产端口即拒(不再 break)----
    pi_prod_ports_of "$pid"; PORTS="$PI_PORTS"; SS_OK="$PI_SS_OK"; PROD_PORT_HIT="$PI_PROD_PORTS"

    # ---- PID 文件:取【全部】命中(不再 head -1;逐行拆开匹配,不再带前导空格 ✗)----
    if pi_pidfiles_of "$pid"; then PIDFILE_READ_OK=1; else PIDFILE_READ_OK=0; fi
    PF_ALL="$PI_PIDFILES"
    pi_prod_pidfiles_of "$pid" >/dev/null 2>&1 || true
    PP_ALL="$PI_PROD_PIDFILES"

    # ═══════════ 判定:默认不可杀,只有明确命中才放行 ═══════════
    if pi_is_self_or_ancestor "$pid"; then
      OWN="⛔ 我自己的进程/祖先链"; REASON="自身或祖先"
    elif [ "$PIDFILE_READ_OK" != "1" ]; then
      OWN="⛔ PID 文件目录读不到(按生产处理)"; REASON="PID 文件证据缺失"
    elif [ -n "$PP_ALL" ]; then
      OWN="⛔⛔ 生产/监控栈"; REASON="PID文件:$(printf '%s' "$PP_ALL" | paste -sd, -)"
    elif [ "$SS_OK" != "1" ]; then
      OWN="⛔ 端口证据读不到(按生产处理)"; REASON="ss 失败/超时"
    elif [ -n "$PROD_PORT_HIT" ]; then
      OWN="⛔⛔ 生产/监控栈"; REASON="端口:$(printf '%s' "$PROD_PORT_HIT" | paste -sd, -)"
    elif pi_cmdline_has_prod_port "$pid"; then
      OWN="⛔⛔ 命令行里写了生产端口"; REASON="cmdline --port"
    elif printf '%s' "$CMD_FULL" | grep -qE "VLLM::EngineCore|VLLM::Worker"; then
      OWN="⛔ vLLM 引擎/worker 子进程"; REASON="引擎子进程"
    elif pi_cmd_is_vllm "$pid"; then
      # ⭐ 唯一放行路径:显式 CUDA_VISIBLE_DEVICES=0(唯一、无逗号)
      if [ "$CV_READ_OK" != "1" ]; then
        OWN="⛔ CUDA_VISIBLE_DEVICES 读不到(按生产处理)"; REASON="environ 不可读/未设置"
      elif [ "$CV_RAW" = "0" ]; then
        KILLABLE=1; OWN="$LBL_ALLOW"; REASON="CV=0"
      else
        OWN="⛔ 非显式 GPU0"; REASON="CV=${CV_RAW:-空}"
      fi
    else
      OWN="⛔ 未知(按生产处理)"; REASON="无身份证据"
    fi
  fi

  printf "  pid=%-9s CV=%-8s 端口=%-12s PID文件=%-26s 归属=%s\n" \
     "$pid" \
     "$( [ "$CV_READ_OK" = 1 ] && printf '%s' "${CV_RAW:-空}" || printf '%s' '不可读' )" \
     "$( [ "$SS_OK" = 1 ] && { [ -n "$PORTS" ] && printf '%s' "$PORTS" | paste -sd, - || printf '%s' '无'; } || printf '%s' 'ss失败' )" \
     "$( [ "$PIDFILE_READ_OK" = 1 ] && { [ -n "$PF_ALL" ] && printf '%s' "$PF_ALL" | paste -sd, - || printf '%s' '无'; } || printf '%s' '目录不可读' )" \
     "$OWN"
  printf "     理由: %s\n" "$REASON"
  printf "     pgid: %s (pid=%s)  cmd: %s\n" \
     "$(pi_pgid "$pid" 2>/dev/null || echo '?')" "$pid" \
     "$(printf '%s' "${CMD_FULL:-}" | cut -c1-100)"

  [ "$KILLABLE" = "1" ] || RC=3
done

echo
if [ "$RC" -eq 0 ]; then
  echo "  ⭐ 结论:全部是【显式 GPU0 的调试实例】⇒ 允许 kill ✓"
else
  echo "  ⛔ 结论:含【生产/未知/自身/非 GPU0/证据缺失】⇒ 不许 kill ✗(退出码 3)"
fi
exit "$RC"
