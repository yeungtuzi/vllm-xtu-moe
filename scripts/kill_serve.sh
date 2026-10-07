#!/usr/bin/env bash
# kill_serve.sh —— 【必须显式给目标】地停掉自己的 vllm 实例(2026-10-07 审计整体重写)。
#
# ⭐ 为什么整体重写:旧版扫描 `/proc/*/cmdline`,用 python `os.kill` 按 argv[0]/命令行
#   特征【选目标】—— 那正是 2026-10-07 一天四次事故的根因模式("按模式找进程再杀"):
#     ① 会把调用者自己算进去(自己的 cmdline 也含同样的字面量)⇒ 自杀 ✗
#     ② 会把【生产 8070】当成"孤儿 vllm"杀掉 ✗✗
#     ③ 匹配条件一改就静默失明/静默误杀 —— 检查器自己不可靠时最危险 ✗
#
# ⭐ 新纪律(与 AGENTS.md 第 0 条一致):
#   * "要杀哪些 PID" 只能由【PID 文件】派生(名字 ⇒ 交给 scripts/proc.sh stop);
#     也允许当次人工显式给出的数字 PID,但必须先过 pi_prod_hit 归属校验 ✓
#   * "能不能杀" 只能由 PID 文件 / 端口 / CUDA_VISIBLE_DEVICES 判定,绝不用进程名 ✗
#   * ⛔ 禁止按名字/命令行模式选目标;禁止无条件清理全卡 GPU 进程
#
# 用法:
#   scripts/kill_serve.sh <name|pid> [<name|pid> ...]     # 必须显式给目标
#     <name>  ⇒ `bash scripts/proc.sh stop <name>`(内部只读 <name>.pid;
#                归属不是"我自己的调试实例"⇒ rc=4 拒绝;杀后未死 ⇒ rc=5)
#     <pid>   ⇒ 先 pi_prod_hit 校验(命中生产证据 ⇒ 拒绝);否则 TERM → 等 → KILL → 验尸
#   WAIT=30 scripts/kill_serve.sh dsv41_prod
#   ⛔ 不带参数 ⇒ 打印用法并 exit 2(**缺省拒绝**,而不是缺省"清全场")✓
#
# 退出码:0 = 全部目标停成功 · 1 = 至少一个目标失败/被拒 · 2 = 用法错
#         3 = 仍有卡占用 >= 1 GiB(M9,只读硬校验)
set -uo pipefail
_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ⭐ 归属证据 / PID 派生 / 存活判定一律走唯一真源 ✓
# shellcheck source=lib_proc_identity.sh
. "$_SELF_DIR/lib_proc_identity.sh"

WAIT="${WAIT:-22}"
RC=0

usage() {
  cat >&2 <<'EOF'
usage: scripts/kill_serve.sh <name|pid> [<name|pid> ...]

  ⛔ 必须【显式】给出要停的目标(实例名或数字 PID);本脚本不做任何模式匹配 ✗
     <name> ⇒ bash scripts/proc.sh stop <name>(只读 <name>.pid,归属门把关)
     <pid>  ⇒ 先做生产归属校验,再 TERM → 等 → KILL → 验尸
  例:scripts/kill_serve.sh dsv41_prod
      WAIT=30 scripts/kill_serve.sh 1234 5678
EOF
}

[ $# -gt 0 ] || { usage; exit 2; }

# ⭐ 等一组 pid 全部消失(最多 WAIT 秒);0 = 全没了
_wait_dead() {
  local i=0 p any
  while [ "$i" -lt $((WAIT*4)) ]; do
    any=0
    for p in "$@"; do pi_alive "$p" && any=1; done
    [ "$any" -eq 0 ] && return 0
    sleep 0.25; i=$((i+1))
  done
  return 1
}

# ⭐ 显式数字 PID:归属校验 ⇒ TERM ⇒ 等 ⇒ KILL ⇒ 验尸(全程只按 PID)✓
stop_pid() {
  local pid="$1" why _g _grc
  case "$pid" in ''|0|1|*[!0-9]*) echo "⛔ 拒绝:非法 PID '$pid'(0/-1 会被 kill 解释成进程组/广播)✗" >&2; return 1 ;; esac
  # ⛔ 命中生产证据(PID 文件/端口/自身祖先链/父链/证据读不到)⇒ 拒绝,exit≠0 ✓
  if why="$(pi_prod_hit "$pid")"; then
    echo "⛔ 拒绝:pid=$pid 命中生产证据:$why" >&2
    echo "   ⇒ 要停生产必须先取得用户【当次】许可(不许用本脚本当旁路)✗" >&2
    return 1
  fi
  # ⭐⭐⭐ 2026-10-07 独立审计(blocker 1):数字 PID 路径原先**只**过 pi_prod_hit ⇒
  #   一个**被 reparent 到 init 的生产 GPU 子进程**(VLLM::EngineCore / VLLM::Worker)
  #   自己不在生产 PID 文件里、不监听端口、cmdline 也没有 --port,父链又断在 init
  #   ⇒ pi_prod_hit 会放行,然后被这里 TERM/KILL 掉 ✗✗
  #   ⇒ 必须再过【权威归属门】whoami_proc.sh:它按名字拦 EngineCore/Worker,
  #     并要求"显式 CUDA_VISIBLE_DEVICES=0 的 vLLM"才放行 ✓(实测:生产子进程 rc=3)
  _g="$(bash "$_SELF_DIR/whoami_proc.sh" "$pid" 2>&1)"; _grc=$?
  if [ "$_grc" -ne 0 ]; then
    echo "⛔ 拒绝:pid=$pid 未通过权威归属门(whoami rc=$_grc)" >&2
    printf '%s\n' "$_g" | sed 's/^/     /' >&2
    return 1
  fi
  if ! pi_alive "$pid"; then echo "  already dead: pid=$pid"; return 0; fi
  kill -TERM "$pid" 2>/dev/null || true
  _wait_dead "$pid" || kill -KILL "$pid" 2>/dev/null || true
  _wait_dead "$pid" || true
  if pi_alive "$pid"; then
    echo "  ✗ FAILED: pid=$pid 发信号后仍存活 ⇒ 需人工处置(不再按模式补杀)✗" >&2
    return 1
  fi
  echo "  stopped pid=$pid(已验尸 ✓)"
  return 0
}

for tgt in "$@"; do
  if pi_valid_pid "$tgt"; then
    stop_pid "$tgt" || RC=1
  elif pi_valid_name "$tgt"; then
    echo "== name=$tgt ⇒ bash scripts/proc.sh stop $tgt"
    bash "$_SELF_DIR/proc.sh" stop "$tgt"
    _rc=$?
    if [ "$_rc" -ne 0 ]; then
      case "$_rc" in
        4) echo "  ✗ name=$tgt 被归属门拒绝(不是自己的调试实例)⇒ 不许停 ✗" >&2 ;;
        5) echo "  ✗ name=$tgt 发信号后仍有存活(proc.sh 已记 KILL-FAILED)✗" >&2 ;;
        *) echo "  ✗ name=$tgt proc.sh stop 失败 rc=$_rc" >&2 ;;
      esac
      RC=1
    fi
  else
    echo "⛔ 非法目标 '$tgt'(既不是数字 PID,也不是合法实例名 [A-Za-z0-9._-])✗" >&2
    RC=1
  fi
done

# ⭐ 只读硬校验(M9):任一卡 used >= 1024 MiB ⇒ exit 3,避免又白等一轮 13 分钟加载
#   ⛔ 2026-10-07 独立审计(major):原写法 `echo "$_used" | awk …` 在 nvidia-smi
#      **不可用/报错/输出为空**时 awk 收到空流 ⇒ rc=0 ⇒ 打印"显存已释放"✗(**fail-open**)
#   ⇒ 现在:nvidia-smi 必须成功且每一行都是数字,否则按"证据不可信"失败退出 ✓
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[kill_serve] ⛔ nvidia-smi 不可用 ⇒ 无法核实显存,按失败退出(M9 fail-closed)✗" >&2; exit 3
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true
_used_raw="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"; _nv_rc=$?
if [ "$_nv_rc" -ne 0 ] || [ -z "$_used_raw" ]; then
  echo "[kill_serve] ⛔ nvidia-smi 读取失败/无输出(rc=$_nv_rc)⇒ 无法核实显存,按失败退出(M9 fail-closed)✗" >&2
  exit 3
fi
_used="$(printf '%s\n' "$_used_raw" | tr -d ' ')"
if printf '%s\n' "$_used" | grep -qvE '^[0-9]+$'; then
  echo "[kill_serve] ⛔ 显存读数含非数字 ⇒ 证据不可信,按失败退出(M9 fail-closed)✗" >&2; exit 3
fi
if printf '%s\n' "$_used" | awk '$1 >= 1024 {exit 1}'; then
  echo "[kill_serve] 完成(显存已释放;只按显式目标停,未做任何模式匹配 ✓)"
else
  echo "[kill_serve] ⛔ 警告:仍有卡占用 >=1 GiB,不要启动(M9)"; exit 3
fi
exit "$RC"
