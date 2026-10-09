#!/usr/bin/env bash
# 进程管理助手 —— 本项目最高优先级纪律的强制执行工具。
#   启动:proc.sh spawn <name> <cmd...>   ⇒ 写 <logdir>/<name>.pid 与 <name>.log(日志持久化)
#   停止:proc.sh stop  <name>            ⇒ 只读 PID 文件 + PID 树,绝不匹配命令行 ✗
#   其它:proc.sh status|pid|adopt <name> [pid]
#
# 背景:按名字/命令行匹配杀进程会杀掉自己(命令行含同一模式),本项目已发生 40+ 次。
#
# ⭐ v4(2026-10-07)。审计判定的 6 处硬伤及修法:
#   ① 门跑在 `kill -0` 之前 ⇒ 陈旧 PID 全部 REFUSE(约 180 个无法清理)
#      ⇒ **先判存活**:死了 ⇒ `not running` + exit 0 ✓
#   ② 组杀假定 `pid==pgid`(实测生产 `pgid≠pid`)⇒ 静默退化成单 PID、且杀后不验尸
#      ⇒ **先比 pgid==pid** 才用 `-$pid`;并显式枚举后代;**杀后【验尸】**,未死 ⇒ exit 5 ✓
#   ③ `status` 恒 exit 0 ⇒ 消费端 `if ! status` 恒假
#      ⇒ RUNNING=0 / 有 PID 文件但已死=1 / 无 PID 文件=3 ✓
#   ④ `spawn` 无条件覆盖仍活着的同名 PID 文件 + `sleep 1` 竞态 ⇒ 制造孤儿
#      ⇒ 同名的**活**进程存在 ⇒ REFUSE exit 4;等 PID 文件出现(最多 5s)才返回 ✓
#   ⑤ `FORCE` 从环境继承(一句 `export FORCE=1` 静默关掉全部门)且不写审计
#      ⇒ **不再读环境变量**;改用显式 `--force --reason=<理由>`,且 `_launch_audit` 记
#        `FORCE-STOP` / `REFUSE` / `KILL-FAILED` / `SPAWN-FAILED` ✓
#   ⑥ 门只判单 pid、proc.sh 却整组击杀
#      ⇒ 杀之前**逐个检查后代**是否命中生产 PID 文件/生产端口/自身祖先链,命中即整单 REFUSE ✓
#
# ⭐ 名单/端口/解析逻辑全部来自 scripts/lib_proc_identity.sh(唯一真源)✓
# ⛔ 旁路只能用于【已取得用户当次许可】的场景;停生产必须先问用户(AGENTS 第 1 条)✗
set -uo pipefail

_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib_proc_identity.sh
. "$_SELF_DIR/lib_proc_identity.sh"
LOGDIR="$PROC_LOGDIR"
mkdir -p "$LOGDIR" 2>/dev/null || true

cmd="${1:-}"; name="${2:-}"; shift 2 2>/dev/null || true
pf="$LOGDIR/${name}.pid"; lf="$LOGDIR/${name}.log"

usage() { echo "usage: proc.sh {spawn|stop|status|pid|adopt} <name> [args...]" >&2; }

# 【2026-10-05】启动/停止审计:一行一条,谁在什么时候动了哪个实例(便于事后定案)
_launch_audit() {
  local act="$1" nm="$2" pid="${3:-}" reason="${4:-}"
  local f="$LOGDIR/_launch_audit.log"
  printf '%s  %-12s %-22s pid=%-9s caller=pid:%s(%s)%s\n' \
    "$(date '+%F %T')" "$act" "$nm" "$pid" "$$" "$(ps -o comm= -p $PPID 2>/dev/null | head -1)" \
    "${reason:+ reason=$reason}" \
    >>"$f" 2>/dev/null || true
}

# ⭐ 等一组 pid 全部消失(最多 n 秒);0=全死了
_dead_wait() {
  local n="$1"; shift
  local i=0 p any
  while [ "$i" -lt $((n*10)) ]; do
    any=0
    for p in "$@"; do pi_alive "$p" && any=1; done
    [ "$any" -eq 0 ] && return 0
    sleep 0.1; i=$((i+1))
  done
  return 1
}

# ⭐ 只按 PID/PGID 发信号(绝不按进程名)✓
#   ⛔⛔ 2026-10-07 独立审计(blocker):原来"pgid==pid 就 `kill -- -pgid`"会把
#      **不属于本实例的进程组成员**一起打(组成员 ≠ 我们的后代;生产的 pgid=106307≠pid 就是这种形状),
#      而且对 pidfile 里的垃圾内容没有校验(`kill -9 0` = 杀本进程组 = 自杀,
#      `kill -9 -1` = 广播级误杀)✗
#   ⇒ 现在:① 显式校验 pgid 是合法 pid 且 ≠0/1;② 把组成员**逐个展开**并过归属门,
#            任一命中生产证据 ⇒ 整单拒绝(refuse 交回调用方),绝不"顺手多杀" ✓
#   $1=信号 $2=主 pid $3..=已体检过的 pid(含后代)
_kill_set() {
  local sig="$1" main="$2"; shift 2
  local p
  # ⛔⛔ 2026-10-07 独立审计(blocker):**不再用 `kill -- -pgid` 整组信号** ✗
  #   实测生产 API 的 `pgid(106307) != pid(114597)`,组长已 DEAD ⇒ 组员集合跨 reparent 存活,
  #   而**进程组成员 ≠ 我们的后代** ⇒ 一旦组里有别人的进程(甚至生产),就被连带杀掉;
  #   上一版"逐成员过 pi_prod_hit"仍挡不住【被 reparent 到 init 的生产 EngineCore/Worker】✗
  #   ⇒ 现在只对【已逐个确认归属的 pid】(main + pi_descendants 得到的后代)发信号 ✓
  #   代价:若真有"reparent 后仍属我们但不在后代列表里"的残留,会由【杀后验尸】抓出来(rc=5)✓
  for p in "$main" "$@"; do
    case "$p" in ''|0|1|*[!0-9]*) continue ;; esac
    kill "-$sig" "$p" 2>/dev/null || true
  done
}

case "$cmd" in
  spawn)
    [ -n "$name" ] && [ $# -gt 0 ] || { echo "usage: proc.sh spawn <name> <cmd...>" >&2; exit 2; }
    pi_valid_name "$name" || { echo "REFUSE: 非法实例名 '$name'(只允许 [A-Za-z0-9._-],且不得为 . / ..)" >&2; exit 3; }
    # ⭐ ④ 不覆盖【活着的】同名 PID 文件
    if [ -f "$pf" ]; then
      _old="$(cat "$pf" 2>/dev/null)"
      if pi_valid_pid "$_old" && pi_alive "$_old"; then
        echo "REFUSE: name=$name 已在运行 pid=$_old(PID 文件 $pf)⇒ 不覆盖 ✗" >&2
        _launch_audit SPAWN-REFUSE "$name" "$_old"
        exit 4
      fi
      echo "  (清除陈旧 PID 文件:pid=${_old:-空} 已不存在)" >&2
      rm -f "$pf"
    fi
    : > "$lf"
    # 由子进程自己写 PID(pid 精确、且 setsid 后自成进程组)
    setsid bash -c 'echo $$ > "$1"; shift; exec "$@"' _ "$pf" "$@" > "$lf" 2>&1 < /dev/null &
    # ⭐ ④ 等 PID 文件出现且进程存在(最多 5s)⇒ 消除 `sleep 1` 竞态
    _new=""
    for _ in $(seq 1 50); do
      _new="$(cat "$pf" 2>/dev/null || true)"
      if pi_valid_pid "$_new" && pi_alive "$_new"; then break; fi
      sleep 0.1
    done
    if ! pi_valid_pid "$_new"; then
      echo "ERROR: spawn 失败(name=$name):5s 内没有写出 PID ⇒ 日志尾部:" >&2
      tail -20 "$lf" 2>/dev/null | sed 's/^/    /' >&2
      _launch_audit SPAWN-FAILED "$name" ""
      exit 6
    fi
    if ! pi_alive "$_new"; then
      echo "WARN: spawn 后 pid=$_new 立刻退出(包装脚本 daemon 化 / 启动即失败)⇒ 请读日志 $lf" >&2
      echo "      ⇒ 真服务若已起来,用【端口派生 PID】再 adopt 认领(禁止按名字匹配)✗" >&2
      _launch_audit SPAWN-DAEMON "$name" "$_new"
      exit 0
    fi
    echo "spawned name=$name pid=$_new log=$lf"
    _launch_audit SPAWN "$name" "$_new"
    ;;

  adopt)
    [ -n "$name" ] && [ -n "${1:-}" ] || { echo "usage: proc.sh adopt <name> <pid>" >&2; exit 2; }
    pi_valid_name "$name" || { echo "REFUSE: 非法实例名 '$name'" >&2; exit 3; }
    _newpid="$1"
    pi_valid_pid "$_newpid" || { echo "REFUSE: 非法 PID '$_newpid'" >&2; exit 3; }
    pi_alive "$_newpid" || { echo "REFUSE: pid=$_newpid 不存在(不许登记死 PID)" >&2; exit 3; }
    if pi_is_self_or_ancestor "$_newpid"; then
      echo "REFUSE: pid=$_newpid 在自身/祖先链上(会把管理脚本自己登记成实例)✗" >&2; exit 3
    fi
    # ⭐ 不覆盖【指向另一个活进程】的 PID 文件(审计:bringup 曾 adopt 别人的 PID)
    if [ -f "$pf" ]; then
      _old="$(cat "$pf" 2>/dev/null)"
      if pi_valid_pid "$_old" && pi_alive "$_old" && [ "$_old" != "$_newpid" ]; then
        echo "REFUSE: $pf 已指向活着的 pid=$_old ⇒ 不覆盖(先停它或换名字)✗" >&2
        _launch_audit ADOPT-REFUSE "$name" "$_newpid"; exit 4
      fi
    fi
    echo "$_newpid" > "$pf"
    echo "adopted name=$name pid=$_newpid log=$lf"
    echo "        cmd=$(pi_cmdline "$_newpid" | cut -c1-90)"
    _launch_audit ADOPT "$name" "$_newpid"
    ;;

  stop)
    [ -n "$name" ] || { usage; exit 2; }
    pi_valid_name "$name" || { echo "REFUSE: 非法实例名 '$name'" >&2; exit 3; }
    [ -f "$pf" ] || { echo "REFUSE: 无 PID 文件 $pf ⇒ 请用端口派生 PID(ss -ltnp | grep :PORT)后 adopt;禁止按名字匹配" >&2; exit 3; }
    _pid="$(cat "$pf" 2>/dev/null)"
    pi_valid_pid "$_pid" || { echo "REFUSE: PID 文件内容非法('${_pid:-空}')" >&2; exit 3; }

    _force=0; _reason=""
    for _a in "$@"; do
      case "$_a" in
        --force)      _force=1 ;;
        --reason=*)   _reason="${_a#--reason=}" ;;
        *) echo "REFUSE: 未知参数 '$_a'(用法: stop <name> [--force --reason=<理由>])" >&2; exit 2 ;;
      esac
    done
    # ⛔ FORCE 环境变量**不再被读取**(审计 ⑤:一句 export FORCE=1 会静默关掉全部门)✗
    case "${FORCE:-}" in ""|0) : ;; *) echo "  (忽略环境变量 FORCE=${FORCE} —— 旁路只能用 --force --reason=…)" >&2 ;; esac

    # ⭐⭐ 2026-10-09(R38 审计 must_fix#1)**无条件**校验:凡给了 `--force` 就必须给 `--reason`,
    #   而不是只在"归属门失败"分支里才要求 —— 否则当目标**能通过归属门**时,
    #   `--force` 会静默跳过下面的后代体检、且一行旁路痕迹都不留 ✗(审计员已沙盒复现)
    if [ "$_force" = "1" ] && [ -z "$_reason" ]; then
      echo "REFUSE: --force 必须同时给 --reason=<理由>(旁路要留痕)" >&2
      _launch_audit REFUSE "$name" "$_pid"; exit 4
    fi

    # ---- ① 先判存活:死了 ⇒ not running + 0(陈旧 PID 文件不再让 stop 恒 REFUSE)----
    if ! pi_alive "$_pid"; then
      echo "not running: name=$name pid=$_pid(PID 文件陈旧,未删)"
      _launch_audit DEAD "$name" "$_pid"
      exit 0
    fi

    # ---- 归属门(默认不可杀)----
    _gate_out="$(bash "$_SELF_DIR/whoami_proc.sh" "$_pid" 2>&1)"; _gate_rc=$?
    if [ "$_gate_rc" -ne 0 ]; then
      if [ "$_force" != "1" ]; then
        echo "REFUSE: name=$name pid=$_pid 归属不是【我自己的调试实例】⇒ 不许停 ✗" >&2
        printf '%s\n' "$_gate_out" | sed 's/^/        /' >&2
        echo "        ⇒ 若是生产/监控栈:必须先取得用户【当次】许可,再由用户决定如何处置 ✗" >&2
        _launch_audit REFUSE "$name" "$_pid"
        exit 4
      fi
      [ -n "$_reason" ] || { echo "REFUSE: --force 必须同时给 --reason=<理由>(旁路要留痕)" >&2; exit 4; }
      echo "  ⚠️ FORCE-STOP 旁路:reason=$_reason" >&2
      _launch_audit FORCE-STOP "$name" "$_pid"
    fi

    # ---- ⑥ 后代逐个体检:命中生产证据/自身链 ⇒ 整单 REFUSE(fail-closed)----
    # ⭐ 2026-10-07 独立审计:原来这里手写了 3 条判据(自身链 / 生产 PID 文件 / 生产端口),
    #   漏掉 lib 的 pi_prod_hit 里另外两条(cmdline 写了生产端口、父链上有生产证据),
    #   且把 pi_prod_pidfiles_of / pi_prod_ports_of 的"证据读不到"当成"没命中"(fail-open)✗
    #   ⇒ 改为直接复用唯一真源 pi_prod_hit(它已是 fail-closed,且含父链检查)✓
    # ⭐⭐ 2026-10-09(R38 审计 must_fix#3,第二版修法):
    #   动机:`--force --reason` 是"已取得用户当次许可后停生产"的**唯一留痕旁路**;而生产
    #   自己的子进程(EngineCore / VLLM::Worker)**必然**命中"父链在生产 PID 文件里"⇒
    #   原先旁路在这条路上**永远无效**(实测:停 8070 被它自己的子进程挡住)✗
    #   ⛔ 第一版修法是"force 就整条跳过 pi_prod_hit",审计判 FAIL:它连
    #     **自身/祖先链(防自杀)**与"自身在生产 PID 文件/端口/cmdline"也一并关掉,
    #     审计员实测复现了"改动版真的 SIGTERM 到 proc.sh 自己的父进程"✗
    #   ⇒ 本版**只豁免【父链】这一类**:授权时改调 `pi_prod_hit_self`(不含父链),
    #     其余四类证据 + fail-closed **一律继续生效** ✓;且**无论归属门是否通过**,
    #     豁免一旦发生就写 `FORCE-DESC-BYPASS` 审计行(含理由)✓
    if ! _desc="$(pi_descendants "$_pid")"; then
      echo "REFUSE: 后代枚举失败(ps 不可用)⇒ 证据缺失,不许停 ✗" >&2
      _launch_audit REFUSE "$name" "$_pid"; exit 4
    fi
    if [ "$_force" = "1" ]; then
      for _d in $_desc; do
        if _why="$(pi_prod_hit_self "$_d")"; then
          echo "REFUSE: 后代 pid=$_d 命中生产证据($_why)⇒ 即使 --force 也不许停 ✗" >&2
          _launch_audit REFUSE "$name" "$_pid"; exit 4
        fi
      done
      echo "  ⚠️ FORCE-DESC-BYPASS:仅豁免【父链类】生产证据(reason=$_reason)" >&2
      _launch_audit FORCE-DESC-BYPASS "$name" "$_pid" "$_reason"
    else
      for _d in $_desc; do
        if _why="$(pi_prod_hit "$_d")"; then
          echo "REFUSE: 后代 pid=$_d 命中生产证据:$_why ⇒ 不许停 ✗" >&2
          _launch_audit REFUSE "$name" "$_pid"; exit 4
        fi
      done
    fi

    # ---- ② 按 PGID/PID 发信号,然后【验尸】----
    _kill_set TERM "$_pid" $_desc
    if ! _dead_wait 10 "$_pid" $_desc; then
      _kill_set KILL "$_pid" $_desc
      _dead_wait 6 "$_pid" $_desc || true
    fi
    _still=""
    for _p in "$_pid" $_desc; do pi_alive "$_p" && _still="$_still $_p"; done
    if [ -n "$_still" ]; then
      echo "FAILED: name=$name 发信号后仍有存活:$_still ⇒ 需人工处置 ✗" >&2
      _launch_audit KILL-FAILED "$name" "$_pid"
      exit 5
    fi
    echo "stopped name=$name pid=$_pid(仅按 PID 文件 + PID 树;已验尸 ✓)"
    _launch_audit STOP "$name" "$_pid"
    ;;

  status)
    [ -n "$name" ] || { usage; exit 2; }
    if [ ! -f "$pf" ]; then
      echo "$name: NOT RUNNING (no pid file: $pf)"; exit 3
    fi
    _pid="$(cat "$pf" 2>/dev/null)"
    if pi_valid_pid "$_pid" && pi_alive "$_pid"; then
      echo "$name: RUNNING pid=$_pid log=$lf"; exit 0
    fi
    echo "$name: NOT RUNNING (pid file: ${_pid:-none})"; exit 1
    ;;

  pid)
    [ -n "$name" ] || { usage; exit 2; }
    [ -f "$pf" ] || { echo "no pid file"; exit 3; }
    cat "$pf"
    ;;

  *)
    usage; exit 2 ;;
esac
