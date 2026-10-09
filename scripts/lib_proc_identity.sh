#!/usr/bin/env bash
# lib_proc_identity.sh —— ⭐⭐【进程归属证据的唯一真源】(2026-10-07)
#
# 为什么要有这个文件(用户 2026-10-07 明令 + 上一个会话连错三版的根因):
#   ⭐ 同一个事实存在【两处表示】,而"分别手写、从不交叉校验" ⇒ 必然对不上 ✗
#   证据:① 赋值 `"✓ 我的调试实例"` / 比较 `"✓ 我自己的调试实例"` ⇒ 门恒拒
#         ② `scripts` 与 `proc.sh` 的分隔符写成点号 / 正确斜杠 ⇒ 4 个文件全错
#         ③ `awk substr($0,21)` / 真实长度 ⇒ off-by-one ⇒ 又一次恒拒
#   ⇒ 所以:**生产名单/端口/解析逻辑只允许在这里定义一次**,别处一律 `. lib` 引用 ✓
#
# 被谁引用:scripts/whoami_proc.sh、scripts/proc.sh 以及各 runner(禁止重复定义名单 ✗)
#
# ⭐ 三条不可违背的设计约束:
#   1. **默认不可杀**:任何"读不到 / 解析不出 / 不确定"一律返回"不可杀" ✓
#   2. **绝不用进程名做【放行】判据** —— 名字只用于"拦截",永不用于"放行" ✓
#   3. **禁止手算偏移**:一律 `cut -d=` / 前缀剥离 / `awk -F`,禁止 `substr` 下标 ✓
#
# 用法:  . "$(dirname "${BASH_SOURCE[0]}")/lib_proc_identity.sh"

# 防止被重复 source(幂等)
[ -n "${_LIB_PROC_IDENTITY_LOADED:-}" ] && return 0
_LIB_PROC_IDENTITY_LOADED=1

_LIB_PI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROC_ROOT="${PROC_ROOT:-$(cd "$_LIB_PI_DIR/.." && pwd)}"
# ⭐ 日志/PID 目录:允许 LOGDIR 覆盖(测试用);解析结果全局唯一 ✓
PROC_LOGDIR="${LOGDIR:-$PROC_ROOT/dev-docs/report/tuning/logs}"

# ─────────────────────────────────────────────────────────────────────────────
# ⭐ 生产判据 A:PID 文件名(唯一真源)
#   ⛔ 名单漂移会误判 ⇒ 由 scripts/gates/check_prod_identity.sh 运行期反查维护 ✓
PROD_PIDFILES_RE='^(vllm_prod_8070|v41_8070|prod768|dsv41_prod|lmcache_server|prometheus|grafana|node_exporter|xtu_exporter|coremap|coremap_png|vram_sampler|vram_watch|hang_watch|sess_watch.*|deg_watch|lmcache.*)\.pid$'

# ⭐ 生产判据 B:端口(生产 vLLM + LMCache + 看板/监控栈,同生命周期 ✓)
PROD_PORTS='8070 5555 8080 9090 3000 9100 8787'
# ─────────────────────────────────────────────────────────────────────────────

# ⭐ 所有"证据输出"全局变量的初值 —— 必须在 lib 里初始化,
#   否则调用方在 `set -u` 下会因"未绑定变量"直接退出(实测踩到 ⇒ 门 rc=1 而不是 rc=3)✗
PI_PORTS=""; PI_SS_OK=0; PI_PORT_LISTEN_N=0
PI_PIDFILES=""; PI_PROD_PIDFILES=""
PI_PROD_PORTS=""
PI_PORT_PIDS=""
PI_CV=""; PI_CV_READ_OK=0
PI_DESC_SEEN=" "

pi_valid_pid() {   # 严格:纯十进制正整数,且拒绝 0
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
    0*)          return 1 ;;
    *)           return 0 ;;
  esac
}

pi_alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# ⭐ 用 sed 解析 /proc/<pid>/stat 的 PPid 字段
#   ⛔ 禁止 `awk '{print $4}'`:comm 含空格时字段会错位(审计 fix #4)✗
#   格式: `pid (comm) state ppid ...` ⇒ 贪婪吃掉 `(...)` 才是对的 ✓
pi_ppid() {
  sed -E 's/^[0-9]+ \(.*\) [A-Z] ([0-9]+) .*/\1/' "/proc/$1/stat" 2>/dev/null
}

pi_pgid() { ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '; }

# ⭐ 自身 + 全部祖先(最多 64 层)⇒ 打印为 " pid pid ... " (两端带空格,便于整词匹配)
pi_self_chain() {
  local p="$$" out=" $$ " n=0 pp
  p="${PPID:-0}"
  while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ] && [ "$n" -lt 64 ]; do
    out="$out$p "
    pp="$(pi_ppid "$p")"
    # 解析失败/不前进 ⇒ 停止(不得死循环)
    [ -n "$pp" ] && [ "$pp" != "$p" ] || break
    p="$pp"; n=$((n+1))
  done
  printf '%s' "$out"
}

pi_is_self_or_ancestor() {   # rc=0 ⇒ 是(⇒ 不许杀)
  local pid="$1" chain
  chain="$(pi_self_chain)"
  case "$chain" in *" $pid "*) return 0 ;; esac
  return 1
}

# ⚠️ 先判可读再读:否则对已死 pid 会在 stderr 冒 `tr: No such file or directory` 噪声
#    (审计 A 报告的问题;功能无影响,但会污染调用方日志)
pi_cmdline() { [ -r "/proc/$1/cmdline" ] || return 0; tr '\0' ' ' < "/proc/$1/cmdline"; }

# ⭐ CUDA_VISIBLE_DEVICES(设置 PI_CV / PI_CV_READ_OK)
pi_cv() {
  local pid="$1" line
  PI_CV=""; PI_CV_READ_OK=0
  [ -r "/proc/$pid/environ" ] || return 1
  line="$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=')"
  [ -n "$line" ] || return 0
  PI_CV_READ_OK=1
  PI_CV="${line#CUDA_VISIBLE_DEVICES=}"      # ⭐ 前缀剥离,禁止手算偏移 ✓
  PI_CV="$(printf '%s' "$PI_CV" | tr -d ' ')"
  return 0
}

# ⭐ 该 pid 监听的全部端口(结果写入 PI_PORTS,每行一个)。
#   ⚠️ 必须用【输出全局变量】而不是 stdout:否则调用方的 $(...) 会开子 shell,
#   PI_SS_OK 传不回来 ⇒ 又会退化成"读不到却静默放行" ✗(审计 fix #2)
#   审计 fix #3:不得 `break` 只取第一条(否则生死由 ss 输出顺序决定 ✗)
pi_all_ports() {
  local pid="$1" tmp rc p
  PI_PORTS=""; PI_SS_OK=0
  tmp="$(mktemp 2>/dev/null)" || { PI_SS_OK=0; return 1; }
  timeout 5 ss -ltnp >"$tmp" 2>/dev/null; rc=$?
  [ "$rc" = "0" ] && PI_SS_OK=1
  # `pid=$pid,` 带逗号 ⇒ 防止 pid=114 误匹配 pid=1145 ✓
  while IFS= read -r p; do
    case "$p" in *:*) PI_PORTS="$PI_PORTS${p##*:}"$'\n' ;; esac
  done < <(awk -v want="pid=$pid," 'index($0,want){print $4}' "$tmp")
  rm -f "$tmp"
  return 0
}

# ⭐ 该 pid 监听的生产端口(写入 PI_PROD_PORTS,每行一个)
#   ⛔ 2026-10-07 独立审计:原来只调用 pi_all_ports 而不看返回码 ⇒
#      pi_prod_hit 里那句"端口证据读不到 ⇒ fail-closed"是**死代码** ✗
#      ⇒ 现在:ss 失败 ⇒ **返回 1**(调用方据此判"证据缺失 ⇒ 不可杀")✓
pi_prod_ports_of() {
  local pid="$1" p
  pi_all_ports "$pid" || return 1
  PI_PROD_PORTS=""
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    case " $PROD_PORTS " in *" $p "*) PI_PROD_PORTS="$PI_PROD_PORTS$p"$'\n' ;; esac
  done <<<"$PI_PORTS"
  [ "$PI_SS_OK" = "1" ] || return 1
  return 0
}

# ⭐ 内容严格等于 <pid> 的 *.pid 文件名 ⇒ 写入 PI_PIDFILES(每行一个)
#   rc=1 ⇒ 目录不可读(证据缺失 ⇒ 调用方必须判"不可杀" ✓)
pi_pidfiles_of() {
  local pid="$1" d="${2:-$PROC_LOGDIR}" f v
  PI_PIDFILES=""
  [ -d "$d" ] && [ -r "$d" ] && [ -x "$d" ] || return 1
  for f in "$d"/*.pid; do
    [ -f "$f" ] || continue
    v="$(cat "$f" 2>/dev/null)" || continue
    [ "$v" = "$pid" ] && PI_PIDFILES="$PI_PIDFILES$(basename "$f")"$'\n'
  done
  return 0
}

# ⭐ 其中命中【生产名单】的 ⇒ 写入 PI_PROD_PIDFILES(每行一个);rc=1 ⇒ 目录不可读
pi_prod_pidfiles_of() {
  local pid="$1" d="${2:-$PROC_LOGDIR}" f
  PI_PROD_PIDFILES=""
  pi_pidfiles_of "$pid" "$d" || return 1
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    if printf '%s\n' "$f" | grep -qE "$PROD_PIDFILES_RE"; then
      PI_PROD_PIDFILES="$PI_PROD_PIDFILES$f"$'\n'
    fi
  done <<<"$PI_PIDFILES"
  return 0
}

# ⭐ 监听 <port> 的 pid(结果写入 PI_PORT_PIDS,每行一个;PI_SS_OK 同步)
#   ⛔ 审计 G9:`awk '$4 ~ ":814"'` 会把 `:8143` 也算命中 ⇒ **必须按 `:` 切分后整段相等** ✗
pi_port_listeners() {
  local port="$1" tmp rc p   # ⭐ 2026-10-07 复审:必须声明 p —— 否则 read -r p 会写进调用者的 local p(bash 动态作用域)⇒ wait_port 循环被清空 ✗
  PI_PORT_PIDS=""; PI_SS_OK=0; PI_PORT_LISTEN_N=0
  tmp="$(mktemp 2>/dev/null)" || return 1
  timeout 5 ss -ltnp >"$tmp" 2>/dev/null; rc=$?
  [ "$rc" = "0" ] && PI_SS_OK=1
  # ⭐ N4 修:单独统计【匹配到的 LISTEN 行数】—— 用于区分
  #   "没人听"(0 行 ⇒ DOWN)与"有人在听但属主读不到"(>0 行且无 pid ⇒ UNKNOWN)✓
  PI_PORT_LISTEN_N="$(awk -v want="$port" '
             $1=="LISTEN" { n=split($4,a,":"); if (n>0 && a[n]==want) c++ }
             END { print c+0 }' "$tmp")"
  while IFS= read -r p; do
    [ -n "$p" ] && PI_PORT_PIDS="$PI_PORT_PIDS$p"$'\n'
  done < <(awk -v want="$port" '
             $1=="LISTEN" { n=split($4,a,":"); if (n>0 && a[n]==want) print $0 }
           ' "$tmp" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  rm -f "$tmp"
  return 0
}

# ⭐ 综合"生产证据"判定 —— 供 runner 在停【自己的】实例之前做归属校验 ✓
#   rc=0 ⇒ 命中生产证据(调用方必须【拒绝停】),并把原因打到 stdout
#   判据:我们自己的祖先链、生产 PID 文件、生产端口、命令行里的生产端口、
#        以及该 pid 的【父链上】有生产证据(挡住"停生产的子进程"这个洞,如 VLLM::EngineCore)✓
#   ⚠️ 证据读不到(PID 文件目录不可读)也算命中 ⇒ **fail-closed** ✓
# ⭐ 2026-10-09(R38 审计 must_fix#3)新增:`pi_prod_hit` 的【非父链】部分 ✓
#   为什么需要拆:授权停生产时,生产**自己的子进程**必然命中"父链在生产 PID 文件里"这一类,
#   而那一类恰恰是我们要豁免的;但同一函数里还有**与豁免动机无关、必须继续生效**的证据:
#     ① 自身/祖先链(防自杀 —— 2026-10-07 第③类"runner 名 = 被管实例名 ⇒ stop 自己"事故)
#     ② 自身在生产 PID 文件里  ③ 自身监听生产端口  ④ 自身 cmdline 写了生产端口
#   以及"证据读不到 ⇒ 命中"(fail-closed)✓
#   ⇒ 拆出本函数后,授权旁路**只换用它**,`pi_prod_hit`(含父链)对所有其它调用点**逐字不变** ✓
pi_prod_hit_self() {
  local pid="$1"
  if pi_is_self_or_ancestor "$pid"; then echo "自身/祖先链"; return 0; fi
  if ! pi_prod_pidfiles_of "$pid"; then echo "PID 文件目录不可读(证据缺失)"; return 0; fi
  if [ -n "$PI_PROD_PIDFILES" ]; then
    echo "生产PID文件:$(printf '%s' "$PI_PROD_PIDFILES" | paste -sd, -)"; return 0
  fi
  if ! pi_prod_ports_of "$pid"; then echo "端口证据读不到(证据缺失)"; return 0; fi
  if [ -n "$PI_PROD_PORTS" ]; then
    echo "生产端口:$(printf '%s' "$PI_PROD_PORTS" | paste -sd, -)"; return 0
  fi
  if pi_cmdline_has_prod_port "$pid"; then echo "命令行写了生产端口"; return 0; fi
  return 1
}

pi_prod_hit() {
  local pid="$1" _p _n
  # ①~④ 与 fail-closed 全部复用 `pi_prod_hit_self`(语义与拆分前逐字相同)✓
  if pi_prod_hit_self "$pid"; then return 0; fi
  # ⭐ 父链检查:生产的【子进程】(EngineCore/Worker)自己既不在生产 PID 文件里、
  #    也不监听生产端口 ⇒ 只看它自己会漏判 ⇒ 必须往上看祖先 ✓
  _p="$(pi_ppid "$pid")"; _n=0
  while [ -n "$_p" ] && [ "$_p" != "0" ] && [ "$_p" != "1" ] && [ "$_n" -lt 64 ]; do
    if ! pi_prod_pidfiles_of "$_p"; then echo "父进程 pid=$_p 的 PID 文件证据读不到(证据缺失)"; return 0; fi
    if [ -n "$PI_PROD_PIDFILES" ]; then
      echo "父进程 pid=$_p 在生产 PID 文件里($(printf '%s' "$PI_PROD_PIDFILES" | paste -sd, -))"; return 0
    fi
    if ! pi_prod_ports_of "$_p"; then echo "父进程 pid=$_p 的端口证据读不到(证据缺失)"; return 0; fi
    if [ -n "$PI_PROD_PORTS" ]; then
      echo "父进程 pid=$_p 监听生产端口($(printf '%s' "$PI_PROD_PORTS" | paste -sd, -))"; return 0
    fi
    if pi_cmdline_has_prod_port "$_p"; then echo "父进程 pid=$_p 命令行写了生产端口"; return 0; fi
    _p="$(pi_ppid "$_p")"; _n=$((_n+1))
  done
  return 1
}

# ⭐ 该 pid 的后代里是否有【生产证据】(停一棵树之前用)✓ rc=0 ⇒ 有(拒停)
pi_prod_hit_tree() {
  local root="$1" d why
  if why="$(pi_prod_hit "$root")"; then echo "pid=$root:$why"; return 0; fi
  for d in $(pi_descendants "$root"); do
    if why="$(pi_prod_hit "$d")"; then echo "后代 pid=$d:$why"; return 0; fi
  done
  return 1
}

# ⭐ 命令行里是否【显式】写了生产端口(防御 ss 看不到属主时的漏判)✓
#   只做"拦截"用途:命中 ⇒ 判不可杀 ✓
pi_cmdline_has_prod_port() {
  local pid="$1" cl p
  cl="$(pi_cmdline "$pid")"
  for p in $PROD_PORTS; do
    case "$cl" in
      *"--port $p "*|*"--port $p"|*"--port=$p "*|*"--port=$p") return 0 ;;
    esac
  done
  return 1
}

# ⭐ 命令行是否像一个 vLLM API server(仅用于"放行路径"的入口判据 ⇒ 保守)
pi_cmd_is_vllm() {
  local cl; cl="$(pi_cmdline "$1")"
  case "$cl" in *"-m vllm.entrypoints"*) return 0 ;; esac
  return 1
}

# ⭐ 该 pid 的全部后代(BFS,纯 PID 派生 ⇒ 不用进程名)✓ 每行一个
#   用 PI_DESC_SEEN 做去重,防止 ppid 环导致重复输出/死循环 ✓
#
# ⛔⛔ 2026-10-07 独立审计抓出的真 bug(我写的):原来把整层 frontier 拼成一个字符串再
#   喂给 `ps --ppid "$frontier"` ⇒ ① 只有【第一层】会被查一次就停(第二层再也查不到)
#    ② 更糟:拼出来的字符串带【前导空格】⇒ 本机 procps-ng 3.3.17 对
#       `ps -o pid= --ppid " 1"` 直接 SIGABRT(core dumped, rc=134)✗
#    ⇒ 实测后果:对生产 114597 只返回 122254/122257,而真正的 Worker 孙进程被漏掉 ⇒
#       "后代逐个体检"与"杀后验尸"都会在第二层静默失效 ✗
#    ⇒ 现在:逐个父 pid 单独查(不拼串),next 用 ${next:+…} 拼接(无前导空格)✓
pi_descendants() {
  local root="$1" frontier="$1" next c f
  PI_DESC_SEEN=" "
  # ⭐ 2026-10-07 独立审计:ps 不可用/失败时**必须返回非 0**,不能静默返回空 ✗
  #   (否则"后代体检"与"杀后验尸"会退化成只查主 pid,却仍打印"已验尸 ✓" ⇒ fail-open)
  ps -o pid= -p $$ >/dev/null 2>&1 || return 1
  while [ -n "$frontier" ]; do
    next=""
    for f in $frontier; do
      # ⚠️ 不能拿 `ps --ppid X` 的退出码当"ps 失败":**没有子进程时它本来就返回 1** ✗
      #   (实测:据此判失败 ⇒ 每个叶子节点都让 pi_descendants 返回 1 ⇒ proc.sh stop 全部 REFUSE)
      #   ⇒ 只在函数开头探一次 ps 可用性;这里只把"空输出"当"无子进程" ✓
      for c in $(ps -o pid= --ppid "$f" 2>/dev/null | tr -d ' '); do
        case "$c" in ''|*[!0-9]*) continue ;; esac
        case "$PI_DESC_SEEN" in *" $c "*) continue ;; esac
        PI_DESC_SEEN="$PI_DESC_SEEN$c "
        printf '%s\n' "$c"
        next="${next:+$next }$c"
      done
    done
    frontier="$next"
  done
}

# ⭐ 名字合法性:实例名会直接拼进 `<name>.pid` 路径 ⇒ 必须拒绝路径穿越/空格 ✓
pi_valid_name() {
  case "${1:-}" in
    ''|*[!A-Za-z0-9._-]*) return 1 ;;
    .|..)                return 1 ;;
    *)                   return 0 ;;
  esac
}

# ⭐⭐ 按【单个 PID】安全停止(2026-10-07 独立审计 F2 的统一修法)✓
#   背景:sweep_spec_k / ab_serve_kernel / run_nat_curve / sweep_spin 原先对
#   **未校验的 pidfile 内容**直接 `kill -TERM/-KILL` 并 `pkill -9 -P`:
#     ① pidfile 是 `0` ⇒ `kill 0` = 杀本进程组(自杀);`-1` ⇒ 广播级误杀 ✗
#     ② 陈旧 pidfile 的 pid 被生产复用时,会先 TERM 生产 ✗
#     ③ `pkill -9 -P` 杀的是"所有子进程",**不过归属门** ✗
#   ⇒ 现在统一走这里:数值守卫 → 归属门(自身/祖先、生产 PID 文件/端口/cmdline、父链)
#      → 只对 main + 已枚举后代发信号 → **验尸**(未死则如实报错,不谎报)✓
#   返回:0 = 已停 / 本就没在跑 ; 1 = 拒绝(非法 PID / 命中生产证据 / 证据缺失)或未死
pi_stop_pid_safe() {
  local pid="$1" desc p why still i
  # ⭐ N1 修:复用唯一真源的 pi_valid_pid(它已拒绝 空/非数字/**前导零 0***)✗
  #   原来手写 `''|0|1|*[!0-9]*` 漏了前导零:`kill -0 00` 会被 kill(2) 当 **pid 0 = 本进程组** ✗
  if [ -z "${pid:-}" ]; then
    echo "  (PID 文件为空 ⇒ 本就没在跑)"; return 0     # ⭐ M3:空值不是"非法",而是"无事可做"
  fi
  if ! pi_valid_pid "$pid"; then
    echo "  ⛔ 非法 PID('$pid')⇒ 不可杀(前导零会被 kill(2) 当 pid 0 = 自杀进程组;-1 = 广播)" >&2
    return 1
  fi
  [ "$pid" != "1" ] || { echo "  ⛔ 拒绝 pid=1" >&2; return 1; }
  # ⭐ N3 修:PID 文件里放的不该是 vLLM 引擎/worker 子进程;若是,极可能是陈旧 pidfile 的 pid
  #   被复用(被 reparent 的生产 EngineCore 会让 pi_prod_hit 漏判)⇒ 一律拒杀 ✓
  case "$(pi_cmdline "$pid")" in
    *"VLLM::EngineCore"*|*"VLLM::Worker"*)
      echo "  ⛔ 拒绝:pid=$pid 是 vLLM 引擎/worker 子进程(PID 文件不该指向它;疑 pid 复用)⇒ 不杀" >&2
      return 1 ;;
  esac
  if why="$(pi_prod_hit "$pid")"; then echo "  ⛔ 拒绝:pid=$pid 命中生产证据:$why ⇒ 不杀任何进程" >&2; return 1; fi
  if ! pi_alive "$pid"; then echo "  already dead: pid=$pid"; return 0; fi
  if ! desc="$(pi_descendants "$pid")"; then echo "  ⛔ 后代枚举失败(ps 不可用)⇒ 证据缺失,不杀" >&2; return 1; fi
  for p in $desc; do
    if why="$(pi_prod_hit "$p")"; then echo "  ⛔ 拒绝:后代 pid=$p 命中生产证据:$why ⇒ 不杀任何进程" >&2; return 1; fi
  done
  for p in "$pid" $desc; do kill -TERM "$p" 2>/dev/null || true; done
  i=0
  while [ "$i" -lt 60 ]; do
    still=""; for p in "$pid" $desc; do pi_alive "$p" && still="$still $p"; done
    [ -z "$still" ] && { echo "  stopped pid=$pid(含后代;已验尸 ✓)"; return 0; }
    sleep 1; i=$((i+1))
  done
  for p in "$pid" $desc; do kill -KILL "$p" 2>/dev/null || true; done
  i=0
  while [ "$i" -lt 10 ]; do
    still=""; for p in "$pid" $desc; do pi_alive "$p" && still="$still $p"; done
    [ -z "$still" ] && { echo "  stopped pid=$pid(强杀后已验尸 ✓)"; return 0; }
    sleep 1; i=$((i+1))
  done
  echo "  ⛔ FAILED:发信号后仍存活:$still ⇒ 需人工处置" >&2; return 1
}
