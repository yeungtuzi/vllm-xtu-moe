#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# random 数据集 12 格基准:{(短 128, 长 16384)} × {C=1, C=2} × 3 个模型。
#
# 为什么用 random 而不是 ShareGPT(2026-09-21 用户决定):评测目标是 **prefill/decode
# 吞吐**,需要**可控的输入长度**;ShareGPT 的长度分布不可控,无法给出"L=16384"这一列。
# prefix cache 保持默认开启;每格用**不同 seed**,避免 C=1/C=2 之间互相吃到缓存。
#
# 本脚本相对 /tmp/rnd.sh(2026-09-21 那次失败 campaign)修掉了四处自伤错误:
#
#   1. **KV cap 必须 ≥ max(MAXLEN, SEQS×L_max)**。原脚本给 MiMo 传 4.0 GiB,
#      而 max_model_len=20480 需要 4.84 GiB ⇒ 引擎**初始化就失败**;给 V4.1 传
#      0.5 GiB 而 32768 需要 0.93 GiB ⇒ 同样失败。这里按**实测 KV 标尺**
#      (tok/GiB)反算并在启动前 **assert**,不满足立刻退出。
#   2. **`waitp` 的早退检测**。原脚本只 `kill -0 "$wrapper_pid"`,而
#      `serve_glm53_mainline.sh` 在非 FOREGROUND 模式会 **fork 出服务后自己退出**
#      ⇒ 该判据恒为假/恒为真都不可靠;而且服务早死了它还会把整个超时轮询完
#      (实测白等 45 分钟)。现在改为:轮询 curl + 检查**服务 pidfile** + 扫日志里的
#      致命初始化错误,任一命中立即失败。
#   3. **V4.1 的 PYTHONPATH**。原脚本没传 ⇒ `serve_v41.sh` 自行解析到了
#      `process_data/ref/repos/vllm-mainline`(**rebase 前的旧树**),数据不能用来
#      验证 rebase。现在三个模型全部显式指向 rebase 树。
#   4. **清场 kill 与观测分离**。原脚本/诊断脚本里 `kill -9 $(nvidia-smi ...)` 会在
#      观测窗口内把 worker 杀掉,而**只有 worker 占显存** ⇒ 存活的 EngineCore 会把它
#      记成 "Worker proc died unexpectedly",制造出与被测对象无关的"崩溃"。
#      本脚本只在**每个阶段加载之前**清场,cell 运行与死亡判定期间**绝不 kill**。
#
# 用法:
#   bash scripts/bench_random_12cells.sh                    # 三个模型全跑
#   MODELS=glm bash scripts/bench_random_12cells.sh         # 只跑一个
#   LONG=16384 SHORT=128 CONC="1 2" N=8 bash scripts/...    # 调参
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# ⭐ 2026-10-07 审计修:归属证据一律走【唯一真源】(pi_prod_hit / pi_alive …)✓
# shellcheck source=lib_proc_identity.sh
. "$ROOT/scripts/lib_proc_identity.sh"

# ⭐ 2026-10-07 审计修(blocker B):就绪与"起服务"都必须由【端口属主】确认 ——
#   裸 curl 成功会被"别人的服务(生产!)"满足 ⇒ 对生产发压测并把结果记成本阶段的数据 ✗
_port_owned_by() {   # $1=port $2=期望 pid(本阶段 pidfile 里的 pid);rc=0 ⇒ 属主就是他 ✓
  local port="$1" want="$2" p
  pi_valid_pid "$want" || return 1
  pi_port_listeners "$port"
  for p in $PI_PORT_PIDS; do [ "$p" = "$want" ] && return 0; done
  return 1
}

# 起服务【之前】断言:该端口没有【陌生属主】(属主是本阶段自己的旧 pid ⇒ 最多等 10s 释放)✓
_assert_port_no_foreign_owner() {   # $1=port $2=pidfile $3=tag
  local port="$1" pidf="$2" tag="$3" p sp="" i=0
  pi_port_listeners "$port"
  [ -z "$PI_PORT_PIDS" ] && return 0
  [ -f "$pidf" ] && sp="$(cat "$pidf" 2>/dev/null || true)"
  if [ -n "$sp" ]; then
    while [ $i -lt 10 ]; do
      _port_owned_by "$port" "$sp" || break      # 已不再由"我们的旧 pid"占用 ⇒ 退出等待
      sleep 1; i=$((i+1))
    done
    pi_port_listeners "$port"
    [ -z "$PI_PORT_PIDS" ] && return 0
  fi
  for p in $PI_PORT_PIDS; do
    if [ -n "$sp" ] && [ "$p" = "$sp" ]; then
      echo "  ⛔ $tag: :$port 仍被【本 tag 的旧实例 pid=$sp】占用 ⇒ 拒绝再起(先停它)✗"
      return 1
    fi
    echo "  ⛔ $tag: :$port 有【陌生属主】pid=$p ⇒ 拒绝起服务(可能是生产/别的实例)✗"
    return 1
  done
  return 0
}

ENVDIR="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENVDIR/bin/python"
export PATH="$ENVDIR/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 FLASHINFER_DISABLE_VERSION_CHECK=1
# 全部指向 rebase 后的树(修掉原脚本 V4.1 漏传 PYTHONPATH 的问题)
export XTU_TREE="${XTU_TREE:-/home/user/lvllm/process_data/ref/repos/vllm-mainline}"

OUT="${OUT:-/tmp/rnd2}"; mkdir -p "$OUT"
LOGD="$ROOT/logs"; mkdir -p "$LOGD"
SHORT="${SHORT:-128}"; LONG="${LONG:-16384}"; CONC="${CONC:-1 2}"; N="${N:-8}"
MODELS="${MODELS:-glm mimo v41}"
MAXSEQS=2

CKPT_G=/home/user/.cache/modelscope/models/ZhipuAI--GLM-5.3-Flash/snapshots/master
CKPT_M=/home/user/.cache/modelscope/models/XiaomiMiMo--MiMo-V2.5/snapshots/master
CKPT_V=/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master
MAXLEN_G=32768; MAXLEN_M=20480; MAXLEN_V=32768

# ---- 实测 KV 标尺(tokens per GiB)------------------------------------------
# GLM 来自 scripts/serve_glm53_mainline.sh 的既有标尺(6 GiB ↔ 293,651 tok)。
# MiMo/V4.1 由 2026-09-21 引擎报错信息反算:
#   MiMo : "max seq len (20480), 4.84 GiB needed" / "estimated max model length 16928" @4.0 GiB
#   V4.1 : "max seq len (32768), 0.93 GiB needed" / "estimated max model length 17472" @0.5 GiB
RULER_GLM=48942
RULER_MIMO=4232
RULER_V41=34944

# 每个模型的"已知可用下限":按标尺算出来的 cap 若低于这个值,取这个值。
# GLM 的 2 GiB 是生产与全部实验验证过的值(且 GPU 预填充 staging 的 preflight 也按
# 这个预算调过),不为了省显存把它压到 0.84 GiB。
FLOOR_GLM=2147483648
FLOOR_MIMO=0
FLOOR_V41=0

GIB=1073741824

assert_kv_cap() { # label cap_bytes needed_tok ruler
  "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import sys
label, cap, need, ruler = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
gib = 1024 ** 3
ok = cap >= need / ruler * gib
print(f"[cfg] {label}: cap={cap/gib:.2f} GiB, 至少需要 {need/ruler:.2f} GiB "
      f"({need} tok @ {ruler:.0f} tok/GiB) -> {'OK' if ok else '不足!!'}")
sys.exit(0 if ok else 1)
PY
}

# resolve_kv_cap <label> <ruler> <maxlen> <seqs> <l_max> <floor> -> 打印最终 cap
#
# 关键:断言**不是**针对自动算出来的值(那个按构造必然 ≥ need,断言是空的),
# 而是针对**用户显式传进来的 `KV_CACHE_BYTES`**。2026-09-21 那次 campaign 恰恰是
# 手工给 MiMo 传了 4.0 GiB、给 V4.1 传了 0.5 GiB,都小于 `max_model_len` 所需 ⇒
# 引擎初始化失败,而脚本还空等了 45/90 分钟。这里显式传入时必须先过断言。
resolve_kv_cap() {
  local label=$1 ruler=$2 maxlen=$3 seqs=$4 lmax=$5 floor=$6
  local need=$(( maxlen > seqs * lmax ? maxlen : seqs * lmax ))
  local auto cap
  auto=$("$PY" -c "import sys;r,m,s,l,f=map(float,sys.argv[1:6]);print(int(max(max(m,s*l)/r*1024**3*1.10,f)))" \
         "$ruler" "$maxlen" "$seqs" "$lmax" "$floor")
  if [ -n "${KV_CACHE_BYTES:-}" ]; then
    cap="$KV_CACHE_BYTES"
    echo "[cfg] $label: 使用显式 KV_CACHE_BYTES=$cap (自动算得 $auto)" >&2
  else
    cap="$auto"
  fi
  assert_kv_cap "$label" "$cap" "$need" "$ruler" >&2 || return 1
  echo "$cap"
}

# ---- 就绪 / 早退检测(修掉原脚本的 waitp)-------------------------------------
# 判据三合一:① curl 就绪;② 服务 pidfile 里的进程消失;③ 日志里出现致命初始化错误。
wait_ready() { # port pidfile log timeout_s
  local port=$1 pidf=$2 logf=$3 tmo=${4:-1800} i=0
  local sp=""
  while [ $i -lt $(( tmo / 5 )) ]; do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      # ⭐ 2026-10-07 审计修(blocker B):curl 成功【必须】同时确认端口属主 == 本阶段 pidfile 的 pid;
      #   否则可能是生产/别的实例在应答(实测路径:压测打到生产并记成 glm53_rnd)✗
      sp=$(cat "$pidf" 2>/dev/null || true)
      if _port_owned_by "$port" "$sp"; then
        echo "  ready=$((i*5))s(端口属主 pid=$sp ✓)"; return 0
      fi
      echo "  ⚠️ :$port 有服务应答,但属主不是本阶段 pidfile(pid='${sp:-空}' 属主='$(printf '%s ' $PI_PORT_PIDS)')⇒ 不认就绪,继续等"
    fi
    if [ -f "$pidf" ]; then
      # ⭐ 2026-10-07 审计修(major):pidfile 内容先过 pi_valid_pid(拒绝空/非数字/`0*`)
      #   ⇒ 非法内容一律【不判死、不猜目标】(fail-safe;超时自然退出)✓
      sp=$(cat "$pidf" 2>/dev/null || true)
      if pi_valid_pid "$sp" && ! pi_alive "$sp"; then
        echo "  ✗ 服务进程($sp)已退出;日志尾部:"; tail -12 "$logf" 2>/dev/null | sed 's/^/    /'
        return 2
      fi
    fi
    # 【2026-09-21 修】致命错误要同时查**包装脚本日志**:各模型的 serve 脚本把服务端
    # 日志写到不同位置(`serve_v41.sh` 写 `dev-docs/report/tuning/logs/<tag>.log`,
    # 不是 `logs/<tag>.log`),只查一个路径会读到空文件、检测静默失效。
    # 包装脚本自己也会打印 "[v41] server exited early; tail:",这一并抓到。
    if grep -qE "Engine core initialization failed|EngineDeadError|CUDA out of memory|larger than the available KV cache|server exited early" "$logf" "${logf}.wrapper" 2>/dev/null; then
      echo "  ✗ 启动即失败(日志命中致命错误):"
      grep -nE "Engine core initialization failed|ValueError|CUDA out of memory|larger than the available KV cache" "$logf" "${logf}.wrapper" 2>/dev/null | tail -4 | sed 's/^/    /'
      return 3
    fi
    sleep 5; i=$((i+1))
  done
  echo "  ✗ 等待就绪超时(${tmo}s)"; return 4
}

freegpu() { # 只在"加载之前"调用 —— 绝不在 cell 运行/判定期间调用
  local f p why
  # 【2026-09-21 修孤儿】先按 pidfile 杀掉已知阶段的残留。只靠 nvidia-smi 会漏掉
  # **"已启动但还在加载、尚未占显存"** 的服务:上一个阶段失败退出时它正在加载,
  # 于是下一阶段的 freegpu 看不见它;等它加载完就变成占着 GPU 的孤儿。
  # 实测:GLM 阶段被陈旧 pidfile 误判失败后,它启动的服务在 GPU 0/1 上活了 12 分钟,
  # 差点让后面用 GPU 0/1 的 V4.1 阶段撞车。
  for f in "$LOGD"/glm53_rnd.pid "$LOGD"/mimo_rnd.pid "$LOGD"/v41_rnd.pid; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    # ⭐ 2026-10-07 审计修(major):PID 文件内容先过 pi_valid_pid ——
    #   `kill -9 0` = 对本进程组发 SIGKILL(自杀);`kill -9 -1` = 广播级误杀 ⇒ 一律拒绝 ✗
    if ! pi_valid_pid "$p"; then
      echo "  ⛔ freegpu 拒绝:PID 文件 $f 内容非法('${p:-空}')⇒ 不可杀,中止本轮 ✗"
      return 1
    fi
    # ⭐ 2026-10-07 审计修:杀【本脚本 PID 文件派生】的 pid 之前先做归属校验;
    #   命中生产证据 ⇒ 打印原因 + 整轮中止(fail-closed:宁可不跑,也不误杀生产)✓
    if why="$(pi_prod_hit "$p")"; then
      echo "  ⛔ freegpu 拒绝:pid=$p(来自 $f)命中生产证据:$why ⇒ 中止本轮,不杀任何进程 ✗"
      return 1
    fi
    if pi_alive "$p"; then
      # ⭐ 2026-10-07 审计修(major):进程组号也过数值守卫(`kill -9 -0` = 打自己的进程组 ⇒ 自杀)✗
      # ⛔ 2026-10-07 独立审计(blocker):不再 `kill -9 -PGID` —— 进程组成员 ≠ 我们的后代,
      #   生产 API 的 pgid!=pid 且组长可能已 DEAD ⇒ 组里会混进别人的进程 ✗
      #   ⇒ 只 kill 本实例 pid 文件里的那个 pid;残留下游由"等显存"超时如实报错 ✓
      if pi_valid_pid "$p"; then kill -9 "$p" 2>/dev/null; fi
      kill -9 "$p" 2>/dev/null
    fi
  done
  # ⛔ 2026-10-07 审计修:删除 `for p in $(nvidia-smi --query-compute-apps=pid …); do kill -9 "$p"; done`
  #   —— 它把【全机所有 GPU 进程】当目标,会杀掉同机共存的其它服务
  #      (LMCache 服务自身也用 CUDA,已被误杀两次)✗✗
  #   ⇒ 改为【只等待显存释放】;超时 ⇒ 醒目报错 + 中止,而不是去杀别的进程 ✓
  local i=0
  while [ $i -lt 60 ]; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || return 0
    sleep 5; i=$((i+1))
  done
  echo "  ⛔ freegpu 超时:GPU 上仍有【不是本脚本 PID 文件派生】的进程占卡 ⇒ 中止本轮,不再按模式补杀 ✗"
  return 1
}

bench_cell() { # port ckpt served L C tag backend extra...
  local port=$1 ckpt=$2 served=$3 L=$4 C=$5 tag=$6 bk=$7; shift 7
  local ep; [ "$bk" = openai-chat ] && ep=/v1/chat/completions || ep=/v1/completions
  local seed=$(( L*7 + C*131 + 17 ))          # 与原 campaign 一致:每格不同 seed
  local logf="$OUT/${tag}_L${L}_C${C}.log"
  timeout 2400 "$PY" -m vllm.entrypoints.cli.main bench serve \
    --backend "$bk" --endpoint "$ep" --host 127.0.0.1 --port "$port" \
    --model "$ckpt" --served-model-name "$served" \
    --dataset-name random --random-input-len "$L" --random-output-len 128 \
    --num-prompts "$N" --max-concurrency "$C" --seed "$seed" \
    --save-result --result-dir "$OUT" --result-filename "${tag}_L${L}_C${C}.json" \
    "$@" > "$logf" 2>&1
  local rc=$?
  local ok; ok=$(grep -c "Successful requests" "$logf" 2>/dev/null || echo 0)
  local t; t=$(grep -oE "Mean TTFT \(ms\): *[0-9.]+" "$logf" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  echo "    ${tag} L=${L} C=${C}: rc=$rc TTFT=${t:-?}ms ok=${ok}"
  # 判定只用日志,不用 kill -0
  if grep -q "ConnectionRefusedError" "$logf" 2>/dev/null; then
    echo "    ✗✗ 服务端拒绝连接(已在运行中死亡)"; return 9
  fi
  return 0
}

cells_for() { # port ckpt served tag backend [extra...]
  local port=$1 ckpt=$2 served=$3 tag=$4 bk=$5; shift 5
  local L C
  for L in "$SHORT" "$LONG"; do
    for C in $CONC; do
      bench_cell "$port" "$ckpt" "$served" "$L" "$C" "$tag" "$bk" "$@" || return 1
    done
  done
}

# 每个模型自己的服务端日志(死亡判定只 grep 它)
srv_log() { echo "$LOGD/$1.log"; }

run_glm() {
  # ⭐ 2026-10-07 审计修(blocker B):GLM 阶段原来用 **生产端口 8070** ⇒ 生产在跑时会对生产发
  #   16K 压测并把结果记成 glm53_rnd ✗ ⇒ 改用非保留端口 **8072**(mimo=8073 / v41=8077 不动;
  #   8071 已被 probe_after_build/serve_mainline 占用,8095-8098 有历史 cellA-D 痕迹 ⇒ 8072 零引用)✓
  local tag=glm53_rnd port=8072 cap
  cap=$(resolve_kv_cap GLM $RULER_GLM $MAXLEN_G $MAXSEQS "$LONG" $FLOOR_GLM) || return 1
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  # ⭐ 2026-10-07 审计修:清场失败(命中生产证据/显存等待超时)⇒ 中止本阶段,不继续起服务 ✓
  freegpu || return 1
  # ⭐ 2026-10-07 审计修(blocker B-3):起服务前断言 :$port 没有陌生属主 ✓
  _assert_port_no_foreign_owner "$port" "$LOGD/$tag.pid" "$tag" || return 1
  # 【必须】先删陈旧 pidfile:serve 脚本要等加载/预热完才写它,而 TAG 可能沿用上一次
  # 运行(同名),于是加载期读到的还是**上一次**的 pid ⇒ 被误判为"服务进程已退出"。
  # 2026-09-21 实测踩到:GLM 阶段因此被误判失败并跳过。
  rm -f "$LOGD/$tag.pid"
  setsid env PYTHONPATH="$XTU_TREE" TAG="$tag" PORT="$port" \
    GPU_UTIL=0.82 SPEC_K=0 SEQS=$MAXSEQS MAXLEN=$MAXLEN_G MBT=4096 \
    KV_CACHE_BYTES="$cap" bash scripts/serve_glm53_mainline.sh > "$(srv_log $tag).wrapper" 2>&1 &
  wait_ready "$port" "$LOGD/$tag.pid" "$(srv_log $tag)" 1800 || return 1
  cells_for "$port" "$CKPT_G" GLM-5.3-Flash glm openai-chat
}

run_mimo() {
  local tag=mimo_rnd port=8073 cap
  cap=$(resolve_kv_cap MiMo $RULER_MIMO $MAXLEN_M $MAXSEQS "$LONG" $FLOOR_MIMO) || return 1
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  freegpu || return 1     # ⭐ 2026-10-07 审计修:清场失败 ⇒ 中止本阶段(见 run_glm 说明)
  _assert_port_no_foreign_owner "$port" "$LOGD/$tag.pid" "$tag" || return 1   # blocker B-3
  rm -f "$LOGD/$tag.pid"        # 见 run_glm 里的说明:陈旧 pidfile 会误判早退
  setsid env CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$XTU_TREE" VLLM_EXPERTS_LOAD_DEVICE=cpu XIAOTU_MOE_THREADS=60 \
    XIAOTU_SPEC_DECODE=1 VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=1500 \
    "$PY" -m vllm.entrypoints.openai.api_server --model "$CKPT_M" --served-model-name MiMo-V2.5 \
    --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 \
    --kv-cache-dtype bfloat16 --max-model-len "$MAXLEN_M" --max-num-batched-tokens 16384 \
    --max-num-seqs $MAXSEQS --kv-cache-memory "$cap" --gpu-memory-utilization 0.72 \
    --language-model-only --trust-remote-code \
    --speculative-config "{\"method\":\"mtp\",\"model\":\"$CKPT_M\",\"num_speculative_tokens\":1}" \
    > "$(srv_log $tag)" 2>&1 &
  echo $! > "$LOGD/$tag.pid"
  wait_ready "$port" "$LOGD/$tag.pid" "$(srv_log $tag)" 2700 || return 1
  cells_for "$port" "$CKPT_M" MiMo-V2.5 mimo openai-chat
}

run_v41() {
  local tag=v41_rnd port=8077 cap
  cap=$(resolve_kv_cap V4.1 $RULER_V41 $MAXLEN_V $MAXSEQS "$LONG" $FLOOR_V41) || return 1
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  freegpu || return 1     # ⭐ 2026-10-07 审计修:清场失败 ⇒ 中止本阶段(见 run_glm 说明)
  _assert_port_no_foreign_owner "$port" "$LOGD/$tag.pid" "$tag" || return 1   # blocker B-3
  rm -f "$LOGD/$tag.pid"        # 见 run_glm 里的说明:陈旧 pidfile 会误判早退
  setsid env PYTHONPATH="$XTU_TREE" TAG="$tag" PORT="$port" GPUS=0,1 TP=2 \
    MAXLEN=$MAXLEN_V MAXSEQS=$MAXSEQS MBT=16384 GPU_UTIL=0.85 \
    SPEC=1 LOAD=auto PREFIX_CACHE=1 KV_CACHE_BYTES="$cap" \
    bash scripts/serve_v41.sh > "$(srv_log $tag).wrapper" 2>&1 &
  wait_ready "$port" "$LOGD/$tag.pid" "$(srv_log $tag)" 5400 || return 1
  # DSV4.1 快照没有 chat_template ⇒ 必须走 /v1/completions 并跳过模板
  cells_for "$port" "$CKPT_V" DeepSeek-V4.1-Flash v41 openai --skip-chat-template
}

echo "══ random 12 格基准 ══ ($(date -Is))"
echo "  短=$SHORT 长=$LONG 并发=[$CONC] N=$N 模型=[$MODELS] 树=$XTU_TREE"
RC=0
for m in $MODELS; do
  echo "── $m ──"
  case "$m" in
    glm)  run_glm  || { echo "  [$m] 阶段失败(已提前退出,不空等)"; RC=1; } ;;
    mimo) run_mimo || { echo "  [$m] 阶段失败"; RC=1; } ;;
    v41)  run_v41  || { echo "  [$m] 阶段失败"; RC=1; } ;;
    *) echo "  未知模型 $m"; RC=1 ;;
  esac
done
echo "══ 完成 $(date -Is) rc=$RC ══"
echo "结果 JSON 在 $OUT/"
[ $RC = 0 ] && echo "RND12_OK" || echo "RND12_PARTIAL"
