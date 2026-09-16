#!/usr/bin/env bash
# §483: **同机 A/B —— lk-moe(LvLLM 参考实现) vs xiaotu-moe(我们的引擎)**。
#
# 设计原则(为什么这样做):
#   * **两个 arm 都跑在 conda env `lvllm`(lvllm-2.5)里**。这样 vLLM 基座
#     (commit 71888f507a + lk 集成补丁)对两边完全相同,**唯一变量 = CPU MoE 引擎**。
#     若一边用 lvllm、另一边用我们自己的 env,差异就无法归因到引擎(基座不同)。
#   * 模型 / GPU / TP / 请求协议 / 线程数 **全部对齐**;每边只保留"引擎自己的开关"。
#   * 参考侧的参数照 `Lvllm/RELEASE_NOTES.md` + `Lvllm/commands/dsv4_0731_serve_tp2_*.sh`
#     (`LVLLM_MOE_NUMA_ENABLED/LK_THREADS/LK_THREAD_BINDING/LVLLM_GPU_PREFETCH_WINDOW/
#     LVLLM_GPU_PREFILL_MIN_BATCH_SIZE/LVLLM_ENABLE_NUMA_INTERLEAVE/LK_POWER_SAVING`),
#     并按本机(2 节点 NPS、192 物理核)做了**最小必要调整**:线程数由"物理核÷GPU 数"
#     的 96 与我们的规则(每 CCD 4-5 核 ⇒ TP=2 每 rank 12 CCD × 5 = 60)**取同一值 60**,
#     否则线程数本身就会造成两倍差异。`GPU_UTIL/MAXLEN` 也两边一致。
#   * 我们的引擎在 lvllm 环境里靠 **`site-packages` 里一个受 `XTU_PLUGIN=1` 门控的 .pth**
#     在每个解释器(含 spawn 出来的 EngineCore)启动时 import 插件 —— 不需要 pip install,
#     也不影响 arm A(arm A 不设 XTU_PLUGIN ⇒ 插件根本不加载)。
#
# 用法:
#   bash scripts/ab_lvllm_vs_xiaotu.sh a        # arm A(lk_moe)跑完并存档
#   bash scripts/ab_lvllm_vs_xiaotu.sh b        # arm B(xiaotu)跑完并存档
#   bash scripts/ab_lvllm_vs_xiaotu.sh both     # 顺序跑两个 arm
#   bash scripts/ab_lvllm_vs_xiaotu.sh report   # 只对比已存档的两边
#   CS="1 2 4" SPEC=1 THREADS=96 bash ... both  # 改协议
# 产物:report/tuning/raw/ab_lvllm_{a,b}_c<N>.json、report/tuning/raw/ab_lvllm_{a,b}_greedy.json
# 日志:report/tuning/logs/abl_{a,b}.log(服务)、report/tuning/logs/abl_{a,b}.bench.log
#
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV="${ENV:-/home/user/anaconda3/envs/lvllm}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash-0731}"     # 两边用**同一个** served-model-name
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-32768}"
MBT="${MBT:-4096}"
SEQS="${SEQS:-2}"
CS="${CS:-1 4}"
L="${L:-512}"
OUT="${OUT:-128}"
N="${N:-8}"
# 两边**必须相同**的 CPU 线程数(见文件头:参考的"核数÷GPU 数"=96 与我们的 60 取 60)
THREADS="${THREADS:-60}"
# 11 = 只测 MoE 层(留 2 核/worker 的规则由 60 满足);0 = 不设常驻层(**参考发布的配置就是没有常驻层**)
RESIDENT="${RESIDENT:-}"
SPEC="${SPEC:-0}"          # 1 = 加 dspark(两边都加;参考发布参数里有它,但会引入投机解码这个额外变量)
SPEC_CFG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
PORT_A="${PORT_A:-8190}"
PORT_B="${PORT_B:-8191}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"

OUTDIR="$ROOT/report/tuning/logs"; RAW="$ROOT/report/tuning/raw"
mkdir -p "$OUTDIR" "$RAW"
PTH="$ENV/lib/python3.12/site-packages/zz_xiaotu_plugin.pth"

note() { printf '\033[36m[ab]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[ab]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[ab] %s\033[0m\n' "$*" >&2; exit 1; }

# --- arm B 的插件装载器:受 XTU_PLUGIN 门控的 .pth(arm A 不受影响) -------------
install_pth() {
  local line="import os,sys; (os.environ.get(\"XTU_PLUGIN\")==\"1\") and (sys.path.insert(0,r\"$ROOT\"), __import__(\"vllm_xiaotu_moe\"))"
  if [ "${XTU_PLUGIN_PTH_REMOVE:-0}" = "1" ]; then rm -f "$PTH"; note "已移除 $PTH"; return 0; fi
  printf '%s\n' "$line" > "$PTH"
  note "已装插件装载器 $PTH(仅 XTU_PLUGIN=1 时生效)"
  # 自检:门控关闭时不能加载插件;打开时必须能加载
  "$PY" -c "import sys; sys.exit(1 if 'vllm_xiaotu_moe' in sys.modules else 0)" || die "装载器在未设 XTU_PLUGIN 时也加载了插件!"
  XTU_PLUGIN=1 timeout 300 "$PY" -c "import sys; sys.exit(0 if 'vllm_xiaotu_moe' in sys.modules else 1)" \
    || die "装载器在 XTU_PLUGIN=1 时没有加载插件(检查 $PTH)"
  note "装载器自检通过(门控开/关都对)"
}

launch_arm() {   # $1=arm(a|b)
  local arm="$1" port log
  if [ "$arm" = a ]; then port="$PORT_A"; else port="$PORT_B"; fi
  log="$OUTDIR/abl_$arm.log"
  rm -f "$log" "$OUTDIR/abl_$arm.pid"
  local -a args=(
    --model "$CKPT" --served-model-name "$MODEL_NAME"
    --host 0.0.0.0 --port "$port"
    --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_UTIL"
    --max-model-len "$MAXLEN" --max-num-batched-tokens "$MBT" --max-num-seqs "$SEQS"
    --dtype bfloat16 --kv-cache-dtype fp8_ds_mla
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":"VLLM_COMPILE"}'
    --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice --trust-remote-code
    --default-chat-template-kwargs '{"enable_thinking": false}'
    --disable-custom-all-reduce
  )
  [ "$SPEC" = "1" ] && args+=(--speculative-config "$SPEC_CFG")
  # A100/SM80:Tilelang 的 PackSeq 等内核不支持 fp8e4nv,关掉 JIT 预热(与我们 serve_mainline.sh 同因)
  args+=(--kernel-config '{"enable_jit_warmup": false}')

  local -a envs=(
    HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
    VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT" OMP_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$GPUS" CUDA_DEVICE_ORDER=PCI_BUS_ID
    VLLM_EXPERTS_LOAD_DEVICE=cpu
  )
  if [ "$arm" = a ]; then
    # ---- lk_moe(参考)------------------------------------------------------
    envs+=( LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="$THREADS" LK_THREAD_BINDING=CPU_CORE
            LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024
            LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_POWER_SAVING=1 )
    [ -n "$RESIDENT" ] && envs+=( LVLLM_GPU_RESIDENT_MOE_LAYERS="$RESIDENT" )
  else
    # ---- xiaotu(我们的):lvk_moe 关掉,只留我们的插件 ------------------------
    envs+=( XTU_PLUGIN=1 LVLLM_MOE_NUMA_ENABLED=0
            XIAOTU_MOE_THREADS="$THREADS" XIAOTU_MOE_NSLICE_SMALL=0 XIAOTU_MOE_ASYNC=0
            XIAOTU_MOE_SPIN_IDLE_US=0 )
    [ -n "$RESIDENT" ] && envs+=( XIAOTU_MOE_GPU_RESIDENT_LAYERS="$RESIDENT" )
    # 我们插件的 env 桥:显式指定文件 ⇒ 覆盖语义(且不会污染别的脚本)
    printf 'XIAOTU_MOE_THREADS=%s\nXIAOTU_MOE_NSLICE_SMALL=0\nXIAOTU_MOE_ASYNC=0\nXIAOTU_MOE_SPIN_IDLE_US=0\nXIAOTU_MOE_GPU_RESIDENT_LAYERS=%s\n' \
      "$THREADS" "$RESIDENT" > "$OUTDIR/abl_b.envfile"
    envs+=( XIAOTU_ENV_FILE="$OUTDIR/abl_b.envfile" )
  fi

  note "启动 arm $arm: port=$port log=$log threads=$THREADS resident='${RESIDENT:-none}' spec=$SPEC"
  nohup env "${envs[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 &
  echo $! > "$OUTDIR/abl_$arm.pid"

  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      note "arm $arm READY(用时 $((SECONDS - t0))s)"; return 0
    fi
    if ! kill -0 "$(cat "$OUTDIR/abl_$arm.pid")" 2>/dev/null; then
      warn "arm $arm 进程已退出;日志尾部:"; tail -30 "$log"; return 1
    fi
    sleep 10
  done
  warn "arm $arm 等待超时;日志尾部:"; tail -20 "$log"; return 1
}

stop_arm() {    # $1=arm
  local arm="$1" pidf="$OUTDIR/abl_$arm.pid"
  [ -f "$pidf" ] || return 0
  local pid; pid="$(cat "$pidf")"
  note "停止 arm $arm(pid=$pid)"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    kill -CHLD 1 2>/dev/null    # vLLM 的 worker 会被 systemd 收养,需要 reap
    sleep 3
  done
  kill -9 "$pid" 2>/dev/null
  rm -f "$pidf"
  # 等显存真的释放
  for _ in $(seq 1 40); do
    local used; used=$(nvidia-smi --id="$(echo "$GPUS" | cut -d, -f1)" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${used:-9999}" -lt 1024 ] && break
    sleep 5
  done
}

measure_arm() {  # $1=arm
  local arm="$1" port
  if [ "$arm" = a ]; then port="$PORT_A"; else port="$PORT_B"; fi
  local blog="$OUTDIR/abl_$arm.bench.log"
  note "arm $arm 性能测量(bench_lat.sh:L=$L OUT=$OUT N=$N CS='$CS')"
  PORT="$port" MODEL="$MODEL_NAME" TAG="ab_lvllm_$arm" SERVER_TAG="abl_$arm" \
    L="$L" OUT="$OUT" N="$N" CS="$CS" TOKENIZER="$CKPT" ENVBENCH="$ENV" \
    bash scripts/bench_lat.sh > "$blog" 2>&1 || warn "arm $arm bench_lat 有失败项(见 $blog)"
  grep -h '^\[bench_lat\]' "$blog" || true
  note "arm $arm 正确性探针(probe_greedy.py → ab_lvllm_${arm}_greedy.json)"
  timeout 3600 "$PY" scripts/probe_greedy.py "$RAW/ab_lvllm_${arm}_greedy.json" "$port" "$MODEL_NAME" \
    >> "$blog" 2>&1 || warn "arm $arm greedy 探针失败(见 $blog)"
  # arm A 不把 JSON 直接比对(不同引擎的逐 token 差异要单独看),这里只确认文件有内容
  [ -s "$RAW/ab_lvllm_${arm}_greedy.json" ] && note "  greedy 存档 OK" || warn "  greedy 存档为空"
}

report() {
  "$PY" - "$RAW" <<'PY'
import json, glob, os, sys
raw = sys.argv[1]
rows = {}
for arm in ("a", "b"):
    for f in sorted(glob.glob(os.path.join(raw, f"ab_lvllm_{arm}_c*.json"))):
        try: d = json.load(open(f))
        except Exception: continue
        c = d.get("C") or d.get("max_concurrency") or "?"
        rows[(arm, str(c))] = d
    g = os.path.join(raw, f"ab_lvllm_{arm}_greedy.json")
    if os.path.exists(g):
        try: rows[(arm, "greedy")] = json.load(open(g))
        except Exception: pass
def g(d, *keys):
    for k in keys:
        if k in d: return d[k]
    return None
print()
print("=== lk-moe(arm A) vs xiaotu-moe(arm B) ===")
print(f"{'C':>4} | {'A agg tok/s':>11} {'A TPOT ms':>9} {'A TTFT ms':>9} | {'B agg tok/s':>11} {'B TPOT ms':>9} {'B TTFT ms':>9} | {'B/A agg':>7}")
cs = sorted({c for (a, c) in rows if c != "greedy"}, key=lambda x: int(x))
for c in cs:
    A, B = rows.get(("a", c)), rows.get(("b", c))
    if not A or not B: continue
    a_agg, b_agg = g(A, "agg"), g(B, "agg")
    ratio = f"{b_agg/a_agg:.2f}x" if (a_agg and b_agg) else "-"
    print(f"{c:>4} | {a_agg or 0:>11.2f} {g(A,'tpot') or 0:>9.2f} {g(A,'ttft') or 0:>9.0f} | "
          f"{b_agg or 0:>11.2f} {g(B,'tpot') or 0:>9.2f} {g(B,'ttft') or 0:>9.0f} | {ratio:>7}")
print("(agg = 服务端聚合 output tok/s;B/A > 1 ⇒ xiaotu 更快)")
ga, gb = rows.get(("a", "greedy")), rows.get(("b", "greedy"))
if ga and gb:
    print("\n--- 正确性(greedy,temperature=0,seed=1234;两边同一份 prompt)---")
    same_txt = same_tok = 0
    for k in sorted(set(ga) | set(gb)):
        ta, tb = (ga.get(k) or {}).get("text", ""), (gb.get(k) or {}).get("text", "")
        ok = ta == tb
        same_txt += ok
        # token 级前缀一致长度(用文本近似:逐字符前缀)
        n = 0
        for x, y in zip(ta, tb):
            if x != y: break
            n += 1
        same_tok += (n > 0)
        print(f"  {k:8s} text_equal={ok}  共同前缀={n}/{max(len(ta),len(tb))}  "
              f"A={ta[:48]!r}")
        if not ok: print(f"           B={tb[:48]!r}")
    print(f"  ⇒ 文本完全一致的 prompt: {same_txt}/{len(set(ga)|set(gb))}")
else:
    print("\n(缺少 greedy 存档,先跑 arm a/b)")
PY
}

case "${1:-both}" in
  a)      install_pth; launch_arm a && measure_arm a; stop_arm a ;;
  b)      install_pth; launch_arm b && measure_arm b; stop_arm b ;;
  both)   install_pth
          launch_arm a && measure_arm a; stop_arm a
          launch_arm b && measure_arm b; stop_arm b
          report ;;
  report) report ;;
  *)      die "用法:$0 {a|b|both|report}" ;;
esac
