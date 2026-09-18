#!/usr/bin/env bash
# ============================================================================
# 「等效参数」同机 A/B:lvllm + lk-moe(arm A) vs 主线 vLLM + xiaotu-moe(arm B)
#
# 为什么这样设计(2026-09-18,用户要求"完整对比,既比 prefill 短/长,也比 output 短/长")
# ---------------------------------------------------------------------------
# * **两个 arm 跑在同一个 conda env(`lvllm`,vllm 2.5.0 基座)里**。这样 vLLM 基座对两边
#   完全相同,**唯一变量 = CPU MoE 引擎**。若一边用 lvllm env、另一边用我们自己的 env,
#   差异无法归因到引擎(基座不同、模型加载路径不同)。arm B 的插件由 site-packages 里
#   受 `XTU_PLUGIN=1` 门控的 .pth 装载(不 pip install,也不影响 arm A)。
# * **参数以 lvllm 自己的 V4.1 启动脚本为准**(`Lvllm/commands/dsv41_serve_tp2_3090_dspark.sh`),
#   只做本机必需的最小调整:TP/MAXLEN/MBT/SEQS/GPU_UTIL/kv-cache-dtype/compilation 全部照抄;
#   线程数取 **60 两边相同**(参考机是 96c、我们是 192c;§483 已论证取同一值才公平)。
#   两个 arm 各用自己的 busy-wait 策略(arm A `LK_POWER_SAVING=1`,
#   arm B `XIAOTU_MOE_SPIN_IDLE_US=300` —— 这是各自文档推荐的稳态值)。
# * **GPU 预填充两边都关**(`GPU_PREFILL=0`),这样量到的是**纯 CPU MoE 引擎**的差异;
#   要量"各arm最佳配置"用 `GPU_PREFILL=1`。
# * 客户端用官方 `vllm bench serve`(`--dataset-name custom` + `--skip-chat-template`),
#   prompt 用 `--dataset-name random` 的固定 token 长度(短=256 / 长=8192),
#   输出用 `--ignore-eos` 强制短=32 / 长=1024。**为什么用 random 而不是 custom/DSH 回放**:
#   ① 参考 env 里**没装 pandas**,`CustomDataset` 会挂在 `pd.read_json`;
#   ② random 让两个 arm 拿到**逐字节相同**的 prompt,是更干净的对照(§483 也用 random)。 ⇒ 得到 2×2 矩阵:
#       prefill 短 × output 短 | prefill 短 × output 长
#       prefill 长 × output 短 | prefill 长 × output 长
#   报三个各自独立的量:`output_throughput` / `mean_ttft_ms` / `mean_tpot_ms`。
#
# 用法:
#   bash scripts/ab_lvllm_matrix.sh            # 两个 arm 顺序跑完(每个 arm 只起一次服务)
#   GPU_PREFILL=1 bash scripts/ab_lvllm_matrix.sh
#   REPS=6 CS="1 4" bash scripts/ab_lvllm_matrix.sh
#   ARMS="a" bash scripts/ab_lvllm_matrix.sh  # 只跑参考侧
#
# 产物:report/tuning/raw/abmx_{a,b}_p<P>_o<L>_c<C>.json
#       日志:report/tuning/logs/abmx_{a,b}.log、raw/abmx_{a,b}.bench.log
#
# License: Apache-2.0
# ============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV="${ENV:-/home/user/anaconda3/envs/lvllm}"
PY="$ENV/bin/python"
CKPT="${CKPT:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master}"
MODEL_NAME="${MODEL_NAME:-dsv41-xtu}"     # 两边同一个 served-model-name
TP="${TP:-2}"
GPUS="${GPUS:-0,1}"
GPU_UTIL="${GPU_UTIL:-0.95}"              # = lvllm 的 dsv41 脚本
MAXLEN="${MAXLEN:-65536}"                 # = lvllm 的 dsv41 脚本
MBT="${MBT:-8192}"                        # = lvllm 的 dsv41 脚本
SEQS="${SEQS:-2}"                         # = lvllm 的 dsv41 脚本
THREADS="${THREADS:-60}"                  # 两边相同(见文件头)
PROMPTS="${PROMPTS:-256 8192}"
OUTPUTS="${OUTPUTS:-32 1024}"
CS="${CS:-1}"
REPS="${REPS:-4}"
GPU_PREFILL="${GPU_PREFILL:-0}"           # 0 = 两边都关(纯 CPU 引擎对比)
ARMS="${ARMS:-a b}"
PORT_A="${PORT_A:-8190}"
PORT_B="${PORT_B:-8191}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
ATAG="${ATAG:-}"                       # 变体扫描时给结果文件名加后缀,避免覆盖基线

OUTDIR="$ROOT/report/tuning/logs"; RAW="$ROOT/report/tuning/raw"
mkdir -p "$OUTDIR" "$RAW"
PTH="$ENV/lib/python3.12/site-packages/zz_xiaotu_plugin.pth"

note() { printf '\033[36m[abmx]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[abmx]\033[0m %s\n' "$*"; }

install_pth() {
  local line="import os,sys; (os.environ.get(\"XTU_PLUGIN\")==\"1\") and (sys.path.insert(0,r\"$ROOT\"), __import__(\"vllm_xiaotu_moe\"))"
  printf '%s\n' "$line" > "$PTH"
  "$PY" -c "import sys; sys.exit(1 if 'vllm_xiaotu_moe' in sys.modules else 0)" \
    || { warn "装载器在未设 XTU_PLUGIN 时也加载了插件!"; return 1; }
  XTU_PLUGIN=1 timeout 300 "$PY" -c "import sys; sys.exit(0 if 'vllm_xiaotu_moe' in sys.modules else 1)" \
    || { warn "装载器在 XTU_PLUGIN=1 时没有加载插件"; return 1; }
  note "插件装载器自检通过(门控开/关都对)"
}

launch_arm() {   # $1=arm
  local arm="$1" port log
  if [ "$arm" = a ]; then port="$PORT_A"; else port="$PORT_B"; fi
  log="$OUTDIR/abmx_$arm.log"
  rm -f "$log"; : > "$log"

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
    --kernel-config '{"enable_jit_warmup": false}'
  )

  local -a envs=(
    HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
    VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT" OMP_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$GPUS" CUDA_DEVICE_ORDER=PCI_BUS_ID
    VLLM_EXPERTS_LOAD_DEVICE=cpu
  )
  if [ "$arm" = a ]; then
    envs+=( LVLLM_MOE_NUMA_ENABLED=1 LK_THREADS="$THREADS" LK_THREAD_BINDING=CPU_CORE
            LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_POWER_SAVING=1 LVLLM_EMBEDDING_NUMA_ENABLED=1
            VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ENGRAM_DROP_PAGE_CACHE=0
            XTU_PLUGIN=0 XIAOTU_MAINLINE_SHIMS=0 XIAOTU_OOT_OVERRIDE=0 )
    if [ "$GPU_PREFILL" = "1" ]; then envs+=( LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1024 )
    else envs+=( LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=1000000000 ); fi
  else
    envs+=( XTU_PLUGIN=1 LVLLM_MOE_NUMA_ENABLED=0
            XIAOTU_MOE_THREADS="$THREADS" XIAOTU_MOE_NSLICE_SMALL=0 XIAOTU_MOE_ASYNC=0
            XIAOTU_MOE_SPIN_IDLE_US=300 XIAOTU_OOT_OVERRIDE=0 )
    if [ "$GPU_PREFILL" = "1" ]; then envs+=( VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=1024 )
    else envs+=( VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=1000000000 ); fi
    # 我们的引擎默认(§5.9 出货配置)。B_* 环境变量可逐项覆盖,用来做"哪个旋钮造成差距"的扫描。
    local B_NSLICE="${B_NSLICE_SMALL:-0}" B_ASYNC="${B_ASYNC:-0}" B_SPIN="${B_SPIN:-300}"
    local B_THREADS="${B_THREADS:-$THREADS}"
    printf 'XIAOTU_MOE_THREADS=%s\nXIAOTU_MOE_NSLICE_SMALL=%s\nXIAOTU_MOE_ASYNC=%s\nXIAOTU_MOE_SPIN_IDLE_US=%s\n' \
      "$B_THREADS" "$B_NSLICE" "$B_ASYNC" "$B_SPIN" > "$OUTDIR/abmx_b.envfile"
    envs+=( XIAOTU_ENV_FILE="$OUTDIR/abmx_b.envfile"
            XIAOTU_MOE_THREADS="$B_THREADS" XIAOTU_MOE_NSLICE_SMALL="$B_NSLICE"
            XIAOTU_MOE_ASYNC="$B_ASYNC" XIAOTU_MOE_SPIN_IDLE_US="$B_SPIN" )
    [ -n "${B_EXTRA_ENV:-}" ] && envs+=( $B_EXTRA_ENV )
  fi

  note "启动 arm $arm (port=$port threads=$THREADS gpu_prefill=$GPU_PREFILL) → $log"
  # ⚠️ 必须在仓库外启动:`python -m` 会把 CWD 放进 sys.path[0],而仓库根有
  # `vllm_xiaotu_moe.egg-info/` ⇒ importlib.metadata 会当成已安装发行版 ⇒
  # load_general_plugins() 自动加载我们的插件,arm A 就被污染了(§483 踩过)。
  ( cd /tmp && nohup env "${envs[@]}" "$PY" -m vllm.entrypoints.openai.api_server "${args[@]}" > "$log" 2>&1 & echo $! > "$OUTDIR/abmx_$arm.pid" )

  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
    if curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      note "arm $arm READY($((SECONDS - t0))s)"
      if grep -qE 'vllm-xtu-moe' "$log"; then
        if [ "$arm" = a ]; then warn "!! arm A 日志出现 vllm-xtu-moe ⇒ 被我们的插件污染,本次作废"; return 1; fi
        note "  arm B 纯度 OK(插件已加载)"
      else
        if [ "$arm" = b ]; then warn "!! arm B 没加载插件 ⇒ 跑的不是 xiaotu 引擎"; return 1; fi
        note "  arm A 纯度 OK(没有我们的插件)"
      fi
      return 0
    fi
    if ! kill -0 "$(cat "$OUTDIR/abmx_$arm.pid" 2>/dev/null)" 2>/dev/null; then
      warn "arm $arm 进程退出;日志尾部:"; tail -25 "$log"; return 1
    fi
    sleep 10
  done
  warn "arm $arm 超时"; tail -20 "$log"; return 1
}

stop_arm() {
  local pidf="$OUTDIR/abmx_$1.pid"
  [ -f "$pidf" ] || return 0
  local pid; pid="$(cat "$pidf")"
  kill "$pid" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; kill -CHLD 1 2>/dev/null; sleep 3; done
  kill -9 "$pid" 2>/dev/null; rm -f "$pidf"
  for _ in $(seq 1 40); do
    local used; used=$(nvidia-smi --id="$(echo "$GPUS" | cut -d, -f1)" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${used:-9999}" -lt 1024 ] && break; sleep 5
  done
}

measure_arm() {  # $1=arm
  local arm="$1" port blog
  if [ "$arm" = a ]; then port="$PORT_A"; else port="$PORT_B"; fi
  blog="$RAW/abmx_$arm.bench.log"; : > "$blog"
  note "  arm $arm: 预热(结果丢弃)"
  "$ENV/bin/vllm" bench serve --backend openai --host 127.0.0.1 --port "$port" \
    --model "$MODEL_NAME" --seed 999999 \
    --dataset-name random --random-input-len 256 --random-output-len 32 --ignore-eos \
    --num-prompts 2 --max-concurrency 1 --percentile-metrics ttft,tpot \
    --metric-percentiles 50 --tokenizer "$CKPT" >>"$blog" 2>&1 || warn "    预热失败"
  for P in $PROMPTS; do
    for O in $OUTPUTS; do
      for C in $CS; do
        local rf="abmx_${arm}${ATAG:-}_p${P}_o${O}_c${C}.json"
        # 【必须给每个格子不同的 seed】random 数据集由 seed 决定 prompt 序列:同 seed ⇒ 同 prompt ⇒
        # 开了前缀缓存的 arm 在第 2 个格子起**整段命中**,TTFT 会假性掉到 ~1 s(实测踩过:
        # 8192/o32 的 TTFT=63.2 s,而同 prompt 的 8192/o1024 只有 1.2 s)。seed 只由 (P,O,C) 决定
        # ⇒ 两个 arm 拿到的 prompt 仍然**逐字节相同**,只是格子之间不再互相命中。
        local SEED=$(( P * 1000 + O * 10 + C ))
        note "  arm $arm: prompt=$P output=$O C=$C N=$REPS seed=$SEED"
        "$ENV/bin/vllm" bench serve --backend openai --host 127.0.0.1 --port "$port" \
          --model "$MODEL_NAME" --seed "$SEED" \
          --dataset-name random --random-input-len "$P" --random-output-len "$O" --ignore-eos \
          --num-prompts "$REPS" --max-concurrency "$C" --request-rate inf \
          --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
          --tokenizer "$CKPT" --save-result --result-dir "$RAW" \
          --result-filename "$rf" >>"$blog" 2>&1 \
          || { warn "    FAIL $rf"; tail -5 "$blog"; continue; }
        "$PY" - "$RAW/$rf" "$arm" "$P" "$O" "$C" <<'PY'
import json,sys
p,arm,P,O,C=sys.argv[1:6]
d=json.load(open(p))
ttft=d.get('mean_ttft_ms',0); tpot=d.get('mean_tpot_ms',0); L=int(O)
share=ttft/(ttft+L*tpot) if (ttft+L*tpot) else 0
print(f"    arm {arm} p={P:>5} o={O:>5} C={C}: out_tput {d.get('output_throughput',0):6.2f} | "
      f"TTFT {ttft:8.0f} ms | TPOT {tpot:6.2f} ms | TTFT占比 {share*100:5.1f}% | "
      f"in {d.get('total_input_tokens')} out {d.get('total_output_tokens')}")
PY
      done
    done
  done
}

note "env=$ENV ckpt=$CKPT"
install_pth || exit 1
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
for arm in $ARMS; do
  if [ "$SKIP_LAUNCH" = "1" ]; then note "SKIP_LAUNCH=1:复用已在跑的 arm $arm"; else
    launch_arm "$arm" || { stop_arm "$arm"; exit 1; }
  fi
  measure_arm "$arm"
  if [ "$SKIP_LAUNCH" = "1" ]; then note "SKIP_LAUNCH=1:不停止 arm $arm"; else stop_arm "$arm"; fi
done

echo
echo "==================== 汇总 ===================="
"$PY" - "$RAW" <<'PY'
import json, os, sys
raw=sys.argv[1]
rows={}
for arm in ("a","b"):
    for f in sorted(os.listdir(raw)):
        if f.startswith(f"abmx_{arm}") and f.endswith(".json") and "_p" in f:
            d=json.load(open(os.path.join(raw,f)))
            key=tuple(f.split("_p",1)[1].replace(".json","").split("_"))
            rows[(arm,key)]=d
keys=sorted({k for (a,k) in rows}, key=lambda t:(int(t[0][1:]),int(t[1][1:]),int(t[2][1:])))
hdr=f"{'prompt':>7} {'out':>5} {'C':>3} | {'A out_tput':>10} {'A TTFT':>8} {'A TPOT':>7} | {'B out_tput':>10} {'B TTFT':>8} {'B TPOT':>7} | {'B/A tput':>8} {'B/A TTFT':>8}"
print(hdr); print("-"*len(hdr))
for k in keys:
    A,B=rows.get(("a",k)),rows.get(("b",k))
    if not A or not B: continue
    P,O,C=(t[1:] for t in k)
    aa,ba=A.get('output_throughput',0),B.get('output_throughput',0)
    at,bt=A.get('mean_ttft_ms',0),B.get('mean_ttft_ms',0)
    ap,bp=A.get('mean_tpot_ms',0),B.get('mean_tpot_ms',0)
    print(f"{P:>7} {O:>5} {C:>3} | {aa:>10.2f} {at:>8.0f} {ap:>7.2f} | {ba:>10.2f} {bt:>8.0f} {bp:>7.2f} | "
          f"{(ba/aa if aa else 0):>7.2f}x {(bt/at if at else 0):>7.2f}x")
print("\n(A = lvllm+lk-moe;B = 主线+xiaotu-moe;B/A>1 表示 B 更高)")
PY
echo "[abmx] DONE"
