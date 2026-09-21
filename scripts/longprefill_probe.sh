#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# GPU long prefill 逐步取证探针(**只测 GLM-5.3-Flash 一个模型**)
#
# 用户要求(2026-09-21):
#   ① 每一步都留追踪记录;② 只测一个模型的 long prefill;
#   ③ **100% 确保走了 GPU 预填充**;④ 每一步都留下日志。
#
# 为什么要"确保走了 GPU 预填充":文档(`mixed_experts.py:1565`)记着同一 13.8K prompt
#   纯 CPU ~54 s / 纯 GPU 应 ~13 s,而当前实测(16.4K)是 105.6 s
#   ⇒ **当前 GPU 预填充比纯 CPU 还慢 ~1.65×**。所以本探针**必须先证明装配真的跑了**,
#   再给 CPU 基线做对照 —— 否则测到的可能根本不是 GPU 路径。
#
# 三个 arm:
#   A: GPU 预填充开(本阶段唯一优化对象) → 逐步 100% 验证装配确实执行
#   B: GPU 预填充关(`GPU_PREFILL=0`)    → 纯 CPU 基线,回答"GPU 到底值不值"
#   C: A + nsys profile                  → 把每层拆成五段,找那 436 ms/层
#
# 每个 arm 的产物都落进一个带时间戳的目录,便于逐步对账:
#   $OUT/run.log              主日志(每步带时间戳)
#   $OUT/00_env.txt           环境与旋钮快照(含 **实际生效值**)
#   $OUT/step<N>*_*.log       各步原始输出
#   $OUT/manifest.txt         本次运行的完整命令与 env-bridge 文件内容
#   $OUT/verify.txt           逐步验证结论(GPU 预填充是否真的走了)
#   $OUT/gap.txt              缺口分析(H2D 下界 vs 实测)
# License: Apache-2.0
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENVDIR="${BENCH_ENV:-/home/user/anaconda3/envs/vllm-xiaotu-moe}"
PY="$ENVDIR/bin/python"
export PATH="$ENVDIR/bin:$PATH" HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
XTU_TREE="${XTU_TREE:-/home/user/lvllm/vllm-up-133b71e0b}"
CKPT=/home/user/.cache/modelscope/models/ZhipuAI--GLM-5.3-Flash/snapshots/master

L="${L:-16384}"                 # 长 prompt 长度
N="${N:-1}"
MBT="${MBT:-4096}"
MAXLEN="${MAXLEN:-32768}"       # 本探针只要求"能放下这条 prompt"
SEQS="${SEQS:-2}"
UTIL="${UTIL:-0.90}"            # 停掉 MTP/常驻后,尽量高

OUT="${OUT:-/tmp/lp/$(date +%Y%m%d-%H%M%S)}"; mkdir -p "$OUT"
LOGD="$ROOT/logs"
RUN="$OUT/run.log"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$RUN"; }
step(){ printf '\n=== STEP %s: %s ===\n' "$1" "$2" | tee -a "$RUN"; }

freegpu() {
  local f p
  for f in "$LOGD"/lp_*.pid; do
    [ -f "$f" ] || continue; p=$(cat "$f" 2>/dev/null)
    [ -n "$p" ] && { kill -9 -"$(ps -o pgid= -p "$p" 2>/dev/null|tr -d ' ')" 2>/dev/null; kill -9 "$p" 2>/dev/null; }
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  local i=0; while [ $i -lt 60 ]; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || return 0
    sleep 5; i=$((i+1)); done
}

wait_ready() { # port pidfile log timeout tagname
  local port=$1 pidf=$2 logf=$3 tmo=${4:-1800} tag=$5 i=0
  while [ $i -lt $(( tmo / 5 )) ]; do
    curl -sf --max-time 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { log "  就绪 ${tag} ($((i*5))s)"; return 0; }
    if [ -f "$pidf" ]; then local sp; sp=$(cat "$pidf" 2>/dev/null||true)
      [ -n "$sp" ] && ! kill -0 "$sp" 2>/dev/null && { log "  ✗ ${tag} 服务进程($sp)已退出"; tail -6 "$logf" 2>/dev/null | sed 's/^/      /' | tee -a "$RUN"; return 2; }; fi
    if grep -qE "Engine core initialization failed|EngineDeadError|CUDA out of memory|larger than the available KV cache|No available memory|server exited early" "$logf" "${logf}.wrapper" 2>/dev/null; then
      log "  ✗ ${tag} 启动即失败"; grep -nE "ValueError|CUDA out of memory|No available memory|Engine core initialization" "$logf" "${logf}.wrapper" 2>/dev/null | tail -3 | sed 's/^/      /' | tee -a "$RUN"; return 3; fi
    sleep 5; i=$((i+1)); done
  log "  ✗ ${tag} 就绪超时"; return 4
}

# ---- GPU 预填充是否真的走了:三路取证 -------------------------------------
# 期望的装配行数 = MoE 层数 × chunk 数(插件每层每 chunk 打一行 [fp8-asm])
LAYERS=$(grep -oE "num_hidden_layers[\"': ]+[0-9]+" "$CKPT/config.json" 2>/dev/null | grep -oE "[0-9]+$" | head -1)
LAYERS=${LAYERS:-45}; MOE_LAYERS=$((LAYERS-1))
CHUNKS=$(( (L + MBT - 1) / MBT ))
EXPECT_ASM=$((MOE_LAYERS * CHUNKS))

verify() { # tag srvlog desc → 写 verify.txt
  local tag=$1 S=$2 desc=$3
  local active asm ctx shape
  active=$(grep -c "GPU prefill ACTIVE" "$S" 2>/dev/null || echo 0)
  asm=$(grep -c "fp8-asm" "$S" 2>/dev/null || echo 0)
  shape=$(grep -c "xtu-eng-shape" "$S" 2>/dev/null || echo 0)
  {
    echo "---- $tag ($desc) ----"
    echo "  服务端日志            : $S"
    echo "  GPU prefill ACTIVE    : $active   (期望 ≥1)"
    echo "  [fp8-asm] 装配行数     : $asm      (期望 ≈ $EXPECT_ASM = $MOE_LAYERS MoE层 × $CHUNKS chunk,两 rank 则 ×2)"
    echo "  [xtu-eng-shape] 引擎行 : $shape   (cpu 引擎存在)"
    if [ "$active" -ge 1 ] && [ "$asm" -ge $((EXPECT_ASM / 2)) ]; then
      echo "  ⇒ 判定:**确实走了 GPU 预填充装配** ✅"
    elif [ "$active" -ge 1 ] && [ "$asm" -eq 0 ]; then
      echo "  ⇒ 判定:⚠️ 决策为 ACTIVE 但**没有任何装配行** —— 必须查(可能 XIAOTU_GPF_STAGE 没生效)"
    else
      echo "  ⇒ 判定:**没有走 GPU 预填充**(期望如此就是对的)"
    fi
  } | tee -a "$OUT/verify.txt" | tee -a "$RUN"
}

bench() { # tag port envdesc -> 打印 TTFT/TPOT 并返回 json 路径
  local tag=$1 port=$2
  local jf="$OUT/${tag}.json" lf="$OUT/step_${tag}.log"
  timeout 2400 "$PY" -m vllm.entrypoints.cli.main bench serve --backend openai-chat \
    --endpoint /v1/chat/completions --host 127.0.0.1 --port "$port" \
    --model "$CKPT" --served-model-name GLM-5.3-Flash \
    --dataset-name random --random-input-len "$L" --random-output-len 128 \
    --num-prompts "$N" --max-concurrency 1 --seed $(( 90000 + RANDOM % 1000 )) \
    --save-result --result-dir "$OUT" --result-filename "${tag}.json" > "$lf" 2>&1
  "$PY" - "$jf" "$tag" "$L" "$MBT" >> "$OUT/gap.txt" <<'PY'
import json,sys
jf,tag,L,MBT=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
d=json.load(open(jf)); c=d.get('completed') or 0; tot=d.get('total_input_tokens') or 0
ttft=d.get('mean_ttft_ms') or 0; tpot=d.get('mean_tpot_ms') or 0
if c and ttft:
    pre=tot/(c*ttft/1000.0)
    print(f"{tag}: ok={c}/{d.get('num_prompts')} tokens={tot} TTFT={ttft:.0f}ms "
          f"prefill={pre:.1f} tok/s decode={1000/tpot if tpot else 0:.1f} tok/s")
else:
    print(f"{tag}: **FAILED** completed={c}")
PY
  grep -q "shard-dma2d\|illegal memory access" "$OUT/step_${tag}.log" 2>/dev/null && log "  ⚠️ $tag bench 侧出现 dma/illegal 关键字"
  tail -1 "$OUT/gap.txt" | sed 's/^/  /' | tee -a "$RUN"
}

launch() { # tagname port extra_env...
  local tag=$1 port=$2; shift 2
  # KV cap 必须与 MAXLEN 匹配:池要装得下 max_model_len 才能起服务。
  # 默认 2 GiB 只够 32K;保证 1M 上下文要 ~19.05 GiB(引擎反算 19,505 B/token)。
  local kv="${KV_CACHE_BYTES:-2147483648}"
  freegpu; rm -f "$LOGD/$tag.pid"
  setsid env PYTHONPATH="$XTU_TREE" TAG="$tag" PORT="$port" \
    GPU_UTIL="$UTIL" SPEC_K=0 SEQS="$SEQS" MAXLEN="$MAXLEN" MBT="$MBT" \
    KV_CACHE_BYTES="$kv" \
    "$@" bash scripts/serve_glm53_mainline.sh > "$LOGD/$tag.log.wrapper" 2>&1 &
  { echo "### launch $tag  port=$port"; echo "命令(等价):"; \
    echo "  TAG=$tag PORT=$port GPU_UTIL=$UTIL SPEC_K=0 SEQS=$SEQS MAXLEN=$MAXLEN MBT=$MBT KV_CACHE_BYTES=$kv $* bash scripts/serve_glm53_mainline.sh"; \
    echo "env-bridge 文件($LOGD/$tag.envfile)内容(权威,插件以它覆盖 os.environ):"; \
    sleep 2; cat "$LOGD/$tag.envfile" 2>/dev/null | sed 's/^/  /'; } >> "$OUT/manifest.txt"
  wait_ready "$port" "$LOGD/$tag.pid" "$LOGD/$tag.log" 1800 "$tag"
}

# ===========================================================================
step 0 "环境与旋钮快照"
{
  echo "## 本次运行 $(date -Is)"
  echo "## L=$L N=$N MBT=$MBT MAXLEN=$MAXLEN SEQS=$SEQS UTIL=$UTIL"
  echo "## 插件 HEAD: $(git -C "$ROOT" log --oneline -1)"
  echo "## 上游树: $XTU_TREE"
  echo "## 关键旋钮(本阶段按用户要求:停 MTP/dspark、停 MoE 常驻、全给 long prefill)"
  env | grep -E "^(XIAOTU_|VLLM_XIAOTU_|GPU_PREFILL|SPEC_K)" | sort | sed 's/^/  /' || echo "  (无)"
  echo "  XIAOTU_MOE_GPU_RESIDENT_LAYERS = ${XIAOTU_MOE_GPU_RESIDENT_LAYERS:-<未设,即 0 层常驻 ✓>}"
  echo "  XIAOTU_MOE_RESIDENT_BUDGET_GB  = ${XIAOTU_MOE_RESIDENT_BUDGET_GB:-<未设>}"
  echo "## GPU"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
  echo "## 期望装配行数 = $MOE_LAYERS MoE 层 × $CHUNKS chunk = $EXPECT_ASM(单 rank)"
} > "$OUT/00_env.txt" 2>&1
cat "$OUT/00_env.txt" | tee -a "$RUN"

step A "GPU 预填充 **开**(唯一优化对象)"
log "  旋钮:XIAOTU_GPF_STAGE=1(打开 [fp8-asm] 逐层计时) XIAOTU_GP_TIMING=1"
# ⚠️ 顺序很重要:GPU prefill ACTIVE」与「[fp8-asm]」都是**请求到达时**才打印的,
# 所以在 bench **之前** verify 必然全为 0(我第一版就栽在这,误判成"没走 GPU")。
launch lp_a 8110 XIAOTU_GPF_STAGE=1 XIAOTU_GP_TIMING=1 && \
  { bench a_gpu 8110; verify "A" "$LOGD/lp_a.log" "GPU 预填充开(**请求之后**计数)"; }
log "  装配计时样本(前 2 + 后 2):"
{ grep -m2 "fp8-asm" "$LOGD/lp_a.log" 2>/dev/null; echo "    ..."; grep "fp8-asm" "$LOGD/lp_a.log" 2>/dev/null | tail -2; } | sed 's/^/    /' | tee -a "$RUN"
log "  每 rank 装配行数:TP0=$(grep "Worker_TP0" "$LOGD/lp_a.log" 2>/dev/null | grep -c 'fp8-asm') TP1=$(grep "Worker_TP1" "$LOGD/lp_a.log" 2>/dev/null | grep -c 'fp8-asm')"
log "  ⇒ 隐含 chunk 数 = 每 rank 装配行数 / ACTIVE 行数×每 rank 的 ACTIVE 数"

step B "纯 CPU 基线(`GPU_PREFILL=0`)—— 回答'GPU 到底值不值'"
if [ "${SKIP_CPU:-0}" = "1" ]; then
  log "  SKIP_CPU=1 ⇒ 跳过纯 CPU 基线(扫 MBT 时用,省一半加载时间)"
else
  launch lp_b 8111 GPU_PREFILL=0 && \
    { bench b_cpu 8111; verify "B" "$LOGD/lp_b.log" "GPU 预填充关(纯 CPU,**请求之后**计数,期望全 0)"; }
fi

step C "把每层拆成五段(nsys profile 单条长请求)"
if command -v nsys >/dev/null 2>&1; then
  log "  nsys 可用,准备 profile(这一步需要另一轮加载;若时间不够可跳过,见下方命令)"
  cat >> "$OUT/manifest.txt" <<'EOF'

### step C 的手动命令(nsys 抓单条长请求)
#   1) 起服务(同 arm A 的旋钮,但不带 XIAOTU_GPF_STAGE 以减少扰动)
#   2) nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
#        --output=$OUT/nsys_gpu --force-overwrite=true \
#        <vllm bench serve 的那条单请求命令>
#   3) nsys stats $OUT/nsys_gpu.nsys-rep > $OUT/nsys_stats.txt
#   目标:把每层 ~600 ms 拆成 H2D / 转置 / MoE 内核 / attention / 空等 五段
EOF
  log "  已把 nsys 手动命令写进 manifest.txt(未自动执行,以免再占一轮加载)"
else
  log "  nsys 不可用,改用 torch.profiler(见 manifest.txt)"
fi

step 9 "汇总"
{
  echo "### 逐步验证(GPU 预填充是否真的走了)"; cat "$OUT/verify.txt" 2>/dev/null
  echo; echo "### 缺口分析"; cat "$OUT/gap.txt" 2>/dev/null
  echo; echo "### 下界参照(见 HANDOFF_GPU_LONGPREFILL.md §2)"
  echo "  每层 H2D 3.38 GiB @ 26.86 GB/s = 135 ms ⇒ 44 层 = 5.9 s/趟"
  echo "  理论 prefill = MBT / 5.9 s = 690 tok/s(与 prompt 长度无关)"
  echo "  文档记录的三种模式(同一 13.8K prompt):纯 CPU ~54 s / 纯 GPU 应 ~13 s / 混合 76.5 s"
} | tee -a "$RUN"
log "全部产物在 $OUT/"
ls -la "$OUT" | sed 's/^/  /' | tee -a "$RUN"
echo "LONGPREFILL_PROBE_DONE"
