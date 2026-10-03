#!/usr/bin/env bash
# ⭐ 生产 8070 固化启动脚本(唯一真源 ✓;用户口径 2026-09-30)
#
# 口径:DeepSeek-V4.1-Flash · **1M 上下文** · GPU 预填 on · dspark k=5 on · LMCache on · 监控栈 on
#
# 【2026-10-03 变更(用户指示)】MAXLEN 768K → **1M**;MBT 4096 → **8192**。
#   为什么现在能开 1M(此前被护栏拒绝):
#     * 方案 B 去掉了每 rank 3.16 GiB 的 raw 重组缓冲(staging 10.73 → 7.56 GiB/rank);
#     * CED 真正生效后每 token KV 从 ~5437 B 降到 ~2106 B(2.6×)⇒ **1M 只需 KV ~2.2 GiB**
#       (而不是 v0.2.5 那套的 5.6 GiB)。
#   实测(TP=2/GPU1,2/MBT=8192/dspark/KV 2.5 GiB):池 1,190,518 tok、
#     16k prefill **988 tok/s**、峰值 **38,399/40,960 MiB = 93.8%** ✅
#   ⚠️ KV 给到 5.3 GiB 时峰值 98.4%(太贴);历史 5.6 GiB 时 98% 且两次 OOM ✗
#     ⇒ **1M + GPU 预填 的成败由 KV 池预算决定,不是由 maxlen 决定。**
#
# 用法:
#   bash scripts/bringup_prod_8070.sh              # 起全套(幂等:已在跑则跳过 ✓)
#   bash scripts/bringup_prod_8070.sh --status     # 只看状态
#   WITH_MONITORING=0 bash scripts/bringup_prod_8070.sh   # 不起监控栈
#
# ⚠️ 8070 是【本 agent 自身的推理后端】(AGENTS.md"自服务环境纪律")⇒ 重启它会打断当前会话 ✗
#    ⇒ 本脚本只用于【机器重启后恢复】;日常调试请用 GPU2 + TP=1,不要动 8070 ✓
#
# 关键词(踩过的坑,别再改错):
#   * LMCache 是口径的一部分 ✗ 不能省(用户明令)⇒ 必须先起 LMCache 服务端(5555),再起 vLLM ✓
#   * MAXLEN = **1048576(1M)** ✓ —— 护栏已改成按 **KV 池预算**判定(见 serve_v41.sh);
#     1M + GPU 预填在 KV ≤ 3 GiB 时成立(实测 KV 2.5 GiB ⇒ 峰值 93.8% ✅)
#   * KV_CACHE_BYTES=2684354560(2.5 GiB):1M 实测只需 ~2.2 GiB;给到 5.3 GiB 会顶到 98.4% ✗
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False 必须带 ✓
#   * EXTRA_ENV 里的变量名是 XIAOTU_GP_ACT_RESERVE_GIB(不是 RESERVED ✗)
#   * proc.sh 的 PID 文件记的会是【包装脚本】✗(serve_v41.sh 内部 nohup 起真服务)
#     ⇒ 必须用【端口派生 PID】adopt 真服务 PID ✓,并把真服务日志路径写进日志头 ✓
set -uo pipefail

R=/home/user/lvllm/vllm-xiaotu-moe
cd "$R" || exit 1
PY=/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python
LOGDIR=dev-docs/report/tuning/logs
PORT="${PORT:-8070}"
WITH_MONITORING="${WITH_MONITORING:-1}"

# ───────── 参数(唯一真源 ✓;改这里就够)─────────
VLLM_ENV=(
  LMCACHE=1
  MAXLEN=1048576
  MBT=8192
  MAXSEQS=1
  GPUS=1,2        # 【2026-10-03 用户明令】TP=2 一律 GPU1+GPU2(GPU0 只有 PCIe x8)⇒ IRON_RULES R19
  TP=2
  GPU_UTIL=0.90
  COMPILE=1
  EAGER=0
  SPEC=1
  KV_DTYPE=fp8_ds_mla
  KV_CACHE_BYTES=2684354560
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=384
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
  # 【2026-10-03 事故修复】索引器 logits 预算 512 → 128 MB。
  #   事故:长 prompt 时 `sparse_attn_indexer` 的 `fp8_fp4_mqa_logits` 回退会分配
  #   `[q, kv]` 的 fp32 logits,分块器把 m×n 填到 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`
  #   (默认 512 MB)⇒ 实测崩溃那次请求 num_computed_tokens=11904,
  #   max_q = 512MB/4 // 11904 = 11274 ⇒ 11274×11904 = 134.2M 元素 = **恰好 512 MB**,
  #   而当时只剩 421.94 MiB 空闲(且 PyTorch 有 2.27 GiB reserved-but-unallocated 的碎片,
  #   因 LMCache 要求 expandable_segments:False 无法回收)⇒ OOM ⇒ EngineCore 死。
  #   为什么以前没预留:显式传 KV_CACHE_BYTES 会让 vLLM **跳过显存剖析**,
  #   而剖析本来会用 dummy 分配为这块 logits 预留 512 MB(见 IRON_RULES R24 第 8 条)。
  #   ⇒ 降到 128 MB 后,同样的请求只申请 ≤128 MB,在碎片空间里就能放下 ✓
  #   (框架按此预算在 query 维分块,可优雅退化到 1 token:vllm/v1/attention/backends/mla/indexer.py:1285-1318)
  EXTRA_ENV="XIAOTU_GP_ACT_RESERVE_GIB=1.5 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128"
  WARMUP=1
  PORT=8070
  TAG=v41_8070
  SERVED=DeepSeek-V4.1-Flash
)
LMCACHE_ENV=(CHUNK_SIZE=2176 TRANSFER_MODE=lmcache_driven ENABLE_MODULES= L1_GB=100 L2_GB=100)
MONITORING_ROOT=/home/user/lvllm/monitoring

port_up() { ss -ltnp 2>/dev/null | grep -q ":$1 "; }
wait_port() { local p="$1" n="${2:-60}"; for _ in $(seq 1 "$n"); do port_up "$p" && return 0; sleep 5; done; return 1; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if [ "${1:-}" = "--status" ]; then
  say "生产状态"
  printf '  %-24s %s\n' "8070 vLLM" "$(port_up 8070 && echo RUNNING || echo DOWN)"
  printf '  %-24s %s\n' "5555 LMCache MP" "$(port_up 5555 && echo RUNNING || echo DOWN)"
  printf '  %-24s %s\n' "8080 LMCache HTTP" "$(port_up 8080 && echo RUNNING || echo DOWN)"
  for p in 9090 3000 9100 8787; do printf '  %-24s %s\n' "监控 $p" "$(port_up $p && echo RUNNING || echo DOWN)"; done
  echo "  真服务 PID 文件: $(cat "$LOGDIR/vllm_prod_8070.pid" 2>/dev/null || echo '无')"
  curl -s --noproxy 127.0.0.1 --max-time 10 "http://127.0.0.1:8070/v1/models" | head -c 200; echo
  exit 0
fi

say "① LMCache 服务端(必须先起 ✓)"
if port_up 5555; then
  echo "  已在跑 ⇒ 跳过 ✓"
else
  bash scripts/proc.sh spawn lmcache_server env "${LMCACHE_ENV[@]}" bash scripts/serve_lmcache.sh
  wait_port 5555 24 && echo "  ✅ 5555 就绪 ✓" || { echo "  ✗ LMCache 未起来"; exit 1; }
fi

say "② vLLM 生产(LMCache + 768K + GPU 预填 + dspark ✓)"
if port_up "$PORT"; then
  echo "  8070 已在跑 ⇒ 跳过启动 ✓(要重启请先 proc.sh stop vllm_prod_8070)"
else
  bash scripts/proc.sh spawn dsv41_prod env "${VLLM_ENV[@]}" bash scripts/serve_v41.sh
  echo "  等待就绪(约 3–6 分钟;先读日志确认加载在推进 ✓)"
  # 同步读日志(纪律:不许"发脚本→等结果" ✗)
  # 【2026-10-03 修】原为 `seq 1 40`(=400 s):而真权重加载实测 **360 s**,再加 KV/CUDA graph
  #   初始化 ⇒ **必然超时** ⇒ 后面"认领真 PID"与"起监控栈"两步被跳过(PID 文件因此留着
  #   上一轮的陈旧 PID、看板一直没起)。改为 150×10 s = 25 min,足够覆盖加载+编译。
  READY_TRIES="${READY_TRIES:-150}"
  for i in $(seq 1 "$READY_TRIES"); do
    lines=$(wc -l < "$LOGDIR/v41_8070.log" 2>/dev/null || echo 0)
    if [ "$i" -le 3 ] || [ $((i % 12)) -eq 0 ]; then
      printf '    [%3d/%s] 日志 %s 行 | %s\n' "$i" "$READY_TRIES" "$lines" \
        "$(tail -1 "$LOGDIR/v41_8070.log" 2>/dev/null | cut -c1-100)"
    fi
    curl -s --noproxy 127.0.0.1 --max-time 8 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q DeepSeek && break
    sleep 10
  done
  curl -s --noproxy 127.0.0.1 --max-time 10 "http://127.0.0.1:$PORT/v1/models" | grep -q DeepSeek \
    || { echo "  ✗ 未就绪 ⇒ 尾部日志:"; tail -15 "$LOGDIR/v41_8070.log" | cut -c1-160 | sed 's/^/    /'; exit 1; }
  echo "  ✅ 8070 就绪 ✓"
fi

say "③ 认领【真】PID + 记日志(按用户定的策略 ✓)"
APIP=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$APIP" ]; then
  PG=$(ps -o pgid= -p "$APIP" 2>/dev/null | tr -d ' ')
  bash scripts/proc.sh adopt vllm_prod_8070 "$APIP"
  {
    echo "[adopt] name=vllm_prod_8070 api_pid=$APIP pgid=$PG adopted_at=$(date -Is)"
    echo "[adopt] real_service_log=$LOGDIR/v41_8070.log"
    echo "[adopt] proc.sh 的 dsv41_prod.pid 是【包装脚本】✗(会立刻退出)⇒ 停服务用 vllm_prod_8070 ✓"
  } >> "$LOGDIR/vllm_prod_8070.log"
  echo "  ✅ 真 PID=$APIP(PGID=$PG)已写入 vllm_prod_8070.pid ✓;真服务日志=$LOGDIR/v41_8070.log ✓"
else
  echo "  ✗ 端口 $PORT 无监听 ⇒ 无法认领"; exit 1
fi

say "④ 监控栈"
if [ "$WITH_MONITORING" = "1" ]; then
  M="$MONITORING_ROOT"
  port_up 9090 || bash scripts/proc.sh spawn prometheus "$M/prometheus-2.45.6.linux-amd64/prometheus" \
    --config.file="$M/prometheus/prometheus.yml" --storage.tsdb.path="$M/prometheus/data" \
    --web.listen-address=127.0.0.1:9090 --web.enable-lifecycle
  port_up 3000 || bash scripts/proc.sh spawn grafana "$M/grafana-v11.4.0/bin/grafana" server \
    --homepath "$M/grafana-v11.4.0" --config "$M/grafana/grafana.ini"
  port_up 9100 || bash scripts/proc.sh spawn node_exporter "$M/node_exporter-1.8.2.linux-amd64/node_exporter" \
    --web.listen-address=127.0.0.1:9100 --collector.textfile.directory="$M/textfile"
  bash scripts/proc.sh spawn xtu_exporter "$PY" "$M/textfile_exporter.py" "$M/textfile/xtu.prom"
  port_up 8787 || bash scripts/proc.sh spawn coremap "$PY" "$M/web/serve.py"
  bash scripts/proc.sh spawn coremap_png "$PY" "$M/coremap_png.py" "$M/web/coremap.png"
  sleep 10
else
  echo "  跳过(WITH_MONITORING=0)✓"
fi

say "⑤ 汇总"
for p in 8070 5555 8080 9090 3000 9100 8787; do
  printf '  %-5s %s\n' "$p" "$(port_up "$p" && echo RUNNING ✓ || echo DOWN)"
done
echo "  看板: http://127.0.0.1:3000/d/dsh-overview"
echo "  冒烟: curl -s --noproxy 127.0.0.1 http://127.0.0.1:8070/v1/models"
