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
# ⭐ 【唯一真源】进程归属证据(生产名单/端口/端口属主解析)全部来自 lib,禁止在此重复定义 ✗
LOGDIR="$R/dev-docs/report/tuning/logs"
# shellcheck source=lib_proc_identity.sh
. "$R/scripts/lib_proc_identity.sh"
PORT="${PORT:-8070}"
WITH_MONITORING="${WITH_MONITORING:-1}"

# ───────── 参数(唯一真源 ✓;改这里就够)─────────
# ⭐ 本次实例计划用的 GPU(与 VLLM_ENV 里的 GPUS 同源,下面有断言强制一致)
GPUS_PLAN=1,2
VLLM_ENV=(
  LMCACHE="${LMCACHE:-0}"   # ⭐ 2026-10-05 目标①:默认关闭 LMCache 连接器
  #   依据:SPEC=0 两次大请求崩溃栈均为 LMCache(retrieve 失败 ⇒ invalid_block_ids ⇒ scheduler 抛)✗
  #   ⇒ 先验证"无 LMCache 时 long prefill 是否稳定" ✓   # 【2026-10-05 修】原来硬编码 ⇒ 覆盖外部 env,令 LMCACHE=0 的 A/B 从未生效 ✗
  MM_IMAGES="${MM_IMAGES:-4}"   # 【2026-10-05】用户被 400 挡住(历史里的图被数成超限)⇒ 提到 4 解阻塞;代价:多图一起过编码器 ⇒ 首 token 慢
  MAXLEN=524288
  MBT=8192   # ⭐⭐ 2026-10-08 【已执行】6144 → 8192:端到端 **−9.9%**(交错 A/B × 3 轮,离散 **0.3%**)✓(台账 B376);原注见下方 C2 勘误

  # ⭐⭐ 【勘误 C2】(2026-10-08,以台账为证)上面那条『回退 8192→6144』的**理由已被推翻** ✗
  #   ① `q_out` 的风险:实测 MBT=8192 下**显存最低可用 10.44 GiB** ⇒ 距 512 MiB 的 q_out **20 倍余量** ✓
  #   ② 但 **MBT=16384 会起不来**(KV 池分配 OOM,只剩 713 MiB)⇒ 天花板仍在 8192~16384 之间 ✓
  #   ③ ⭐ **端到端实测**(交错 A/B × 3 轮,**离散 0.3%**):**MBT 6144→8192 = −9.9%** ✓(B376)
  #   ⇒ ✅ **已于 2026-10-08 执行(本行现为 `MBT=8192`)**;⚠️ 执行前须过 R38 复审;⚠️ 上一版的『−12.2%』是**单次样本,已撤回** ✗

              #   按 A41 斜率(−722 MiB/档)预计谷底 ≈1.6 GiB ⇒ 兼顾速度与安全
              #   4096 的实测谷底 3,110 MiB;8192 崩时 223 MiB(两点外推,需实测确认 ✓)
              #   触发事故:00:07:51 aten::new_empty 要 512 MiB 而只剩 223.5 MiB ⇒ 崩(A39)
              #   ⚠️ 代价:prefill 约 −34%(CHANGELOG);用户要保留投机 ⇒ 用 MBT 换安全 ✓
              #    GPU 预填触发率=0(device=cuda 0 次 ✗)⇒ 丢掉 1.41× 预填优化  ⛔ **此处已失效,见文件末尾【勘误 C1】(2026-10-08)**
              #    安全改由 ACT_RESERVE 承担(见下)
              #    原 8192(用户 2026-10-03 指示)⇒ 崩溃回归;详见 BUG_REGISTRY A19/A20/A21/A24
  MAXSEQS=2    # ⭐⭐ 2026-10-09 用户定案:并发 4 → **2**
  #   ⛔ 来由(2026-10-09 04:36 崩溃):4 路 sub-agent 同时打本机,每个 prompt 18 万 token 且
  #      **前缀缓存零命中(完整预填)** ⇒ 显存爬到 34.2 GiB(上限~36)⇒ attention 里 q_padded
  #      分配失败(`aten::new_empty`)⇒ **EngineCore 死、服务整体下线** ✗
  #   ⭐ 选择只降并发、**不动 MBT**(保住预填速度):见 USER_QA_LEDGER #133
  #   旧值(保留):4 —— 2026-10-08 用户定案(窗口期):并发 → 4;配合 MAXLEN=524288
               #   用户口径:「512K 天花板 + 短任务多路」;⚠️ **长请求超池会排队**
               # ⚠️ 算术(实测):池 = 1,080,480 token(2.5 GiB,2.426 KiB/tok)
               #   4 × 512K = 2,097,152 tok ⇒ **只有 ~2 路满长能真同时跑**,其余**排队**(不报错)✓
               #   短任务(如每路 ≤32K)⇒ 4 路都跑得动 ✓
               # ⭐ 口径唯一真源:docs/MODEL_GUIDES.md §0.1b
               #   规则:seqs×maxlen ≤ 池容量 ⇒ 满载;超过 ⇒ **只排队、不报错**
               #   换档要同时改 MAXLEN,不必动 KV;⚠️ 本改动**下次 bringup 才生效**(需用户当次许可)
               # ⚠️ 未设 LPT(=vLLM 原行为):serve_v41.sh:139-151 要求"多路并发时设 LPT=2048+adaptive",
               #   否则复现 p99 ITL 8.2–8.7 s;但 B273 实测 LPT=2048 **净更差**(p99 +25%、TTFT +113%)
               #   ⇒ 两处口径冲突,已登记待裁。**本轮把"饿死解码"作为已知代价接受**,
               #   并已量出该代价(饿死探针),由 R4(在 chunk 间插入纯解码步)单独解决 ✓
               # (历史:2026-10-07 曾定案 1 → 2;实测运行实例始终是 1,未生效)
  GPUS=1,2        # 【2026-10-03 用户明令】TP=2 一律 GPU1+GPU2(GPU0 只有 PCIe x8)⇒ IRON_RULES R19
  TP=2
  GPU_UTIL=0.90
  COMPILE=1
  EAGER=0
  SPEC=1   # ⭐ 2026-10-05 用户指示:生产【开启】投机解码(DSpark k=5)
            #   理由:投机收益太大(实测 mean acceptance ≈3.59 ⇒ decode 约 2–3×)
            #   ⭐ 保留 LMCACHE=0(已确认它是 long prefill 崩溃的元凶 ✓,见 docs/KNOWN_ISSUES_LMCACHE.md)
            #   ⇒ 本配置用于明天验证【输出退化】是否仍出现
            #   若本臂 long prefill 稳定 ⇒ 按目标③把此配置留在生产供明天测退化
            #   崩溃栈:LMCache retrieve 失败 ⇒ invalid_block_ids ⇒ scheduler 抛 ⇒ EngineCore 死(A34)
            #   ⇒ 暂回 SPEC=1(已知可用);SPEC=0 需先解决 LMCache 崩才能用于退化验证
            #   目的①:减每步激活(不再被 draft token 放大)⇒ 放开显存闸门
            #   目的②:⭐ 验证模型退化是否由投机解码引起(sample #3 的对照臂)
            #   ⚠️ 代价:decode 吞吐下降(acceptance 均值 3.59 ⇒ 预期 decode 慢 ~2x)
  KV_DTYPE=fp8_ds_mla
  KV_CACHE_BYTES=2684354560   # ⚠️ 2026-10-05:1.5 GiB 导致 EngineCore 初始化失败 ✗ ⇒ 回退已知可用值 ✓(见 A19)
  # ⭐ 2026-10-05 三方审核定案:原 2.5 GiB ⇒ 可用显存只剩 ~0.14-0.42 GiB
  #   ⇒ 512 MiB 的 per-forward q_out(MBT=8192 ⇒ 8188x64KiB)分配失败 ⇒ EngineDeadError
  #   ⇒ 降到 1.5 GiB 永久多出 ~1 GiB 余量 ⇒ 确定性 >=512 MiB ✓
  #   ⇒ 代价:KV 池 119 万 → ~71 万 token(agent 实际只需 ~31 万 ✓)
  #   ⇒ 依据:BUG_REGISTRY A17/A18 + EXPERIMENTS B25 + CHANGELOG 2026-09-21
  # 【2026-10-05 改】原 384 是 **ds-v4-flash** 时代的甜点 ✗(该模型已退役)⇒ 归档 ✓
  # 现依据【本仓实测】`dev-docs/PREFILL_CPU_VS_GPU_2026-10-03.md` §3:
  #   TP=1 交叉点 ≈ **5700** token/层;TP=2 每 rank 的 DMA 减半 ⇒ 交叉点 **~2500–2900** ✓
  #   ⇒ 生产 TP=2 取 **2560**(正好在交叉点)✓:≥2560 走 GPU(实测 chunk=8192 时 GPU 快 1.41× ✓),
  #      <2560 走 CPU(实测 chunk=1019 时 GPU 只有 0.40× ✗)
  #   机理:胜负由 **chunk = min(prompt, MBT)** 决定,不是 prompt 总长 ✓
  VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="${GPU_PREFILL_MIN_TOKENS:-2560}"
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
  EXTRA_ENV="XIAOTU_GP_ACT_RESERVE_GIB=1.5 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 XIAOTU_GPF_STAGE_TILE_E=96"   # ⛔ 2026-10-08 **回退**:3.0 被 A29 实测否决(GPU 预填 device=cuda **0** 次);2.0 亦"可能完全不触发"⇒ **只有 1.5 下实测 ACTIVE(346 次)** ⇒ 不动 ✓
  # ⭐ 【勘误 C3】(2026-10-08)`ACT_RESERVE` **保持 1.5** 的回退理由仍然成立 ✓
  #   (A29 实测:3.0 ⇒ GPU 预填 `device=cuda` **0 次**;2.0 亦'可能完全不触发')✓
  #   ⚠️ 但**不要**据此以为 GPU 预填没在跑 —— 见下面的 C4 ✓

  # ⭐ 2026-10-05 定案(A28):ACT_RESERVE 原为 1.5 GiB,而代码默认 3.0 GiB(gpu_prefill.py:886)✗
  #   ⇒ 预检过松 ⇒ 显存极紧时仍放行 GPU 预填 ⇒ 随后的几百 MiB 分配失败 ⇒ EngineCore 死 ✗
  #   ⇒ 恢复 3.0(与"peak activation 2.9 GiB"同量级 ✓)⇒ 保住 MAXLEN=1M ✓
  #   ⇒ B 臂修正:3.0 过严 ⇒ GPU 预填【完全禁用】(实测 device=cuda 0 次 ✗)
  #   ⭐ 2026-10-06 用户定案:取 2.0 —— 保持 MAXLEN=1M(不削上下文 ✓),
  #     用【更严的预检】防谷底崩溃;降级为【每 forward 重判】(见 mixed_experts.py §602)
  #     降级时会打印「显存不足,本次降级至 CPU Prefill」+ 缺口 + 建议压缩多少上下文 ✓
  WARMUP=1
  # ⭐ 审计修:原为硬编码 `PORT=8070` ⇒ 外层 `PORT=8071` 被覆盖 ⇒ **在活生产上起第二个全量实例** ✗
  #    ⇒ 现在沿用外层 PORT(唯一真源)✓
  PORT="${PORT}"
  TAG=v41_8070
  SERVED=DeepSeek-V4.1-Flash
)
LMCACHE_ENV=(CHUNK_SIZE=2176 TRANSFER_MODE=lmcache_driven ENABLE_MODULES= L1_GB=64 L2_GB=100)
  # 【2026-10-05】L1 100→64 GiB:各 node 仅剩 5.9–20.1 GiB 空闲 ⇒ 100 GiB 即使 interleave 也偏紧;
  # 64 GiB ⇒ 每节点 8 GiB ✓ 宽裕;保留 interleave(用户:lmcache 性能要求不高,不必锁 local ✓)
MONITORING_ROOT=/home/user/lvllm/monitoring

# ⭐ 自检:GPUS_PLAN 必须等于 VLLM_ENV 里的 GPUS —— 防止"同一事实两处表示" ✗
#   (来由:上一个会话连错三版,根因全是"两处手写、从不交叉校验" ⇒ 这里强制交叉校验 ✓)
_gpus_in_env="$(printf '%s\n' "${VLLM_ENV[@]}" | grep -m1 '^GPUS=' | cut -d= -f2-)"
if [ "$_gpus_in_env" != "$GPUS_PLAN" ]; then
  echo "⛔ 内部不一致:VLLM_ENV GPUS=$_gpus_in_env 但 GPUS_PLAN=$GPUS_PLAN ⇒ 拒绝执行 ✗" >&2
  exit 1
fi

# ⛔⛔ 2026-10-07 独立审计(blocker 3):原实现 `ss -ltnp | grep -q ":$1 "` **丢弃 ss 的退出码、
#   无 timeout、且是子串匹配** ⇒ ss 失败/超时/被裁 ⇒ 判"端口没人听" ⇒ **同机多实例硬门形同不存在**,
#   会在活生产 8070 上照常起第二个全量实例(2026-10-04 OOM 事故同型)✗
#   ⇒ 改为唯一真源:pi_port_listeners(`:` 整段相等 + timeout + PI_SS_OK)✓
#   ⭐ 证据读不到(ss 失败)一律【当作在跑】⇒ fail-closed ✓
# ⭐⭐ 2026-10-07 独立审计 F1:原来的二元 port_up 把"ss 读不到"当成"在跑" ✗
#   对【安全门】(非 8070 + 生产在跑 ⇒ 拒绝)方向正确,但同一函数被 12 个调用点复用 ⇒
#   在【启动决策】处方向相反:会**静默跳过**启动监控栈/LMCache/coremap,
#   而 --status/⑤汇总还会**反报 RUNNING** ✗
#   ⇒ 改为三态:UP / DOWN / UNKNOWN(= 证据读不到)✓
port_state() {   # 结果写入 PORT_STATE;rc:0=UP 1=DOWN 2=UNKNOWN
  pi_port_listeners "$1"
  if [ "$PI_SS_OK" != "1" ]; then PORT_STATE=UNKNOWN; return 2; fi
  if [ -n "$PI_PORT_PIDS" ]; then PORT_STATE=UP; return 0; fi
  # ⭐ N4 修:有 LISTEN 行、但属主 pid 读不到(如别的 uid 的 socket)⇒ **UNKNOWN,不是 DOWN** ✗
  #   原先判 DOWN ⇒ 安全门不拒、启动决策"该起"、--status 反报 DOWN,与 F1 同类 fail-open
  if [ "${PI_PORT_LISTEN_N:-0}" -gt 0 ]; then PORT_STATE=UNKNOWN; return 2; fi
  PORT_STATE=DOWN; return 1
}
# 兼容旧调用:只有【确定在跑】才为真(UNKNOWN 不再算"在跑")
port_up() { port_state "$1" || return 1; }
# ⭐ 启动决策用这个:确定 DOWN ⇒ 该起;UP ⇒ 跳过;UNKNOWN ⇒ **拒绝静默跳过,报错中止** ✓
port_maybe_start() {   # rc:0=该起(确定 DOWN) 1=已在跑 2=无法判定(调用方必须中止)
  port_state "$1" && return 1
  [ "$PORT_STATE" = "DOWN" ] && return 0
  return 2
}
# ⭐ 状态展示用:UNKNOWN 必须显示 UNKNOWN,不得显示 RUNNING ✗
port_label() { port_state "$1"; case "$PORT_STATE" in UP) echo RUNNING;; DOWN) echo DOWN;; *) echo UNKNOWN;; esac; }
wait_port() {   # ⭐ UNKNOWN 不算就绪(继续等),等不到就失败 ✓
  local p="$1" n="${2:-60}" _
  for _ in $(seq 1 "$n"); do port_up "$p" && return 0; sleep 5; done
  return 1
}
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ⭐ 统一 spawn:被拒(rc≠0)⇒ 立刻退出 —— 绝不在"门拒绝"之后继续后面的动作 ✓
spawn_or_die() {   # <name> <cmd...>
  local n="$1"; shift
  if ! bash scripts/proc.sh spawn "$n" "$@"; then
    echo "  ⛔ spawn $n 被拒绝/失败 ⇒ 立刻停止(不得在门拒绝后继续)✗" >&2
    exit 1
  fi
}

# ⭐ 回收+重启一个【看板栈】助手
#   审计修:原代码 `stop … >/dev/null 2>&1 || true` 之后**无条件 spawn**
#   ⇒ 停不掉也照起 ⇒ 每次 bringup 多 2 个孤儿(实测一天 33 个)✗
#   ⇒ 现在:停不掉(门判为生产/监控栈)就**不重启它** ✓
recycle_helper() {   # <name> <cmd...>
  local n="$1"; shift
  if bash scripts/proc.sh status "$n" >/dev/null 2>&1; then
    if ! bash scripts/proc.sh stop "$n"; then
      echo "  ⚠️ 无法安全停止 $n(门判为生产/监控栈 ⇒ 需用户【当次】许可)⇒ 不重启它,避免制造孤儿 ✗" >&2
      return 1
    fi
  fi
  spawn_or_die "$n" "$@"
}

if [ "${1:-}" = "--status" ]; then
  say "生产状态"
  printf '  %-24s %s\n' "8070 vLLM" "$(port_label 8070)"
  printf '  %-24s %s\n' "5555 LMCache MP" "$(port_label 5555)"
  printf '  %-24s %s\n' "8080 LMCache HTTP" "$(port_label 8080)"
  for p in 9090 3000 9100 8787; do printf '  %-24s %s\n' "监控 $p" "$(port_label $p)"; done
  port_state 8070; [ "$PORT_STATE" != "UNKNOWN" ] || echo "  ⚠️ 8070 端口状态无法判定(ss 失败)⇒ 上面的 RUNNING/DOWN 不可信" >&2
  echo "  真服务 PID 文件: $(cat "$LOGDIR/vllm_prod_8070.pid" 2>/dev/null || echo '无')"
  curl -s --noproxy 127.0.0.1 --max-time 10 "http://127.0.0.1:8070/v1/models" | head -c 200; echo
  exit 0
fi

# ⭐ 同机多实例硬门(AGENTS「同机多实例纪律」)—— 必须放在【任何 spawn 之前】✓
#   来由(2026-10-07 自查):原先它排在 ①LMCache 之后 ⇒ 用 PORT=8071 起第二个实例时,
#   会先把 LMCache 服务端拉起来、再拒绝 ⇒ "拒绝了却已经动了东西" ✗ ⇒ 现在提到最前 ✓
# ⭐ 安全门:生产"确定在跑"或"无法判定"都拒绝(fail-closed)✓
port_state 8070; _prod_st=$?
if [ "$PORT" != "8070" ] && { [ "$_prod_st" -eq 0 ] || [ "$PORT_STATE" = "UNKNOWN" ]; }; then
  echo "⛔ 生产 8070 正在运行,而本次要起 PORT=$PORT 的第二个实例 ⇒ 拒绝执行(同机多实例纪律)✗" >&2
  echo "   若确需如此:必须由用户决定是否暂停生产(AGENTS 第 1 条:每次都要当次许可)✗" >&2
  exit 3
fi

say "① LMCache 服务端"
# ⭐ 数组元素【不是】shell 变量 ⇒ 必须先取出真实变量(否则 set -u 会直接退出 ✗)
_LMCACHE_ON="${LMCACHE:-0}"
if [ "$_LMCACHE_ON" = "1" ]; then
  port_maybe_start 5555; _lm=$?
  if [ "$_lm" = "2" ]; then echo "  ⛔ :5555 端口状态无法判定(ss 失败)⇒ 中止 ✗" >&2; exit 1; fi
  if [ "$_lm" = "1" ]; then
    echo "  已在跑 ⇒ 跳过 ✓"
  else
    spawn_or_die lmcache_server env "${LMCACHE_ENV[@]}" bash scripts/serve_lmcache.sh
    wait_port 5555 24 && echo "  ✅ 5555 就绪 ✓" || { echo "  ✗ LMCache 未起来"; exit 1; }
  fi
else
  echo "  LMCACHE=0 ⇒ 不启动 lmcache_server(端口 5555/8080 预期 DOWN ✓);也不等待它 ✓"
fi

say "② vLLM 生产(LMCache + 512K + GPU 预填 + dspark ✓)"
# (同机多实例硬门已提到 ① 之前 —— 见上面)✓
port_maybe_start "$PORT"; _vp=$?
if [ "$_vp" = "2" ]; then echo "  ⛔ :$PORT 端口状态无法判定(ss 失败)⇒ 中止(不静默跳过启动)✗" >&2; exit 1; fi
if [ "$_vp" = "1" ]; then
  echo "  $PORT 已在跑 ⇒ 跳过启动 ✓(要重启请先 proc.sh stop vllm_prod_8070)"
else
  spawn_or_die dsv41_prod env "${VLLM_ENV[@]}" bash scripts/serve_v41.sh
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
# ⭐ 审计修:认领名不得在非 8070 端口时仍写成生产名(否则用调试实例覆盖生产 PID 文件)✗
ADOPT_NAME="$([ "$PORT" = "8070" ] && echo vllm_prod_8070 || echo "probe_${PORT}")"
# ⭐ 审计修:原代码用 `ss | grep ":$PORT " … | head -1` 且**只看"有人听"**就 adopt
#   ⇒ 会**认领别人的 PID**(事故④"把生产当孤儿"的同一类错误)✗
#   ⇒ 现在:① 端口属主用 lib 的**锚定**解析 ② 必须证明它是 vLLM 且 CV 与计划一致 ✓
pi_port_listeners "$PORT"
APIP=""
if [ "$PI_SS_OK" = "1" ] && [ -n "$PI_PORT_PIDS" ]; then
  APIP="$(printf '%s\n' "$PI_PORT_PIDS" | head -1)"
fi
if [ -z "$APIP" ]; then
  echo "  ✗ 端口 $PORT 无监听(或 ss 读不到属主)⇒ 拒绝认领(取不到属主不等于没人听)✗" >&2; exit 1
fi
if ! pi_cmd_is_vllm "$APIP"; then
  echo "  ✗ 端口 $PORT 的属主 pid=$APIP 不是 vLLM API server ⇒ 拒绝认领(不 adopt 别人的 PID)✗" >&2
  echo "      cmd=$(pi_cmdline "$APIP" | cut -c1-120)" >&2; exit 1
fi
pi_cv "$APIP"
if [ "$PI_CV_READ_OK" != "1" ]; then
  echo "  ✗ pid=$APIP 的 CUDA_VISIBLE_DEVICES 读不到 ⇒ 拒绝认领(无法证明是计划内的实例)✗" >&2; exit 1
fi
case ",${PI_CV}," in
  *",${GPUS_PLAN},"*) : ;;
  *) echo "  ✗ pid=$APIP 的 CV='${PI_CV}' 与本次计划 '$GPUS_PLAN' 不一致 ⇒ 拒绝认领 ✗" >&2; exit 1 ;;
esac
PG="$(pi_pgid "$APIP")"
if ! bash scripts/proc.sh adopt "$ADOPT_NAME" "$APIP"; then
  echo "  ✗ adopt 被拒绝 ⇒ 停止(不得留下无主的服务)✗" >&2; exit 1
fi
{
  echo "[adopt] name=$ADOPT_NAME api_pid=$APIP pgid=$PG adopted_at=$(date -Is)"
  echo "[adopt] real_service_log=$LOGDIR/v41_8070.log"
  echo "[adopt] proc.sh 的 dsv41_prod.pid 是【包装脚本】✗(会立刻退出)⇒ 停服务用 $ADOPT_NAME ✓"
} >> "$LOGDIR/$ADOPT_NAME.log"
echo "  ✅ 真 PID=$APIP(PGID=$PG)已写入 $ADOPT_NAME.pid ✓;真服务日志=$LOGDIR/v41_8070.log ✓"

say "④ 监控栈"
if [ "$WITH_MONITORING" = "1" ]; then
  M="$MONITORING_ROOT"
  port_maybe_start 9090; case $? in 0) : ;; 1) : ;; *) echo "  ⛔ :9090 端口状态无法判定(ss 失败)⇒ 拒绝静默跳过,中止 ✗" >&2; exit 1 ;; esac
  [ "$PORT_STATE" = "UP" ] || spawn_or_die prometheus "$M/prometheus-2.45.6.linux-amd64/prometheus" \
    --config.file="$M/prometheus/prometheus.yml" --storage.tsdb.path="$M/prometheus/data" \
    --web.listen-address=127.0.0.1:9090 --web.enable-lifecycle
  port_maybe_start 3000; case $? in 0) : ;; 1) : ;; *) echo "  ⛔ :3000 端口状态无法判定(ss 失败)⇒ 中止 ✗" >&2; exit 1 ;; esac
  [ "$PORT_STATE" = "UP" ] || spawn_or_die grafana "$M/grafana-v11.4.0/bin/grafana" server \
    --homepath "$M/grafana-v11.4.0" --config "$M/grafana/grafana.ini"
  port_maybe_start 9100; case $? in 0) : ;; 1) : ;; *) echo "  ⛔ :9100 端口状态无法判定(ss 失败)⇒ 中止 ✗" >&2; exit 1 ;; esac
  [ "$PORT_STATE" = "UP" ] || spawn_or_die node_exporter "$M/node_exporter-1.8.2.linux-amd64/node_exporter" \
    --web.listen-address=127.0.0.1:9100 --collector.textfile.directory="$M/textfile"
  # 【2026-10-05 修】回收上一批监控助手 —— 否则 **每次 bringup 都留 2 个孤儿** ✗
  # 实测:一天内 10 次 bringup ⇒ 33 个孤儿(`coremap_png.py` / `textfile_exporter.py`)
  # 【2026-10-07 再修(审计)】原代码"停不掉(|| true)也照起" ⇒ 孤儿照旧 ✗
  #   ⇒ 改 `recycle_helper`:停不掉就**不重启** ✓
  recycle_helper coremap_png "$PY" "$M/coremap_png.py" "$M/web/coremap.png" || true
  # 【规则·用户 2026-10-05】把"进程已消失满 3 天"的陈旧 .pid/日志打包归档 ✓
  bash scripts/archive_stale_logs.sh >/dev/null 2>&1 || true
  recycle_helper xtu_exporter "$PY" "$M/textfile_exporter.py" "$M/textfile/xtu.prom" || true
  port_maybe_start 8787
  case $? in
    0) _start_coremap=1 ;;
    1) _start_coremap=0 ;;
    *) echo "  ⛔ :8787 端口状态无法判定(ss 失败)⇒ 中止 ✗" >&2; exit 1 ;;
  esac
  if [ "$_start_coremap" = "1" ]; then
    if bash scripts/proc.sh status coremap >/dev/null 2>&1; then
      echo "  ⚠️ coremap 进程在但 8787 未监听 ⇒ 不重复起(请排查)" >&2
    else
      spawn_or_die coremap "$PY" "$M/web/serve.py"
    fi
  fi
  sleep 10
else
  echo "  跳过(WITH_MONITORING=0)✓"
fi

say "⑤ 汇总"
for p in 8070 5555 8080 9090 3000 9100 8787; do
  printf '  %-5s %s\n' "$p" "$(port_label "$p")"
done
echo "  看板: http://127.0.0.1:3000/d/dsh-overview"
echo "  冒烟: curl -s --noproxy 127.0.0.1 http://127.0.0.1:8070/v1/models"

# ══════════════════════════════════════════════════════════════════════════════
# 【勘误 C1】(2026-10-08 行间注)关于上面 MBT 段的 "GPU 预填触发率=0" 与 "prefill −34%"
#
# ⛔ **此处不对 —— 以【活日志】为证,不再引用注释**:
#   ① **"触发率=0" 已失效**:`device=cuda` **346** 次 / `device=cpu` 348 次,
#      `GPU prefill ACTIVE` **42** 次 ⇒ **GPU 预填在跑** ✓
#      (门:`first 6140 tokens >= threshold 2560`;`staging ~7.56 GiB`)
#   ② **"prefill 约 −34%" 口径错**:那是【4096 vs 8192】的数 ——
#      CHANGELOG 原文 *"MBT = 8192(**原 4096**):16k prefill ~648 → 982 tok/s"* ⇒ 648/982−1 = −34.0% ✗
#      本档(6144)的正确代价 ≈ **−16%**:实测 `t_chunk` **7.31 s** ⇒ **840 tok/s**;8192 ≈ **999** ✓
#   ③ ⭐ **且当时迫使降档的约束已变**:那次 OOM(A39)时 free 仅 **223.5 MiB**;
#      而今 GPU 预填 preflight 报 **`had 18.94 GiB, slack +9.12 GiB`** ⇒
#      **把 MBT 拿回 8192 值得重测**(模型 **+18.9%** prefill,只需多 ~128 MiB 的 `q_out`)✓
#   ⭐ 待办:下一个窗口把 **MBT 6144 → 8192** 作为【单变量】跑 A/B(见 `docs/EXPERIMENTS.md` **B345**)✓
#
# ⚠️ **为什么这条放在文件末尾而不是原地**:原地插入 11 行会把 `VLLM_ENV=( … )` 的闭合括号
#    从 **130** 推到 **141**(基线见下) ⇒ `preflight_bringup.sh` 的 `sed -n '1,130p'` 干跑
#    拿到**未闭合的数组** ⇒ **前置检查失败** ✗(实测)。
#    ⇒ 故:原地只留**同行短指针**;全文放末尾 ⇒ **行数零增**,horizon 不受影响 ✓
# ══════════════════════════════════════════════════════════════════════════════
