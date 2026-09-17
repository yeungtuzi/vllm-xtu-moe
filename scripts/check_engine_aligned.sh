#!/usr/bin/env bash
# 第一条(把 CPU 解码引擎效率对齐 lk_moe)的**回归门禁** —— 一条命令跑完两项检查:
#
#   1) 数值门禁:`test_block23_equiv.py`(受控 me=2/3 路由 + numpy golden 逐元素对拍)。
#      R55 规定:**任何改动 `numa_pool.hpp` 同步结构或 `moe_v2_packed4.hpp` 内层循环的提交,
#      都必须先过这一项** —— 轮 68/70 的两次事故都是 bench 三次全过而对拍立刻挂/算错。
#   2) 性能门禁:`bench_engine_ab.py`(BS=6/K=6/THREADS=120/真实层权重/DEDUP=12),
#      验收 **≤0.70 ms/层**(NOTES §119/§73)。
#
# 用法:
#   scripts/check_engine_aligned.sh
#   NPZ=<真实模型目录> NPZ_EQ=<real_layer1_model.npz> THRESH=0.70 REP=60 scripts/check_engine_aligned.sh
#
# 当前基线:
#   * **8 NUMA node 时代(NPS4)** BS=6/DEDUP=12/THREADS=120:xiaotu 0.65-0.68 ms/层
#     (229-232 GB/s、1.9 GB/s·线程) vs lk_moe 0.57 ms/层(265 GB/s)。
#   * **2 NUMA node 时代(NPS1,2026-09-16 起)**:同一份代码 0.94-0.97 ms/层、156 GB/s
#     —— nshard_ 由 8 降到 2,每个 node 的工作集大 4×,DEDUP=12 的 L3 口径失效(§498)。
#     阈值因此按拓扑自动取,别再拿 0.70 在 2-node 机器上报警。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/user/anaconda3/envs/vllm-xiaotu-moe/bin/python}"
# 两项检查需要**不同的 fixture**:对拍要小 npz,bench 要真实模型目录。
NPZ="${NPZ:-/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/master}"
NPZ_EQ="${NPZ_EQ:-/home/user/lvllm/vllm-xiaotu-moe/fixtures/real_layer1_model.npz}"
# ---------------------------------------------------------------- 拓扑先决条件
# `nshard_ = max(1, numa_node_count()/world)`(moe_v2.hpp:461),所以 **NUMA node 数是引擎
# 行为的输入**:同一份代码在 8-node(NPS4)与 2-node(NPS1)上,每个 node 要读的行数差 4×,
# DEDUP=12 这个"L3 驻留交付"口径必然不同。2026-09-16 实测(机器重启后变 NPS1):
#     nshard=2(自适应、正确) : 0.97 ms/层、156 GB/s
#     XIAOTU_MOE_NSHARD=8     : 44.4 ms/层(灾难 —— node 2..7 没有 worker)
# 故阈值按拓扑取:>=8 node 沿用历史 0.70(232 GB/s 口径);<=2 node 用 1.15
#   (实测静默机器 3 次:DEDUP=12 = 0.95,DEDUP=23 = 1.07/1.08/1.09 ⇒ 取 1.15 留 ~6% 余量)。
# **这不是放宽门禁**:它把"跨拓扑拿旧阈值报警"这个假信号去掉,同时保留同拓扑下的比较力。
NNODES="$(numactl --hardware 2>/dev/null | sed -n 's/^available: \([0-9]\+\) nodes.*/\1/p')"
NNODES="${NNODES:-0}"
if [ -z "${THRESH:-}" ]; then
  if [ "$NNODES" -ge 8 ]; then THRESH=0.70; else THRESH=1.15; fi
  echo "== [0/2] 拓扑:${NNODES:-?} 个 NUMA node ⇒ 阈值 ${THRESH} ms/层(8+ node 的历史口径是 0.70)"
else
  echo "== [0/2] 拓扑:${NNODES:-?} 个 NUMA node,阈值由调用方指定 ${THRESH} ms/层"
fi
echo "   (node 数是引擎行为的输入:nshard_=max(1,node/world);跨会话比数字前先比它)"
REP="${REP:-60}"
FAIL=0

echo "== [1/2] 数值门禁 test_block23_equiv.py =="
XIAOTU_LAYER1_NPZ="$NPZ_EQ" timeout 300 "$PY" "$ROOT/scripts/test_block23_equiv.py" > /tmp/eq_aligned.txt 2>&1
if [ ! -s /tmp/eq_aligned.txt ]; then echo "   !! 对拍无输出(检查 NPZ_EQ=$NPZ_EQ)"; tail -3 /tmp/eq_aligned.txt; fi
NOK=$(grep -acE '^\[OK' /tmp/eq_aligned.txt || true)
NBAD=$(grep -acE '^\[BAD' /tmp/eq_aligned.txt || true)
echo "   OK=$NOK BAD=$NBAD (me=1 的 BAD 是既有的 NR=8 fp32 重结合偏差,不计入)"
# me=2/3/混合/4..7 必须全 OK(共 7 项)
if [ "${NOK:-0}" -lt 7 ]; then echo "   !! 数值门禁未通过(需 7 OK)"; FAIL=1; else echo "   数值门禁通过"; fi

echo "== [2/2] 性能门禁 bench_engine_ab.py (阈值 ${THRESH} ms/层;每线程目标 2.2 GB/s) =="
# 每个 (列, k 组) 的专家权重字节:gate+up = 2*I*H/2、down = H*I/2 ⇒ 每专家 12.58 MB(I=2048,H=4096)
PER_EXP_MB=12.58
# 【§566】**阈值必须按形状(DEDUP)分别标定**。原来把只在 DEDUP=12 上标定过的
# `0.70 ms` / `1.20×` 直接套到 DEDUP=23,而**历史记录里 DEDUP=23 本来就是**
# "xiaotu 0.82-0.85 vs lk 0.67"(见 bench_engine_ab.py docstring 与 NOTES §119)⇒ 比值 1.24-1.27
# ⇒ 那个形状**永远报 FAIL**。一条永远红的门禁等于没有门禁(与"跨会话拿绝对值比"同类错误)。
# 记录基线(2026-09-17 热态复测与历史一致):
#   DEDUP=12: ours 0.65-0.69 / lk 0.57-0.61 ⇒ 比值 1.14-1.19
#   DEDUP=23: ours 0.82-0.85 / lk 0.66-0.67 ⇒ 比值 1.24-1.27
# 故 DEDUP=23 的 ms 门 = DEDUP=12 的 ×1.36(随拓扑一起缩放)。
# **比值门的余量按实测噪声带取**(2026-09-17 同日 6 次:x 0.69-0.85 / lk 0.56-0.66
# ⇒ 比值带 1.14-1.35,而 lk 自身的 run-to-run 漂移就有 12%)。取 1.30/1.40 留 ~5% 余量,
# 同时**仍能抓住 §505 那种 1.43× 的分片回归**(这是比值门存在的唯一理由)。
THRESH_12="${THRESH_12:-$THRESH}"
THRESH_23="${THRESH_23:-$(awk -v t="$THRESH" 'BEGIN{printf "%.2f", t*1.36}')}"
RATIO_MAX_12="${RATIO_MAX_12:-1.30}"
RATIO_MAX_23="${RATIO_MAX_23:-1.40}"
# 【§566】**必须给预热**:我们的引擎在**前几十次调用**里逐次变快
# (WARMUP=1→1.01、50→0.70、200→0.70、800→0.69 ms/层),而 lk 与调用次数无关(恒 0.58)。
# 原来两臂都不给 WARMUP(默认 1)⇒ 短跑把冷启动暂态摊进均值,报出**假 FAIL**
# (REP=60 报比值 1.62×;同尺子热态是 1.19-1.21×,与历史 0.65-0.68 一致 ⇒ 其实没有回归)。
# 服务里引擎每秒被调用上万次、始终热态 ⇒ **验收口径必须是热态**,故两臂都强制给 WARMUP。
WARMUP="${WARMUP:-200}"
for D in 12 23; do
  if [ "$D" = "12" ]; then TH="$THRESH_12"; RM="$RATIO_MAX_12"; else TH="$THRESH_23"; RM="$RATIO_MAX_23"; fi
  OUT=$(ENG=xiaotu XIAOTU_MOE_THREADS=120 XIAOTU_MOE_PROFILE=1 XIAOTU_LAYER1_NPZ="$NPZ" \
        BS=6 DEDUP="$D" REP="$REP" WARMUP="$WARMUP" timeout 600 "$PY" "$ROOT/scripts/bench_engine_ab.py" 2>&1)
  MS=$(echo "$OUT" | grep -E '^ +6 ' | tail -1 | awk '{print $2}')
  # 【§566】**必须只从聚合行 `[MOE-PROF]` 取 `na=`**:`[NS-PROF]` 也会按分桶打 `na=`,
  # 用 `tail -1` 会抓到最后一个**桶**行(实测 DEDUP=12 被读成 na=2 ⇒ 带宽算成 36 GB/s)。
  NA=$(echo "$OUT" | grep -o '\[MOE-PROF\].*' | tail -1 | grep -oE 'na=[0-9]+' | cut -d= -f2)
  [ -z "${NA:-}" ] && NA=$(echo "$OUT" | grep -oE 'na=[0-9]+' | tail -1 | cut -d= -f2)
  if [ -z "${MS:-}" ] || [ -z "${NA:-}" ]; then echo "   !! DEDUP=$D 未取到结果(ms=${MS:-?} na=${NA:-?})"; FAIL=1; continue; fi
  PT=$(awk -v na="$NA" -v ms="$MS" 'BEGIN{printf "%.2f", na*12.58/(ms*120)}')
  AGG=$(awk -v na="$NA" -v ms="$MS" 'BEGIN{printf "%.0f", na*12.58/ms}')
  MSOK=$(awk -v ms="$MS" -v th="$TH" 'BEGIN{print (ms<=th)?"PASS":"FAIL"}')
  PTOK=$(awk -v pt="$PT" 'BEGIN{print (pt>=2.2)?"PASS":"WARN"}')
  printf '   DEDUP=%-3s na=%-3s %s ms/层(门限 %s) 聚合 %s GB/s  每线程 %s GB/s  ms=%s per-thread=%s\n' \
         "$D" "$NA" "$MS" "$TH" "$AGG" "$PT" "$MSOK" "$PTOK"
  # ms 条款是硬门禁;每线程条款只告警:它在 DEDUP=12 这个"L3 驻留/每 CCD 交付"口径上受硬件限制
  # (见 NOTES §130/§133),而在服务端真实形状(na≈32)已达 2.9 GB/s·线程。
  # 【§525/§526 新增】**同日 lk 对照 + 比值门**(RATIO=0 可关,RATIO_MAX 默认 1.20)。
  # 为什么必须加:绝对阈值 0.70 在机器状态漂移时会把环境变化算在我们头上(2026-09-17 实测
  # 同日 lk 也慢了 1.14×);而 2026-09-16 把阈值放宽到 1.15 又**放过了 §505 的分片回归**
  # (同日比值 1.43×,绝对 ms 1.16 却"低于放宽后的阈值")。⇒ 判据改成**同日、同参、
  # 同机的 xiaotu/lk 比值**;绝对 ms 仍然打印并保留原阈值告警(便于和 §119 的历史数字比)。
  if [ "${RATIO:-1}" = "1" ] && [ -x "${LK_PY:-/home/user/anaconda3/envs/lvllmds4-x/bin/python}" ]; then
    LKMS="$(ENG=lk LK_THREADS="${LK_THREADS:-120}" CUDA_VISIBLE_DEVICES="${LK_GPU:-0}" \
            WARMUP="$WARMUP" \
            XIAOTU_LAYER1_NPZ="$NPZ" BS=6 DEDUP="$D" REP="$REP" timeout 600 \
            "${LK_PY:-/home/user/anaconda3/envs/lvllmds4-x/bin/python}" "$ROOT/scripts/bench_engine_ab.py" 2>&1 \
            | grep -E "^ +6 " | awk '{print $2}' | head -1)"
    if [ -n "${LKMS:-}" ]; then
      RAT=$(awk -v a="$MS" -v b="$LKMS" 'BEGIN{printf "%.2f", a/b}')
      ROK=$(awk -v r="$RAT" -v m="$RM" 'BEGIN{print (r<=m)?"PASS":"FAIL"}')
      printf '   â³ 同日 lk 对照 %s ms/层 ⇒ 比值 %s× (门限 ≤%s) %s\n' \
             "$LKMS" "$RAT" "$RM" "$ROK"
      [ "$ROK" = "PASS" ] || FAIL=1
    else
      echo "   ↳ 同日 lk 对照未取到结果(跳过比值门)"
    fi
  fi
  [ "$MSOK" = "PASS" ] || FAIL=1
done

[ "$FAIL" = 0 ] && echo "== 全部门禁通过 ==" || echo "== 有门禁未通过 =="
exit "$FAIL"
