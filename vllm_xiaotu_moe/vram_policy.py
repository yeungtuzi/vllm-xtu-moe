#!/usr/bin/env python3
"""显存分配的**优先级规划器** —— 把 IRON_RULES 的 R-VRAM 变成可执行的判定。

用户 2026-09-16 明确指示(以后一律照此办理),优先级**固定**为:

    1. 保证 **1M 上下文** —— KV cache 必须先按 max_model_len 留够(**不可降级**)
    2. 在此基础上,优先安排 **GPU 预填充**
    3. 若显存还够,**开启 GPU 投机解码**(DSpark drafter 放 GPU)
    4. 若显存还够,**多放一些专家层**(放收益最高的 = 解码器 20-39)

任何一项不满足 ⇒ **默认关闭**(fallback):
    * GPU 预填充 → CPU 预填充
    * 投机解码 → 关闭
    * 专家层常驻 → 全部放主机内存(常驻 0)

**总约束:不允许额外多占系统内存。**

这条总约束有一个立刻可判定的后果:今天的 GPU 预填充会给 K-major 缓存**再复制一份权重**
(实测 **+253 GiB/rank**,NOTES §505g)。在把这份拷贝消掉之前(修法:用引擎 C++ 侧的权重快照 +
`copy_hostbuf_to_device()`,不要在 Python 里 clone),按本规则 **GPU 预填充应当判为"不可用"
⇒ 退回 CPU 预填充**;而不是"开了 GPU 预填充再多占 253 GiB/rank"。
想有意破例:`XIAOTU_VRAM_ALLOW_HOST_COST=1`(会把理由打进日志)。

用法(自检,不需要起服务):
    python -m vllm_xiaotu_moe.vram_policy                      # 用本机默认账本打印规划
    python -m vllm_xiaotu_moe.vram_policy --maxlen 1048576
    python -m vllm_xiaotu_moe.vram_policy --emit-env           # 只输出可 export 的 env
    python -m vllm_xiaotu_moe.vram_policy --free-gib 12.5      # 用**实测**的 KV 之后剩余显存

License: Apache-2.0
"""
from __future__ import annotations

import argparse
import os

# ---- 账本默认值(全部来自本机实测;可用 CLI/env 覆盖)---------------------------
CARD_TOTAL_GIB = 39.49          # A100-40G 实际可用
WEIGHTS_GIB = 7.4               # V4.1 TP=2 每 rank 的模型权重(日志 "Model loading took 7.79 GiB")
# 1M 上下文 KV —— **实测,§509**(V4.1 / TP=2 / MBT=8192 / maxlen=1M):
#   `Available KV cache memory 2.71 GiB → GPU KV cache size 1,290,154 tokens`
#   ⇒ 2.20 KiB/token ⇒ **1M 上下文 ≈ 2.2 GiB/卡**(不是 16.4!)
# 注:早先记的 17.45 GiB/1.116M tok 来自别的配置(TP=1 的 v41el 口径)⇒ 用量级差 7×。
# 这个数直接决定"优先级 2/3/4 还剩多少显存",所以必须用**本配置**实测值。
KV_GIB_PER_MTOKEN = 2.71 / 1.290154      # ≈2.1 GiB / 1M tokens
# 预填充/激活/CUDA graph 之外的**保守余量**:§509 实测"解析式说 6 层常驻、6 层直接 OOM(34.8/40GB);
# 2 层稳过(27.1GB)" ⇒ 解析式必须留安全垫,否则会把用户推到 OOM。
VRAM_RESERVE_GIB = 4.0
# 【§551 修 §513c 的 32K 预填充 OOM】**序列长度相关的工作区**必须进预算,否则 KV 会把卡占满、
# 长预填充的 attention/indexer 工作区无处可放。实测锚点(§513c):maxlen≥32768 的那次请求
# 在 reserve=4 GiB + GPU_PREFILL=3 GiB 下 OOM,报错时**只差 508 MiB** ⇒ 32K 档至少要多留 ~5 GiB。
# 做法:按 32K 线性、但**封顶**(分块预填充下每个 chunk 的工作区有界,不能随 maxlen 无界增长;
# 若线性外推到 1M 会得出 ~160 GiB 的荒谬值)。两个数都可用 env 覆盖以便实测标定。
LONG_SEQ_WORKSPACE_GIB_PER_32K = 5.0
LONG_SEQ_WORKSPACE_CAP_GIB = 6.0
GPU_PREFILL_GIB = 3.0           # MBT=8192 的激活/工作区 + ping/pong 双槽的额外部分(待精确测,先保守按 3.0)
                                # 注:双槽本身 ≈ 2×一层 K-major(TP=2 每层 ~1.7 GB)≈ 3.4 GB,已含在此数内
DRAFT_GIB_PER_RANK = 7.388 / 2  # mtp 全量 7.388 GiB,TP=2 ⇒ 每 rank 一半
RESIDENT_GIB_PER_LAYER = 3.36   # V4.1:6.72/TP(=2)
GPU_PREFILL_HOST_GIB = 0.0      # 【§508 更正】GPU 预填充走 ping/pong(2 槽)+ 从**引擎分片**直接填 K-major
                                # ⇒ 主机侧代价 **0**。此前记的 253 GiB/rank 是"Python 侧再 clone 一份"
                                # 的**兜底路径**被我的条件默认误开所致,不是 GPU 预填充的必要代价。

# 优先级 4 的"收益最高"顺序:解码只跑解码器 ⇒ 解码器(20-39)在前,且靠前的更早被执行
RESIDENT_PRIORITY = [f"{i}" for i in range(20, 40)] + [f"{i}" for i in range(0, 20)]


def _env_gib(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def plan(*, maxlen: int, free_gib: float | None = None,
         kv_gib_per_mtoken: float | None = None,
         gpu_prefill_gib: float | None = None,
         draft_gib: float | None = None,
         resident_per_layer_gib: float | None = None,
         allow_host_cost: bool | None = None) -> dict:
    """按 R-VRAM 顺序给出判定。

    `free_gib` 给了就用**实测**口径(KV 之后的剩余显存);否则用解析口径
    `CARD_TOTAL − WEIGHTS − KV(maxlen)`。
    """
    kv_per_m = kv_gib_per_mtoken if kv_gib_per_mtoken is not None else _env_gib("XIAOTU_KV_GIB_PER_MTOKEN", KV_GIB_PER_MTOKEN)
    pref = gpu_prefill_gib if gpu_prefill_gib is not None else _env_gib("XIAOTU_GPUPREFILL_GIB", GPU_PREFILL_GIB)
    draft = draft_gib if draft_gib is not None else _env_gib("XIAOTU_DRAFT_GIB_PER_RANK", DRAFT_GIB_PER_RANK)
    per_layer = resident_per_layer_gib if resident_per_layer_gib is not None else _env_gib("XIAOTU_RESIDENT_GIB_PER_LAYER", RESIDENT_GIB_PER_LAYER)
    if allow_host_cost is None:
        allow_host_cost = os.environ.get("XIAOTU_VRAM_ALLOW_HOST_COST") == "1"
    host_cost = _env_gib("XIAOTU_GPUPREFILL_HOST_GIB", GPU_PREFILL_HOST_GIB)

    kv_gib = kv_per_m * (maxlen / 1_000_000.0)
    budget = free_gib if free_gib is not None else (CARD_TOTAL_GIB - WEIGHTS_GIB - kv_gib)
    # 留出保守余量(激活/cudagraph/碎片);§509 的实测证据见 VRAM_RESERVE_GIB 注释
    # 【§551】基础安全垫 + 序列长度相关工作区(按 32K 锚点线性、封顶;§513c 的 32K OOM 处方)
    per32k = _env_gib("XIAOTU_VRAM_RESERVE_WORKSPACE_GIB_PER_32K", LONG_SEQ_WORKSPACE_GIB_PER_32K)
    cap_ws = _env_gib("XIAOTU_VRAM_RESERVE_WORKSPACE_CAP_GIB", LONG_SEQ_WORKSPACE_CAP_GIB)
    long_ws = min(cap_ws, per32k * (maxlen / 32768.0))
    # 【§552 实测】maxlen=1M 时 vLLM 自己的"非 KV"占用约 20 GiB(KV 上限 2.53 GiB、权重 ~8 GiB 的前提下
    # 实测峰值 30.7 GiB)⇒ 1M 档只剩很少空间给常驻层。对照组:3 层常驻(10 GiB)时 **32K 预填充 OOM**,
    # 0 层时通过(剩 9.7 GiB)。所以按 R-VRAM 优先级(1M 上下文 > 常驻层)在这里**给 maxlen 加硬上限**。
    # 边界待更细的 maxlen 扫描来替换成公式;当前用保守阶跃(实测边界在 2-3 层之间)。
    # 【§554 实测,TP 相关】1M 档:TP=2 实测 2 层通过(峰值 38.25/38.28 GiB,余 ~2.2 GiB)、3 层 OOM;
    # TP=1 实测 **0 层**时峰值 35.4 GiB(余 ~5 GiB)⇒ 再加 1 层(+6.72)必超 ⇒ TP=1 上限 0。
    # 根因:maxlen 很长时 vLLM 自身的非 KV 占用(~20 GiB)不在解析式里,只能靠实测标定。
    _tp_eff = max(1, int(_env_gib("XIAOTU_T_VRAM_TP", os.environ.get("TP", 2))))
    resident_cap = None
    if maxlen >= 1_000_000:
        resident_cap = 2 if _tp_eff <= 2 else 0
        if _tp_eff == 1:
            resident_cap = 0
    elif maxlen >= 262_144:
        resident_cap = 3 if _tp_eff >= 2 else 1
    reserve = VRAM_RESERVE_GIB + long_ws
    budget = max(0.0, budget - reserve)

    steps: list[tuple[str, bool, str]] = []

    # 优先级 1:1M 上下文 —— 不可降级
    kv_ok = kv_gib <= (free_gib if free_gib is not None else CARD_TOTAL_GIB - WEIGHTS_GIB)
    steps.append((
        "1. 1M 上下文(KV)",
        kv_ok,
        f"需要 {kv_gib:.1f} GiB/卡(maxlen={maxlen});"
        + ("" if kv_ok else " **不满足 ⇒ 整条链终止(本规则不允许降级上下文)**"),
    ))
    if not kv_ok:
        return {"kv_gib": kv_gib, "budget_gib": budget, "kv_ok": False, "steps": steps,
                "gpu_prefill": False, "spec_on_gpu": False, "resident_layers": []}

    # 优先级 2:GPU 预填充(受"不得额外占系统内存"约束)
    host_ok = allow_host_cost or host_cost <= 0.0
    pref_ok = (budget >= pref) and host_ok
    why = f"剩余 {budget:.1f} GiB,需要 {pref:.1f} GiB"
    if not host_ok:
        why += (f"; **额外占主机内存 {host_cost:.0f} GiB/rank ⇒ 按总约束判为不可用**"
                f"(除非 XIAOTU_VRAM_ALLOW_HOST_COST=1)")
    elif not (budget >= pref):
        why += " ⇒ 不够"
    steps.append(("2. GPU 预填充", pref_ok, why + ("(启用)" if pref_ok else " ⇒ fallback: CPU 预填充")))
    if pref_ok:
        budget -= pref

    # 优先级 3:GPU 投机解码
    spec_ok = budget >= draft
    steps.append(("3. GPU 投机解码(draft=mtp)",
                  spec_ok,
                  f"剩余 {budget:.1f} GiB,需要 {draft:.1f} GiB"
                  + ("(启用)" if spec_ok else " ⇒ fallback: 关闭投机解码")))
    if spec_ok:
        budget -= draft

    # 优先级 4:专家层常驻(收益最高 = 解码器靠前的层)
    n = int(budget // per_layer) if per_layer > 0 else 0
    n = max(0, min(n, len(RESIDENT_PRIORITY)))
    # 【§552 实测硬上限】maxlen 很长时 vLLM 自己的"非 KV"占用很大(1M 实测峰值 30.7 GiB),
    # 解析式会高估可用空间 ⇒ 用实测标定的阶跃上限兜住(3 层常驻时 32K 预填充 OOM,0 层通过,边界 2-3)。
    if resident_cap is not None and n > resident_cap:
        n = resident_cap
    taken = RESIDENT_PRIORITY[:n]
    # 把连续的编号压成区间写法(20,21,22 -> "20-22")
    spec = _ranges([int(x) for x in taken])
    steps.append(("4. 专家层常驻",
                  n > 0,
                  f"剩余 {budget:.1f} GiB / 每层 {per_layer:.2f} GiB ⇒ 可放 {n} 层:{spec or '(0)'}"
                  + ("" if n > 0 else " ⇒ fallback: 全部放主机内存")))
    return {"kv_gib": kv_gib, "budget_gib": budget, "kv_ok": True, "steps": steps,
            "gpu_prefill": pref_ok, "spec_on_gpu": spec_ok,
            "resident_layers": taken, "resident_spec": spec}


def _ranges(xs: list[int]) -> str:
    """[20,21,22,25] -> '20-22,25'"""
    if not xs:
        return ""
    xs = sorted(set(xs))
    out, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x == p + 1:
            p = x
            continue
        out.append(str(s) if s == p else f"{s}-{p}")
        s = p = x
    out.append(str(s) if s == p else f"{s}-{p}")
    return ",".join(out)


def emit_env(p: dict) -> str:
    """把判定转成可 export 的 env(服务脚本直接 eval 即可)。"""
    lines = []
    # 总约束:GPU 预填充要么用"零主机代价"的实现,要么不开
    lines.append(f"VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS={1024 if p['gpu_prefill'] else 0}")
    lines.append(f"XIAOTU_GPU_RESIDENT_LAYERS={p.get('resident_spec','')}")
    # 【§508 更正】draft 上不上 GPU **不需要新旋钮**:hybrid_model.py:669-686 里
    # "同一 prefix 第二次构造 = draft",默认 `XIAOTU_MOE_RESIDENT_DRAFT=1` ⇒
    # **draft 层永远在 GPU**(实测 draft 走 CPU 时投机净负:7.11 vs 10.76 t/s)。
    # 所以这里只输出"要不要显式关掉它",而 spec 本身由 `--speculative-config` 决定。
    if p["spec_on_gpu"]:
        lines.append("XIAOTU_MOE_RESIDENT_DRAFT=1")   # 默认值,显式写出便于审计
    else:
        lines.append("XIAOTU_MOE_RESIDENT_DRAFT=0")   # 显存真不够时:关掉投机(优先级 3 的 fallback)
    # 【§508】**恒 0**:GPU 预填充的活跃路径从引擎分片直接填 K-major,不需要 Python 副本;
    # 只有"引擎没有分片"的兜底路径才建副本(那时会打告警)。
    lines.append("XIAOTU_GPUPREFILL_WCOPY=0")
    # 【§509】**顺序问题的唯一可靠解法**:常驻层是在模型加载期分配的,而 KV cache
    # 是**之后**才由 vLLM 定尺寸的 ⇒ 若不设预算,常驻层会先吃掉 KV 的空间,
    # 就会出现"1M 上下文(优先级 1)反而被挤掉"——违反 R-VRAM。
    # 所以这里把"优先级 1/2/3 之后剩下的 GiB"显式交给引擎的预算旋钮,
    # 让常驻层**只能在这个额度内**贪心放置(引擎会逐层试、超了就跳过)。
    lines.append(f"XIAOTU_MOE_RESIDENT_BUDGET_GB={max(0.0, p.get('budget_gib', 0.0)):.1f}")
    # 【§551 新增,修 §513c 的 32K 预填充 OOM】**显式给 KV cache 设上限**。
    # 为什么必须这样做:vLLM 按 `--gpu-memory-utilization` 把卡填到 95%,留给激活的只有它自己
    # 剖析出的那点余量;而**长序列预填充的 attention/indexer 工作区随序列长度增长**(§513c:
    # 32K 输入 OOM 时只差 508 MiB)⇒ KV 必须让出一块。这里按"maxlen 真正需要多少 KV"再给 15% 余量
    # 作为上限,超出的显存全部留给工作区/激活。上限可用 XIAOTU_KV_CACHE_BYTES 覆盖(便于实测标定)。
    _slack = _env_gib("XIAOTU_KV_CACHE_SLACK", 1.15)
    _kv_bytes = int(p.get("kv_gib", 0.0) * _slack * (1 << 30))
    if _kv_bytes < (1 << 29):          # 至少 0.5 GiB,避免极端 maxlen 下把 KV 压到不可用
        _kv_bytes = 1 << 29
    lines.append(f"XIAOTU_KV_CACHE_BYTES={_kv_bytes}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="R-VRAM 显存优先级规划器")
    ap.add_argument("--maxlen", type=int, default=int(os.environ.get("MAXLEN", 1048576)))
    ap.add_argument("--free-gib", type=float, default=None,
                    help="KV 之后**实测**的剩余显存(给了就用它,否则用解析口径)")
    ap.add_argument("--emit-env", action="store_true")
    ap.add_argument("--tp", type=int, default=int(__import__("os").environ.get("TP", 2)),
                    help="张量并行度:每层常驻/draft 的每卡占用按 6.72/TP 与 7.388/TP 缩放(默认 2)")
    a = ap.parse_args()
    _tp = max(1, int(getattr(a, "tp", 2)))
    os.environ["TP"] = str(_tp)   # 让 plan() 内部的 TP 相关上限能读到(§554)
    _scale = 2.0 / _tp
    p = plan(maxlen=a.maxlen, resident_per_layer_gib=6.72 / _tp, gpu_prefill_gib=None, draft_gib=7.388 / _tp, free_gib=a.free_gib)
    if a.emit_env:
        print(emit_env(p))
        return 0
    print(f"[vram-policy] maxlen={a.maxlen}  1M-KV 需要 {p['kv_gib']:.1f} GiB/卡  "
          f"{'OK' if p['kv_ok'] else '**不满足(不可降级)**'}")
    for name, ok, why in p["steps"]:
        print(f"  {'✅' if ok else '❌'} {name:28s} {why}")
    print(f"  ⇒ 结论: gpu_prefill={p['gpu_prefill']}  spec_on_gpu={p['spec_on_gpu']}  "
          f"resident='{p.get('resident_spec','')}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
