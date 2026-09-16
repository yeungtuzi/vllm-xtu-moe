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
# 1M 上下文 KV:实测 `Available KV cache memory 17.45 GiB → 1,116,005 tokens`
KV_GIB_PER_MTOKEN = 17.45 / 1.116005      # ≈15.6 GiB / 1M tokens
GPU_PREFILL_GIB = 3.0           # MBT=8192 的激活/工作区(待精确测,先按 3.0 保守)
DRAFT_GIB_PER_RANK = 7.388 / 2  # mtp 全量 7.388 GiB,TP=2 ⇒ 每 rank 一半
RESIDENT_GIB_PER_LAYER = 3.36   # V4.1:6.72/TP(=2)
GPU_PREFILL_HOST_GIB = 253.0    # 当前实现下 GPU 预填充要额外占的**主机**内存/rank(§505g)

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
        why += f"; **额外占主机内存 {host_cost:.0f} GiB/rank ⇒ 按总约束判为不可用**(除非 XIAOTU_VRAM_ALLOW_HOST_COST=1)"
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
    # 投机解码:draft 上不上 GPU 由插件侧预留额度决定;这里给出"是否允许"
    lines.append(f"XIAOTU_DRAFT_ON_GPU={1 if p['spec_on_gpu'] else 0}")
    lines.append(f"XIAOTU_GPUPREFILL_WCOPY={1 if p['gpu_prefill'] else 0}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="R-VRAM 显存优先级规划器")
    ap.add_argument("--maxlen", type=int, default=int(os.environ.get("MAXLEN", 1048576)))
    ap.add_argument("--free-gib", type=float, default=None,
                    help="KV 之后**实测**的剩余显存(给了就用它,否则用解析口径)")
    ap.add_argument("--emit-env", action="store_true")
    a = ap.parse_args()
    p = plan(maxlen=a.maxlen, free_gib=a.free_gib)
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
