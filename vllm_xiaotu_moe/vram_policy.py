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
WEIGHTS_GIB = 9.78              # 【§592 实测修正】V4.1/TP=2 每 rank 的**非 KV 常驻占用**
                                # = vLLM 启动日志 "Actual usage is 9.78 GiB for consumed memory
                                #   (weights + non-torch)"(gpf3,util=0.90,无常驻)。
                                # 原值 7.4 只等于 "Model loading took 7.79 GiB" 里的一部分,
                                # 漏掉了 non-torch(CUDA context/通信缓冲等)⇒ 预算偏乐观。
                                # 峰值激活(0.73~1.22)+ CUDAGraph(0.12~0.70)由下面 reserve 覆盖。
# 1M 上下文 KV —— **实测,§509**(V4.1 / TP=2 / MBT=8192 / maxlen=1M):
#   `Available KV cache memory 2.71 GiB → GPU KV cache size 1,290,154 tokens`
#   ⇒ 2.20 KiB/token ⇒ **1M 上下文 ≈ 2.2 GiB/卡**(不是 16.4!)
# 注:早先记的 17.45 GiB/1.116M tok 来自别的配置(TP=1 的 v41el 口径)⇒ 用量级差 7×。
# 这个数直接决定"优先级 2/3/4 还剩多少显存",所以必须用**本配置**实测值。
KV_GIB_PER_MTOKEN = 2.71 / 1.290154      # ≈2.1 GiB / 1M tokens
# ⚠️【§592 实测异常,未解释】KV 的"每 token 成本"**随 maxlen 变化**,不是常数:
#   maxlen=1M  : 2.53 GiB → 1,206,255 tok(2.25 KiB/tok);12.01 GiB → 6,724,586 tok(1.79 KiB/tok);
#                 2.71 GiB → 1,290,154 tok(2.20 KiB/tok)   ← 三次互洽
#   maxlen=8192: 25.03 GiB → 1,600,788 tok(16.79 KiB/tok);11.21 GiB → 716,861 tok(16.74 KiB/tok)
#   ⇒ 1M 档的有效开销 ≈ 4 个独立 KV 组(正好是 `kv_source_layer_ids=[2,8,14,20]`),
#     而 8K 档 ≈ 26 层各自独立 —— 疑似 vLLM 的 hybrid allocator 在短 maxlen 下没走 KV 共享。
#   **本常量按 maxlen=1M 的口径标定**(对 1M 是保守的:实测 1.79 < 2.1);
#   短 maxlen 下 2.1 GiB/Mtoken 也只是 0.017 GiB,而 8K 真正需要的上界是
#   8192 × 16.8 KiB = 0.13 GiB ⇒ 两者都远小于 0.5 GiB 的下限,不影响判定。
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
# 【§589 修】**按实测口径**改为 6.7 GiB:实际 staging 是 `ns=4`(TP=2+node 分片)下的 6.7 GiB,
# 加 25% 余量需 8.4 GiB 空闲显存;原值 3.0 只有真实需求的一半 ⇒ 优先级 2 预留了显存
# 却永远过不了 preflight(且'部分层 GPU/部分层 CPU'的混合模式比纯 CPU 更慢)。
# 【§590,用户要求"按模型动态算,别写死"】**GPU 预填充的预留由模型维度现算**:
#   staging = 2 × (w13 + w2 + s13 + s2)   ← 开头的 2 就是 **ping/pong 双槽**
#   (与 `gpu_prefill.staging_bytes()` 同一公式,即 preflight 实际用的那个;单槽 = 3.36 GiB,
#    正是 RESIDENT_GIB_PER_LAYER,可见口径自洽。)
#   ⇒ 预留 = staging × 1.25(preflight 的 margin,默认 1.25)
# 维度来源:env `XIAOTU_VRAM_{EXPERTS,HIDDEN,INTER,GROUP_K}`;缺省用 V4.1 的值,
# 且 **inter 取每 rank 分片**(I_full/tp)。找不到维度时才回落到写死的 6.72 GiB。
GPU_PREFILL_GIB_FALLBACK = 6.72
GPU_PREFILL_MARGIN = 1.25     # preflight 的口径:fits_device() 要求 free ≥ staging × 1.25


def gpu_prefill_gib(n_experts: int, hidden: int, inter: int, group_k: int = 32) -> float:
    """与 `gpu_prefill.staging_bytes()` 同式的 staging 估算(GiB),不依赖 torch。"""
    E, H, I, gk = int(n_experts), int(hidden), int(inter), max(1, int(group_k))
    w13d = E * (2 * I) * (H // 2)
    w2d = E * H * (I // 2)
    s13d = E * (2 * I) * (H // gk)
    s2d = E * H * (I // gk)
    return 2 * (w13d + w2d + s13d + s2d) / 2**30


def _dims_from_ckpt() -> tuple[int, int, int] | None:
    """从 checkpoint 的 config.json 读 (n_routed_experts, hidden_size, moe_intermediate_size)。

    路径:`XIAOTU_VRAM_MODEL`(默认用 V4.1 的 model scope 快照)。读不到返回 None。
    **目的是让用户什么都不用填** —— 策略按当前模型自己算预留。
    """
    import json
    p = os.environ.get("XIAOTU_VRAM_MODEL") or (
        "/home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master")
    try:
        with open(os.path.join(p, "config.json")) as fh:
            c = json.load(fh)
        t = c.get("text_config", c)
        E = int(t.get("n_routed_experts", 0))
        H = int(t.get("hidden_size", 0))
        I = int(t.get("moe_intermediate_size", 0))
        return (E, H, I) if (E and H and I) else None
    except Exception:  # noqa: BLE001
        return None


def _prefill_staging_gib(tp: int = 2) -> float:
    """按模型维度算 staging;dim 不可得时回落到实测值。"""
    def _e(k, dflt):
        try:
            return int(os.environ.get(k, dflt))
        except (TypeError, ValueError):
            return int(dflt)
    E = _e("XIAOTU_VRAM_EXPERTS", 0)
    H = _e("XIAOTU_VRAM_HIDDEN", 0)
    I = _e("XIAOTU_VRAM_INTER", 0)
    GK = _e("XIAOTU_VRAM_GROUP_K", 32)
    if not (E > 0 and H > 0 and I > 0):
        dims = _dims_from_ckpt()          # ← 用户不填:从 checkpoint config 自取
        if dims:
            E, H, I = dims
    if E > 0 and H > 0 and I > 0:
        return gpu_prefill_gib(E, H, I // max(1, int(tp)), GK)
    return GPU_PREFILL_GIB_FALLBACK           # MBT=8192 的激活/工作区 + ping/pong 双槽的额外部分(待精确测,先保守按 3.0)
                                # 注:双槽本身 ≈ 2×一层 K-major(TP=2 每层 ~1.7 GB)≈ 3.4 GB,已含在此数内
DRAFT_GIB_PER_RANK = 7.388 / 2  # **草稿 = DSpark 投机解码**(算法名);其权重在 checkpoint 里
                                # 存为 `mtp.0/1/2`(config `num_nextn_predict_layers=3`),
                                # 沿用 DeepSeek-V3 的 MTP 命名 ⇒ **mtp 是权重命名,不是算法名**。
                                # 全量 7.388 GiB,TP=2 ⇒ 每 rank 一半
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
         allow_host_cost: bool | None = None,
         max_num_seqs: int | None = None,
         gpu_mem_util: float | None = None) -> dict:
    """按 R-VRAM 顺序给出判定。

    `free_gib` 给了就用**实测**口径(KV 之后的剩余显存);否则用解析口径
    `CARD_TOTAL − WEIGHTS − KV(maxlen)`。
    """
    kv_per_m = kv_gib_per_mtoken if kv_gib_per_mtoken is not None else _env_gib("XIAOTU_KV_GIB_PER_MTOKEN", KV_GIB_PER_MTOKEN)
    # 【§590】**先解析 tp**,后面所有"按模型现算"的量都依赖它(原先 _tp_eff 在下面才定义,
    # 导致这里 UnboundLocalError)。
    _tp_eff = max(1, int(_env_gib("XIAOTU_T_VRAM_TP", os.environ.get("TP", 2))))
    # GPU 预填充 staging(含 ping/pong 双槽)与"单层专家权重"(= staging/2)都按模型维度现算;
    # 显式传参或 env 仍可覆盖(调试用)。
    _dyn_pref = _env_gib("XIAOTU_GPUPREFILL_GIB", _prefill_staging_gib(_tp_eff))
    pref = gpu_prefill_gib if gpu_prefill_gib is not None else _dyn_pref
    draft = draft_gib if draft_gib is not None else _env_gib("XIAOTU_DRAFT_GIB_PER_RANK", DRAFT_GIB_PER_RANK)
    # 【§590】单层专家权重(每 rank)= staging/2(那一半就是单槽;实测 3.36 GiB @TP=2),
    # 同样**按模型现算**,不再让用户填 RESIDENT_GIB_PER_LAYER。
    per_layer = (resident_per_layer_gib if resident_per_layer_gib is not None
                 else _env_gib("XIAOTU_RESIDENT_GIB_PER_LAYER", _dyn_pref / 2.0))
    if allow_host_cost is None:
        allow_host_cost = os.environ.get("XIAOTU_VRAM_ALLOW_HOST_COST") == "1"
    host_cost = _env_gib("XIAOTU_GPUPREFILL_HOST_GIB", GPU_PREFILL_HOST_GIB)

    # 【§592,用户要求"用户不知道填多少 ⇒ 我们动态算"】**KV 需求量按 maxlen × 并发**算:
    # 服务承诺的是"每个请求都能用满 maxlen",所以 KV 必须至少装下 `maxlen × max_num_seqs`。
    # 这样"优先级 1 不被降级"是可验证的,而且剩余量可以**显式**留给预填充/投机
    # (见下面的 `--kv-cache-memory` 建议)。
    _seqs = max(1, int(max_num_seqs if max_num_seqs is not None else _env_gib(
        "XIAOTU_VRAM_MAX_SEQS", float(os.environ.get("MAXSEQS") or os.environ.get("MAX_NUM_SEQS") or 1.0))))
    _util = float(gpu_mem_util if gpu_mem_util is not None else _env_gib(
        "XIAOTU_VRAM_GPU_UTIL", float(os.environ.get("GPU_UTIL") or 0.90)))
    kv_gib = kv_per_m * (maxlen / 1_000_000.0) * _seqs
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
        f"需要 {kv_gib:.1f} GiB/卡(maxlen={maxlen} × {_seqs} 并发);"
        + ("" if kv_ok else " **不满足 ⇒ 整条链终止(本规则不允许降级上下文)**"),
    ))
    if not kv_ok:
        return {"kv_gib": kv_gib, "budget_gib": budget, "kv_ok": False, "steps": steps,
                "gpu_prefill": False, "spec_on_gpu": False, "resident_layers": []}

    # 优先级 2:GPU 预填充(受"不得额外占系统内存"约束)
    host_ok = allow_host_cost or host_cost <= 0.0
    # 【§589 修】必须按"实测 staging × 1.25"判(preflight 就是按这个口径卡的),
    # 否则会出现"策略说启用、preflight 全拒、退化成混合模式(比纯 CPU 还慢)"。
    pref_need = pref * GPU_PREFILL_MARGIN
    pref_ok = (budget >= pref_need) and host_ok
    why = f"剩余 {budget:.1f} GiB,需要 {pref_need:.1f} GiB(staging {pref:.2f}×{GPU_PREFILL_MARGIN})"
    # 【§592,用户裁决】TP=1 **不支持** GPU 预填充:staging 不随 TP 摊薄
    # (TP=2 每 rank 是 I/2,TP=1 是 I ⇒ staging 翻倍到 13.4 GiB、preflight 要 16.8),
    # 单卡还要放 KV+投机 ⇒ 必然逐层被拒、静默退化成 CPU。用户明确"没指望 TP=1 跑起来"。
    # 所以这里**直接判否**(而不是给个注定的乐观值),让调用方一眼看到要 TP≥2。
    if _tp_eff <= 1:
        pref_ok = False
        why = (f"TP=1:staging {pref:.2f} GiB ⇒ preflight 要 {pref_need:.1f} GiB 空闲,单卡放不下"
               f" ⇒ **GPU 预填充要求 --tensor-parallel-size >= 2**")
    if not host_ok:
        why += (f"; **额外占主机内存 {host_cost:.0f} GiB/rank ⇒ 按总约束判为不可用**"
                f"(除非 XIAOTU_VRAM_ALLOW_HOST_COST=1)")
    elif not (budget >= pref_need):
        why += " ⇒ 不够"
    steps.append(("2. GPU 预填充", pref_ok, why + ("(启用)" if pref_ok else " ⇒ fallback: CPU 预填充")))
    if pref_ok:
        budget -= pref_need

    # 优先级 3:GPU 投机解码
    spec_ok = budget >= draft
    steps.append(("3. GPU 投机解码(draft=mtp)",
                  spec_ok,
                  f"剩余 {budget:.1f} GiB,需要 {draft:.1f} GiB"
                  + ("(启用)" if spec_ok else " ⇒ fallback: 关闭投机解码")))
    if spec_ok:
        budget -= draft

    # 优先级 4:专家层常驻。
    # 【§584 实测】**默认改为 0 层** —— "常驻收益最高"这个前提被推翻:
    #   * 服务级(TP=2/8K/5 层常驻):TPOT 43.19 vs 无常驻 42.86 ms ⇒ **+0.33 ms(中性偏负)**;
    #     聚合更明显:C=4 41.42 vs **46.92 t/s(−12%)**;
    #   * 原因:GPU 小 M MoE 比 CPU 引擎**慢 3-5×**(M=1:1.272 vs 0.38 ms/层,随 M 线性),
    #     常驻是把"0.38 ms 的 CPU 计算"换成"1.27 ms 的 GPU 计算";
    #   * ⇒ 这块显存**改投 KV** 收益确定且单调(每层 3.36 GiB/rank ≈ 1.5M token KV)。
    # 想恢复旧行为:显式给 RESIDENT_LAYERS(或 `XIAOTU_RESIDENT_POLICY=1`)。
    n = 0
    if os.environ.get("XIAOTU_RESIDENT_POLICY") == "1":
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
    # ---- 【§590,用户要求】**容量自检 + 可执行提示** ----------------------------
    # 只要启用了 GPU 预填充,它的安全预留必须**在 KV 之外**留出来;留不出来时:
    #   ①显式给出该留的 KV 预算(供 `--kv-cache-memory`,否则 vLLM 会把空闲显存全吃进 KV,
    #     预填充的 staging 就过不了 preflight,退化成"部分层 GPU/部分层 CPU"的混合模式 —— 更慢);
    #   ②告诉用户两条可执行出路:关掉 GPU 预填充,或缩小 `--max-model-len`。
    held = (pref_need if pref_ok else 0.0) + (draft if spec_ok else 0.0)
    free_for_kv = max(0.0, CARD_TOTAL_GIB - WEIGHTS_GIB - reserve - held)
    kv_bytes = int(max(0.5, min(kv_gib, free_for_kv)) * 2**30)
    warn = None
    # 【§592 实测】**净空显存 = (1 − util) × CARD_TOTAL**,因为 vLLM 会把 KV 池填到 util 为止
    # (gpf3: util=0.90 ⇒ 只剩 3.09 GiB;gpf_on: util=0.55 ⇒ 17.77 GiB)。
    # 所以"策略说开、preflight 逐层拒绝"的根因就是**没显式给 --kv-cache-memory**。
    _guard_free = max(0.0, CARD_TOTAL_GIB * (1.0 - max(0.0, min(1.0, _util))))
    _need_all = pref_need + (draft if spec_ok else 0.0) + 0.5
    if pref_ok and _guard_free < _need_all:
        warn = (
            f"⚠️ `--gpu-memory-utilization={_util}` 会让 vLLM 把 KV 池填到 "
            f"{CARD_TOTAL_GIB * _util:.1f} GiB/卡,启动后只剩 {_guard_free:.1f} GiB 空闲;"
            f"而 GPU 预填充的 preflight 要 {pref_need:.1f} GiB"
            + (f" + 投机 {draft:.1f} GiB" if spec_ok else "")
            + " ⇒ **会被逐层拒绝并静默退回 CPU 预填充(比纯 CPU 还慢)**。\n"
            f"      解法:显式传 `--kv-cache-memory {kv_bytes}`(把 KV 池按 maxlen×{_seqs} 并发封顶),"
            f"剩下的 {CARD_TOTAL_GIB - WEIGHTS_GIB - reserve - kv_gib:.1f} GiB 才留给预填充/投机。\n"
            f"      这就是 R-VRAM 优先级 1(KV)之后、优先级 2/3 能拿到显存的**唯一**途径。"
            f"(详见 docs/RUNBOOK.md §5.8)"
        )
    if warn is None and pref_ok and kv_gib > free_for_kv + 1e-9:
        maxlen_ok = int(free_for_kv / max(1e-9, kv_per_m) * 1_000_000)
        warn = (
            f"⚠️ 显存放不下:maxlen={maxlen} 需要 KV {kv_gib:.1f} GiB/卡,"
            f"但 GPU 预填充安全预留 {pref_need:.1f} GiB(单层 staging {pref:.2f}×{GPU_PREFILL_MARGIN})"
            + (f" + 投机 {draft:.1f} GiB" if spec_ok else "")
            + f" 之后只剩 {free_for_kv:.1f} GiB ⇒ 请二选一:\n"
            f"      (a) 关掉 GPU 预填充:VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0"
            f"(上下文仍可 {maxlen},预填充回 CPU);或\n"
            f"      (b) 缩小上下文:--max-model-len ≤ {maxlen_ok}(KV ≤ {free_for_kv:.1f} GiB),"
            f"保留 GPU 预填充。\n"
            f"      另:务必把 KV 预算显式传进去(--kv-cache-memory {kv_bytes}),"
            f"否则 vLLM 会把空闲显存吃进 KV、预填充 staging 过不了 preflight。"
            f"(详见 docs/RUNBOOK.md §5.8)")
    return {"kv_gib": kv_gib, "budget_gib": budget, "kv_ok": True, "steps": steps,
            "gpu_prefill": pref_ok, "spec_on_gpu": spec_ok,
            "resident_layers": taken, "resident_spec": spec,
            "kv_bytes": kv_bytes, "kv_free_gib": free_for_kv, "warning": warn,
            "max_num_seqs": _seqs, "gpu_mem_util": _util,
            "guard_free_gib": _guard_free, "prefill_need_gib": pref_need}


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
    # 【§603】**阈值 1024 偏低**:实测 GPU 预填充是"每 chunk ≈8.9 s 固定 + 0.79 ms/token",
    # 而 CPU 是 ≈3.9 ms/token ⇒ 盈亏平衡在 **~2860 token**。用 1024 会让 1-3K 的 prompt
    # 白付 ~9 s(实测 L=1024 的 TTFT 从 CPU 的 ~4 s 变成 10.03 s)。这里取 **4096**。
    _gpm = _env_gib("XIAOTU_GPU_PREFILL_SWITCH_TOKENS", 4096)
    lines.append(f"VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS={int(_gpm) if p['gpu_prefill'] else 0}")
    lines.append(f"XIAOTU_GPU_RESIDENT_LAYERS={p.get('resident_spec','')}")
    # 【§508 更正】draft 上不上 GPU **不需要新旋钮**:hybrid_model.py:669-686 里
    # "同一 prefix 第二次构造 = draft",默认 `XIAOTU_MOE_RESIDENT_DRAFT=1` ⇒
    # **draft 层永远在 GPU**(实测 draft 走 CPU 时投机净负:7.11 vs 10.76 t/s)。
    # 所以这里只输出"要不要显式关掉它",而 spec 本身由 `--speculative-config` 决定。
    if p["spec_on_gpu"]:
        lines.append("XIAOTU_MOE_RESIDENT_DRAFT=1")   # 默认值,显式写出便于审计
    else:
        # 【§594 修】**绝不能输出 DRAFT=0**:那表示"draft 层不进 GPU",会走 CPU 引擎,
        # 而 §508 实测 draft 走 CPU 时投机是**净负**(7.11 vs 10.76 t/s)。
        # 正确做法是**根本不传 `--speculative-config`**(整块关掉投机),而不是把 draft 放 CPU。
        lines.append("# ⚠️ 优先级 3 不足:请**不要**传 --speculative-config(整块关掉投机);"
                     "把 draft 放 CPU 是净负(§508:7.11 vs 10.76 t/s)")
        lines.append("XIAOTU_MOE_RESIDENT_DRAFT=1")
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
    # 【B134 修复】下限必须**高于引擎的最低需求**,否则服务会以
    #   ValueError: To serve at least one request (N tokens), X GiB KV cache is needed
    # 直接起不来(实测:vLLM 对 131072 token 要 0.66 GiB,而原下限 0.5 GiB < 它)。
    # 另:原来算出 _kv_bytes 却下发 p['kv_bytes'] ⇒ 设计好的 slack 从未生效,这里一并修掉。
    # ⚠️ 这仍是**保底**;每-token KV 的模型常量(KV_GIB_PER_MTOKEN)按 V4.1 标定,对 GQA 系模型
    #   会低估(见 EXPERIMENTS B134/B135),把长上下文容量算准是独立工作项。
    _kv_floor = max(1 << 30, int(_env_gib("XIAOTU_KV_CACHE_FLOOR", 1.0) * (1 << 30)))
    if _kv_bytes < _kv_floor:
        _kv_bytes = _kv_floor
    lines.append(f"XIAOTU_KV_CACHE_BYTES={_kv_bytes}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="R-VRAM 显存优先级规划器")
    ap.add_argument("--maxlen", type=int, default=int(os.environ.get("MAXLEN", 1048576)))
    ap.add_argument("--free-gib", type=float, default=None,
                    help="KV 之后**实测**的剩余显存(给了就用它,否则用解析口径)")
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("MAXSEQS") or os.environ.get("MAX_NUM_SEQS") or 1),
                    help="KV 需要装下 maxlen × 该并发数(默认取 MAXSEQS/MAX_NUM_SEQS,否则 1)")
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("GPU_UTIL") or 0.90),
                    help="用于算「vLLM 填池后还剩多少净空」=(1-util)×39.49 GiB")
    ap.add_argument("--emit-env", action="store_true")
    ap.add_argument("--tp", type=int, default=int(__import__("os").environ.get("TP", 2)),
                    help="张量并行度:每层常驻/draft 的每卡占用按 6.72/TP 与 7.388/TP 缩放(默认 2)")
    a = ap.parse_args()
    _tp = max(1, int(getattr(a, "tp", 2)))
    os.environ["TP"] = str(_tp)   # 让 plan() 内部的 TP 相关上限能读到(§554)
    _scale = 2.0 / _tp
    p = plan(maxlen=a.maxlen, resident_per_layer_gib=None, gpu_prefill_gib=None,
             draft_gib=7.388 / _tp, free_gib=a.free_gib,
             max_num_seqs=a.max_num_seqs, gpu_mem_util=a.gpu_mem_util)
    if a.emit_env:
        print(emit_env(p))
        return 0
    print(f"[vram-policy] maxlen={a.maxlen} × {p['max_num_seqs']} 并发  util={p['gpu_mem_util']:.2f}"
          f"  ⇒ 填池后净空 {p['guard_free_gib']:.1f} GiB(预填充 preflight 要 {p['prefill_need_gib']:.1f})")
    print(f"[vram-policy] maxlen={a.maxlen}  KV 需要 {p['kv_gib']:.1f} GiB/卡  "
          f"{'OK' if p['kv_ok'] else '**不满足(不可降级)**'}")
    if p.get("warning"):
        print(p["warning"])
    for name, ok, why in p["steps"]:
        print(f"  {'✅' if ok else '❌'} {name:28s} {why}")
    print(f"  ⇒ 结论: gpu_prefill={p['gpu_prefill']}  spec_on_gpu={p['spec_on_gpu']}  "
          f"resident='{p.get('resident_spec','')}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
