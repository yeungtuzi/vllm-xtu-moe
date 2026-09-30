# SPDX-License-Identifier: Apache-2.0
"""UVA 零拷贝预填路径(实验一产物 ✓;`XIAOTU_GPF_UVA=1` 打开,默认关 ✓)。

思路(用户 2026-09-30 定的两条原则 ✓):
1. **内存廉价、prefill 期 CPU 廉价** ⇒ 让 CPU 把权重预排成 GPU 最舒服的形态:
   **严格连续读** ⇒ CPU 侧按 ``[E, n_block, rows, BLK]`` 重排一次(一次性、可缓存 ✓)
2. **永远有准备好的层** ⇒ 深环常备(由调用方 `mixed_experts` 的 slot 机制提供 ✓)

实测依据(见 `dev-docs/UVA_ZEROCOPY_EXPERIMENT.md` ✓):
* E1:UVA **顺序读 = 25.02 GiB/s**(= 批量 DMA 线速 ✓);跨步读崩塌 ✗
* E2:原寻址 UVA = 9.47 GiB/s ✗(内核是"读 64B 连续 + 跳 4096B" ✗)
* E3:仅换寻址 ⇒ **25.03 GiB/s = 线速 100%** ✓
* E3-full:真实整层 ⇒ 710 → **389 ms**(1.83× ✓),但仍比 230 ms 基线慢 1.7× ✗
  ⇒ 缺口在"读与计算串行" ✗ ⇒ 本模块用 `num_stages` 流水 + 深环来补 ✓

⚠️ 与主路径的关系:本模块**不改变**默认行为 ✓;`XIAOTU_GPF_UVA=0`(默认)时一切照旧 ✓。
"""

from __future__ import annotations

import os
import time

import torch
import triton
import triton.language as tl

from vllm_xiaotu_moe.gpu_prefill import _dequant_e2m1_gather, _dequant_e2m1_nibble

UVA_BLK = 64


def uva_enabled() -> bool:
    return os.environ.get("XIAOTU_GPF_UVA", "0") == "1"


def _permute_kmajor(t: torch.Tensor, blk: int = UVA_BLK) -> torch.Tensor:
    """把 K-major ``[E, rows, cols]`` 预排成 ``[E, n_block, rows, blk]``(CPU ✓)。

    目的:内核按"固定 n_block、遍历 rows"读取时**地址连续** ✓(E3 实测 25.03 GiB/s ✓)。
    """
    e, r, c = t.shape
    if c % blk:
        raise ValueError(f"cols {c} 不能被 blk {blk} 整除")
    return t.reshape(e, r, c // blk, blk).permute(0, 2, 1, 3).contiguous()


def build_seq_mirror(w13, s13, w2, s2, blk: int = UVA_BLK):
    """输入 4 个 **宿主 K-major** 张量(如 `_pinned_kmajor` 的产物 ✓)⇒ 返回 UVA 视图四元组 ✓。"""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    out = []
    for t in (w13, s13, w2, s2):
        host = t.detach().to("cpu", torch.uint8).contiguous()
        p = _permute_kmajor(host, blk)
        if not p.is_pinned():
            p = p.pin_memory()
        out.append(get_accelerator_view_from_cpu_tensor(p))
    return tuple(out)


@triton.jit
def _gate_up_uva(x_ptr, x_stride, tok_ptr, seg_ptr, w13_ptr, s13_ptr,
                 inter_ptr, inter_ld, W13_E, S13_E,
                 H: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 ROWS: tl.constexpr, SROWS: tl.constexpr, NS: tl.constexpr, LUT: tl.constexpr):
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * W13_E + pid_n * (ROWS * BN)
    s_off = pid_e.to(tl.int64) * S13_E + pid_n * (SROWS * BN)
    c = tl.arange(0, BN)
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.range(0, H, BK, num_stages=NS):
            a = tl.load(x_ptr + toks[:, None] * x_stride + (kk + tl.arange(0, BK))[None, :],
                        mask=rmask[:, None], other=0.0)
            r = (kk >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w13_ptr + w_off + r[:, None] * BN + c[None, :])
            if LUT:
                w_e = _dequant_e2m1_gather(b_lo & 0x0F, None)
                w_o = _dequant_e2m1_gather((b_lo >> 4) & 0x0F, None)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BN))
            sr = (kk >> 5) + tl.arange(0, BK // 32)
            scale = tl.load(s13_ptr + s_off + sr[:, None] * BN + c[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BN)), (BK, BN))
            acc = tl.dot(a, (b_val * sc).to(tl.bfloat16), acc)
        tl.store(inter_ptr + g_rows[:, None] * inter_ld + (pid_n * BN + c)[None, :],
                 acc.to(tl.bfloat16), mask=rmask[:, None])
        mt += 1


@triton.jit
def _down_uva(inter_ptr, inter_ld, tok_ptr, wts_ptr, seg_ptr, w2_ptr, s2_ptr,
              out_ptr, out_ld, W2_E, S2_E,
              I: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
              ROWS: tl.constexpr, SROWS: tl.constexpr, NS: tl.constexpr, LUT: tl.constexpr):
    pid_e = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * W2_E + pid_h * (ROWS * BH)
    s_off = pid_e.to(tl.int64) * S2_E + pid_h * (SROWS * BH)
    c = tl.arange(0, BH)
    mt = 0
    while mt * BM < (end - base):
        g_rows = base + mt * BM + tl.arange(0, BM)
        rmask = g_rows < end
        toks = tl.load(tok_ptr + g_rows, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g_rows, mask=rmask, other=0.0).to(tl.float32)
        acc = tl.zeros((BM, BH), dtype=tl.float32)
        for kk in tl.range(0, I, BK, num_stages=NS):
            gate = tl.load(inter_ptr + g_rows[:, None] * inter_ld + (kk + tl.arange(0, BK))[None, :],
                           mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(inter_ptr + g_rows[:, None] * inter_ld
                         + (I + kk + tl.arange(0, BK))[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            act = (gate / (1.0 + tl.math.exp(-gate))) * up
            r = (kk >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w2_ptr + w_off + r[:, None] * BH + c[None, :])
            if LUT:
                w_e = _dequant_e2m1_gather(b_lo & 0x0F, None)
                w_o = _dequant_e2m1_gather((b_lo >> 4) & 0x0F, None)
            else:
                w_e = _dequant_e2m1_nibble(b_lo & 0x0F)
                w_o = _dequant_e2m1_nibble((b_lo >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BH))
            sr = (kk >> 5) + tl.arange(0, BK // 32)
            scale = tl.load(s2_ptr + s_off + sr[:, None] * BH + c[None, :])
            sc = tl.exp2(scale.to(tl.float32) - 127.0)
            sc = tl.reshape(tl.broadcast_to(sc[:, None, :], (BK // 32, 32, BH)), (BK, BH))
            acc = tl.dot(act.to(tl.bfloat16), (b_val * sc).to(tl.bfloat16), acc)
        tl.atomic_add(out_ptr + toks[:, None] * out_ld + (pid_h * BH + c)[None, :],
                      acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


def gpu_moe_layer_uva(x, topk_ids, topk_weights, mirror, H: int, I: int, K: int, *,
                      device: torch.device, inter=None, out=None,
                      BM: int | None = None, BN: int = 64, BK: int = 64, BH: int | None = None,
                      NS: int = 2, LUT: bool = False):
    # 默认值 = 实测定稿(dev-docs/UVA_ZEROCOPY_EXPERIMENT.md §10 ✓):
    #   BM=128 关键 —— `mt` 在 `kk` 外层 ⇒ BM=64 会把权重读【2 遍】✗ ⇒ 128 只读 1 遍 ⇒ 2× ✓
    #   warps=8 —— 管用的是占用率,不是流水深度 ✓
    # 全部可用 XIAOTU_GPU_PREFILL_* 覆盖 ✓
    if BM is None:
        BM = int(os.environ.get("XIAOTU_GPU_PREFILL_BM", "128") or 128)
    if BH is None:
        BH = int(os.environ.get("XIAOTU_GPU_PREFILL_BH", "64") or 64)
    """UVA 版一层 MoE(语义与 `gpu_prefill.gpu_moe_layer` 对齐 ✓,权重从宿主直读 ✓)。"""
    from vllm_xiaotu_moe.gpu_prefill import _build_segmentation

    device = torch.device(device)
    w13, s13, w2, s2 = mirror
    T, E = x.shape[0], w13.shape[0]
    tok, wts, seg, A = _build_segmentation(topk_ids, topk_weights, E, device)
    if out is None:
        out = torch.zeros((T, H), dtype=torch.bfloat16, device=device)
    if T == 0 or K == 0:
        return out
    if inter is None:
        inter = torch.empty((A, 2 * I), dtype=torch.bfloat16, device=device)
    if os.environ.get("XIAOTU_GPF_UVA_DBG", "1") == "1":
        print(f"[uva] use BM={BM} BH={BH} NS={NS} mirrors={len(_MIRRORS)}", flush=True)
    _gate_up_uva[(E, triton.cdiv(2 * I, BN))](
        x, x.stride(0), tok, seg, w13, s13, inter, inter.stride(0),
        w13.stride(0), s13.stride(0),
        H=H, BM=BM, BN=BN, BK=BK, ROWS=H // 2, SROWS=H // 32, NS=NS, LUT=LUT,
        num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "8")))
    _down_uva[(E, triton.cdiv(H, BH))](
        inter, inter.stride(0), tok, wts, seg, w2, s2, out, out.stride(0),
        w2.stride(0), s2.stride(0),
        I=I, BM=BM, BH=BH, BK=BK, ROWS=I // 2, SROWS=I // 32, NS=NS, LUT=LUT,
        num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "8")))
    return out

# ── 镜像缓存:每个 (engine, layer) 只做一次「装配 → 拷宿主 → 预排 → pin → UVA」✓ ──
_MIRRORS: dict = {}
_MIRROR_LOCK = None


def get_or_build_mirror(cache_key, build_device_fn, *, blk: int = UVA_BLK, free_device: bool = True):
    """返回 UVA 镜像(4 元组 ✓);首次调用时用 ``build_device_fn()`` 造设备端 K-major ✓。

    ★ 关键:建成后**释放设备端那几份** ✓ ⇒ 稳态下设备侧不再持有 staging ✓
    (这正是"省 10.73 GiB/卡"的落地方式 ✓;官方 design 见 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md ✓)

    ⚠️ 必须在 **CUDA graph 捕获之前**调用 ✗(cudaHostRegister 会作废捕获 ✓)。
    """
    global _MIRROR_LOCK
    import threading

    if _MIRROR_LOCK is None:
        _MIRROR_LOCK = threading.Lock()
    # 诊断绑到 UVA 开关本身 ✓(STAGE 走 env 桥时会被丢 ✗,害我看不到证据)
    _dbg = os.environ.get("XIAOTU_GPF_UVA_DBG", "1") == "1" and uva_enabled()
    hit = _MIRRORS.get(cache_key)
    if hit is not None:
        if _dbg:
            print(f"[uva] cache HIT key={cache_key} (已建 {len(_MIRRORS)} 个)", flush=True)
        return hit
    if _dbg:
        print(f"[uva] cache MISS key={cache_key} ⇒ 开始构建(已建 {len(_MIRRORS)} 个)", flush=True)
    with _MIRROR_LOCK:
        hit = _MIRRORS.get(cache_key)
        if hit is not None:
            return hit
        devbufs = build_device_fn()
        if devbufs is None:
            return None
        # ⚠️ 内存纪律(2026-09-30 实测教训 ✗):曾同时持有 4 份(设备 + to(cpu) + 置换
        # contiguous + pin)⇒ 每 worker RSS 达 【482 GiB】✗(应为 134 ✓)。
        # 现改为:逐张量「设备→pin 缓冲」**一次跨步拷贝**完成,随即放掉设备份 ✓
        devbufs = list(devbufs)
        n = len(devbufs)
        pinned = []
        for idx, t in enumerate(devbufs):
            shape = tuple(t.shape)                       # [E, rows, cols] ✓
            e, r, c = shape
            if c % blk:
                raise ValueError(f"cols {c} 不能被 blk {blk} 整除")
            pin = torch.empty(shape, dtype=torch.uint8, pin_memory=True)   # 目标(锁页 ✓)
            view = pin.view(e, r, c // blk, blk).permute(0, 2, 1, 3)       # [E,n_blk,rows,blk] 视图 ✓
            view.copy_(t.detach().to(view.device).view(e, r, c // blk, blk).permute(0, 2, 1, 3))
            devbufs[idx] = None                          # ⭐ 立刻放掉设备份 ✓
            pinned.append(pin.contiguous())              # 已就位;contiguous 对刚写好的张量是 no-op ✓
        del devbufs
        if free_device:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        mirror = tuple(get_accelerator_view_from_cpu_tensor(p) for p in pinned)
        _MIRRORS[cache_key] = mirror
        _MIRROR_IDS.add(id(mirror))
        if _dbg:
            print(f"[uva] 构建完成 key={cache_key} 宿主 {mirror_bytes(mirror)/2**30:.2f} GiB "
                  f"设备 {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
        return mirror


_MIRROR_IDS: set = set()


def is_mirror(obj) -> bool:
    """该对象是否为 `get_or_build_mirror` 返回的镜像(按 id 登记 ✓)。"""
    return isinstance(obj, tuple) and len(obj) == 4 and id(obj) in _MIRROR_IDS


def mirror_bytes(mirror) -> int:
    return sum(t.numel() * t.element_size() for t in mirror) if mirror else 0

# ── 【验收④】启动期预建全部层镜像(取代"用到才建" ✗) ───────────────────────────
def prebuild_all(device, log=print) -> int:
    """⚠️ 默认关闭(需 XIAOTU_GPF_UVA_PREBUILD_ALL=1 ✓)。

    2026-09-30 用户澄清 ✗:"永远有准备好的层"原意是
    **GPU 算第 1 层时 CPU 正在转置第 2、3 层**(流水线、几层窗口 ✓),
    **不是**把 40 层一次全建完 ✗("那太蠢了" ✗)。
    实测代价:全量预建 ⇒ 269 GiB pin + 瞬时 482 GiB/worker ✗ ⇒ 机器被推到 free 47 GiB ✗。
    ⇒ 本函数仅保留作对照;主线转向"**单一原生副本 + 顺序流式/stride 读**" ✓(见 §20/§21 ✓)。
    """
    if os.environ.get("XIAOTU_GPF_UVA_PREBUILD_ALL", "0") != "1":
        return 0
    """把所有已注册的 MoE 层的镜像**一次建完** ✓(在任何请求之前 ✓)。

    为什么必须这样(实测 ✓,见 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md §15):
    * 懒构建 = 每个"首见层"在**请求内**花 ~7 s(装配+拷宿主+预排+pin+UVA ✓)
      ⇒ 40 层 ≈ **280 s** ⇒ 直接把第一发请求拖到引擎 RPC 超时 ✗(实测 282.6 s / 296 s ✓)
    * 本函数把这段代价挪到**启动期** ✓ ⇒ 请求内零构建 ✓ ⇒ 真正做到"CPU 永远有准备好的层" ✓
    ⚠️ 注意:它必须在 **CUDA graph 捕获结束后**才做吗?——**不必** ✗:实测 prefill 是 eager 的
       (只有 2 个 FULL 图,属 decode ✓),所以捕获后、开始服务前即可 ✓。
    """
    # ⚠️ 无条件诊断(曾因"空 registry 静默 return"而完全看不到证据 ✗)
    print(f"[uva] prebuild_all 被调用: uva_enabled={uva_enabled()} "
          f"cwd={os.getcwd()}", flush=True)
    if not uva_enabled():
        print("[uva] prebuild_all: UVA 关 ⇒ 跳过", flush=True)
        return 0
    from vllm_xiaotu_moe import mixed_experts as _mx  # 延迟导入,避免循环依赖 ✓
    layers = dict(getattr(_mx, "_GP_LAYERS", {}) or {})
    print(f"[uva] prebuild_all: _GP_LAYERS 条目={len(layers)} "
          f"module={_mx.__file__} pid={os.getpid()}", flush=True)
    if not layers:
        print("[uva] prebuild_all: registry 为空 ⇒ 本次无层可建(可能这是 EngineCore 进程 ✗)", flush=True)
        return 0
    import time as _t
    import torch as _torch
    t0 = _t.perf_counter()
    n = 0
    for li in sorted(layers):
        mod = layers[li]
        eng = getattr(mod, "engine", None)
        if eng is None:
            continue
        try:
            _E, _I, _H = mod.gp_shapes()
            if not (_E and _I and _H):
                continue
            _gk = int(getattr(mod, "_group_k", 128) or 128)
            d = _torch.device(device)
            got = get_or_build_mirror(
                (d.index, li),
                lambda eng=eng, d=d, _H=_H, _I=_I, _E=_E, _gk=_gk: (
                    __import__("vllm_xiaotu_moe.gpu_prefill", fromlist=["x"])
                    .kmajor_from_engine_shards(eng, d, _H, _I, _E, _gk)))
            if got is not None:
                n += 1
                if n % 8 == 0:
                    log(f"[uva] prebuild {n} 层 …({_t.perf_counter()-t0:.0f}s)")
        except Exception as _e:  # noqa: BLE001
            log(f"[uva] prebuild 层 {li} 失败: {type(_e).__name__}: {_e}")
    log(f"[uva] 启动期预建完成: {n}/{len(layers)} 层,耗时 {_t.perf_counter()-t0:.0f}s,"
        f"宿主共 {sum(mirror_bytes(v) for v in _MIRRORS.values())/2**30:.1f} GiB ✓")
    return n

# ── 【验收④】一次性扫描:引擎是惰性建的 ⇒ 只能在"第一次 forward"里扫全部层 ✓ ──
_SWEPT = [False]


def sweep_all(device, log=print, why: str = "") -> int:
    """⚠️ 默认关闭(需 XIAOTU_GPF_UVA_SWEEP_ALL=1 ✓);理由同 `prebuild_all` ✗。"""
    if os.environ.get("XIAOTU_GPF_UVA_SWEEP_ALL", "0") != "1":
        return 0
    """把 **全部已注册层** 的镜像一次建完(幂等 ✓;仅第一次真正干活 ✓)。

    为什么不在启动期做(实测 ✓,见 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md §17):
    * `self._xiaotu_engine` 在**第一次 forward** 里才被创建(惰性 ✓)
      ⇒ 启动期 `_GP_LAYERS` 里 40 层的 `engine` **全是 None** ✗ ⇒ `prebuild_all` 完成 0/40 ✗
    * ⚠️ 且**不能**在启动期强行建引擎 ✗ —— 原码注释:「直接 `_ensure_engine` 会 **SIGSEGV**,
      崩在 C++」(2026-09-15 实测 ✓)
    ⇒ ⇒ 所以本函数在**第一次 forward** 时统一扫描 ✓;若服务脚本带 `WARMUP=1` ✓,
       这次扫描就落在**预热期** ⇒ 真实请求内零构建 ✓✓
    """
    if _SWEPT[0] or not uva_enabled():
        return 0
    _SWEPT[0] = True
    from vllm_xiaotu_moe import mixed_experts as _mx
    import time as _t
    layers = dict(getattr(_mx, "_GP_LAYERS", {}) or {})
    d = device if isinstance(device, torch.device) else torch.device(device)
    t0 = _t.perf_counter()
    n = 0
    for li in sorted(layers):
        mod = layers[li]
        eng = getattr(mod, "engine", None)
        if eng is None:
            continue
        try:
            _E, _I, _H = mod.gp_shapes()
            if not (_E and _I and _H):
                continue
            _gk = int(getattr(mod, "_group_k", 128) or 128)
            import vllm_xiaotu_moe.gpu_prefill as _gp
            got = get_or_build_mirror(
                (d.index, li),
                lambda eng=eng, _H=_H, _I=_I, _E=_E, _gk=_gk:
                    _gp.kmajor_from_engine_shards(eng, d, _H, _I, _E, _gk))
            if got is not None:
                n += 1
        except Exception as _e:  # noqa: BLE001
            log(f"[uva] sweep 层 {li} 失败: {type(_e).__name__}: {_e}")
    log(f"[uva] 【一次性扫描】完成 {n}/{len(layers)} 层,耗时 {_t.perf_counter()-t0:.1f}s,"
        f"宿主共 {sum(mirror_bytes(v) for v in _MIRRORS.values())/2**30:.1f} GiB"
        f"{(' | ' + why) if why else ''} ✓")
    return n

# ══════════════════════════════════════════════════════════════════════════
# 【0.2.5 主线】直接读【引擎分片】的 UVA 路径(无需镜像、无需设备 staging ✓)
#
# 依据(全部实测/原码 ✓,见 dev-docs/UVA_ZEROCOPY_EXPERIMENT.md):
#   * 引擎布局(C++ 注释权威 ✓):每 node 缓冲 = [E, 2*crows, Last],crows = I/NS ✓
#     每专家 [gate crows][up crows];node n 的 gate 行 → 全局 [n*crows,…)、up 行 → [I+n*crows,…)
#     ⇒ 一个 node 的缓冲含【两段不相连的全局行】✗ ⇒ **按半区启动**(局部行 + 全局列 ✓)
#   * 引擎自有分片就是"本来就在"的那份 ⇒ 宿主 **1×** ✓(零新增内存 ✓)
#   * 就地锁页已实现(`pin_hostbufs` ✓,**不额外占内存** ✓)
#   * `hostbuf_ptr(which,node)` 已由本仓 C++ 暴露(6 变体全部生效 ✓)
#   * 分片寻址已对拍通过:**误差 0.0000** ✓(§29.16)
# ══════════════════════════════════════════════════════════════════════════

def _view_from_ptr(ptr: int, nbytes: int) -> torch.Tensor:
    """宿主裸指针 → UVA 视图(零拷贝 ✓;生命周期由引擎保证 ✓;必须先锁页 ✓)。"""
    import ctypes
    import numpy as np
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
                                shape=(int(nbytes),))
    return get_accelerator_view_from_cpu_tensor(torch.from_numpy(arr))


def shard_views(engine, device=None, pin: bool = True):
    """给引擎的宿主分片建 UVA 视图 ✓(返回 dict,含几何 ✓)。

    ⚠️ 先 `pin_hostbufs()` ✓(幂等;不额外占内存 ✓)—— UVA 访问要求锁页 ✗。
    """
    # ⚠️【退化假设】每次 shard_views 都 pin ⇒ 可能触发驱动重新映射/迁移页 ✗
    #   ⇒ 改为【每引擎只 pin 一次】✓(实测:此前日志里 "host 分片锁页" 出现 80 次 ✗)
    if pin:
        _pk = id(engine)
        if _pk not in _PINNED:
            try:
                engine.pin_hostbufs()
                _PINNED[_pk] = True
            except Exception:  # noqa: BLE001
                pass
    # ⚠️ 原生 MOE_MXFP4 类【没有】shard_ns()/shard_w13_node_bytes()/scale_*_bytes() ✗
    #    (那些是 `gpu_prefill_bridge.wrap_engine_class` 包装器才加的 ✓)
    #    ⇒ ⇒ 一律从 `shard_geometry()` 取 ✓(原生类也有 ✓,已实测 ✓)
    _g = engine.shard_geometry()
    ns = int(_g["ns"])
    crows13, crows2 = int(_g["w13_crows"]), int(_g["w2_crows"])
    w13nb, w2nb = int(_g["w13_node_bytes"]), int(_g["w2_node_bytes"])
    s13b, s2b = int(_g["w13_scale_bytes"]), int(_g["w2_scale_bytes"])
    kh13 = int(_g["w13_cbytes"]) // max(1, crows13)          # rowbytes = KH ✓(打包维 ✓)
    # ⭐ E:从引擎配置取(权威 ✓);取不到则从几何反推 ✓
    try:
        E = int(engine.config().expert_num)
    except Exception:  # noqa: BLE001
        E = int(w13nb // (2 * crows13 * kh13))
    # ⚠️ 关键修复 ✗:`_view_from_ptr` 给的是【1 维扁平字节】✗ ⇒ 驱动读 shape[0..2] ⇒ IndexError ✗
    #    ⇒ 必须按权威几何 **reshape 成 3 维** ✓(这也解释了此前一切"维度怪现象" ✗)
    kh2 = int(w2nb // (E * crows2))                          # w2 的打包 K ✓
    # ⚠️ scales 是【单份、不随节点切】✓(C++ 注释:"single full copy, indexed by absolute row" ✓)
    #    ⇒ 行数 = 【全部】输出行(2I / H ✓),不是单节点的 crows ✗
    #    (曾按节点算 ⇒ sr13 = 640 ✗ 而正确的 H/32 = 160 ✗✓)
    sr13 = int(s13b // (E * 2 * crows13 * ns))               # = H/32 = 160 ✓
    sr2 = int(s2b // (E * crows2 * ns))                      # = I/32 ✓
    w13v = [_view_from_ptr(engine.hostbuf_ptr(0, n), w13nb).view(E, 2 * crows13, kh13)
            for n in range(ns)]
    w2v = [_view_from_ptr(engine.hostbuf_ptr(1, n), w2nb).view(E, crows2, kh2)
           for n in range(ns)]
    # ⭐ scale 视图用【全量行数】(单份 ✓)
    s13v = _view_from_ptr(engine.hostbuf_ptr(2, 0), s13b).view(E, 2 * crows13 * ns, sr13)
    s2v = _view_from_ptr(engine.hostbuf_ptr(3, 0), s2b).view(E, crows2 * ns, sr2)
    return {
        "ns": ns, "E": E,
        "w13": w13v, "w2": w2v, "s13": s13v, "s2": s2v,
        "crows13": crows13, "crows2": crows2, "kh13": kh13, "kh2": kh2,
        "sr13": sr13, "sr2": sr2,
    }


@triton.jit
def _sh_gu(w_ptr, s_ptr, a_ptr, a_ld, tok_ptr, seg_ptr, out_ptr, inter_ld,
           lrow0, grow0, nrows, w_stride, s_stride, W_E, S_E,
           I2: tl.constexpr, KH: tl.constexpr, SR: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr):
    """gate+up:读某 node 的【一个半区】✓(局部行 → 输出全局列 ✓)。"""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * w_stride
    s_off = pid_e.to(tl.int64) * s_stride
    # ⭐【结构杠杆】一次 launch 覆盖 gate+up 两半(同一缓冲的相邻行段 ✓)
    #   局部行 r ∈ [0, 2·crows):r < crows ⇒ gate ⇒ 全局 n*crows + r
    #                          r ≥ crows ⇒ up   ⇒ 全局 I + n*crows + (r − crows)
    #   ⇒ 每层 launch 12 → 8(−33% ✓);每列仍走同一 k 循环 ⇒ 数值不变 ✓
    _rr = pid_n * BN + tl.arange(0, BN)
    _cc = nrows // 2                                  # = crows ✓(调用方传 2·crows ✓)
    n_loc = lrow0 + _rr                               # 分片内局部行 ✓
    g_col = tl.where(_rr >= _cc, grow0 + _cc + (_rr - _cc), grow0 + _rr)   # 全局列 ✓
    # ⭐【行掩码】必须 ✗:nrows(= crows13 = 288)不是 BN(64)的整数倍 ✓
    #   ⇒ cdiv(288,64)=5 个块覆盖 320 行 ✗ ⇒ up 半(lrow0=288)最大到 607 ✗ > 视图 576 ✗✓
    #   ⇒ Triton illegal memory access ✓✓(这是 4 次崩溃的真正根因 ✓)
    #   ⚠️ down 段没有此问题 ✓:crows2 = 1280 = 20×64 ✓ 正好整除 ✓
    nmask = (pid_n * BN + tl.arange(0, BN)) < nrows
    mt = 0
    while mt * BM < (end - base):
        g = base + mt * BM + tl.arange(0, BM)
        rmask = g < end
        toks = tl.load(tok_ptr + g, mask=rmask, other=0)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, 2 * KH, BK, num_stages=NS):
            # ⚠️ 行距必须用【实际 stride】✗(插件也是传 x.stride(0) ✓)
            #    我曾写死 2*KH ✗ ⇒ 若输入是非连续视图 ⇒ 越界读 ✗✓
            a = tl.load(a_ptr + toks[:, None] * a_ld + (k0 + tl.arange(0, BK))[None, :],
                        mask=rmask[:, None], other=0.0)
            kr = (k0 >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w_ptr + w_off + n_loc[:, None] * KH + kr[None, :],
                           mask=nmask[:, None], other=0)
            bt = tl.trans(b_lo)
            w_e = _dequant_e2m1_nibble(bt & 0x0F)
            w_o = _dequant_e2m1_nibble((bt >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BN))
            # ⚠️ scales 是【单份、按绝对行索引】(C++ 注释 ✓)⇒ 必须用【全局行 g_col】✗
            #    (我先前用局部的 n_loc ✗ ⇒ 分片路径下必错 ✓)
            sc_raw = tl.load(s_ptr + s_off + g_col[:, None] * SR
                             + ((k0 >> 5) + tl.arange(0, BK // 32))[None, :],
                             mask=nmask[:, None], other=127)   # 掩掉 ⇒ 127 ⇒ 缩放 1 ✓
            # ⭐ 与插件【逐字同序】:先 exp2 再 reshape ✓(插件:sc = tl.exp2(scale-127) → reshape ✓)
            sc = tl.exp2(sc_raw.to(tl.float32) - 127.0)
            sc_t = tl.trans(sc)
            sc = tl.reshape(tl.broadcast_to(sc_t[:, None, :], (BK // 32, 32, BN)), (BK, BN))
            # ⭐ 与插件【逐字一致】用融合累加 ✓(acc += dot 的舍入不同 ✗ ⇒ 40 层可翻转 token ✓)
            acc = tl.dot(a, (b_val * sc).to(tl.bfloat16), acc)
        tl.store(out_ptr + g[:, None] * inter_ld + g_col[None, :], acc.to(tl.bfloat16),
                 mask=rmask[:, None] & nmask[None, :])
        mt += 1


@triton.jit
def _sh_down(w_ptr, s_ptr, x_ptr, tok_ptr, wts_ptr, seg_ptr, out_ptr, out_ld,
             lrow0, grow0, nrows, w_stride, s_stride, W_E, S_E,
             H: tl.constexpr, K2: tl.constexpr, SR: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr):
    """down:读某 node 的【一个半区】✓(w2 的分片按输出行 H 切 ✓)。"""
    pid_e = tl.program_id(0)
    pid_n = tl.program_id(1)
    base = tl.load(seg_ptr + pid_e)
    end = tl.load(seg_ptr + pid_e + 1)
    if end <= base:
        return
    w_off = pid_e.to(tl.int64) * w_stride
    s_off = pid_e.to(tl.int64) * s_stride
    n_loc = lrow0 + pid_n * BN + tl.arange(0, BN)
    g_col = grow0 + pid_n * BN + tl.arange(0, BN)
    mt = 0
    while mt * BM < (end - base):
        g = base + mt * BM + tl.arange(0, BM)
        rmask = g < end
        toks = tl.load(tok_ptr + g, mask=rmask, other=0)
        wts = tl.load(wts_ptr + g, mask=rmask, other=0.0).to(tl.float32)   # ⭐ 路由权重 ✓
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in tl.range(0, 2 * K2, BK, num_stages=NS):
            # ⭐ SwiGLU(照抄插件 _down_kernel ✓):gate = 前半、up = 后半 ✓
            # ⚠️ 行距必须是【gate_up 输出的行宽】= 4·K2(= I2 = 2I ✓)✗
            #    而 up 的偏移 = 【2·K2】(= I ✓,即插件里的 `I + kk` ✓)
            #    (我曾用 2·K2 当行距、K2 当 up 偏移 ✗ ⇒ 两者都差一半 ⇒ 越界读 ✗✓)
            _ld = 4 * K2
            # ⚠️⭐【真实 bug,已修】:读 inter 必须带 rmask ✗
            #   `g = base + mt*BM + arange(BM)` 在 (end-base) < BM 时会【超过 A】✗
            #   真实场景每专家段仅 ~27 行 < BM=64 ⇒ 【生产路径同样越界读】✓✓
            #   (写 out 一直有 rmask ✓;只有这两处读 [inter] 漏了 ✗)
            gate = tl.load(x_ptr + g[:, None] * _ld + (k0 + tl.arange(0, BK))[None, :],
                           mask=rmask[:, None], other=0.0).to(tl.float32)
            up = tl.load(x_ptr + g[:, None] * _ld + (2 * K2 + k0 + tl.arange(0, BK))[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            sig = 1.0 / (1.0 + tl.math.exp(-gate))
            a = ((gate * sig) * up).to(tl.bfloat16)
            kr = (k0 >> 1) + tl.arange(0, BK // 2)
            b_lo = tl.load(w_ptr + w_off + n_loc[:, None] * K2 + kr[None, :])
            bt = tl.trans(b_lo)
            w_e = _dequant_e2m1_nibble(bt & 0x0F)
            w_o = _dequant_e2m1_nibble((bt >> 4) & 0x0F)
            b_val = tl.reshape(tl.trans(tl.join(w_e, w_o)), (BK, BN))
            sc = tl.load(s_ptr + s_off + g_col[:, None] * SR      # 绝对行 ✓
                         + ((k0 >> 5) + tl.arange(0, BK // 32))[None, :])
            sc_t = tl.trans(sc)
            sc = tl.reshape(tl.broadcast_to(sc_t[:, None, :], (BK // 32, 32, BN)), (BK, BN))
            # ⭐ 与插件【逐字一致】用融合累加 ✓(acc += dot 的舍入不同 ✗ ⇒ 40 层可翻转 token ✓)
            acc = tl.dot(a, (b_val * tl.exp2(sc.to(tl.float32) - 127.0)).to(tl.bfloat16), acc)
        # ⭐ 必须【按 token 原子累加 + 乘路由权重】(照抄插件 _down_kernel ✓)
        #    我先前用 tl.store 覆盖写 ✗ ⇒ 同一 token 的 K 份贡献被丢掉 K-1 份 ✗✓
        tl.atomic_add(out_ptr + toks[:, None] * out_ld + g_col[None, :],
                      acc * wts[:, None], mask=rmask[:, None], sem="relaxed")
        mt += 1


def gpu_moe_layer_shards(x, topk_ids, topk_weights, sh, H: int, I: int, K: int, *,
                         device: torch.device, BM: int | None = None, BN: int | None = None,
                         BK: int | None = None, NS: int | None = None):
    # ⭐【验收①】读粒度是主因(§20:顺序读=100% 线速;跨步读崩到 0.38 GiB/s ✗)
    #   现状每次只读 BK//2 = 32 字节的连续段 ✗ ⇒ 正是"跨步小读" ✗
    #   ⇒ 故把 BM/BN/BK/NS 接到环境变量以便扫描 ✓(默认与插件一致 ✓)
    # BM 官方默认 64(与既有 GPU 预填路径一致 => 数值已验证等价)。
    #   BM=128 更快(稳态 595->425ms)但同会话 md5 复现性下降 => 未验证等价,
    #   故【仅作可选旋钮】(XIAOTU_GPF_UVA_BM=128),不设为默认。见 §75/§76
    BM = int(os.environ.get("XIAOTU_GPF_UVA_BM", "64")) if BM is None else BM
    BN = int(os.environ.get("XIAOTU_GPF_UVA_BN", "64")) if BN is None else BN
    BK = int(os.environ.get("XIAOTU_GPF_UVA_BK", "64")) if BK is None else BK
    NS = int(os.environ.get("XIAOTU_GPF_UVA_STAGES", "2")) if NS is None else NS
    """用【引擎分片的 UVA 视图】算一层 MoE ✓(逐半区启动 ✓;无需设备 staging ✓)。"""
    _t0 = time.perf_counter()          # 【验收①】逐层计时起点 ✓
    _layertag = sh.get("_tag", "?")    # 层名(装配处塞入 ✓)
    _CALL_SEQ[0] += 1
    _seq = _CALL_SEQ[0]
    _SEQ[0] += 1
    from vllm_xiaotu_moe.gpu_prefill import _build_segmentation
    device = torch.device(device)
    # ⚠️ 几何一律【从张量形状推导】✓ —— 禁止用公式 ✗
    #    (2026-09-30 实测教训 §32 ✓:驱动曾用 I2=2*I / KH2=I//2 等公式 ✗,
    #     而真实 w2 的打包 K = 1152 ✗ ≠ I//2 = 576 ✗ ⇒ 直接算错 ✓。
    #     更严重的是:合成测试用【同一个错误假设】造数据 ⇒ 于是"验证"了错误 ✗✓)
    ns = int(sh["ns"])
    w13_0, s13_0 = sh["w13"][0], sh["s13"]
    E, KH = int(w13_0.shape[0]), int(w13_0.shape[2])
    # ⚠️ I2 是【全局 gate_up 输出宽】= 2·I ✗ —— 【不是】分片张量的第 1 维 ✗✓
    #    w13_0.shape[1] = 2·crows13 = 2·(I/ns) = 576 ✗ ⇒ 若拿它当输出宽 ⇒
    #    store 用 inter_ld=576 而 g_col 最大 2303 ⇒ 【写越界】✓✓(实测 §43 ✓)
    I2 = 2 * int(I)
    SR13 = int(s13_0.shape[2]) if s13_0.dim() == 3 else int(s13_0.shape[1]) // I2
    crows13 = (2 * KH and I2 // 2) if False else int(sh["crows13"])
    w2_0, s2_0 = sh["w2"][0], sh["s2"]
    KH2 = int(w2_0.shape[2])                                 # 打包 K ✓(分片维 = 全局维 ✓)
    SR2 = int(s2_0.shape[2]) if s2_0.dim() == 3 else int(s2_0.shape[1]) // int(w2_0.shape[1])
    crows2 = int(sh["crows2"])
    # ⚠️ H 是【全局 hidden】✗ —— w2_0.shape[1] = crows2 = H/ns = 1280 ✗ 不是 H ✓
    #    (与 I2 同一类错误 ✓:分片维 ≠ 全局维 ✗;实测 §44:out 被写成 (T,1280) ✗✓)
    H = int(H)                                               # ⭐ 用调用方传入的 hidden ✓
    tok, wts, seg, A = _build_segmentation(topk_ids, topk_weights, E, device)
    if _UVA_TIME:
        torch.cuda.synchronize()
    _t_seg = time.perf_counter()
    # ⭐ 假设(默认关):40 层共用一块 inter ⇒ 跨层 WAR 依赖可能抬高稳态墙钟
    inter = _pooled(("inter", device.index, I2, _inter_slot()), (A, I2), torch.bfloat16, device)
    W13_E = int(w13_0.numel() // E)
    S13_E = int(s13_0.numel() // E)
    for n in range(ns):
        # ⭐ 合并 gate/up 两半 ⇒ 每 node 只 launch 一次 ✓
        # ⭐ 合并开关:默认【开】✓;=0 时回退为分两半(用于同口径对照 ✓)
        #   ⚠️ 改动必须打印实际文本核对 ✗(§65.3 的教训 ✓)
        _merge_gu = os.environ.get("XIAOTU_GPF_UVA_MERGE_GU", "1") == "1"
        _halves = ((0, n * crows13, 2 * crows13),) if _merge_gu else \
                  ((0, n * crows13, crows13), (crows13, I + n * crows13, crows13))
        for (lrow0, grow0, nrows) in _halves:
            _sh_gu[(E, triton.cdiv(nrows, BN))](
                sh["w13"][n], sh["s13"], x, x.stride(0), tok, seg, inter, inter.stride(0),
                lrow0, grow0, nrows, W13_E, S13_E, W13_E, S13_E,
                I2=I2, KH=KH, SR=SR13, BM=BM, BN=BN, BK=BK, NS=NS,
                # ⚠️ 必须与插件【同默认】✗:插件 NW = XIAOTU_GPU_PREFILL_WARPS 默认【4】✓
                #    我曾写死 8 ✗ ⇒ tl.dot 归约划分不同 ⇒ 舍入不同 ⇒ 40 层可翻转 token ✓✓
                num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4")))
    # ⭐ 诊断:XIAOTU_GPF_UVA_SYNC=1 时【逐段同步】✓ ⇒ 把 Triton 的【异步归属】钉死 ✗
    #   (Triton kernel 异步入队 ⇒ 日志里的崩溃帧可能归属到【后面】那次 launch ✗✓)
    if _UVA_SYNC:
        torch.cuda.synchronize()
        print(f"[uva-sync] gate_up 段完成 ✓ ns={ns} I2={I2} KH={KH} SR={SR13} "
              f"crows13={crows13} W13_E={W13_E} S13_E={S13_E} "
              f"inter={tuple(inter.shape)}", flush=True)
    if _UVA_TIME:
        torch.cuda.synchronize()
    _t_gu = time.perf_counter()
    out = torch.zeros((x.shape[0], H), dtype=torch.bfloat16, device=device)
    W2_E = int(sh["w2"][0].numel() // E)   # ⭐ 几何已在上方从张量推导 ✓
    S2_E = int(sh["s2"].numel() // E)
    for n in range(ns):
        # ⚠️ w2 的分片是【单段】(moe_v2.hpp:1392:"node n stores ONE cbytes slice per expert" ✓)
        #    ⇒ 每 node 只启动一次 ✗(我曾错用 gate/up 两半 ✗)
        for (lrow0, grow0, nrows) in ((0, n * crows2, crows2),):
            _sh_down[(E, triton.cdiv(nrows, BN))](
                sh["w2"][n], sh["s2"], inter, tok, wts, seg, out, out.stride(0),
                lrow0, grow0, nrows, W2_E, S2_E, W2_E, S2_E,
                H=H, K2=KH2, SR=SR2, BM=BM, BN=BN, BK=BK, NS=NS,
                num_warps=int(os.environ.get("XIAOTU_GPU_PREFILL_WARPS", "4")))
    if _UVA_SYNC:
        torch.cuda.synchronize()
        print(f"[uva-sync] down 段完成 ✓ H={H} K2={KH2} SR={SR2} crows2={crows2} "
              f"W2_E={W2_E} S2_E={S2_E} out={tuple(out.shape)} "
              f"w2[0]={tuple(sh['w2'][0].shape)} s2={tuple(sh['s2'].shape)}", flush=True)
    # ⭐【验收①】逐层计时 ✓(门控 ✓,默认关 ⇒ 不影响生产 ✓)
    #   口径:UVA 真实整层(与本函数同范围 ✓)⇒ 与现状基线 366.29 ms 比 ✓
    if _UVA_TIME:
        torch.cuda.synchronize()
        _t_end = time.perf_counter()
        # ⚠️ 三个打点【必须共用同一基准 _t0】✗:曾把"时长"当"绝对时刻"用 ✗
        #    (同一行改错 3 次:未定义 ✓ → 单位混用 ✓ → 基准混用 ✓)
        print(f"[uva-time] seq={_seq} layer={_layertag} A={A} T={int(x.shape[0])} "
              f"uva_layer_ms={(_t_end - _t0) * 1e3:.2f} "
              f"seg_ms={(_t_seg - _t0) * 1e3:.2f} "
              f"gu_ms={(_t_gu - _t_seg) * 1e3:.2f} "
              f"down_ms={(_t_end - _t_gu) * 1e3:.2f}", flush=True)
    return out

_UVA_SYNC = os.environ.get("XIAOTU_GPF_UVA_SYNC", "0") == "1"   # 诊断:逐段同步 ✓
_UVA_TIME = os.environ.get("XIAOTU_GPF_UVA_TIME", "0") == "1"   # 验收①:逐层计时 ✓
_CALL_SEQ = [0]        # 全局调用序
_SEQ = [0]             # 层调用序(供 parity 池键 ✓)(查清"6 遍"是什么 ✓)


# ── 【0.2.5 主线】分片视图缓存:每 (engine, layer) 建一次 ✓ ─────────────────────
_SHARD_CACHE: dict = {}

# ⭐【验收①】中间缓冲池 ✓:去掉每次调用的分配(实测退化 + 显存增长的可疑主因 ✓)
#   ⚠️ 复用是【安全】的 ✓:`_sh_gu` 会写满 inter 的【所有列】✓,
#      且 `_sh_down` 只读属于各专家分段的那些行 ✓ ⇒ 未被写的行永不参与计算 ✓
#   ⚠️ `out` 不复用 ✗(它是返回值,调用方会持有 ✓)⇒ 仍每次 zeros ✓(仅 17.7 MB ✓)
_BUF_POOL: dict = {}
_PINNED: dict = {}   # ⭐ 每引擎只 pin 一次(§68.3 ✓)
# §78.4/§79 缓冲复用模式:0=40 层共用(已发布 ✓)/ 1=每层一块(实测崩 ✗)/ 2=按层奇偶双缓冲 ⭐
    # ⭐ 默认 8(实测 N=1→8 单调变快:req1 350→285 ms ✓ 且不崩 ✓);只影响缓冲分配,计算不变 ✓
_POOL_N = max(1, int(os.environ.get("XIAOTU_GPF_UVA_POOL_N", "8") or 1))  # 池槽数(1=共用 ✓)


def _inter_slot():
    """inter 的池槽:把"跨层 WAR 依赖"按 N 个槽轮转打断 ✓。

    实测规律(§79/§80):
      N=1(共用)⇒ req1 350 ms · 不崩 ✓
      N=2(奇偶)⇒ req1 319.6 ms · 不崩 ✓
      每层一块(≈N=80)⇒ req1 267-268 ms(≈线速理想 ✓)但【崩 16 次】✗
    ⇒ ⇒ 本函数按 `XIAOTU_GPF_UVA_POOL_N` 轮转,用于找"不崩前提下的最优 N" ✓
    """
    n = _POOL_N
    if n <= 1:
        return 0
    return int(_SEQ[0]) % n



def _pooled(key, shape, dtype, device, zero=False):
    # ⚠️【必须保留旧缓冲】✗:形状变化时若【丢掉】旧张量 ⇒ 分配器复用其内存 ✗,
    #   而排队的 kernel 可能仍在读它 ⇒ 【use-after-free】⇒ illegal memory access ✓
    #   (实测:per-layer 池键换形状时崩 16 次 ✓;故改为【每个 key 保留一个列表】✓)
    slot = _BUF_POOL.get(key)
    if not isinstance(slot, list):
        slot = [] if slot is None else [slot]
        _BUF_POOL[key] = slot
    for t in slot:
        if tuple(t.shape) == tuple(shape) and t.dtype == dtype:
            if zero:
                t.zero_()
            return t
    t = torch.zeros(shape, dtype=dtype, device=device) if zero \
        else torch.empty(shape, dtype=dtype, device=device)
    slot.append(t)          # ⭐ 保留(不释放 ✓)
    return t


def shard_views_cached(engine, device, key):
    """给引擎分片建 UVA 视图并缓存 ✓(幂等;宿主 1×;零新增锁页 ✓)。

    ⚠️ 与镜像路线的区别 ✓:这里**不复制任何数据** ✗ —— 直接给引擎已有的宿主分片建视图 ✓
       ⇒ 不再需要 269 GiB pin ✗、不再需要 sweep/预建 ✓(镜像路线已退为对照 ✓)
    """
    hit = _SHARD_CACHE.get(key)
    if hit is not None:
        return hit
    sh = shard_views(engine, device)
    _SHARD_CACHE[key] = sh
    if os.environ.get("XIAOTU_GPF_UVA_DBG", "1") == "1":
        print(f"[uva] 分片视图已建 key={key} ns={sh['ns']} "
              f"w13[0]={tuple(sh['w13'][0].shape)} w2[0]={tuple(sh['w2'][0].shape)}", flush=True)
    return sh
