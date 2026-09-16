# xiaotu-moe: `gpu_prefill` 桥 —— 让引擎满足 lk 编排链的 ABI。
#
# 背景(为什么需要它)
# ------------------
# lk 的 vLLM 侧(`routed_experts.py:_gpu_prefill`)对大批量 prefill 会直接调用
# **引擎自己的** GPU 实现:
#
#     def _gpu_prefill(self, hidden_states, topk_weights, topk_ids):
#         output = torch.empty_like(hidden_states)              # bf16 [qlen, H]
#         self.lk_moe.gpu_prefill(
#             hidden_states.data_ptr(), output.data_ptr(),
#             topk_ids.data_ptr(), topk_weights.data_ptr(),
#             hidden_states.size(0), topk_ids.size(1),
#             torch.cuda.current_stream().cuda_stream,          # 第 7 个参数:stream
#         )
#         return output
#
# 也就是说:**大 prefill 的权重流式 + GPU 计算由引擎负责**,而不是走 vLLM 的标准 GPU MoE。
# 原因是混合模式下专家权重在 CPU 上(43 层 × 3.2 GB ≈ 137 GiB 放不下显存),
# 标准 GPU MoE 需要 `moe_kernel`(MARLIN 重打包要求权重在 CUDA 上)
# ⇒ 实测 `NotImplementedError: Could not run '_C::gptq_marlin_repack' with arguments
#    from the 'CPU' backend`(见 NOTES §282)。
#
# 我们的 `gpu_prefill.py`(970 行、5 个 Triton kernel、权重流式 + side stream +
# PrefetchSlot)本来就是同一套设计 —— 本桥只是把它接到 lk 的 ABI 上,**没有新算法**。
#
# 桥怎么工作
# ----------
# * lk 构造引擎时传的是 **CPU 裸指针**(`w13_weight.data_ptr()` 等),本桥在构造时把这些
#   指针连同 cfg 记下来,惰性建成 numpy 视图 —— `gpu_moe_layer` 正好吃 numpy 数组;
# * `gpu_prefill(...)` 收到的是 **device 裸指针**,用 `__cuda_array_interface__` 零拷贝
#   包成 torch 张量,再调 `gpu_moe_layer`,最后把结果写回 out 指针;
# * 其它方法(`cpu_decode` / `cpu_prefill` / `configure_ep` / `prepare_decode_buffers` …)
#   一律通过 `__getattr__` 转发给原生对象,行为完全不变。
#
# License: Apache-2.0
from __future__ import annotations

import ctypes
import os

__all__ = ["wrap_engine_class"]

# MXFP4 打包布局(与 checkpoint 一致;见 docs/ARCHITECTURE.md):
#   w13 [E, 2I, H/2] uint8   w2 [E, H, I/2] uint8
#   s13 [E, 2I, H/groupK]    s2 [E, H, I/groupK]   (e8m0 字节)
_SHAPE_RULES = {
    "MXFP4": lambda E, I, H, gK: ((E, 2 * I, H // 2), (E, H, I // 2),
                                  (E, 2 * I, H // gK), (E, H, I // gK)),
    "NVFP4": lambda E, I, H, gK: ((E, 2 * I, H // 2), (E, H, I // 2),
                                  (E, 2 * I, H // gK), (E, H, I // gK)),
}


# 保活池:torch.frombuffer 建出来的张量引用外部缓冲,必须自己持有,否则可能被回收。
_KEEPALIVE: list = []


def _cpu_view(ptr: int, shape, dtype=None):
    """在 CPU 裸指针上建 **torch** 视图(零拷贝)。

    必须是 torch 张量而非 numpy:`gpu_prefill._pinned()` 会取 `untyped_storage()`
    做锁页缓存(就地 cudaHostRegister),numpy 数组没有这个属性。
    """
    import torch

    if dtype is None:
        dtype = torch.uint8
    n = 1
    for d in shape:
        n *= int(d)
    nbytes = n * dtype.itemsize
    buf = (ctypes.c_ubyte * nbytes).from_address(int(ptr))
    t = torch.frombuffer(buf, dtype=dtype)
    _KEEPALIVE.append((buf, t))
    return t.view(shape)


class _CudaArray:
    """把 device 裸指针包成 `__cuda_array_interface__`,供 `torch.as_tensor` 零拷贝使用。"""

    __slots__ = ("_d",)

    def __init__(self, ptr: int, shape, typestr: str):
        n = 1
        for d in shape:
            n *= int(d)
        self._d = {
            "shape": tuple(int(d) for d in shape),
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
            "strides": None,
        }

    @property
    def __cuda_array_interface__(self):
        return self._d


def _dev_tensor(ptr: int, shape, dtype, device):
    """device 裸指针 → torch 张量(零拷贝,经 __cuda_array_interface__)。

    bf16 没有 numpy 等价 dtype(用 "<V2" 会得到 numpy.void,Torch 拒绝转换),
    所以按 uint16 建张量再 view 成 bfloat16。
    """
    import torch

    typestr = {torch.bfloat16: "<u2", torch.float32: "<f4",
               torch.int32: "<i4"}[dtype]
    t = torch.as_tensor(_CudaArray(ptr, shape, typestr))
    if dtype is torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t.to(device) if t.device != device else t


# 引擎按构造顺序登记 —— lk 的链逐层构造,所以这个顺序就是层顺序;
# 桥靠它实现 **prefetch window = 1**(本层计算时预取下一层权重,对应 lk 的
# `LVLLM_GPU_PREFETCH_WINDOW=1`)。
_ENGINES: list = []


def _gpu_prefill_requested() -> bool:
    """GPU 预填充**在启动时**是否被打开(决定要不要在构造期保住一份权重副本)。

    `XIAOTU_GPUPREFILL_WCOPY` 的默认值必须跟着这个走(§504):
      * GPU 预填充 **关**(本项目默认,`XIAOTU_MOE_GPU_PREFILL_MIN_TOKENS=0`)
        ⇒ 构造期那份 Python 侧副本**永远用不到**,而 V4.1 是 ~6.7 GiB/层/rank,
          40 层就是 ~270 GiB 白占 ⇒ 默认 **0**;
      * GPU 预填充 **开** ⇒ 必须默认 **1**:vLLM 随后会 `clean_weights_after_loading`
        删掉 CPU 侧 `w13_weight/w2_weight`,源页会被回收/置 PROT_NONE ⇒
        惰性复制就会读到已释放的内存(该文件下方注释记录的 lkport21 原生崩溃)。
    阈值可能来自环境变量,也可能来自运行时**文件桥**(两条都要看)。
    """
    # 三个名字都要认:`vllm_xiaotu_moe/gpu_prefill.py:80` 读的是 VLLM_ 前缀那个
    # (规范名),而文档/脚本里这三个都出现过 —— 少认一个就会在"用户以为开了 GPU 预填充"
    # 时仍然不复制权重 ⇒ 惰性复制读到已释放内存 ⇒ 原生崩溃。
    for k in ("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS",
              "XIAOTU_MOE_GPU_PREFILL_MIN_TOKENS",
              "XIAOTU_GPU_PREFILL_MIN_TOKENS"):
        try:
            if int(os.environ.get(k) or 0) > 0:
                return True
        except ValueError:
            pass
    f = (os.environ.get("XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE")
         or os.environ.get("VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE"))
    if f:
        try:
            with open(f) as fh:
                if int((fh.read() or "0").strip() or 0) > 0:
                    return True
        except (OSError, ValueError):
            pass
    return False


def wrap_engine_class(native_cls, kind: str):
    """把原生引擎类包一层:加 `gpu_prefill`,其余属性原样转发。"""

    shape_rule = _SHAPE_RULES.get(kind)

    class _Engine:
        def __init__(self, cfg, w13, w2, w13_g=0, w2_g=0, w13_gs=0, w2_gs=0):
            self._impl = native_cls(cfg, w13, w2, w13_g, w2_g, w13_gs, w2_gs)
            self._cfg = cfg
            self._wptrs = (w13, w2, w13_g, w2_g)
            self._warr = None
            # 【必须在构造期就把权重复制出来】vLLM 随后会走
            # `clean_weights_after_loading`,对 CPU 层 **删除** w13_weight/w2_weight,
            # 底层页会被回收/置 PROT_NONE ⇒ 之后再用这些指针做 GPU prefill 会**原生崩溃**
            # (实测 lkport21:栈上全是 Py_BytesMain 之类的裸帧)。我们引擎的 C++ 侧早就在
            # 构造期快照过一份(见 moe_v2.hpp 的 "COPY the weight blocks" 注释);这里为
            # GPU prefill 也做同样的事,只是把副本放在 Python 侧(惰性就会太晚)。
            self._idx = len(_ENGINES)
            _ENGINES.append(self)
            self._slot = None            # 上一层为本层预取好的设备槽
            self._km = None              # K-major 锁页缓存(惰性)
            _wcopy_default = "1" if _gpu_prefill_requested() else "0"
            if os.environ.get("XIAOTU_GPUPREFILL_WCOPY", _wcopy_default) == "1":
                try:
                    self._warr = tuple(_cpu_view(int(p), sh)
                                       for p, sh in zip(self._wptrs, self._shapes()))
                    self._warr = tuple(a.clone() for a in self._warr)
                except Exception:  # noqa: BLE001  形状不匹配等:留给 gpu_prefill 报错
                    self._warr = None

        # 除本类自己定义的以外,一切交给原生对象(保持 ABI 完全一致)
        def __getattr__(self, name):
            return getattr(self._impl, name)

        def _shapes(self):
            if shape_rule is None:
                raise RuntimeError(f"gpu_prefill 未支持量化类型 {kind}")
            c = self._cfg
            E, I, H, gK = (int(c.expert_num), int(c.intermediate_size),
                           int(c.hidden_size), int(c.groupK) or 32)
            return shape_rule(E, I, H, gK)

        def _weights(self):
            if self._warr is None:
                self._warr = tuple(_cpu_view(int(p), s)
                                   for p, s in zip(self._wptrs, self._shapes()))
            return self._warr

        def _kmajor(self):
            """本层权重的 K-major 锁页副本(缓存;`prefetch_layer` 要求这种输入)。"""
            if self._km is None:
                from .gpu_prefill import _pinned_kmajor
                w13, w2, s13, s2 = self._weights()
                self._km = (_pinned_kmajor(w13), _pinned_kmajor(s13),
                            _pinned_kmajor(w2), _pinned_kmajor(s2))
            return self._km

        def prefetch(self, device):
            """异步发起本层权重的 H2D(由上一层调用 ⇒ 与本层计算重叠)。

            即 lk 的 `LVLLM_GPU_PREFETCH_WINDOW=1` 语义。关:`XIAOTU_GPUPREFILL_WINDOW=0`。
            """
            if os.environ.get("XIAOTU_GPUPREFILL_WINDOW", "1") == "0":
                return
            if self._slot is not None:
                return
            try:
                from .gpu_prefill import prefetch_layer
                a, b, c, d = self._kmajor()
                self._slot = prefetch_layer(a, b, c, d, device)
            except Exception:  # noqa: BLE001  显存不够等 ⇒ 退回逐层同步 H2D
                self._slot = None

        def gpu_prefill(self, x_ptr, out_ptr, ids_ptr, wts_ptr, qlen, k, stream=0):
            """lk ABI:大 prefill 的 GPU 路径(权重按层流式 H2D + Triton 分组 GEMM)。

            参数都是 **device 裸指针**;`out` 是 bf16 [qlen, H],与 hidden_states 同布局。
            """
            import torch
            from .gpu_prefill import gpu_moe_layer

            qlen, k = int(qlen), int(k)
            H = int(self._cfg.hidden_size)
            I = int(self._cfg.intermediate_size)
            dev = torch.device("cuda", torch.cuda.current_device())

            x = _dev_tensor(x_ptr, (qlen, H), torch.bfloat16, dev)
            out = _dev_tensor(out_ptr, (qlen, H), torch.bfloat16, dev)
            ids = _dev_tensor(ids_ptr, (qlen, k), torch.int32, dev)
            wts = _dev_tensor(wts_ptr, (qlen, k), torch.float32, dev)
            w13, w2, s13, s2 = self._weights()

            ctx = None
            if stream:
                ctx = torch.cuda.stream(torch.cuda.ExternalStream(int(stream)))
            if ctx is not None:
                ctx.__enter__()
            try:
                # 用上一层为本层预取好的槽(没有就同步 H2D),并顺手预取下一层,
                # 让它的 H2D 与本层内核重叠 —— 这就是把 6.4 s 固定开销压下去的关键。
                slot, self._slot = self._slot, None
                if self._idx + 1 < len(_ENGINES):
                    _ENGINES[self._idx + 1].prefetch(dev)
                y = gpu_moe_layer(x, ids, wts, w13, s13, w2, s2,
                                  H=H, I=I, K=k, device=dev, slot=slot)
                out.copy_(y.to(torch.bfloat16))
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)

    _Engine.__name__ = f"Wrapped{native_cls.__name__}"
    _Engine.__qualname__ = _Engine.__name__
    return _Engine
