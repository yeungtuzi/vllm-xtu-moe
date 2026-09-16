"""GPU/CPU Mixed Mode 的通用 CPU 计算后端(XiaotuCPUExperts*)。

背景:主线已引入 `VLLM_EXPERTS_LOAD_DEVICE=cpu`(见 vllm/envs.py + routed_experts.py),
让 routed-expert 权重从构造起就在 CPU(不 OOM)。但计算仍走 quant_method.apply(GPU 内核)
会设备不匹配。本模块提供 CPU 计算后端,并**按量化格式注册到主线的 CPU 后端槽位**,
因此任何使用 `FusedMoEFactory` 的 MoE 模型(DS-V4、GLM-5.x、Qwen、Mixtral…)只要
权重格式在支持列表里,就会自动选中 xiaotu 引擎,**不需要模型级 OOT 覆盖**。

主线侧需要配合的一点:oracle 在 GPU 平台上默认不会选 CPU 后端(它要求
`current_platform.is_cpu()`),所以混合模式下需要把 CPU 后端前置。MXFP4 的改动在
`patches/mainline_sm80_mixed_mode.patch`(PR1),FP8 的对称改动在同一补丁里。

支持格式(引擎 → 基线类):
  bf16   MOE_BF16   CPUUnquantizedExperts 无缩放        权重 bf16 [E,2I,H]/[E,H,I]
  mxfp4  MOE_MXFP4  CPUExpertsMxfp4  groupN=1    groupK=32   权重 u8 nibble + e8m0
  fp8    MOE_FP8    CPUExpertsFp8    groupN=128  groupK=128  权重 e4m3 + fp32 块缩放
  int4   MOE_WNA16  CPUExpertsInt4   groupN=32/128 groupK=32/128 (按 quant_config 取)

路由:主线把路由拆在 `fused_moe/router/*`(softmax / sigmoid / sqrtsoftplus /
grouped-topk / noaux_tc / custom_routing_function)。monolithic 后端拿到的只有
router_logits,所以本模块**复用主线的 router 对象**(`create_fused_moe_router`),
而不是像主线 cpu_moe 那样硬编码 softmax —— 后者对 GLM(sigmoid+noaux_tc)和
DS-V4(sqrtsoftplus)都会选错专家。

激活:引擎实现**packed 布局**的 gated 激活 `out = clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L))
* (clamp(up,±L)+beta)`(与主线 `silu_and_mul_with_clamp` 同语义),因此支持主线的
`SILU` 与 `SWIGLUOAI_UNINTERLEAVE`(两者都是 packed);**不支持 `SWIGLUOAI`**
(gpt-oss 把 gate/up 交错存在 w13 里,我们按 packed 取数会静默算错)—— 由
`_supports_activation` 显式拒绝,让 oracle 去选别的后端。

在 AMD(无 AMX)上,这是唯一能跑的 CPU MoE 内核;主线 CPUExperts* 要 Intel AMX。
"""
from __future__ import annotations

import os

import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUExpertsFp8,
    CPUExpertsInt4,
    CPUExpertsMxfp4,
    CPUUnquantizedExperts,
)

# Upstream moved the monolithic routing helper out of `experts/cpu_moe.py` into
# `router/cpu_router.py` (mainline >= 0.29.1rc1.dev95, commit dabc4362b). Keep
# both paths so the plugin works on older and newer mainline trees alike.
try:
    from vllm.model_executor.layers.fused_moe.router.cpu_router import (
        select_experts,
    )
except ImportError:  # mainline <= 6c73b08dec
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        select_experts,
    )


_VERIFY_LAYER = os.environ.get("XIAOTU_VERIFY_LAYER", "") == "1"
_VERIFY_MAX = int(os.environ.get("XIAOTU_VERIFY_MAX", "1"))
_DUMP_LAYER = os.environ.get("XIAOTU_DUMP_LAYER", "")
_HID_LAYER = os.environ.get("XIAOTU_HID_LAYER", "")


def mixed_mode_enabled() -> bool:
    """True when experts are configured to live/compute on CPU on a GPU run.

    Delegates to `mainline_shims.mixed_mode_enabled`, which reads the env var
    directly so the plugin also works on a **stock** vLLM (whose `vllm.envs`
    has no `VLLM_EXPERTS_LOAD_DEVICE`).
    """
    from .mainline_shims import mixed_mode_enabled as _m

    return _m()


def _supports_mixed_device() -> bool:
    from vllm.platforms import CpuArchEnum, current_platform

    return (
        mixed_mode_enabled()
        and current_platform.get_cpu_architecture() == CpuArchEnum.X86
    )


# 主线 RoutedExperts 上保存的路由/激活配置(见 routed_experts.py __init__)。
_ROUTING_ATTRS = (
    "use_grouped_topk",
    "renormalize",
    "scoring_func",
    "custom_routing_function",
    "num_expert_group",
    "topk_group",
    "e_score_correction_bias",
    "routed_scaling_factor",
)
_SWIGLU_ATTRS = ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
# 少数模型的 router 需要额外参数(本插件只在 layer 上存在时才透传)。
_ROUTER_EXTRA_ATTRS = (
    "hash_indices_table",
    "num_fused_shared_experts",
    "shared_expert_weight",
    "bias_vl",
    "image_sentinel_lo",
)


def _host_mib() -> dict:
    """读 /proc/self/status 的互斥计数(只用 XIAOTU_MEM_DIAG=1 时)。"""
    out: dict = {}
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                k, _, rest = ln.partition(":")
                if k in ("RssAnon", "RssShmem", "RssFile"):
                    out[k] = int(rest.split()[0]) >> 10
    except Exception:  # noqa: BLE001
        pass
    return out


_LAYER_IDX_RE = __import__("re").compile(r"layers\.(\d+)\.")
_RELEASE_MISSES = 0   # 见 _release_source_weights:静默失败的可观测性
_RELEASE_FILE = "/tmp/xiaotu_release_source"


def _release_source_enabled() -> bool:
    """env 优先,其次看标记文件(两者都没有 = 关闭)。

    ⚠️ **为什么必须有文件开关**:2026-09-15 实测,EngineCore 子进程的 environ 与
    launcher **不是同一份**(launcher 76 项 / EngineCore 2711 项),
    `XIAOTU_RELEASE_SOURCE` 在 EngineCore 里**根本不存在** —— 而同一批 `XIAOTU_MOE_*`
    (THREADS/ASYNC/NSLICE_SMALL/SPIN_IDLE_US)却都在。于是 `os.environ.get(...)` 恒为
    "0",**开关打开了也永远不释放**,而且不留任何日志。

    插件在 GPU prefill 阈值上早就踩过同一个坑并用文件开关绕过(gpu_prefill.
    gpu_prefill_min_tokens 读 `XIAOTU_GPU_PREFILL_MIN_TOKENS_FILE`),这里沿用同一手法。
    开启方式:`echo 1 > /tmp/xiaotu_release_source`。
    """
    v = os.environ.get("XIAOTU_RELEASE_SOURCE")
    if v is not None:
        return v == "1"
    try:
        with open(_RELEASE_FILE) as fh:
            return fh.read().strip() == "1"
    except OSError:
        return False


class _XiaotuExpertsMixin:
    """Shared behaviour: build the xiaotu engine from the layer's raw CPU weights.

    Subclasses set ``_engine_attr`` (engine class on the ``xiaotu_moe`` module) and
    ``_scale_attrs`` (candidate attribute names for w13/w2 scales, in order).
    """

    _engine_attr = ""
    _scale_attrs: tuple[str, str] = ("w13_weight_scale", "w2_weight_scale")
    # scale grouping passed to the engine (groupN, groupK)
    _group_n = 1
    _group_k = 32
    # weight element width in bytes (u8 for mxfp4/int4-packed, 1 for fp8)
    _w_bytes = 1
    # expected dtype of the layer's w13_weight (None = unchecked)
    _expect_dtype = None
    # dtype the engine reads block scales as; None = pass the layer's tensor
    # through unchanged (MXFP4 uses uint8 e8m0 bytes). Checkpoints disagree on
    # this (GLM-5.3 ships fp32 scale_inv, Qwen3 ships bf16), so we normalize to
    # fp32 for the formats whose engine kernel reads `const float*`.
    _scale_dtype = None

    def __init__(self, moe_config, quant_config):
        _dbg = os.environ.get("XIAOTU_MEM_DIAG") == "1"
        if _dbg:
            _b = _host_mib()
        super().__init__(moe_config, quant_config)
        if _dbg:
            _m = _host_mib()
            print(f"[xtu-seg] experts.__init__: dAnon="
                  f"{_m.get('RssAnon',0)-_b.get('RssAnon',0):+d}MiB "
                  f"dShmem={_m.get('RssShmem',0)-_b.get('RssShmem',0):+d}MiB", flush=True)
        self._xiaotu_engine = None
        self._layer_ref: torch.nn.Module | None = None
        self._router = None
        # Mirror the router configuration that monolithic apply() must carry
        # itself (mainline captures the same fields in CPUUnquantizedExperts).
        self.use_grouped_topk = False
        self.renormalize = False
        self.scoring_func = "softmax"
        self.custom_routing_function = None
        self.num_expert_group = None
        self.topk_group = None
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0
        self.swiglu_limit = None
        self.swiglu_alpha = None
        self.swiglu_beta = None
        self._router_extra: dict = {}

    # ---- device / weight handling -------------------------------------
    @staticmethod
    def _supports_current_device() -> bool:
        return _supports_mixed_device()

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        # Routing is delegated to mainline's router objects (see _get_router),
        # so every routing method the layer can be built with is supported —
        # including DeepSeekV3 (sigmoid+noaux_tc), DeepseekV4 (sqrtsoftplus),
        # grouped top-k and custom routing functions.
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # The engine consumes the *packed* w13 layout (gate rows first, then up)
        # and applies an optional clamp; that is exactly mainline's SILU and
        # SWIGLUOAI_UNINTERLEAVE. SWIGLUOAI (gpt-oss) interleaves gate/up inside
        # w13, so accepting it here would produce silently wrong results.
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Do NOT AMX-prepack; remember the layer so the engine can use the raw
        CPU parameters (the checkpoint layout the engine consumes directly)."""
        self._layer_ref = layer
        for name in _ROUTING_ATTRS:
            if hasattr(layer, name):
                setattr(self, name, getattr(layer, name))
        # DeepSeek-V4 routes with `sqrtsoftplus` + a per-expert correction bias;
        # mainline expresses exactly that as `FusedTopKBiasRouter`
        # (fused_topk_bias_router.py, which handles the bias, sqrtsoftplus, the
        # hash table and the vision bias). `router_factory.create_fused_moe_router`
        # only picks it when `e_score_correction_bias is not None`; otherwise it
        # falls back to `FusedTopKRouter`, whose `fused_topk` has no sqrtsoftplus
        # support and raises `ValueError: Unsupported scoring function`.
        # `RoutedExperts` does not always expose the bias as a top-level attribute
        # (DeepSeek-V4 keeps it on the gate), so resolve it from the usual owners.
        if getattr(self, "e_score_correction_bias", None) is None:
            gate = getattr(layer, "gate", None)
            for owner, attr in (
                (layer, "e_score_correction_bias"),
                (gate, "e_score_correction_bias"),
                (gate, "bias"),
                (getattr(layer, "router", None), "e_score_correction_bias"),
            ):
                if owner is None:
                    continue
                bias = getattr(owner, attr, None)
                if isinstance(bias, torch.Tensor):
                    self.e_score_correction_bias = bias
                    break
        for name in _SWIGLU_ATTRS:
            if hasattr(layer, name):
                setattr(self, name, getattr(layer, name))
        self._router_extra = {
            name: getattr(layer, name)
            for name in _ROUTER_EXTRA_ATTRS
            if getattr(layer, name, None) is not None
        }
        # Resolve block scales once (converted to the dtype the engine reads).
        # We keep our own tensors instead of mutating vLLM's parameters so the
        # GPU path (if used for the same layer elsewhere) is unaffected.
        self._scales: tuple = (None, None)
        if self._scale_dtype is not None:
            resolved = []
            for name in self._scale_attrs:
                cands = name if isinstance(name, (tuple, list)) else (name,)
                t = self._find_scale(layer, cands)
                if t is not None and t.dtype != self._scale_dtype:
                    t = t.to(self._scale_dtype).contiguous()
                resolved.append(t)
            self._scales = tuple(resolved)
        self._router = None
        # ---- 切分一层、释放一层(内存;默认关闭,未 A/B)------------------------
        # vLLM 会把**每层**的主机专家张量留到运行结束,而我们的引擎又持有一份
        # per-NUMA 分片副本 ⇒ 专家权重 **2×**(IRON_RULES R9);
        # 对 V4.1-Flash 就是 271 GB 白白占用。
        # 引擎构造时已把字节拷进自己的分片区,之后源张量就是死重量 ——
        # 所以在这里**提前构造引擎并释放源存储**,而不是等第一次 forward 懒加载。
        # 开关:XIAOTU_RELEASE_SOURCE=1(默认 0,待 A/B 后再改默认)。
        # ⚠️ 模块化(Mode B)路径下主线**不会在装载期调用本钩子**:日志实测
        # `Model loading took ...`(v41n2.log:52)出现在第一条 `[SHARD-DIAG]`(:57)
        # **之前**,说明引擎是在 profile run 的第一次 forward 里惰性建的。
        # 所以真正生效的释放点是 `apply()` 里的 `_maybe_release_source`(NOTES §378)。
        # 【2026-09-15 cellC】但**也不能无条件在这里提前建**:那样会 SIGSEGV,见
        # `_eager_build_ok` 的文档。先验证,不满足就退回惰性路径并打印原因。
        if self._is_resident_layer(layer):
            # 常驻层:不建 CPU 引擎、不释放源(见 _maybe_release_source 的注释)。
            print(
                f"[vllm-xtu-moe] resident layer keeps host sources: "
                f"{getattr(layer, 'layer_name', '?')}",
                flush=True,
            )
            self._src_released = True
            return
        if _release_source_enabled() and self._xiaotu_engine is None:
            ok, why = self._eager_build_ok(layer)
            if ok:
                self._ensure_engine(layer)
            else:
                print(
                    f"[vllm-xtu-moe] deferring engine build to first forward "
                    f"({getattr(layer, 'layer_name', '?')}): {why}",
                    flush=True,
                )
        self._maybe_release_source(layer)

    @staticmethod
    def _stage_src(layer: torch.nn.Module, nm: str):
        """取用于建引擎的源张量:优先 shim 在 upstream 钩子**之前**保活的那份。

        原因见 `_eager_build_ok` 与 NOTES §385:pwal 返回后 `layer.w13_weight`
        可能指向一块临时/已释放的存储,直接读会 SIGSEGV。
        """
        # ⚠️ 只在**释放开关打开时**才优先用快照。否则会改变其它模型(V4 等)建引擎
        # 所依据的张量 —— 那正是"适配新模型不能破坏已有模型"的红线。
        if _release_source_enabled():
            snap = getattr(layer, "_xiaotu_pre_pwal_src", None) or {}
            t = snap.get(nm)
            if isinstance(t, torch.Tensor):
                return t
        return getattr(layer, nm, None)

    def _eager_build_ok(self, layer: torch.nn.Module) -> tuple[bool, str]:
        """能否在 `process_weights_after_loading` 里**安全**地提前建引擎。

        2026-09-15 cellC 实测:这里直接 `_ensure_engine` 会 **SIGSEGV**,崩在 C++
        引擎构造函数里的 `shard_fill_w13` memcpy —— 寄存器 `RSI`(源指针)与故障地址
        **完全相等**(0x7ecc84000000,页对齐),`RDX = 0x2d0000 = 2949120 = cbytes`,
        即第一次 gate 拷贝就读到不可读内存 ⇒ 那一刻 `layer.w13_weight` 给不出可读的主机指针。

        这也顺带解释了历史上"释放了 0 字节且完全静默":`_release_source_weights` 的
        `p.device.type != "cpu"` 会直接 `continue` —— **两者是同一个原因**。

        所以先验证再建;不满足就退回惰性路径(在 `apply()` 里建,那里已验证可用)并打印原因。
        宁可晚一点释放,也不要 segfault。
        """
        for nm in ("w13_weight", "w2_weight"):
            p = self._stage_src(layer, nm)
            if not isinstance(p, torch.Tensor):
                return False, f"{nm} is {type(p).__name__}"
            if p.device.type != "cpu":
                return False, f"{nm} on device={p.device.type}"
            if p.numel() == 0:
                return False, f"{nm} is empty"
            if not p.is_contiguous():
                return False, f"{nm} not contiguous"
            # 子代理指出:cellC 那种坏张量恰好能通过上面所有检查。补两项 ——
            # (a) 非零 storage_offset 会让"从 data_ptr 起读满 E*stride"越出视图;
            # (b) 长度必须够引擎按定长步长读满,否则就是读到别人的内存。
            if p.storage_offset() != 0:
                return False, f"{nm} storage_offset={p.storage_offset()}"
        w13 = self._stage_src(layer, "w13_weight")
        w2 = self._stage_src(layer, "w2_weight")
        if isinstance(w13, torch.Tensor) and isinstance(w2, torch.Tensor):
            if w13.dim() != 3 or w2.dim() != 3:
                return False, f"rank: w13 {w13.dim()}D w2 {w2.dim()}D (want 3D)"
            e = int(self.moe_config.num_experts)
            # H/I 必须**从 w13 推导**,再拿去验 w2;若反过来用 w2 的形状推导就是自证。
            h = int(w13.shape[2]) * 2
            i = int(w13.shape[1]) // 2
            need13 = e * 2 * i * (h // 2)
            need2 = e * h * (i // 2)
            if int(w13.shape[0]) != e or int(w2.shape[0]) != e:
                return False, (f"expert dim: w13 {int(w13.shape[0])} w2 "
                               f"{int(w2.shape[0])} vs E={e}")
            if int(w13.numel()) < need13 or int(w2.numel()) < need2:
                return False, (f"short source: w13 {int(w13.numel())}<{need13} or "
                               f"w2 {int(w2.numel())}<{need2} (E={e} H={h} I={i})")
        return True, ""

    def _maybe_release_source(self, layer: torch.nn.Module) -> int:
        """切分一层、释放一层(env 门控、每层幂等)。

        必须同时挂在 `process_weights_after_loading` 与惰性 `apply()` 两处:
        不同主线版本走哪条路不一样,漏一处就会像 2026-09-15 那样**静默零释放**。
        """
        if not _release_source_enabled():
            return 0
        if getattr(self, "_src_released", False):
            return 0
        # ⚠️ **常驻层绝不能释放**:它的权重还要被一次性搬到 GPU 常驻(NOTES §418)。
        # 实测踩过:装载期的 eager 路径不知道常驻层的存在,把 layers.38/39 的源张量
        # 换成了 1 字节 `empty_strided` 空壳 ⇒ 常驻槽位拿到垃圾/直接报错。
        if self._is_resident_layer(layer):
            self._src_released = True          # 标记已处理,避免反复判
            return 0
        self._src_released = True
        return self._release_source_weights(layer)

    def _release_source_weights(self, layer: torch.nn.Module) -> int:
        """释放引擎已分片的**主机源张量**,只保留参数对象本身。

        ⚠️ 这是**未经端到端验证**的优化(2026-09-15 加入,默认关闭)。
        参数对象会变成 0 元素张量:模块化链路只把 w1/w2 **透传**给我们的
        `apply()`,而 `apply()` 用的是引擎、**从不读它们** —— 但如果上游某处读了
        `.shape`/`.data_ptr()`,就会出问题。A/B 时先看能不能跑通再谈收益。
        """
        global _RELEASE_MISSES
        freed = 0
        self._released_shapes = getattr(self, "_released_shapes", {})
        names = ["w13_weight", "w2_weight"]
        for attr in getattr(self, "_scale_attrs", ()):
            names.extend(attr if isinstance(attr, (tuple, list)) else (attr,))
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            p = getattr(layer, name, None)
            if not isinstance(p, torch.Tensor) or p.device.type != "cpu":
                continue
            nbytes = p.numel() * p.element_size()
            if nbytes < (1 << 28):  # 小张量不值得冒险
                continue
            try:
                # 模块化链路仍会把已置空的 w1/w2 **透传**进 apply(),而 apply 要从
                # w2 的 dim 1 取 hidden_size ⇒ 必须在置空前把形状记下来。
                self._released_shapes[name] = tuple(p.shape)
                # ⚠️ **不能**用 `torch.empty(0)`:那是 **0 维**张量,而模块化链路
                # (`modular_kernel.moe_problem_size`)会断言
                # `len(w1.shape) == 3 and len(w2.shape) == 3` —— 0 维会直接
                # AssertionError 打挂 EngineCore(cellI 实测)。
                # 用全零 stride 的 `empty_strided`:保留真实 shape/ndim,
                # 但底层 storage 只有 **1 个元素**,内存照样放掉。
                p.data = torch.empty_strided(
                    tuple(p.shape), (0,) * p.dim(), dtype=p.dtype, device=p.device
                )
            except Exception:  # noqa: BLE001
                continue
            freed += nbytes
        if freed:
            print(
                f"[vllm-xtu-moe] released {freed / 2**30:.2f} GiB host source "
                f"expert weights ({getattr(layer, 'layer_name', '?')})",
                flush=True,
            )
        elif _RELEASE_MISSES < 3:
            # 释放路径**静默失败**过一次(2026-09-15):XIAOTU_RELEASE_SOURCE=1
            # 明明生效、引擎也建了,日志里却一条 release 都没有。原因是
            # `if freed:` 把 freed==0 的情况完全吞掉了。这里把"为什么一个字节
            # 都没释放"直接打出来,否则无从判断。
            _RELEASE_MISSES += 1
            seen = []
            for name in names:
                p = getattr(layer, name, None)
                if isinstance(p, torch.Tensor):
                    seen.append(
                        f"{name}:{tuple(p.shape)}/{p.dtype}/"
                        f"{p.numel() * p.element_size() / 2**20:.0f}MiB/{p.device.type}"
                    )
                else:
                    seen.append(f"{name}:{type(p).__name__}")
            print(
                f"[vllm-xtu-moe] release found NOTHING to free on "
                f"{getattr(layer, 'layer_name', '?')} (miss {_RELEASE_MISSES}): "
                + ", ".join(seen),
                flush=True,
            )
        # ⚠️ **必须清掉"钩子前快照"**。它持有**原始**专家张量的引用,而上面的
        # 释放只是把参数指向了 1 字节的 `empty_strided` —— 不清引用,那
        # 6.33 GiB/层(40 层 = 264 GiB)就永远回不到 OS。
        # 这正是 cellJ/cellK 里 RSS(1064~1116 GB)比账面(~467 GiB)高出几百 GB 的主因。
        try:
            layer._xiaotu_pre_pwal_src = None
        except Exception:  # noqa: BLE001
            pass
        return freed

    # ---- routing -------------------------------------------------------
    def _get_router(self):
        if self._router is not None:
            return self._router
        from vllm.model_executor.layers.fused_moe.router.router_factory import (
            create_fused_moe_router,
        )

        kwargs = dict(
            top_k=self.moe_config.experts_per_token,
            global_num_experts=self.moe_config.num_experts,
            eplb_state=None,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )
        for name, value in self._router_extra.items():
            kwargs.setdefault(name, value)
        try:
            self._router = create_fused_moe_router(**kwargs)
            print(
                f"[vllm-xtu-moe] router={type(self._router).__name__} "
                f"scoring_func={self.scoring_func} "
                f"top_k={self.moe_config.experts_per_token} "
                f"bias={'yes' if self.e_score_correction_bias is not None else 'no'} "
                f"hash={'yes' if self._router_extra.get('hash_indices_table') is not None else 'no'} "
                f"vl={'yes' if self._router_extra.get('bias_vl') is not None else 'no'} "
                f"grouped={self.use_grouped_topk}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[vllm-xtu-moe] router factory failed ({exc}); falling back to "
                f"cpu_moe.select_experts(scoring_func={self.scoring_func})",
                flush=True,
            )
            self._router = False  # sentinel: use the legacy fallback
        return self._router

    def _select_topk(self, hidden_states, router_logits, input_ids=None):
        router = self._get_router()
        if router:
            return router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=torch.int32,
                # DeepSeek-V4's first `num_hash_layers` layers route by table
                # lookup on the token ids (upstream `fused_topk_bias`); without
                # this the hash router has nothing to look up. `input_ids` never
                # reaches fused_experts through the monolithic chain, so shim 6
                # (mainline_shims) parks it on the layer.
                input_ids=input_ids,
            )
        # `select_experts` only implements softmax/sigmoid. Silently mapping any
        # other scoring function onto softmax picks the WRONG experts (see
        # docs/UPSTREAM.md §2.1), so refuse instead of degrading quietly.
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(
                f"xiaotu CPU backend has no fallback routing for "
                f"scoring_func={self.scoring_func!r}; the upstream "
                f"FusedTopKBiasRouter (router_factory) should have been selected "
                f"instead — check that e_score_correction_bias reached the router "
                f"factory."
            )
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

    def _local_expert_map(self, expert_map, device):
        """把 expert_map 缓存在 ids 所在的设备上(EP 下每层一份)。"""
        cached = getattr(self, "_expert_map_cached", None)
        if cached is not None and cached[0] == device:
            return cached[1]
        em = expert_map
        if em.device != device:
            em = em.to(device)
        if em.dtype not in (torch.int32, torch.int64):
            em = em.to(torch.long)
        self._expert_map_cached = (device, em)
        return em

    # ---- engine construction ------------------------------------------
    def _validate_weights(self, layer, ex_w13, ex_w2) -> None:
        """Per-format layout guard, run before the engine is built.

        Called with the tensors the engine will actually consume (i.e. after
        `_prepare_weights`). The engine reads the checkpoint layout directly (it
        never calls mainline's `prepare_*_for_cpu` AMX repack), so a format whose
        layout differs from the engine's must be converted or rejected here
        instead of silently computing garbage.
        """

    def _assert_host_source(self, ex_w13, ex_w2, ex_s13=None, ex_s2=None) -> None:
        """引擎只接受**连续、非空、在主机上**的源张量。

        这些指针会被直接交给 C++ 引擎,引擎在里面按 `[E][2I][H/2]` 的定长步长
        `memcpy`。一旦拿到设备指针/空张量/非连续视图,C++ 侧只会在
        `shard_fill_w13` 里读到不可读地址然后 **SIGSEGV**(2026-09-15 cellC 实测),
        没有 Python 栈可看。所以在这里先拦下来,给出可读的错误。
        """
        for nm, t in (("w13", ex_w13), ("w2", ex_w2),
                      ("s13", ex_s13), ("s2", ex_s2)):
            if t is None:
                if nm in ("s13", "s2"):
                    continue      # 未量化后端可以没有 scale
                raise RuntimeError(f"xiaotu {self._engine_attr}: {nm} is None")
            bad = (
                t.device.type != "cpu"
                or t.numel() == 0
                or not t.is_contiguous()
            )
            if bad:
                raise RuntimeError(
                    f"xiaotu {self._engine_attr}: refusing to build the engine "
                    f"from a non-host source ({nm}: device={t.device.type} "
                    f"numel={t.numel()} contiguous={t.is_contiguous()} "
                    f"shape={tuple(t.shape)}); handing this to the C++ engine "
                    f"would segfault in shard_fill_*."
                )
            # ⚠️【NOTES §486】**必须**检查 storage 大小。`XIAOTU_RELEASE_SOURCE=1`
            # 的释放用 `empty_strided(shape, (0,)*ndim)` —— **shape/ndim/contiguous
            # 都还正常**,只有 storage 掉到近 0 字节。上面三项检查全部通过,
            # 但 `data_ptr()` 已经悬空,引擎按 cfg 算出的长度 memcpy 就会读到映射尽头
            # 而 **SIGSEGV**(无 Python 栈)。只查 w13/w2 也不够:释放同时覆盖
            # `_scale_attrs`(w13_weight_scale / w2_weight_scale),而实测 DSpark draft
            # 崩的那一次 memcpy 长度恰好 == E·H·(I/gk),**就是 w2 的 scale 拷贝**。
            _need = t.numel() * t.element_size()
            _have = t.untyped_storage().nbytes()
            if _have < _need:
                raise RuntimeError(
                    f"xiaotu {self._engine_attr}: {nm} was RELEASED before the engine "
                    f"was built (need {_need} bytes, storage {_have} bytes, "
                    f"shape={tuple(t.shape)}) — reading it would SIGSEGV in the C++ "
                    f"engine. Fix the release ordering for this layer."
                )

    def _prepare_weights(self, layer) -> None:
        """Resolve the tensors/grouping the engine consumes.

        Default: the layer's own tensors — BF16/FP8/MXFP4/NVFP4 checkpoints
        already match the engine's layout. Formats that differ override this and
        set `self._engine_w13/_engine_w2`, `self._scales`,
        `self._group_n/_group_k` (INT4/WNA16 repacks the GPTQ int32 packing).
        """
        self._engine_w13 = None
        self._engine_w2 = None

    def _find_scale(self, layer, names):
        for n in names:
            if not n:  # unquantized backends have no scale tensors
                continue
            t = getattr(layer, n, None)
            if t is not None:
                return t
        return None

    # ---- GPU 常驻层(V4.1 模块化路径;NOTES §417)---------------------------
    def _is_resident_layer(self, layer: torch.nn.Module) -> bool:
        """本层是否在 `XIAOTU_MOE_GPU_RESIDENT_LAYERS` 里。

        ⚠️ 这条功能原先只存在于 `hybrid_model.py`(DeepSeek-**V4** 的 OOT 路径),
        V4.1 走的模块化路径**根本没有实现** —— 这正是 §414/§416 两轮排查的收敛点。
        默认空集合 ⇒ 恒为 False ⇒ 非常驻路径逐字不变。
        """
        try:
            from vllm_xiaotu_moe.hybrid_model import gpu_resident_layers

            spec = gpu_resident_layers()
            if not spec:
                return False
            m = _LAYER_IDX_RE.search(getattr(layer, "layer_name", "") or "")
            return bool(m) and int(m.group(1)) in spec
        except Exception:  # noqa: BLE001
            return False

    def _ensure_resident(self, layer: torch.nn.Module):
        """把本层权重**一次性**搬到 GPU 并转成 K-major,做成常驻槽位。

        复用 `gpu_prefill.gpu_moe_layer` 的 `slot=` 接口(它已处理 CUDA graph 捕获:
        常驻槽位的 ready 事件在**捕获外** record,图内不 wait_event)。常驻之后这些层
        **不再走 CPU 引擎**,因此贡献 0 往返、0 DRAM 权重流量。
        """
        if getattr(self, "_resident_slot", None) is not None:
            return self._resident_slot
        # ⚠️ 逐字照抄 V4 的 `_build_resident_slot`(hybrid_model.py:1086):
        #   * K-major 在**主机**上做(`_pinned_kmajor`,顺带锁页),再阻塞 H2D;
        #     我先前写成"先 .to(cuda) 再在设备上转",会走到另一条内核路径 ⇒ 曾触发
        #     `Triton Error [CUDA]: an illegal memory access`(cellAI);
        #   * 设备缓冲用 `slot.alloc(src, dev)` 按 src 的形状/dtype 分配;
        #   * `slot.busy = None`(不是空 Event)。
        from vllm_xiaotu_moe.gpu_prefill import PrefetchSlot, _pinned_kmajor

        # 解析本层实际要用的权重/尺度(与 _ensure_engine 同源;INT4 等格式的
        # 重排由 _prepare_weights 负责)。
        self._prepare_weights(layer)
        w13 = getattr(self, "_engine_w13", None) if getattr(self, "_engine_w13", None) is not None else layer.w13_weight
        w2 = getattr(self, "_engine_w2", None) if getattr(self, "_engine_w2", None) is not None else layer.w2_weight
        s13, s2 = self._scales
        if s13 is None:
            s13 = self._find_scale(layer, (self._scale_attrs[0],))
        if s2 is None:
            s2 = self._find_scale(layer, (self._scale_attrs[1],))
        for _nm, _t in (("w13", w13), ("w2", w2), ("s13", s13), ("s2", s2)):
            if not isinstance(_t, torch.Tensor) or _t.device.type != "cpu":
                raise RuntimeError(
                    f"xiaotu resident layer needs a host tensor for {_nm}; got "
                    f"{type(_t).__name__} device={getattr(_t, 'device', None)}"
                )
            # ⚠️ 这里**不能**要求 `is_contiguous()`:`_kmajor_bytes` 内部会
            # `transpose(1,2).contiguous()`,本来就容忍非连续输入;而实测
            # `w13_weight` 恰恰是非连续的(带 stride 的视图)。但必须确认它不是
            # 已被 `_release_source_weights` 换成 1 字节 storage 的空壳,否则常驻层
            # 会拿垃圾数据算。所以打印/校验 storage 大小。
            _need = _t.numel() * _t.element_size()
            _have = _t.untyped_storage().nbytes()
            if _have < _need:
                raise RuntimeError(
                    f"xiaotu resident layer {_nm} has a released/undersized storage: "
                    f"need {_need} bytes, storage {_have} bytes "
                    f"(shape={tuple(_t.shape)} stride={tuple(_t.stride())})"
                )
            if not getattr(self, "_resident_dbg", False):
                self._resident_dbg = True
                print(
                    f"[vllm-xtu-moe] resident src probe: {_nm} "
                    f"shape={tuple(_t.shape)} stride={tuple(_t.stride())} "
                    f"storage={_have} contiguous={_t.is_contiguous()}",
                    flush=True,
                )

        dev = torch.device("cuda", torch.cuda.current_device())
        src = (
            _pinned_kmajor(w13), _pinned_kmajor(s13),
            _pinned_kmajor(w2), _pinned_kmajor(s2),
        )
        slot = PrefetchSlot()
        slot.alloc(src, dev)
        for buf, s_ in zip(slot.bufs, src):
            buf.copy_(s_, non_blocking=False)
        slot.ready = torch.cuda.Event()
        slot.ready.record()
        slot.busy = None
        self._resident_slot = slot
        self._slot_device = dev
        self._resident_keep = src          # 保活(主机 pinned 源)
        self._resident_I = int(w13.shape[1]) // 2
        nbytes = sum(b.numel() * b.element_size() for b in slot.bufs)
        print(
            f"[vllm-xtu-moe] GPU-resident(V4.1) {getattr(layer, 'layer_name', '?')}: "
            f"{nbytes / 2**30:.2f} GiB on {dev}",
            flush=True,
        )
        return slot

    def _ensure_engine(self, layer: torch.nn.Module):
        if self._xiaotu_engine is not None:
            return self._xiaotu_engine
        import xiaotu_moe

        self._prepare_weights(layer)
        ex_w13 = (getattr(self, "_engine_w13", None) if getattr(self, "_engine_w13", None) is not None
                  else self._stage_src(layer, "w13_weight"))
        ex_w2 = (getattr(self, "_engine_w2", None) if getattr(self, "_engine_w2", None) is not None
                 else self._stage_src(layer, "w2_weight"))
        self._validate_weights(layer, ex_w13, ex_w2)
        self._assert_host_source(ex_w13, ex_w2, s13, s2)
        if self._expect_dtype is not None and ex_w13.dtype != self._expect_dtype:
            raise NotImplementedError(
                f"xiaotu {self._engine_attr} backend expects "
                f"{self._expect_dtype} weights, got {ex_w13.dtype}"
            )
        s13, s2 = self._scales
        if s13 is None and self._scale_dtype is None:
            # unquantized / e8m0 formats: use the layer tensors as-is
            s13 = self._find_scale(layer, (self._scale_attrs[0],))
            s2 = self._find_scale(layer, (self._scale_attrs[1],))
        num_local_experts = int(ex_w13.shape[0])
        if num_local_experts != int(self.moe_config.num_experts):
            # 专家并行:本 rank 只持有 local_num_experts 个专家,id 已在 apply 里
            # 按 expert_map 映射成局部 id。
            print(
                f"[vllm-xtu-moe] expert parallelism: local={num_local_experts} "
                f"global={self.moe_config.num_experts}",
                flush=True,
            )
        if not self._supports_activation(self.moe_config.activation):
            raise NotImplementedError(
                f"xiaotu CPU experts backend implements the packed gated "
                f"activation only (SILU / SWIGLUOAI_UNINTERLEAVE), got "
                f"{self.moe_config.activation}"
            )

        cfg = xiaotu_moe.MOEConfigV2()
        # 【v0.2】把 TP rank 告诉引擎,**只用于 NUMA/核放置**(每个 rank 只占自己那半
        # 12 个 CCD + 4 个 NUMA node),归约由主线自己做(专家按 I 切分 + expert_map)
        # ⇒ 必须同时关掉引擎的自建 EP 归约,否则会重复归约。
        # 实测动机:num_processes=1 时池横跨全部 24 CCD,与 vLLM 自己的线程抢核,
        # 每层 compute 呈双峰(MIN 0.41 ms vs 典型 9.5 ms,见 NOTES §334p)。
        os.environ.setdefault("XIAOTU_MOE_NO_AUTO_EP", "1")
        _tp, _rank = 1, 0
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            _tp = int(get_tensor_model_parallel_world_size())
            _rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            _tp, _rank = 1, 0
        cfg.num_processes = max(1, _tp)
        cfg.process_id = max(0, _rank)
        cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True
        cfg.expert_num = num_local_experts
        cfg.top_k = self.moe_config.experts_per_token
        cfg.hidden_size = int(ex_w2.shape[1])            # w2: [E, H, I//2] or [E, H, I]
        cfg.intermediate_size = int(ex_w13.shape[1] // 2)  # w13: [E, 2I, ...]
        # 每 token 缓冲在 V2 引擎里是按调用分配的,这两个值只影响预分配提示;
        # 仍按 vLLM 的调度配置设置,避免与真实 batch 规模脱节。
        sched = getattr(self.moe_config, "scheduler_config", None)
        cfg.max_batch_size = int(getattr(sched, "max_num_batched_tokens", 0) or 8192)
        cfg.max_num_seqs = int(getattr(sched, "max_num_seqs", 0) or 256)
        cfg.stride = int(self._group_k)
        cfg.group_min_len = 10
        cfg.group_max_len = int(getattr(sched, "max_num_batched_tokens", 0) or 4096) + 128
        # 激活:0=plain SiLU(gate*up),1=clamped SwiGLU(vLLM silu_and_mul_with_clamp
        # 同语义,GLM/DS-V4/MiniMax 的 swiglu_limit)。
        limit = float(self.swiglu_limit) if self.swiglu_limit else 0.0
        alpha = float(self.swiglu_alpha) if self.swiglu_alpha else 1.0
        beta = float(self.swiglu_beta) if self.swiglu_beta else 0.0
        clamped = limit > 0.0 or alpha != 1.0 or beta != 0.0
        cfg.activation_type = 1 if clamped else 0
        cfg.swiglu_limit = limit
        cfg.swiglu_alpha = alpha
        cfg.use_gpu_prefill = False
        cfg.groupN = int(self._group_n)
        cfg.groupK = int(self._group_k)
        engine_cls = getattr(xiaotu_moe, self._engine_attr)
        # 【诊断·NOTES §484】ctor 只拿到 **裸 data_ptr**,而 shard_fill 会在里面按
        # cfg 推出来的几何 memcpy;DSpark 的 draft 层让这个 memcpy 读到了映射尽头
        # (SIGSEGV in MOE_V2<MXFP4>::MOE_V2)。这里把"cfg 期望的字节数"与
        # "张量实际的字节数/形状"摆在一起打一次 —— 不一致一眼可见。
        if not getattr(self, "_eng_shape_dbg", False):
            self._eng_shape_dbg = True
            _e13 = int(cfg.expert_num) * 2 * int(cfg.intermediate_size) * (int(cfg.hidden_size) // 2)
            _e2 = int(cfg.expert_num) * int(cfg.hidden_size) * (int(cfg.intermediate_size) // 2)
            print(
                f"[xtu-eng-shape] {getattr(layer, 'layer_name', '?')} "
                f"cfg(E={cfg.expert_num} H={cfg.hidden_size} I={cfg.intermediate_size} "
                f"gN={cfg.groupN} gK={cfg.groupK}) "
                f"| w13 shape={tuple(ex_w13.shape)} dtype={ex_w13.dtype} "
                f"bytes={ex_w13.numel() * ex_w13.element_size()} contig={ex_w13.is_contiguous()} "
                f"| w2 shape={tuple(ex_w2.shape)} bytes={ex_w2.numel() * ex_w2.element_size()} "
                f"| expect(w13)={_e13} expect(w2)={_e2}",
                flush=True,
            )
        self._xiaotu_engine = engine_cls(
            cfg,
            ex_w13.data_ptr(), ex_w2.data_ptr(),
            s13.data_ptr() if s13 is not None else 0,
            s2.data_ptr() if s2 is not None else 0,
            0, 0,
        )
        print(
            f"[vllm-xtu-moe] xiaotu {self._engine_attr} engine: "
            f"E={cfg.expert_num} H={cfg.hidden_size} I={cfg.intermediate_size} "
            f"topk={cfg.top_k} group={self._group_n}x{self._group_k} "
            f"scales={'yes' if s13 is not None else 'no'} "
            f"routing={self.scoring_func}"
            f"{f'/grouped{self.num_expert_group}x{self.topk_group}' if self.use_grouped_topk else ''}"
            f"{f'/bias' if self.e_score_correction_bias is not None else ''} "
            f"swiglu={'clamp@' + str(limit) if clamped else 'plain'}",
            flush=True,
        )
        return self._xiaotu_engine

    # ---- optional in-process self-verification -------------------------
    def _verify_once(self, layer, hidden_states, topk_ids, topk_weights, out):
        """对比引擎输出与 torch 参考(同一批已加载权重),定位数值差异。

        XIAOTU_VERIFY_LAYER=1 打开;每层只跑一次(跳过路由 id 无效的 profile 调用)。
        参考实现(对 token 0 的全部 top-k 专家求和):
          gate/up = dequant(w13[e]) @ x ; act = silu / clamped swiglu
          out += w * (dequant(w2[e]) @ bf16(act))
        """
        try:
            w13 = self._engine_w13 if getattr(self, "_engine_w13", None) is not None \
                else layer.w13_weight
            w2w = self._engine_w2 if getattr(self, "_engine_w2", None) is not None \
                else layer.w2_weight
            s13, s2 = self._scales
            unquant = (self._scale_dtype is None) and (s13 is None)
            gn, gk = int(self._group_n), int(self._group_k)
            I = int(w13.shape[1] // 2)
            H = int(w2w.shape[1])
            int4 = self._engine_attr == "MOE_WNA16" and w13.dtype == torch.uint8

            def _unpack4(w):   # [N, K/2] u8 -> [N, K] f32 with value = nibble - 8
                b = w.to(torch.int16)
                lo = (b & 0x0F) - 8
                hi = ((b >> 4) & 0x0F) - 8
                return torch.stack((lo, hi), dim=-1).reshape(w.shape[0], -1).float()

            x = hidden_states[0].float().cpu()
            ref = torch.zeros(H, dtype=torch.float32)
            ids_row = [int(v) for v in topk_ids[0]]
            for r, e in enumerate(ids_row):
                w = float(topk_weights[0, r])
                if w == 0.0:
                    continue
                w13_e = _unpack4(w13[e]) if int4 else w13[e].float()
                if unquant:
                    deq13 = w13_e
                else:
                    s13_full = s13[e].float().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
                    deq13 = w13_e * s13_full[: 2 * I, :H]
                gate = deq13[:I] @ x
                up = deq13[I:] @ x
                if self.swiglu_limit or self.swiglu_alpha or self.swiglu_beta:
                    limit = float(self.swiglu_limit or 0.0)
                    alpha = float(self.swiglu_alpha or 1.0)
                    beta = float(self.swiglu_beta or 0.0)
                    if limit > 0:
                        gate = torch.clamp(gate, max=limit)
                        up = torch.clamp(up, -limit, limit)
                    act = (gate / (1 + torch.exp(-alpha * gate))) * (up + beta)
                else:
                    act = (gate / (1 + torch.exp(-gate))) * up
                act = act.to(torch.bfloat16).float()
                w2_e = _unpack4(w2w[e]) if int4 else w2w[e].float()
                if unquant:
                    down = w2_e @ act
                else:
                    s2_full = s2[e].float().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
                    down = (w2_e * s2_full[:H, :I]) @ act
                ref += w * down
            got = out[0].float().cpu()
            rms = float(torch.sqrt(torch.mean(ref * ref)))
            rel = float(torch.sqrt(torch.mean((got - ref) ** 2)) / (rms + 1e-12))
            print(
                f"[vllm-xtu-moe/verify] {self._engine_attr} "
                f"layer={getattr(layer, 'layer_name', '?')} M={hidden_states.shape[0]} ids0={ids_row} "
                f"w0={[round(float(v), 4) for v in topk_weights[0]]} "
                f"ref_rms={rms:.4f} rel_rms={rel:.4e} "
                f"max_abs={float((got - ref).abs().max()):.4e}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[vllm-xtu-moe/verify] failed: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()

    # ---- optional one-shot dump/compare against the checkpoint -----------
    def _dump_layer(self, layer):
        """XIAOTU_DUMP_LAYER=<layer_name 子串> 时,打印已加载张量的形状/样例值;
        若 XIAOTU_CKPT_DIR 指向 checkpoint 目录,再与原始 per-expert 张量逐项比对。
        用途:判断"引擎与 torch 参考一致但模型输出不对"是否来自权重加载布局。"""
        import json
        import re

        import numpy as np

        name = getattr(layer, "layer_name", "")
        w13, w2 = layer.w13_weight, layer.w2_weight
        s13, s2 = self._scales
        print(
            f"[dump] layer={name} w13={tuple(w13.shape)}{w13.dtype} "
            f"w2={tuple(w2.shape)}{w2.dtype} "
            f"s13={None if s13 is None else tuple(s13.shape)} "
            f"s2={None if s2 is None else tuple(s2.shape)} "
            f"attrs_scale=({getattr(layer, 'w13_weight_scale_inv', None) is not None},"
            f"{getattr(layer, 'w2_weight_scale_inv', None) is not None})",
            flush=True,
        )
        ckpt = os.environ.get("XIAOTU_CKPT_DIR")
        m = re.search(r"layers\.(\d+)", name)
        if not ckpt or not m:
            return
        li = m.group(1)
        try:
            from safetensors import safe_open
        except ImportError:
            return
        idx_path = os.path.join(ckpt, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            return
        wm = json.load(open(idx_path))["weight_map"]

        def load(key, dtype_cast=None):
            with safe_open(os.path.join(ckpt, wm[key]), framework="pt") as f:
                t = f.get_tensor(key)
            return t

        import torch as _t

        pre = f"model.layers.{li}.mlp.experts.0."
        pairs = [
            ("w13[:I] vs gate", w13[0][: w13.shape[1] // 2],
             load(pre + "gate_proj.weight")),
            ("w13[I:] vs up", w13[0][w13.shape[1] // 2:],
             load(pre + "up_proj.weight")),
            ("w2 vs down", w2[0], load(pre + "down_proj.weight")),
        ]
        for label, got, want in pairs:
            g = got.view(_t.uint8).numpy() if got.dtype == _t.float8_e4m3fn else got.numpy()
            w = want.view(_t.uint8).numpy() if want.dtype == _t.float8_e4m3fn else want.numpy()
            same = g.shape == w.shape
            diff = float(np.abs(g.astype(np.int16) - w.astype(np.int16)).max()) if same else -1
            print(f"[dump]   {label}: shape got={g.shape} ckpt={w.shape} "
                  f"byte_max_diff={diff}", flush=True)
        if s13 is not None:
            gs = s13[0].float().numpy()
            gs_ck = load(pre + "gate_proj.weight_scale_inv").float().numpy()
            us_ck = load(pre + "up_proj.weight_scale_inv").float().numpy()
            d2 = s2[0].float().numpy()
            ds_ck = load(pre + "down_proj.weight_scale_inv").float().numpy()
            print(f"[dump]   s13 got={gs.shape} ckpt_cat={np.concatenate([gs_ck, us_ck], 0).shape} "
                  f"gate_diff={float(np.abs(gs[:gs_ck.shape[0]] - gs_ck).max()):.3e} "
                  f"up_diff={float(np.abs(gs[gs_ck.shape[0]:] - us_ck).max()):.3e} "
                  f"s2 got={d2.shape} ckpt={ds_ck.shape} "
                  f"diff={float(np.abs(d2 - ds_ck).max()):.3e}", flush=True)

    # ---- compute ------------------------------------------------------
    # Two upstream call conventions must both work:
    #   * monolithic (shim path): apply(hidden_states, w1, w2, router_logits, ...)
    #     -> we compute routing here from router_logits.
    #   * modular (upstream RoutedExperts + our CPU backend, mainline >= 0.29):
    #     apply(output=..., hidden_states=..., w1=..., w2=..., topk_weights=...,
    #     topk_ids=..., ..., workspace13=..., expert_tokens_meta=...)
    #     -> the router already produced topk_weights/topk_ids AND already applied
    #     expert_map (upstream's own CPUExperts*.apply does not remap either), so
    #     we must NOT remap them again. The result is written into `output`.
    # The first nine parameters keep the monolithic order so existing positional
    # calls are unaffected; everything new is keyword-only.
    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        activation: MoEActivation | None = None,
        global_num_experts: int = 0,
        expert_map: torch.Tensor | None = None,
        a1q_scale: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
        *,
        output: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        workspace13: torch.Tensor | None = None,
        workspace2: torch.Tensor | None = None,
        expert_tokens_meta: object | None = None,
    ) -> torch.Tensor:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "xiaotu CPU experts backend does not support "
                "apply_router_weight_on_input"
            )
        if topk_weights is not None and topk_ids is not None:
            # Modular path: routing is done, expert ids are already local.
            topk_weights = topk_weights.to(torch.float32)
            topk_ids = topk_ids.to(torch.int32)
        else:
            # Monolithic path: compute routing ourselves.
            # monolithic apply() 拿不到 input_ids,若模型的路由需要它(hash routing),
            # router 会在这里报错——比静默选错专家好。
            # shim 6(mainline_shims)把 monolithic 链路丢掉的 input_ids 暂存在 layer 上。
            _ids = getattr(self._layer_ref, "_xiaotu_input_ids", None)
            topk_weights, topk_ids = self._select_topk(
                hidden_states, router_logits, _ids
            )

            # 专家并行(EP):路由给出的是全局 expert id,需要按 expert_map 映射到本
            # rank 的本地 id;映射为 -1 表示该专家不在本 rank,把权重置 0。
            if expert_map is not None:
                em = self._local_expert_map(expert_map, topk_ids.device)
                local = em[topk_ids.to(torch.long)]
                miss = local < 0
                if bool(miss.any()):
                    topk_weights = torch.where(
                        miss, torch.zeros_like(topk_weights), topk_weights
                    )
                    local = torch.where(miss, torch.zeros_like(local), local)
                topk_ids = local.to(torch.int32)

        layer = self._layer_ref
        if layer is None:
            raise RuntimeError(
                "XiaotuCPUExperts.apply called before process_weights_after_loading"
            )
        # GPU 常驻层:不建 CPU 引擎、也不释放源张量(它还要被常驻槽位用)。
        _resident = self._is_resident_layer(layer)
        qlen = hidden_states.size(0)
        # ---- 长 prefill 的 GPU **流式**路径(阈值门控) ----------------------
        # 与"常驻层"是两件事:常驻层把权重永久留在显存;这里每次 forward 把本层
        # 原始 MXFP4 权重 H2D 一遍、算完即弃(V4.1 = 6.72 GiB/层)。
        # 因此它只在 batch 足够大、能把这次**固定 DMA** 摊薄时才划算 ⇒ 必须阈值门控;
        # 阈值怎么按 PCIe 带宽/显卡能力选,见 docs/GPU_PREFILL.md。
        # 三个硬性前提(任一不满足就留在 CPU):
        #   * 只是 MXFP4 后端 —— gpu_moe_layer 的 Triton 内核只实现了 fp4+e8m0;
        #   * 不能在图捕获里(V4.1 走图,捕获期开新 H2D 会作废捕获);
        #   * 源权重必须还在(见下面 storage 校验)。
        _gp_min = 0
        _gp_on = False          # 本模块是否启用了 GPU prefill(与本次 qlen 无关)
        _gpu_pf = False         # 本次调用是否真的走 GPU
        if getattr(self, "_engine_attr", "") == "MOE_MXFP4" and not _resident:
            from vllm_xiaotu_moe.gpu_prefill import (
                gpu_prefill_min_tokens,
                in_profile_run,
            )

            _gp_min = gpu_prefill_min_tokens()
            _gp_on = _gp_min > 0
            # ⚠️ profile run 期间**必须**留在 CPU:vLLM 用那次 forward 的峰值显存
            # 给 KV cache 定容,而流式 staging 有 ~14 GiB(V4.1@TP=1);不挡住就会
            # 得到 `Available KV cache memory: -1.57 GiB` ⇒ **服务起不来**
            # (实测,NOTES §460)。挡住之后 KV 按 CPU 路径定容(正常),运行时再预检。
            _gpu_pf = (
                _gp_on
                and qlen >= _gp_min
                and not in_profile_run()
                and not torch.cuda.is_current_stream_capturing()
            )
        # ⚠️ 释放条件是 `_gp_on`,不是 `_gpu_pf`。引擎是**惰性**建的:第一个请求
        # 若是长 prefill 就走 GPU、不建引擎;而紧接着的解码步(qlen < 阈值)会把引擎
        # 建起来 —— 那一刻若把源权重释放掉,下一次长 prefill 就没得流式读了。
        # 所以只要这个模块启用了 GPU prefill,就**永不释放**源权重。
        # 引擎**始终**要建(非常驻层):GPU 流式路径读的就是引擎自有的紧凑分片
        # (NOTES §459),而且解码步无论如何都需要它。
        engine = None if _resident else self._ensure_engine(layer)
        if not _resident and not _gp_on:
            # 惰性建引擎的这一刻,主机源张量已经没有任何消费者了 —— 立即释放。
            # 这是 V4.1(模块化路径)上唯一真正会执行的释放点,见 NOTES §378。
            self._maybe_release_source(layer)
        # Use the tensor the ENGINE was built from: for formats whose checkpoint
        # layout differs (INT4/WNA16), the `w2` argument is the raw packed tensor
        # ([E, I/8, H]) and its dim 1 is NOT the hidden size.
        w2_engine = getattr(self, "_engine_w2", None) if getattr(self, "_engine_w2", None) is not None else w2
        if w2_engine.numel() == 0:
            # XIAOTU_RELEASE_SOURCE=1 已把源张量置空,而模块化链路仍把它透传进来。
            # 用释放时记下的形状,否则 shape[1] 直接越界(会让第二次 forward 挂掉)。
            hidden_size = int(
                getattr(self, "_released_shapes", {}).get("w2_weight", (0, 0))[1]
            )
            if hidden_size <= 0:
                raise RuntimeError(
                    "xiaotu: w2 is empty and no released shape was recorded "
                    "(XIAOTU_RELEASE_SOURCE released the source without "
                    "capturing its shape)"
                )
        else:
            hidden_size = int(w2_engine.shape[1])
        out = torch.empty(qlen, hidden_size, dtype=torch.float32,
                          device=hidden_states.device)
        stream = torch.cuda.current_stream()
        if os.environ.get("XIAOTU_STREAM_TRACE") and not getattr(self, "_st_traced", False):
            self._st_traced = True
            print(f"[stream] cur={stream.cuda_stream:#x} "
                  f"default={torch.cuda.default_stream().cuda_stream:#x} "
                  f"cur_device={stream.device} layer={getattr(layer,'layer_name','?')}",
                  flush=True)
        # The binding's pointer extractor accepts numpy arrays or integer
        # data_ptr() values (NOT torch tensors -> nullptr), so pass data_ptr().
        h_bf16 = hidden_states.to(torch.bfloat16)
        ids_i32 = topk_ids.to(torch.int32)
        wts_f32 = topk_weights.to(torch.float32)
        # 传裸指针给引擎做异步 D2H/H2D:必须保证这些张量在拷贝完成前不被释放或复用。
        # ① 保留上一代的强引用(同一层下一次调用在流序上晚于本次拷贝);
        # ② record_stream 告知分配器这些块仍被当前流使用。
        self._keepalive = (h_bf16, ids_i32, wts_f32, out)
        for t in (h_bf16, ids_i32, wts_f32, out):
            if t.is_cuda:
                t.record_stream(stream)
        # 引擎调用前后可选地与设备同步。跨请求状态污染已在 binding.cpp 中从根上修复
        # (每次调用使用不可变的参数块,回调不再读 per-engine 的可变字段),因此默认
        # 关闭;调试或怀疑时序问题时设 XIAOTU_SYNC_DECODE=1 可强制串行化。
        _sync = os.environ.get("XIAOTU_SYNC_DECODE", "0")
        _is_first = ".0." in getattr(layer, "layer_name", "")
        if _sync == "1" or (_sync == "pre") or (_sync == "first" and _is_first):
            torch.cuda.synchronize()
        if _resident:
            # 常驻层:整层 MoE 在 GPU 上算(权重已在 GPU 且已转 K-major,无 H2D)。
            from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

            # ⚠️ 必须先建槽位:`_resident_I` 是在 `_ensure_resident` 里设置的,
            # 而 kwargs 从左到右求值 —— 写成 slot=... 在同一行会在 I= 之后被求值 ⇒ AttributeError。
            _slot = self._ensure_resident(layer)
            out = gpu_moe_layer(
                h_bf16, ids_i32, wts_f32,
                getattr(self, "_engine_w13", None) if getattr(self, "_engine_w13", None) is not None else layer.w13_weight,
                None,
                getattr(self, "_engine_w2", None) if getattr(self, "_engine_w2", None) is not None else layer.w2_weight,
                None,
                H=hidden_size, I=self._resident_I,
                K=int(self.moe_config.experts_per_token),
                device=h_bf16.device, slot=_slot,
            )
        elif _gpu_pf:
            # ---- 长 prefill:本层专家在 GPU 上算(逐层流式权重) ----------------
            # 权重来源优先用**引擎自有的紧凑分片**(NOTES §459):这些分片合起来正好
            # 是一份完整拷贝,所以 GPU 流式在 XIAOTU_RELEASE_SOURCE=1 下也能工作,
            # 主机专家内存 522 -> 253 GiB(实测保留源张量会让 EngineCore 在加载完成前
            # 就到 1073 GB、每 node 只剩 ~3 GB)。分片不存在(如 NOSHARD)时才退回源张量。
            from vllm_xiaotu_moe.gpu_prefill import (
                PrefetchSlot,
                gpu_moe_layer,
                kmajor_from_engine_shards,
            )

            # 需要的只有形状:E/H/I。源张量可能已被释放,所以优先用释放时记下的形状。
            _shp = getattr(self, "_released_shapes", {}).get("w2_weight")
            if _shp is None:
                _w2live = getattr(self, "_engine_w2", None)
                if _w2live is None:
                    _w2live = layer.w2_weight
                _shp = tuple(_w2live.shape)
            _E, _I = int(_shp[0]), 2 * int(_shp[2])
            _dev = h_bf16.device
            # 运行时预检:腾不出 staging 就**优雅放弃**(慢但能跑),并给用户选择。
            from vllm_xiaotu_moe.gpu_prefill import fits_device, staging_bytes

            try:
                _ns = int(engine.shard_geometry()["ns"]) or 2
            except Exception:  # noqa: BLE001
                _ns = 2
            _need = staging_bytes(_E, hidden_size, _I, int(self._group_k), _ns)
            # 判定**每个模块只做一次**:预检看的是瞬时空闲显存,逐次判定会让同一层
            # 在 GPU/CPU 之间来回跳(实测一次 forward 内 24 层走分片、16 层退回源张量,
            # 另一半 forward 全部 SKIPPED,NOTES §464),行为不可复现。
            _ok = getattr(self, "_gpu_pf_ok", None)
            if _ok is None:
                _ok, _free = fits_device(_need, _dev)
                self._gpu_pf_ok = bool(_ok)
                if not _ok:
                    print(
                        f"[vllm-xtu-moe] GPU prefill DISABLED for this layer -> staying "
                        f"on CPU (slower but correct).\n"
                        f"    per-layer staging ~{_need / 2**30:.1f} GiB; preflight wants "
                        f"~{_need * 1.25 / 2**30:.1f} GiB free VRAM (25% margin), "
                        f"only {_free / 2**30:.1f} GiB is free.\n"
                        f"    Options: --tensor-parallel-size 2 together with "
                        f"XIAOTU_MOE_RANK_SPLIT=0 (halves staging AND keeps the shard "
                        f"path), a larger-VRAM GPU, or a lower "
                        f"--gpu-memory-utilization.\n"
                        f"    VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=0 silences this.",
                        flush=True,
                    )
            if not _ok:
                _gpu_pf = False
            _km = None
            _pf_reason = "engine shards"
            if _gpu_pf:
                try:
                    import time as _time

                    _t_split = os.environ.get("XIAOTU_GP_SPLIT") == "1"
                    _t0 = _time.perf_counter() if _t_split else 0.0
                    _km = kmajor_from_engine_shards(
                        engine, _dev, hidden_size, _I, _E, int(self._group_k)
                    )
                    if _t_split:
                        torch.cuda.synchronize()
                        _asm = (_time.perf_counter() - _t0) * 1e3
                        _free, _tot = torch.cuda.mem_get_info(_dev)
                        print(f"[gp-split] layer={getattr(layer,'layer_name','?')} "
                              f"qlen={qlen} asm={_asm:.1f}ms free={_free/2**30:.2f}GiB "
                              f"alloc={torch.cuda.memory_allocated(_dev)/2**30:.2f}GiB "
                              f"reserved={torch.cuda.memory_reserved(_dev)/2**30:.2f}GiB",
                              flush=True)
                    if _km is None:
                        _pf_reason = "checkpoint source (engine has no shards)"
                except torch.OutOfMemoryError:
                    # 预检与真实分配之间可能被别的分配抢走 -> 同样优雅退回 CPU,
                    # 并且**粘住**这个否定结论,避免每次 forward 反复试错。
                    torch.cuda.empty_cache()
                    _gpu_pf = False
                    _km = None
                    if getattr(self, "_gpu_pf_ok", None) is not False:
                        self._gpu_pf_ok = False
                        print(
                            "[vllm-xtu-moe] GPU prefill OOM while staging a layer -> "
                            "falling back to CPU prefill for this layer and disabling "
                            "GPU prefill from here on (slower but correct). Consider "
                            "TP=2 (+ XIAOTU_MOE_RANK_SPLIT=0), a larger-VRAM GPU, or a "
                            "lower --gpu-memory-utilization.",
                            flush=True,
                        )
            if _km is not None:
                _slot = PrefetchSlot()
                _slot.bufs = _km
                _slot.ready = torch.cuda.Event()
                _slot.ready.record(torch.cuda.current_stream(_dev))
                _t1 = _time.perf_counter() if _t_split else 0.0
                out = gpu_moe_layer(
                    h_bf16, ids_i32, wts_f32, _km[0], _km[1], _km[2], _km[3],
                    H=hidden_size, I=_I,
                    K=int(self.moe_config.experts_per_token),
                    device=_dev, slot=_slot,
                )
                if _t_split:
                    torch.cuda.synchronize()
                    print(f"[gp-split] layer={getattr(layer,'layer_name','?')} "
                          f"kernels={(_time.perf_counter()-_t1)*1e3:.1f}ms", flush=True)
            elif _gpu_pf:
                # 退回源张量(需要源没被释放——`_gp_on` 已保证这一点)。
                self._prepare_weights(layer)
                w13h = getattr(self, "_engine_w13", None)
                if w13h is None:
                    w13h = layer.w13_weight
                w2h = getattr(self, "_engine_w2", None)
                if w2h is None:
                    w2h = layer.w2_weight
                s13h, s2h = self._scales
                if s13h is None:
                    s13h = self._find_scale(layer, (self._scale_attrs[0],))
                if s2h is None:
                    s2h = self._find_scale(layer, (self._scale_attrs[1],))
                for _nm, _t in (("w13", w13h), ("w2", w2h), ("s13", s13h), ("s2", s2h)):
                    if not isinstance(_t, torch.Tensor):
                        raise RuntimeError(
                            f"xiaotu gpu-prefill needs a host tensor for {_nm}; got "
                            f"{type(_t).__name__}"
                        )
                    _need = _t.numel() * _t.element_size()
                    _have = _t.untyped_storage().nbytes()
                    if _have < _need:
                        raise RuntimeError(
                            f"xiaotu gpu-prefill {_nm} has a released/undersized storage "
                            f"(need {_need} bytes, storage {_have} bytes, "
                            f"shape={tuple(_t.shape)})"
                        )
                out = gpu_moe_layer(
                    h_bf16, ids_i32, wts_f32, w13h, s13h, w2h, s2h,
                    H=hidden_size, I=int(w13h.shape[1]) // 2,
                    K=int(self.moe_config.experts_per_token),
                    device=_dev, slot=None,
                )
            if not getattr(self, "_gpu_pf_dbg", False):
                self._gpu_pf_dbg = True
                print(
                    f"[vllm-xtu-moe] GPU prefill ACTIVE: first {qlen} tokens >= "
                    f"threshold {_gp_min}; weights from {_pf_reason}",
                    flush=True,
                )
        if not _resident and not _gpu_pf:
            engine.cpu_decode(
                stream.cuda_stream, qlen, self.moe_config.experts_per_token,
                h_bf16.data_ptr(), ids_i32.data_ptr(), wts_f32.data_ptr(),
                out.data_ptr(),
            )
        if _sync == "post":
            torch.cuda.synchronize()
        if _HID_LAYER and _HID_LAYER in getattr(layer, "layer_name", ""):
            hs = h_bf16
            rows = [0] if hs.shape[0] < 5 else [0, 4]
            bits = hs.view(torch.uint16)  # bf16 原始位模式
            fp = " ".join(
                f"r{r}:[" + ",".join(f"{int(v):04x}" for v in bits[r, :4].tolist()) + "]"
                for r in rows
            )
            print(f"[hid] layer={getattr(layer, 'layer_name', '?')} M={hs.shape[0]} {fp}",
                  flush=True)
        if _DUMP_LAYER and _DUMP_LAYER in getattr(layer, "layer_name", "") \
                and not getattr(self, "_dumped", False):
            self._dumped = True
            self._dump_layer(layer)
        if _VERIFY_LAYER and getattr(self, "_verified_n", 0) < _VERIFY_MAX:
            # 跳过 profile/warmup 等路由 id 无效的调用(此时 topk_ids 为 -1 哨兵)
            if bool((ids_i32 >= 0).all()):
                self._verified_n = getattr(self, "_verified_n", 0) + 1
                self._verify_once(layer, h_bf16, ids_i32, wts_f32, out)
        result = out.to(hidden_states.dtype)
        if output is not None:
            # Modular upstream API: the caller owns the output buffer and uses it
            # directly, so the result must land in it (upstream ignores our return).
            output.copy_(result)
            return output
        return result


class XiaotuCPUExpertsBF16(_XiaotuExpertsMixin, CPUUnquantizedExperts):
    """无量化(BF16)专家:引擎 MOE_BF16。覆盖 Mixtral / Qwen2-MoE / Qwen3-MoE(bf16)等。"""

    _engine_attr = "MOE_BF16"
    _scale_attrs = ("", "")
    _group_n, _group_k = 1, 1
    _expect_dtype = torch.bfloat16


class XiaotuCPUExpertsMxfp4(_XiaotuExpertsMixin, CPUExpertsMxfp4):
    """MXFP4 W4A16 专家:权重在 CPU,计算用 xiaotu AVX512-VNNI 引擎(无需 AMX)。"""

    _engine_attr = "MOE_MXFP4"
    _scale_attrs = ("w13_weight_scale", "w2_weight_scale")
    _group_n, _group_k = 1, 32
    _expect_dtype = torch.uint8


class XiaotuCPUExpertsFp8(_XiaotuExpertsMixin, CPUExpertsFp8):
    """FP8 W8A16 块量化专家(GLM-5.x / DS-V4 等):引擎 MOE_FP8,块 128x128。"""

    _engine_attr = "MOE_FP8"
    _scale_attrs = ("w13_weight_scale_inv", "w2_weight_scale_inv")
    _group_n, _group_k = 128, 128
    _expect_dtype = torch.float8_e4m3fn
    _scale_dtype = torch.float32


class XiaotuCPUExpertsInt4(_XiaotuExpertsMixin, CPUExpertsInt4):
    """INT4 W4A16 组量化专家:引擎 MOE_WNA16。

    检查点布局与引擎布局不同,`_prepare_weights` 在**引擎构造时做一次重排**:

      GPTQ / compressed-tensors:`w13 [E, K/8, 2I] int32`(每 int32 沿 K 打包 8 个
      nibble,小端下 int32→u8 视图即为"低 nibble = 偶 k")⇒ `transpose(1,2).contiguous()
      .view(uint8)` 直接得到引擎的 `[E, 2I, K/2] u8`;`w2` 同理。缩放 `[E, G, N]`
      `transpose(1,2)` 得到引擎的 `[E, N, G]`(即 groupN=1、groupK=group_size)。

    零点:引擎的 int4 表是"中心 8"(`value = nibble - 8`),只等价于**对称**量化。
    GPTQ 检查点的 `qzeros` 存的是 `zp - 1`(主线 AMX 重排里 `+1` 还原),所以对称
    模型的 qzeros 全为 7 → zp=8;若出现非 8 的零点,`_require_symmetric` 会显式报错
    (逐组零点需要内核侧支持,见 docs/ROADMAP.md)。
    """

    _engine_attr = "MOE_WNA16"
    # Attribute names differ per quant method: auto_gptq / moe_wna16 use
    # w13_qweight / w13_scales / w13_qzeros, compressed-tensors uses
    # w13_weight_packed / w13_weight_scale / w13_weight_zero_point.
    _scale_attrs = (("w13_scales", "w13_weight_scale"),
                    ("w2_scales", "w2_weight_scale"))
    _w_names = (("w13_qweight", "w13_weight_packed", "w13_weight"),
                ("w2_qweight", "w2_weight_packed", "w2_weight"))
    _z_names = ("w13_qzeros", "w13_weight_zero_point",
                "w2_qzeros", "w2_weight_zero_point")
    _group_n, _group_k = 1, 128
    _scale_dtype = torch.float32

    def _prepare_weights(self, layer) -> None:
        w13 = self._find_scale(layer, self._w_names[0])
        w2 = self._find_scale(layer, self._w_names[1])
        if w13 is None or w2 is None:
            raise NotImplementedError(
                "xiaotu INT4 backend found no int4 weight tensors on the layer "
                f"(looked for {self._w_names[0]} / {self._w_names[1]})"
            )
        group = int(getattr(self.quant_config, "group_size", 0) or 0)
        if group <= 0:
            group = 128
        if bool(getattr(self.quant_config, "desc_act", False)):
            raise NotImplementedError(
                "xiaotu INT4 backend does not support GPTQ desc_act=True "
                "(g_idx reordering); the engine assumes sequential K order."
            )
        if w13.dtype == torch.int32:
            # GPTQ / compressed-tensors packing: nibbles along K, N last. On a
            # little-endian host the int32 -> uint8 view exposes two nibbles per
            # byte with the low nibble = lower K index, which is exactly the
            # engine's [E, 2I, K/2] layout.
            self._require_symmetric_zero_points(layer)
            w13 = torch.empty(w13.transpose(1, 2).shape, dtype=torch.int32).copy_(
                w13.transpose(1, 2)).view(torch.uint8)
            w2 = torch.empty(w2.transpose(1, 2).shape, dtype=torch.int32).copy_(
                w2.transpose(1, 2)).view(torch.uint8)
            s13, s2 = self._scales
            if s13 is not None and s2 is not None:
                # GPTQ scales are [E, K/group, N]; the engine wants [E, N, K/group].
                # Materialise into FRESH buffers with an explicit copy: chaining
                # .transpose().contiguous() can leave the result as a view whose
                # storage is owned elsewhere (observed as PROT_NONE pages once the
                # owner is released -> SIGSEGV inside the engine's weight copy).
                self._scales = (self._fresh_f32(s13.transpose(1, 2)),
                                self._fresh_f32(s2.transpose(1, 2)))
        elif w13.dtype != torch.uint8:
            raise NotImplementedError(
                f"xiaotu INT4 backend expects int32 (GPTQ/compressed-tensors) or "
                f"uint8 (engine layout) weights, got {w13.dtype}"
            )
        self._engine_w13, self._engine_w2 = w13, w2
        self._group_n, self._group_k = 1, group

    @staticmethod
    def _fresh_f32(t):
        out = torch.empty(t.shape, dtype=torch.float32)
        out.copy_(t)
        return out

    def _require_symmetric_zero_points(self, layer) -> None:
        """Reject asymmetric int4: the engine's table is centered at 8.

        GPTQ stores `zero_point - 1` (mainline's AMX repack adds 1 back), so a
        symmetric checkpoint shows nibbles of 7; compressed-tensors may store 8
        directly. Either way the effective zero point must be 8.
        """
        for name in self._z_names:
            qz = getattr(layer, name, None)
            if qz is None:
                continue
            b = qz.detach().to("cpu").contiguous()
            packed = b.dtype in (torch.int32, torch.int16) and b.shape[-1] * 8 == qz.shape[-1]
            if b.dtype != torch.uint8:
                b = b.to(torch.int32).view(torch.uint8)
            lo = (b & 0x0F).to(torch.int16)
            hi = (b >> 4).to(torch.int16)
            if packed:
                lo = lo + 1   # mainline's unpack adds 1 to GPTQ zeros
                hi = hi + 1
            vals = torch.unique(torch.cat((lo.reshape(-1), hi.reshape(-1))))
            vmin, vmax = int(vals.min()), int(vals.max())
            symmetric = bool(torch.all((vals == 8) | (vals == 7)))
            if not symmetric or vmin < 7:
                raise NotImplementedError(
                    f"xiaotu INT4 backend only implements symmetric quantization "
                    f"(effective zero point 8); {name} has zero points in "
                    f"[{vmin}, {vmax}]. Per-group zero points need kernel "
                    "support; see docs/ROADMAP.md (INT4 checkpoint adaptation)."
                )

    def _validate_weights(self, layer, ex_w13, ex_w2) -> None:
        # Engine layout: w13 [E, 2I, H/2] u8, w2 [E, H, I/2] u8 (nibbles along K).
        layout_ok = (
            ex_w13.dtype == torch.uint8
            and ex_w13.dim() == 3
            and ex_w2.dim() == 3
            and ex_w13.shape[2] * 2 == ex_w2.shape[1]
            and ex_w2.shape[2] * 2 == ex_w13.shape[1] // 2
        )
        if not layout_ok:
            raise NotImplementedError(
                "xiaotu INT4/WNA16 backend cannot consume this checkpoint: "
                f"w13 shape={tuple(ex_w13.shape)} dtype={ex_w13.dtype}, "
                f"w2 shape={tuple(ex_w2.shape)} dtype={ex_w2.dtype}. "
                "Expected the byte-packed [E, 2I, K/2] / [E, H, I/2] layout "
                "(GPTQ/compressed-tensors are repacked automatically; AWQ's "
                "N-packed layout is not supported yet). "
                "See docs/KNOWN_LIMITATIONS.md (INT4/WNA16)."
            )


# 格式 → 需要替换的主线 CPU 后端类
_BACKENDS = (
    ("CPUUnquantizedExperts", XiaotuCPUExpertsBF16, "BF16"),
    ("CPUExpertsMxfp4", XiaotuCPUExpertsMxfp4, "MXFP4"),
    ("CPUExpertsFp8", XiaotuCPUExpertsFp8, "FP8"),
    ("CPUExpertsInt4", XiaotuCPUExpertsInt4, "INT4"),
)


def register_mixed_cpu_backend() -> None:
    """混合模式下把主线的 CPU experts 后端换成 xiaotu 引擎实现。

    oracle 对这些后端是**惰性 import** 模块属性,所以替换 `cpu_moe` 里的类即可
    让 oracle 选中我们的实现(而不是仅 Intel AMX 的主线内核)。
    注:oracle 里加一个正式的后端注册表是更干净的上游扩展点;这里先做 OOT 等价物。
    """
    if not mixed_mode_enabled():
        return
    import vllm.model_executor.layers.fused_moe.experts.cpu_moe as cpu_moe

    patched = []
    for attr, cls, label in _BACKENDS:
        if getattr(cpu_moe, attr, None) is cls:
            continue
        if not hasattr(cpu_moe, attr):
            continue  # 该主线版本没有这个后端
        setattr(cpu_moe, attr, cls)
        patched.append(label)
    if patched:
        print(
            "[vllm-xtu-moe] GPU/CPU Mixed: CPU backends -> xiaotu engine "
            f"({', '.join(patched)}; AVX512, no AMX required)",
            flush=True,
        )


__all__ = [
    "mixed_mode_enabled",
    "register_mixed_cpu_backend",
    "XiaotuCPUExpertsBF16",
    "XiaotuCPUExpertsMxfp4",
    "XiaotuCPUExpertsFp8",
    "XiaotuCPUExpertsInt4",
    # backwards-compatible alias (older code imported XiaotuCPUExperts)
    "XiaotuCPUExpertsMxfp4",
]
XiaotuCPUExperts = XiaotuCPUExpertsMxfp4
