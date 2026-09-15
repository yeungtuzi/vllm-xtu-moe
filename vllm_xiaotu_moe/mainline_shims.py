"""让**原生(未打补丁)**的 vLLM 主线也能跑混合模式:4 处薄壳 monkey-patch。

背景:混合模式需要主线的 4 处配合(见 `docs/ARCHITECTURE.md` §5):
  1. routed-expert 权重必须建在 CPU(否则 100+ GB 专家在构造时就 OOM);
  2. oracle 必须优先选 CPU 后端(GPU 平台默认不选);
  3. CPU 后端的 AMX 重打包必须跳过(它会破坏原始权重布局,且依赖可能未编译的
     `torch.ops._C.convert_weight_packed`);
  4. 量化方法的 `process_weights_after_loading` 必须通知 experts 后端(fp8/wna16 主线不调);
  5. `oracle/mxfp4.convert_weight_to_mxfp4_moe_kernel_format` 必须为 CPU 后端原样返回
     原始权重(否则 `Mxfp4MoEMethod._setup_kernel` 直接
     `raise ValueError("Unsupported mxfp4_backend ... CPU")`)。
     第 5 处对应上游 PR #56118 的第三段;缺了它,**主线 RoutedExperts 路径根本起不来**
     (实测 `lkpath1`:权重加载完后在 `process_weights_after_loading` 处失败)。

这 4 处目前尚未合并进上游。为了让使用
**只装插件就能用**,这里用 monkey-patch 提供等价行为:

  shim 1  给每个 `FusedMoEMethodBase` 子类的 `create_weights` 套一层
          `with torch.device("cpu")`;对**之后才导入**的量化类用
          `__init_subclass__` 钩子自动补上(量化模块是惰性导入的)。
  shim 2  包住 `oracle/{fp8,mxfp4,int_wna16,unquantized}.py` 的
          `_get_priority_backends`,把 CPU 后端移到队首。
  shim 3  包住 `cpu_moe.prepare_{fp8,mxfp4,int4}_moe_layer_for_cpu`,混合模式下
          原样返回(不做 AMX 重打包)。
  shim 4  包住 `process_weights_after_loading`,调用 experts 后端的同名钩子
          (若源码里已经调过则跳过,避免重复)。

全部是**幂等**的:在已合并相应改动的 vLLM 上重复应用等于 no-op。
关闭:`XIAOTU_MAINLINE_SHIMS=0`。自检:`python -m vllm_xiaotu_moe.mainline_shims`。
"""
from __future__ import annotations

import functools
import importlib
import inspect
import os

import torch

_SHIM_FLAG = "XIAOTU_MAINLINE_SHIMS"


def mixed_mode_enabled() -> bool:
    """True when experts are configured to live/compute on CPU on a GPU run.

    Reads the env var directly (a stock vLLM has no `envs.VLLM_EXPERTS_LOAD_DEVICE`).
    """
    try:
        from vllm import envs

        if getattr(envs, "VLLM_EXPERTS_LOAD_DEVICE", None) == "cpu":
            return True
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("VLLM_EXPERTS_LOAD_DEVICE", "gpu").lower() == "cpu"


def _log(msg: str) -> None:
    print(f"[vllm-xtu-moe/shims] {msg}", flush=True)


# ---------------------------------------------------------------------------
# shim 1 + 4: quant method classes
# ---------------------------------------------------------------------------
_PWAL_DIAG = {"n": 0, "src": 0}


def _host_mem_mib() -> dict:
    """Read the disjoint host-memory counters from /proc/self/status."""
    out: dict = {}
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                key, _, rest = ln.partition(":")
                if key in ("RssAnon", "RssFile", "RssShmem", "VmRSS"):
                    out[key] = int(rest.split()[0]) >> 10  # MiB
    except Exception:  # noqa: BLE001
        pass
    return out


def _mem_diag_on() -> bool:
    """内存诊断探针是否开启(默认否)。

    这些探针会给**每个模型的每一层**加一次 `/proc/self/status` 读取与若干行打印。
    V4/V4.1 的默认路径必须零额外开销,所以按 env 门控,只有
    `XIAOTU_MEM_DIAG=1` 时才打开(排查内存时用)。
    """
    return os.environ.get("XIAOTU_MEM_DIAG") == "1"


def _release_on() -> bool:
    """释放开关是否打开(与 mixed_experts._release_source_enabled 同一判据)。

    ⚠️ 这里必须门控:下面的"钩子前快照"会**持有**权重张量的引用。若对所有模型
    (含 V4)都抓,就会把它们本可释放的存储多留一份 —— 那正是
    "适配新模型不能破坏已有模型"的红线。
    """
    try:
        from vllm_xiaotu_moe.mixed_experts import _release_source_enabled

        return _release_source_enabled()
    except Exception:  # noqa: BLE001
        return False


def _notify_experts(method, layer) -> None:
    kernel = getattr(method, "moe_kernel", None)
    experts = getattr(kernel, "fused_experts", None)
    fn = getattr(experts, "process_weights_after_loading", None)

    # Observability for the "shard one layer, release one layer" path: without
    # this, XIAOTU_RELEASE_SOURCE produces no output at all and there is no way
    # to tell whether the hook ran, which backend class is in use, or whether
    # the host expert tensors are anonymous or shmem-backed.
    _PWAL_DIAG["n"] += 1
    nbytes = 0
    for nm in ("w13_weight", "w2_weight"):
        t = getattr(layer, nm, None)
        if isinstance(t, torch.Tensor):
            nbytes += t.numel() * t.element_size()
    _PWAL_DIAG["src"] += nbytes
    if _mem_diag_on() and (_PWAL_DIAG["n"] == 1 or _PWAL_DIAG["n"] % 8 == 0):
        m = _host_mem_mib()
        print(
            f"[xtu-diag] pwal#{_PWAL_DIAG['n']} method={type(method).__name__} "
            f"experts={type(experts).__name__} hook={callable(fn)} "
            f"layer_src={nbytes / 2**30:.2f}GiB cum_src={_PWAL_DIAG['src'] / 2**30:.1f}GiB "
            f"RssAnon={m.get('RssAnon', -1)}MiB RssFile={m.get('RssFile', -1)}MiB "
            f"RssShmem={m.get('RssShmem', -1)}MiB",
            flush=True,
        )

    if callable(fn):
        fn(layer)


def _already_notifies(func) -> bool:
    try:
        return "process_weights_after_loading(layer)" in inspect.getsource(func)
    except Exception:  # noqa: BLE001
        return False


def _patch_quant_method_cls(cls) -> list[str]:
    out: list[str] = []
    cw = cls.__dict__.get("create_weights")
    if cw is not None and not getattr(cw, "_xtu_shim", False):
        @functools.wraps(cw)
        def create_weights(self, layer, *a, **kw):
            if mixed_mode_enabled():
                with torch.device("cpu"):
                    return cw(self, layer, *a, **kw)
            return cw(self, layer, *a, **kw)

        create_weights._xtu_shim = True  # type: ignore[attr-defined]
        cls.create_weights = create_weights
        out.append(f"{cls.__name__}.create_weights")

    pwal = cls.__dict__.get("process_weights_after_loading")
    if (
        pwal is not None
        and not getattr(pwal, "_xtu_shim", False)
        and not _already_notifies(pwal)
    ):
        @functools.wraps(pwal)
        def process_weights_after_loading(self, layer, *a, **kw):
            # Split the per-layer memory growth between UPSTREAM's hook
            # (`_setup_kernel` -> make_mxfp4_moe_kernel -> experts/kernel ctor)
            # and OUR hook (Mixin.process_weights_after_loading -> _ensure_engine),
            # so the pinned/shmem growth is attributed to a concrete site.
            b = _host_mem_mib() if _mem_diag_on() else {}
            # 记录 upstream 钩子**前后**的权重指针。若指针变了(或被释放),就证明
            # "pwal 换掉了/搬走了原始存储",这正好解释 cellC 为什么在 pwal 返回后
            # 立刻建引擎会 SIGSEGV,而惰性路径(看到的是稳定后的张量)却没事。
            _pre = {}
            for _nm in (("w13_weight", "w2_weight") if _release_on() else ()):
                _t = getattr(layer, _nm, None)
                if isinstance(_t, torch.Tensor):
                    _pre[_nm] = (_t.data_ptr(), tuple(_t.shape), _t.storage_offset())
            if _pre:
                # 【保活】若 upstream 在 pwal 里换掉/搬走/释放了原始存储,那么 pwal
                # 返回后 `layer.w13_weight` 可能指向一块临时内存(cellC 的 SIGSEGV
                # 正是如此)。把**钩子前**的张量引用存下来:既让源保持可读,
                # 也给 `_ensure_engine` 一个安全的构建来源。
                layer._xiaotu_pre_pwal_src = {
                    _nm: getattr(layer, _nm) for _nm in _pre
                }
            res = pwal(self, layer, *a, **kw)
            m = _host_mem_mib() if _mem_diag_on() else {}
            if _pre and _PWAL_DIAG.get("n", 0) <= 2:
                for _nm, (_p0, _s0, _o0) in _pre.items():
                    _t = getattr(layer, _nm, None)
                    if not isinstance(_t, torch.Tensor):
                        print(f"[xtu-diag-ptr] {_nm}: {_p0:#x} {_s0} off={_o0} "
                              f"-> GONE ({type(_t).__name__})", flush=True)
                        continue
                    _p1, _s1, _o1 = _t.data_ptr(), tuple(_t.shape), _t.storage_offset()
                    if (_p0, _s0, _o0) != (_p1, _s1, _o1):
                        print(f"[xtu-diag-ptr] {_nm}: {_p0:#x} {_s0} off={_o0} "
                              f"-> {_p1:#x} {_s1} off={_o1}  ** REPLACED by pwal **",
                              flush=True)
                    else:
                        print(f"[xtu-diag-ptr] {_nm}: unchanged {_p1:#x} {_s1} off={_o1}",
                              flush=True)
            if mixed_mode_enabled():
                _notify_experts(self, layer)
            a2 = _host_mem_mib() if _mem_diag_on() else {}
            _PWAL_DIAG["split"] = _PWAL_DIAG.get("split", 0) + 1
            if _mem_diag_on() and (
                _PWAL_DIAG["split"] == 1 or _PWAL_DIAG["split"] % 8 == 0
            ):
                print(
                    f"[xtu-diag-split] l#{_PWAL_DIAG['split']} "
                    f"upstream: dShmem={m.get('RssShmem', 0) - b.get('RssShmem', 0):+d}MiB "
                    f"dAnon={m.get('RssAnon', 0) - b.get('RssAnon', 0):+d}MiB | "
                    f"our_hook: dShmem={a2.get('RssShmem', 0) - m.get('RssShmem', 0):+d}MiB "
                    f"dAnon={a2.get('RssAnon', 0) - m.get('RssAnon', 0):+d}MiB",
                    flush=True,
                )
            return res

        process_weights_after_loading._xtu_shim = True  # type: ignore[attr-defined]
        cls.process_weights_after_loading = process_weights_after_loading
        out.append(f"{cls.__name__}.process_weights_after_loading")
    return out


def _walk_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_subclasses(sub)


def _install_quant_method_shims() -> list[str]:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    applied: list[str] = []
    for cls in _walk_subclasses(FusedMoEMethodBase):
        applied += _patch_quant_method_cls(cls)

    if not getattr(FusedMoEMethodBase, "_xtu_subclass_hook", False):
        def __init_subclass__(cls, **kwargs):  # noqa: N807
            try:
                object.__init_subclass__(**kwargs)
            except TypeError:
                pass
            _patch_quant_method_cls(cls)

        FusedMoEMethodBase.__init_subclass__ = classmethod(__init_subclass__)
        FusedMoEMethodBase._xtu_subclass_hook = True  # type: ignore[attr-defined]
        applied.append("FusedMoEMethodBase.__init_subclass__")
    return applied


# ---------------------------------------------------------------------------
# shim 2: oracle priority lists
# ---------------------------------------------------------------------------
_ORACLE_MODULES = (
    "vllm.model_executor.layers.fused_moe.oracle.fp8",
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.fused_moe.oracle.unquantized",
)


def _cpu_first(backends: list):
    if not backends:
        return backends
    try:
        enum_cls = type(backends[0])
        cpu = next((m for m in enum_cls if m.name == "CPU"), None)
    except Exception:  # noqa: BLE001
        return backends
    if cpu is None:
        return backends
    rest = [b for b in backends if b is not cpu]
    return [cpu] + rest


def _install_oracle_shims() -> list[str]:
    applied: list[str] = []
    for mod_name in _ORACLE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            _log(f"skip {mod_name}: {type(exc).__name__}: {exc}")
            continue
        orig = getattr(mod, "_get_priority_backends", None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue

        @functools.wraps(orig)
        def wrapper(*a, _orig=orig, **kw):
            backends = list(_orig(*a, **kw))
            return _cpu_first(backends) if mixed_mode_enabled() else backends

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(mod, "_get_priority_backends", wrapper)
        applied.append(f"{mod_name.rsplit('.', 1)[-1]}._get_priority_backends")
    return applied


# ---------------------------------------------------------------------------
# shim 3: skip the AMX prepack
# ---------------------------------------------------------------------------
_PREPACK_PASSTHROUGH = {
    # name: indices of the tensors that are passed through unchanged
    "prepare_fp8_moe_layer_for_cpu": ("w13", "w2"),
    "prepare_mxfp4_moe_layer_for_cpu": ("w13", "w2", "w13_scale", "w2_scale"),
    "prepare_int4_moe_layer_for_cpu": (
        "w13_packed", "w2_packed", "w13_scale", "w2_scale", "w13_zeros", "w2_zeros",
    ),
}


def _install_prepack_shims() -> list[str]:
    try:
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe
    except Exception as exc:  # noqa: BLE001
        _log(f"skip prepack shims: {type(exc).__name__}: {exc}")
        return []

    applied: list[str] = []
    for name, params in _PREPACK_PASSTHROUGH.items():
        orig = getattr(cpu_moe, name, None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue
        try:
            sig = inspect.signature(orig)
        except (TypeError, ValueError):
            continue

        @functools.wraps(orig)
        def wrapper(*a, _orig=orig, _sig=sig, _params=params, **kw):
            if not mixed_mode_enabled():
                return _orig(*a, **kw)
            bound = _sig.bind_partial(*a, **kw)
            bound.apply_defaults()
            return tuple(bound.arguments[p] for p in _params)

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(cpu_moe, name, wrapper)
        applied.append(f"cpu_moe.{name}")
    return applied


# ---------------------------------------------------------------------------
# shim 5: CPU 后端跳过 mxfp4 kernel-format 转换
#
# 等价于上游 PR #56118 的第三处改动。上游在 `oracle/mxfp4.py` 里让
# `select_mxfp4_moe_backend()` 直接返回 CPU 后端,并在
# `convert_weight_to_mxfp4_moe_kernel_format()` 里为 CPU 后端原样返回原始权重:
#   "the CPU backend is an out-of-tree engine that consumes the raw
#    [E, 2I, H//2] / [E, H, I//2] uint8 weights and raw e8m0 scales directly,
#    so skip the AMX prepack."
# 主线(本仓 pin 的 commit)还没有这段,于是 `Mxfp4MoEMethod._setup_kernel`
# 会走到 `raise ValueError("Unsupported mxfp4_backend ... Mxfp4MoeBackend.CPU")`。
# 这里补上同样的语义。
#
# 注意:`quantization/mxfp4.py` 是 `from ...oracle.mxfp4 import (...)` **按名字**导入的,
# 所以两个模块的绑定都要替换,只改 oracle 不影响调用点。
_SHIM5_MODULES = (
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.quantization.mxfp4",
)


def _install_mxfp4_cpu_convert_shim() -> list[str]:
    applied: list[str] = []
    for mod_name in _SHIM5_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            _log(f"skip {mod_name}: {type(exc).__name__}: {exc}")
            continue
        orig = getattr(mod, "convert_weight_to_mxfp4_moe_kernel_format", None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue

        @functools.wraps(orig)
        def wrapper(mxfp4_backend, layer, w13_weight, w2_weight,
                    w13_weight_scale, w2_weight_scale, w13_bias=None,
                    w2_bias=None, *a, _orig=orig, **kw):
            # CPU 后端 = OOT 引擎,直接吃 checkpoint 原始布局,不做任何重打包。
            if mixed_mode_enabled() and getattr(mxfp4_backend, "name", "") == "CPU":
                return (w13_weight, w2_weight, w13_weight_scale,
                        w2_weight_scale, w13_bias, w2_bias)
            return _orig(
                mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale,
                w2_weight_scale, w13_bias, w2_bias, *a, **kw,
            )

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(mod, "convert_weight_to_mxfp4_moe_kernel_format", wrapper)
        applied.append(f"{mod_name.rsplit('.', 1)[-1]}."
                       "convert_weight_to_mxfp4_moe_kernel_format")
    return applied


# ---------------------------------------------------------------------------
# shim 6: 把 monolithic 路径丢掉的 `input_ids` 存到 layer 上
#
# DeepSeek-V4 的前 3 层(`config.num_hash_layers=3`)是 **hash MoE**:
#   `DeepseekV4MoE.__init__`: `is_hash_moe = layer_idx < num_hash_layers`
#       ⇒ `gate.tid2eid` 是路由表,`gate.e_score_correction_bias = None`
#   `DeepseekV4MoE.forward` → `self.experts(x, router_logits=x, input_ids=input_ids)`
# 而 hash 路由必须查表:`fused_topk_bias(..., input_tokens=input_ids,
# hash_indices_table=...)`。链路是
#   RoutedExperts.forward_monolithic(x, router_logits, input_ids)
#     → quant_method.apply_monolithic(layer, x, router_logits, input_ids)   ← 收下了
#       → moe_kernel.apply_monolithic(...)                                  ← 没有 input_ids 这个参数
#         → fused_experts.apply(...)                                        ← 于是永远拿不到
# 即**两端都有、中间断了**。这里在断点上把 input_ids 暂存到 layer 上,
# 让 OOT 后端(我们的 experts)能取到,而不必改写整条调用链。
def _patch_apply_monolithic_input_ids(cls) -> list[str]:
    fn = cls.__dict__.get("apply_monolithic")
    if fn is None or getattr(fn, "_xtu_ids_shim", False):
        return []

    @functools.wraps(fn)
    def apply_monolithic(self, layer, x, router_logits, input_ids=None, *a, **kw):
        if input_ids is not None:
            try:
                layer._xiaotu_input_ids = input_ids
            except Exception:  # noqa: BLE001
                pass
        return fn(self, layer, x, router_logits, input_ids, *a, **kw)

    apply_monolithic._xtu_ids_shim = True  # type: ignore[attr-defined]
    cls.apply_monolithic = apply_monolithic
    return [f"{cls.__name__}.apply_monolithic"]


def _install_input_ids_shim() -> list[str]:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    applied = _patch_apply_monolithic_input_ids(FusedMoEMethodBase)
    for cls in _walk_subclasses(FusedMoEMethodBase):
        applied += _patch_apply_monolithic_input_ids(cls)
    return applied


# ---------------------------------------------------------------------------
# shim 7: 把路由 extras(hash 表 / vision bias)也放到 `RoutedExperts` 上
#
# `FusedMoEFactory` 收下 `hash_indices_table` / `bias_vl` / `image_sentinel_lo`,
# 但它们**只**被送进 `create_fused_moe_router()` —— 也就是跑在 runner 上的那个
# router(modular 路径用)。`RoutedExperts.__init__` 只保存 `e_score_correction_bias`,
# **不保存这三样**。
# ⇒ monolithic 后端(我们的 CPU experts)自己在 `_get_router()` 里建 router 时,
#   既拿不到 hash 表也拿不到 vision bias,于是 factory 兜底选 `FusedTopKRouter`
#   ⇒ DeepSeek-V4 的 sqrtsoftplus 直接 `ValueError`。
# 实测诊断行:`router=FusedTopKRouter … bias=no hash=no vl=no`(lkpath4)。
# 这里在 factory 出口把这些 extras 补挂到 `RoutedExperts` 实例上,让
# `mixed_experts._ROUTER_EXTRA_ATTRS` 能取到 ⇒ 选中上游的 `FusedTopKBiasRouter`。
_ROUTER_EXTRAS = ("hash_indices_table", "bias_vl", "image_sentinel_lo")


def _install_router_extras_shim() -> list[str]:
    import sys

    from vllm.model_executor.layers.fused_moe import layer as _layer

    orig = getattr(_layer, "FusedMoEFactory", None)
    if orig is None or getattr(orig, "_xtu_extras_shim", False):
        return []

    @functools.wraps(orig)
    def FusedMoEFactory(*a, **kw):
        mod = orig(*a, **kw)
        if mixed_mode_enabled():
            extras = {n: kw.get(n) for n in _ROUTER_EXTRAS if kw.get(n) is not None}
            if extras:
                try:
                    for m in mod.modules():
                        if type(m).__name__ == "RoutedExperts":
                            for n, v in extras.items():
                                if getattr(m, n, None) is None:
                                    setattr(m, n, v)
                            break
                except Exception:  # noqa: BLE001
                    pass
        return mod

    FusedMoEFactory._xtu_extras_shim = True  # type: ignore[attr-defined]

    # 定义处 + 所有**按名字导入过**它的已加载模块(DS-V4 的 model.py 就是这种)。
    patched = ["fused_moe.layer.FusedMoEFactory"]
    for mod_name, m in list(sys.modules.items()):
        if m is None or m is _layer:
            continue
        try:
            if getattr(m, "FusedMoEFactory", None) is orig:
                setattr(m, "FusedMoEFactory", FusedMoEFactory)
                patched.append(f"{mod_name}.FusedMoEFactory")
        except Exception:  # noqa: BLE001
            continue
    setattr(_layer, "FusedMoEFactory", FusedMoEFactory)
    return patched


# ---------------------------------------------------------------------------
def apply_mainline_shims() -> list[str]:
    """Idempotently install all shims; returns the list of things applied."""
    if not mixed_mode_enabled():
        return []
    if os.environ.get(_SHIM_FLAG, "1") == "0":
        _log(f"disabled via {_SHIM_FLAG}=0")
        return []

    applied: list[str] = []
    for step in (
        _install_quant_method_shims,
        _install_oracle_shims,
        _install_prepack_shims,
        _install_mxfp4_cpu_convert_shim,
        _install_input_ids_shim,
        _install_router_extras_shim,
    ):
        try:
            applied += step()
        except Exception as exc:  # noqa: BLE001
            _log(f"{step.__name__} failed: {type(exc).__name__}: {exc}")
    if applied:
        _log(f"mainline shims applied ({len(applied)}): " + ", ".join(applied))
    return applied


def self_check() -> int:
    """Print what the plugin will do on this machine/checkout."""
    print(f"mixed_mode_enabled = {mixed_mode_enabled()}")
    print(f"{_SHIM_FLAG}     = {os.environ.get(_SHIM_FLAG, '1')}")
    try:
        from vllm import envs

        print(f"vllm.envs.VLLM_EXPERTS_LOAD_DEVICE = "
              f"{getattr(envs, 'VLLM_EXPERTS_LOAD_DEVICE', '<absent: stock vLLM>')}")
    except Exception as exc:  # noqa: BLE001
        print(f"vllm.envs unavailable: {exc}")
    try:
        import xiaotu_moe

        print(f"xiaotu_moe variant = {getattr(xiaotu_moe, '__variant__', '?')}")
        print(f"xiaotu_moe version = {getattr(xiaotu_moe, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"xiaotu_moe unavailable: {exc}")
    applied = apply_mainline_shims()
    print(f"shims applied now  = {applied or '(none)'}")
    try:
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe

        for attr in ("CPUUnquantizedExperts", "CPUExpertsMxfp4",
                     "CPUExpertsFp8", "CPUExpertsInt4"):
            cls = getattr(cpu_moe, attr, None)
            print(f"  {attr:26s} -> {cls.__name__ if cls else '<absent>'}")
    except Exception as exc:  # noqa: BLE001
        print(f"cpu_moe unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
