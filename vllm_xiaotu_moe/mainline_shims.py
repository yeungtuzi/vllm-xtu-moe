"""让**原生(未打补丁)**的 vLLM 主线也能跑混合模式:4 处薄壳 monkey-patch。

背景:混合模式需要主线的 4 处配合(见 `docs/ARCHITECTURE.md` §5):
  1. routed-expert 权重必须建在 CPU(否则 100+ GB 专家在构造时就 OOM);
  2. oracle 必须优先选 CPU 后端(GPU 平台默认不选);
  3. CPU 后端的 AMX 重打包必须跳过(它会破坏原始权重布局,且依赖可能未编译的
     `torch.ops._C.convert_weight_packed`);
  4. 量化方法的 `process_weights_after_loading` 必须通知 experts 后端(fp8/wna16 主线不调)。

这 4 处目前尚未合并进上游。为了让用户
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
def _notify_experts(method, layer) -> None:
    kernel = getattr(method, "moe_kernel", None)
    experts = getattr(kernel, "fused_experts", None)
    fn = getattr(experts, "process_weights_after_loading", None)
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
            res = pwal(self, layer, *a, **kw)
            if mixed_mode_enabled():
                _notify_experts(self, layer)
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
