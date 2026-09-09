"""Oracle 探测:不加载权重,直接问主线"这个 MoE 配置会选哪个后端"。

用途:在花 1 小时加载 306 GB 之前,先确认某个模型(config 决定的量化/路由形态)
在混合模式下确实被分到 xiaotu CPU 后端。

用法:
  VLLM_EXPERTS_LOAD_DEVICE=cpu python scripts/probe_oracle.py [fp8|mxfp4|wna16]
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vllm_xiaotu_moe  # noqa: F401  (registers the mixed CPU backends)

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kInt4Static,
    kInt8DynamicTokenSym,
)

CASES = {
    "bf16": (
        "无量化 BF16 (Mixtral/Qwen2-MoE/Qwen3-MoE bf16)",
        4096, 14336, 8, 2, RoutingMethodType.Renormalize,
        None, None, None,
    ),
    # (label, hidden, inter, experts, topk, routing, weight_key, act_key, swiglu_limit)
    "fp8-glm53": (
        "GLM-5.3-Flash (fp8 e4m3 128x128, sigmoid+noaux_tc)",
        4096, 2048, 288, 8, RoutingMethodType.DeepSeekV3,
        kFp8Static128BlockSym, kFp8Dynamic128Sym, 10.0,
    ),
    "fp8-dsv4": (
        "DS-V4 (fp8, sqrtsoftplus routing)",
        4096, 2048, 256, 6, RoutingMethodType.DeepseekV4,
        kFp8Static128BlockSym, kFp8Dynamic128Sym, 10.0,
    ),
    "mxfp4-dsv4": (
        "DS-V4 (mxfp4, sqrtsoftplus routing)",
        4096, 2048, 256, 6, RoutingMethodType.DeepseekV4,
        kMxfp4Static, None, 10.0,
    ),
    "wna16-int4": (
        "WNA16 int4 group 128",
        4096, 2048, 256, 6, RoutingMethodType.Renormalize,
        kInt4Static, None, None,
    ),
    "int8": (
        "INT8 per-channel (unsupported by xiaotu engine)",
        4096, 2048, 256, 6, RoutingMethodType.Renormalize,
        kInt8DynamicTokenSym, kInt8DynamicTokenSym, None,
    ),
}


def make_config(hidden, inter, experts, topk, routing, swiglu_limit):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    return FusedMoEConfig(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=inter,
        num_local_experts=experts,
        num_logical_experts=experts,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=routing,
        moe_backend="auto",
        swiglu_limit=swiglu_limit,
    )


def check_activation_guard() -> int:
    """T04:引擎只做 packed 布局的 gated 激活;SWIGLUOAI(交错)必须被拒绝。"""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import cpu_moe

    expect = {
        MoEActivation.SILU: True,
        MoEActivation.SWIGLUOAI_UNINTERLEAVE: True,
        MoEActivation.SWIGLUOAI: False,   # gpt-oss: w13 内 gate/up 交错
        MoEActivation.GELU: False,
        MoEActivation.GELU_TANH: False,
        MoEActivation.RELU2: False,
    }
    rc = 0
    for cls in (
        cpu_moe.CPUUnquantizedExperts,
        cpu_moe.CPUExpertsMxfp4,
        cpu_moe.CPUExpertsFp8,
        cpu_moe.CPUExpertsInt4,
    ):
        bad = [
            a.name
            for a, want in expect.items()
            if bool(cls._supports_activation(a)) != want
        ]
        ok = not bad
        rc |= 0 if ok else 1
        print(
            f"[{'OK ' if ok else 'BAD'}] activation guard {cls.__name__:26s} "
            f"{'packed-only (SILU + SWIGLUOAI_UNINTERLEAVE)' if ok else 'mismatch: ' + ','.join(bad)}",
            flush=True,
        )
    return rc


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.experts import cpu_moe
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import select_fp8_moe_backend
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        select_mxfp4_moe_backend,
    )
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        select_unquantized_moe_backend,
    )

    rc = check_activation_guard()
    for key, (
        label, hidden, inter, experts, topk, routing, wkey, akey, swiglu,
    ) in CASES.items():
        if only and only not in key:
            continue
        cfg = make_config(hidden, inter, experts, topk, routing, swiglu)
        try:
            if "fp8" in key:
                backend, cls = select_fp8_moe_backend(cfg, wkey, akey)
            elif "mxfp4" in key:
                backend, cls = select_mxfp4_moe_backend(cfg, akey)
            elif key == "bf16":
                backend, cls = select_unquantized_moe_backend(cfg)
            else:
                # WNA16 的 oracle 需要 quant_config 对象,这里直接问 CPU 后端
                # 自己的能力检查(等价于 oracle 里的那一步)。
                base = (
                    cpu_moe.CPUExpertsInt4
                    if "int4" in key
                    else cpu_moe.CPUExpertsInt8
                )
                ok, reason = base.is_supported_config(
                    base, cfg, wkey, akey, mk.FusedMoEActivationFormat.Standard
                )
                print(
                    f"[{'OK ' if ok else 'NOT'}] {key:12s} -> "
                    f"is_supported_config={ok} reason={reason}  ({label})",
                    flush=True,
                )
                continue
            name = cls.__module__.split(".")[-1] + "." + cls.__name__
            is_xiaotu = cls.__name__.startswith("Xiaotu")
            print(
                f"[{'OK ' if is_xiaotu else 'NOT'}] {key:12s} -> "
                f"{backend.value:14s} {name}  ({label})",
                flush=True,
            )
            if key in ("fp8-glm53",) and not is_xiaotu:
                rc = 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] {key:12s} -> {type(exc).__name__}: {exc}", flush=True)
            if key in ("fp8-glm53",):
                rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
