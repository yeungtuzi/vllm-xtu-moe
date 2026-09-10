# SPDX-License-Identifier: Apache-2.0
"""把 Qwen3.8-Flash-Next 的 PLE n-gram 嵌入表放到主机内存(`XIAOTU_PLE_CPU=1`)。

背景:该模型的 PLE 表约 **51.2 GB**(20M n-gram × 2560 维 fp8),而单张 A100-40GB
装不下;但它每次前向只做一次 `F.embedding` 查表(每 token 读几行),放在主机内存里
用 **UVA**(Unified Virtual Addressing)访问的代价可以接受 —— 省下的是**显存**,
换来的是每 token 几 KB 的 PCIe 读。

直接依赖 vLLM 的 `--cpu-offload-params` 不够:UVA offloader 是在**模块构造之后**把
参数搬到 CPU 的,而构造期 `create_weights` 已经先在显存里分配了 47.7 GB → 单卡必 OOM
(实测 `torch.OutOfMemoryError: Tried to allocate 47.69 GiB`)。所以这里做两件事:

1. `create_weights` 在 `torch.device("cpu")` 下执行 ⇒ 参数直接建在内存里(与 MoE
   专家权重走的是同一条思路,见 `mainline_shims.py` 的 `create_weights` 包装);
2. 权重加载完成后(`process_weights_after_loading`)把参数 pin 到锁页内存并换成
   **UVA 视图**(`p.device` 仍是 cuda,gather 内核直接读主机内存)。

环境变量:
  `XIAOTU_PLE_CPU=1`        打开(默认关闭,行为与上游一致)
  `XIAOTU_PLE_NO_PIN=1`     不 pin(调试用;UVA 需要锁页内存,一般不要设)
"""

from __future__ import annotations

import functools
import os

import torch


def ple_cpu_enabled() -> bool:
    return os.environ.get("XIAOTU_PLE_CPU", "0") == "1"


def _to_uva(layer: torch.nn.Module) -> None:
    """把该层 CPU 上的大参数换成锁页内存的 UVA 视图(显存里不占空间)。"""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    targets = []
    for mod_name, mod in [("", layer)] + list(layer.named_modules()):
        for name, p in mod.named_parameters(recurse=False):
            targets.append((f"{mod_name}.{name}".lstrip("."), p))
    for name, p in targets:
        if p is None or not isinstance(p, torch.Tensor):
            continue
        if p.device.type != "cpu":
            continue          # 已经在显存里(没走 offload) → 保持原样
        nbytes = p.numel() * p.element_size()
        if nbytes < (1 << 30):
            # 小参数(FP8 的 weight_scale 等)必须回到输出设备:主线的 FP8 PLE
            # 会检查 "FP8 PLE embedding scale must be on the output device"。
            if p.device.type == "cpu" and torch.cuda.is_available():
                p.data = p.data.to(torch.device("cuda", torch.cuda.current_device()))
            continue
        data = p.data
        if not data.is_pinned() and os.environ.get("XIAOTU_PLE_NO_PIN", "0") != "1":
            data = data.pin_memory()
        view = get_accelerator_view_from_cpu_tensor(data)
        p.data = view
        print(
            f"[vllm-xtu-moe/ple] {name} {tuple(p.shape)} {p.dtype} "
            f"{nbytes / 2**30:.1f} GiB → CPU 锁页 + UVA 视图(显存不占)",
            flush=True,
        )


def install() -> list[str]:
    """包装 PLE 嵌入的量化方法;返回已安装的钩子名(便于日志/自检)。"""
    if not ple_cpu_enabled():
        return []
    try:
        from vllm.models.qwen4_exp.nvidia.ple_layer import (
            Qwen4ExpPLEFp8EmbeddingMethod,
        )
    except Exception:          # noqa: BLE001 —— 非 Qwen4Exp 版本没有这个模块
        return []

    applied: list[str] = []
    for cls in (Qwen4ExpPLEFp8EmbeddingMethod,):
        cw = cls.__dict__.get("create_weights")
        if cw is not None and not getattr(cw, "_xtu_ple", False):
            @functools.wraps(cw)
            def create_weights(self, layer, *a, **kw):
                with torch.device("cpu"):
                    return cw(self, layer, *a, **kw)

            create_weights._xtu_ple = True  # type: ignore[attr-defined]
            cls.create_weights = create_weights
            applied.append(f"{cls.__name__}.create_weights")

    # 关键:UVA 转换必须发生在 **process_weights_after_loading 之前**。
    # vLLM 的 `device_loading_context()` 会在调用该钩子前把所有 CPU 参数搬到
    # 目标设备("量化方法期望参数在目标设备上")——51 GB 的表会在这一步 OOM。
    # 只要参数已经变成 UVA 视图(p.device 是 cuda、物理内存在主机),该上下文就
    # 不会搬它,显存也不占。因此挂在 `Qwen4ExpNGramEmbedding.load_weights` 后面。
    try:
        from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
    except Exception:          # noqa: BLE001
        return applied
    lw = Qwen4ExpNGramEmbedding.__dict__.get("load_weights")
    if lw is not None and not getattr(lw, "_xtu_ple", False):
        @functools.wraps(lw)
        def load_weights(self, weights, *a, **kw):
            res = lw(self, weights, *a, **kw)
            _to_uva(self)
            return res

        load_weights._xtu_ple = True  # type: ignore[attr-defined]
        Qwen4ExpNGramEmbedding.load_weights = load_weights
        applied.append("Qwen4ExpNGramEmbedding.load_weights")
    return applied
