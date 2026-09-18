# xiaotu-moe: open-source reimplementation of the (closed) lk-moe CPU MoE engine.
#
# On import, the ISA-specific native extension is selected at runtime by
# `loader` (best supported SIMD for the host CPU) and exposed transparently.
#
# License: Apache-2.0

from . import loader  # noqa: F401

# Re-export the loaded module's public API so `import xiaotu_moe` works like the
# native module. Attribute access not present here is forwarded by loader.
from .loader import choose_variant, load  # noqa: F401

# Import eagerly so `import xiaotu_moe as m; m.MOE_BF16` works immediately and
# the chosen module (and its __version__) is pinned at import time.
_load = loader.load()
MOEConfigV2 = _load.MOEConfigV2

# 【lk 编排链接入】给引擎类包一层 `gpu_prefill`:lk 的 `routed_experts._gpu_prefill`
# 直接调用 `self.lk_moe.gpu_prefill(x_ptr, out_ptr, ids_ptr, wts_ptr, qlen, k, stream)`,
# 大 prefill 的"权重流式 H2D + GPU 计算"由引擎负责(混合模式下权重在 CPU,
# 走不了 vLLM 标准 GPU MoE —— 见 gpu_prefill_bridge 顶部注释与 NOTES §282)。
# 包装只**新增**这一个方法,其余一切经 __getattr__ 原样转发,ABI 不变。
from .gpu_prefill_bridge import wrap_engine_class as _wrap  # noqa: E402

MOE_BF16 = _wrap(_load.MOE_BF16, "BF16")
MOE_FP16 = _wrap(_load.MOE_FP16, "FP16")
MOE_BF16_FP16 = _wrap(_load.MOE_BF16_FP16, "BF16")
MOE_FP8 = _wrap(_load.MOE_FP8, "FP8")
MOE_FP8_FP16 = _wrap(_load.MOE_FP8_FP16, "FP8")
MOE_MXFP4 = _wrap(_load.MOE_MXFP4, "MXFP4")
MOE_MXFP4_FP16 = _wrap(_load.MOE_MXFP4_FP16, "MXFP4")
MOE_WNA16 = _wrap(_load.MOE_WNA16, "WNA16")
MOE_WNA16_FP16 = _wrap(_load.MOE_WNA16_FP16, "WNA16")
MOE_NVFP4 = _wrap(_load.MOE_NVFP4, "NVFP4")
MOE_NVFP4_FP16 = _wrap(_load.MOE_NVFP4_FP16, "NVFP4")
__version__ = "0.2.1"
__variant__ = loader._chosen
