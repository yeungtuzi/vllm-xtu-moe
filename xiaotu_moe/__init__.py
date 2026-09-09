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
MOE_BF16 = _load.MOE_BF16
MOE_FP16 = _load.MOE_FP16
MOE_BF16_FP16 = _load.MOE_BF16_FP16
MOE_FP8 = _load.MOE_FP8
MOE_FP8_FP16 = _load.MOE_FP8_FP16
MOE_MXFP4 = _load.MOE_MXFP4
MOE_MXFP4_FP16 = _load.MOE_MXFP4_FP16
MOE_WNA16 = _load.MOE_WNA16
MOE_WNA16_FP16 = _load.MOE_WNA16_FP16
MOE_NVFP4 = _load.MOE_NVFP4
MOE_NVFP4_FP16 = _load.MOE_NVFP4_FP16
__version__ = "0.1.0"
__variant__ = loader._chosen
