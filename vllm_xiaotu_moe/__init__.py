"""vllm-xtu-moe: vLLM 主线混合推理插件(CPU 专家 + GPU 注意力/长 prefill)。

本包是 vLLM 主线的 OOT 插件,通过官方 `vllm.general_plugins` 入口加载。
内置的 xiaotu CPU 引擎(AVX-512 VNNI/BF16;MXFP4/FP8/INT4/BF16)作为计算内核,
由 `mixed_experts` 注册到主线的 CPU experts 后端槽位;`mainline_shims` 提供
混合模式所需的主线配合(上游尚未合并)。用法见 README 与 docs/RUNBOOK.md。
"""
from . import hybrid_model  # noqa: F401
from . import mixed_experts  # noqa: F401
from . import mainline_shims  # noqa: F401

# 混合模式(VLLM_EXPERTS_LOAD_DEVICE=cpu)下,把主线的 CPU experts 后端注册为
# xiaotu 引擎(无 AMX 依赖)。非混合模式时是 no-op。
mixed_experts.register_mixed_cpu_backend()

# 在原生 vLLM 上补齐混合模式所需的 4 处主线行为:
# 专家权重建在 CPU / oracle 优先选 CPU 后端 / 跳过 AMX 重打包 / 通知 experts 后端。
# 已合并相应改动的主线上重复应用等于 no-op(XIAOTU_MAINLINE_SHIMS=0 可关闭)。
mainline_shims.apply_mainline_shims()
