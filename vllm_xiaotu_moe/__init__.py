"""vllm-xtu-moe: vLLM mainline MoE plugin (compute core = xiaotu-moe engine).

把 xiaotu-moe 引擎(CPU AVX512-VNNI/BF16 fp4 MXFP4)作为主线 vLLM 的 MoE 计算模块/模型覆盖挂进来,
实现无 AMX x86(AMD EPYC)上的混合推理。用法见 README / docs/hybrid_adapter_design.md。
"""
from . import adapter  # noqa: F401
from . import hybrid_model  # noqa: F401
from . import mixed_experts  # noqa: F401

# GPU/CPU Mixed 模式(VLLM_EXPERTS_LOAD_DEVICE=cpu)下,把 MXFP4 的 CPU 计算后端注册为
# xiaotu 引擎(无 AMX)。非混合模式时是 no-op。
mixed_experts.register_mixed_cpu_backend()
