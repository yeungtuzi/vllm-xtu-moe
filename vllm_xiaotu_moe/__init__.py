"""vllm-xtu-moe: vLLM 主线混合推理插件。

项目边界:本包属于 **vllm-xtu-moe**(vLLM 主线插件),不是 xiaotu-moe 项目。
内置的 xiaotu 引擎(CPU AVX512-VNNI/BF16, MXFP4/FP8/INT4/BF16)源码来自独立项目
`xiaotu-moe`(闭源 lk_moe 的开源重实现,现已转私有),作为**计算内核**被本插件使用。
本插件把该引擎挂到主线的 `FusedMoEFactory`(通用 CPU experts 后端)并实现混合推理。
用法与项目关系见 README / docs/BACKLOG.md。
"""
from . import adapter  # noqa: F401
from . import hybrid_model  # noqa: F401
from . import mixed_experts  # noqa: F401
from . import mainline_shims  # noqa: F401

# GPU/CPU Mixed 模式(VLLM_EXPERTS_LOAD_DEVICE=cpu)下,把 MXFP4 的 CPU 计算后端注册为
# xiaotu 引擎(无 AMX)。非混合模式时是 no-op。
mixed_experts.register_mixed_cpu_backend()

# 混合模式下,用 monkey-patch 让**原生 vLLM**(未打我们的主线补丁)也能工作:
# 专家权重建在 CPU / oracle 优先选 CPU 后端 / 跳过 AMX 重打包 / 通知 experts 后端。
# 已打过补丁的主线上重复应用等于 no-op(XIAOTU_MAINLINE_SHIMS=0 可关闭)。
mainline_shims.apply_mainline_shims()
