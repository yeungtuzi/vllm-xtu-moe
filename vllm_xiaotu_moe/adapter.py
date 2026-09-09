# vllm-xtu-moe: xiaotu 引擎挂进 vLLM 主线的 FusedMoEExperts 计算模块。
#
# 设计目标(详见 ref/fork_vs_mainline_plugin_decision.md §9 / README.md):
#   - 复用主线战育:serving/KV/scheduling/TP 全部用主线;
#   - 提供主线在本机【没有】的 CPU 量化内核:无 AMX 的 x86 上跑 MXFP4/FP8(AMX-only 之外)。
# 本文件是适配层占位/骨架:以主线 cpu_moe.py 的 CPUExpertsMxfp4/FP8 为蓝本,把计算内核
# 换成 xiaotu 引擎。尚未能运行(依赖 xiaotu csrc 被识别为 vLLM CPU 内核,见表单,见 DOC 文档)。
# 实现状态:WIP。

from __future__ import annotations

import torch

# vLLM 主线模块化 MoE 的契约基类与工具
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExperts,
    FusedMoEExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey, kMxfp4Static
from vllm.platforms import CpuArchEnum, current_platform

# 计算内核 = xiaotu 引擎(CPU AVX512-VNNI/BF16),not vLLM 的 fused_experts_cpu
try:
    import xiaotu_moe  # noqa: F401  (vllm-xtu-moe 项目内的引擎包)
    _has_xiaotu = True
except Exception:
    _has_xiaotu = False


class XiaotuMxfp4CPUExperts(FusedMoEExpertsMonolithic):
    """在无 AMX 的 x86 上,用 xiaotu 引擎计算 MXFP4 MoE 的模块(替代主线 CPUExpertsMxfp4)。

    与主线 CPUExpertsMxfp4 的唯一关键差异:_supports_current_device 不要求 AMX
    (本机 AMD EPYC 9654 无 AMX,主线因此在本机无 MXFP4 CPU 内核;本类补上)。
    """

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config, quant_config)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        # 关键放宽:只要求 x86 CPU,**不要求 AMX**(与主线 CPUExpertsMxfp4 仅此一处不同)。
        return (
            current_platform.is_cpu()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
        )
        # 可选再加: and （_has_xiaotu）

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) in [(kMxfp4Static, None)]

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method in [
            RoutingMethodType.Default,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_router_logits_dtype(router_logits_dtype: torch.dtype, routing_method: RoutingMethodType) -> bool:
        return True

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """按主线契约选 topk,然后用 xiaotu 引擎算该层 MoE。

        TODO(实现):把主线 apply 里的 `fused_experts_cpu(..., CPUQuantMethod.MXFP4, ..., is_vnni=True)`
        换成对 xiaotu 引擎的调用(挪出独立 compute 函数,见下方 _compute_via_xiaotu)。
        依赖:xiaotu csrc 需要以 vLLM CPU 内核/算子形式接入(见 docs/ 与 csrc/pybind),或本模块
        直接持有时 torch:: 调用 xiaotu 的 C++ API。
        """
        topk_weights, topk_ids = self._select_experts_xiaotu(
            hidden_states, router_logits,
            num_expert_group, topk_group, e_score_correction_bias, routed_scaling_factor,
        )
        return self._compute_via_xiaotu(
            hidden_states, topk_weights, topk_ids, w1, w2, apply_router_weight_on_input,
        )

    def _select_experts_xiaotu(self, hidden_states, router_logits, num_expert_group,
                               topk_group, e_score_correction_bias, routed_scaling_factor):
        # 复用主线 cpu_moe.py 的 select_experts(与主线 CPUExpertsMxfp4 一致)
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import select_experts
        renormalize = self.moe_config.routing_method in (
            RoutingMethodType.Renormalize, RoutingMethodType.RenormalizeNaive,
        )
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            e_score_correction_bias=e_score_correction_bias,
        )

    def _compute_via_xiaotu(self, hidden_states, topk_weights, topk_ids, w1, w2, apply_router_weight_on_input):
        # TODO(实现):对 xiaotu 引擎算该层 MoE。引擎 python API(已从 csrc/python_binding/binding.cpp 实证):
        #   engine = xiaotu_moe.MOEV2(cfg, w13_weight, w2_weight, w13_scale, w2_scale,
        #                             w13_global_scale, w2_global_scale)
        #     -- 构造时持 CPU 权重镜像(as_ptr(data_ptr));cfg=MOEConfigV2
        #     -- (注意:绑定模板名由 XIAOTU_MOE_MODULE_NAME 决定;loader 导出的类名见
        #        xiaotu_moe/__init__.py: MOEV2/MOE_BF16 等,转发到 xiaotu_moe.load())
        #   engine.cpu_decode(stream, qlen, top_k, hidden[DEVICE bf16], expert_ids[DEVICE int32],
        #                     weights[DEVICE fp32], out_gpu[DEVICE fp32])
        #     -- GPU tensor I/O + CPU host-func compute(D2H -> forward_many -> H2D)
        #   engine.cpu_prefill(qlen, top_k, expert_ids, weights, input, output)
        #     -- 纯 CPU I/O
        # 桥接(adapter 层要做的):主线的 apply() 拿到的是主线已按 tp_rank 分片的权重(w1/w2,
        # 已是主线布局);要用 xiaotu 引擎,需(a)把主线权重喂给 engine 构造,或(b)让引擎
        # 直接读主线持有的 CPU 权重镜像。两者互排:主线 weights 在 vLLM 的 per-rank 张量,
        # 引擎要自己持 CPU mirror。=> 二选一(见下方 TODO),联调时定。
        #
        # 候选实现路径(优先其在主线能跑):
        #   (a) CustomOp.register_oot 覆盖 modular_fused_moe,forward_oot() 里读主线给的权重、
        #       调 xiaotu cpu_decode/cpu_prefill;
        #   (b) 在量化方法层注入 experts_cls = XiaotuMxfp4CPUExperts(仿 ZenCPUExpertsInt8 先例)。
        raise NotImplementedError(
            "XiaotuMxfp4CPUExperts._compute_via_xiaotu 尚未连到 xiaotu 引擎;"
            "(实现细节见 binding.cpp 的 MOE.cpu_decode/cpu_prefill 签名)"
        )


# ---- 注册:让主线分发器在 CPU+x86 上选中本类 ----
# 分两条(见 ref/fork_vs_mainline_plugin_decision.md §3):
#  A. CustomOp 覆盖:"modular_fused_moe" 已被主线注册;外部包需用 vLLM 的插件注册机制覆盖。
#  B. 作为子类进注册表:需接入主线 FusedMoEExperts 的 class 发现/枚举清单。
# 两者都需要主线开放"如何发现额外 CPU expert 类"的挂点 —— 这正是下一步要在实测里确认的
# (主线 is_supported_config 遍历哪个类列表;外部类能否插进去)。
def _register():
    # TODO(实现):接入主线 FusedMoEExperts 注册/枚举机制;验证 CustomOp 覆盖是否足以让本类被选中。
    pass
