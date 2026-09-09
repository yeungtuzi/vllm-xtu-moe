"""GPU/CPU Mixed Mode 的 CPU 计算后端(XiaotuCPUExperts)。

背景:主线已引入 `VLLM_EXPERTS_LOAD_DEVICE=cpu`(见 vllm/envs.py + routed_experts.py),
让 routed-expert 权重从构造起就在 CPU(不 OOM)。但计算仍走 quant_method.apply(GPU 内核)
会设备不匹配。本模块提供 CPU 计算后端:
  - 复用主线 CPU experts 的契约(FusedMoEExpertsMonolithic)与 support 判定;
  - 覆盖 process_weights_after_loading(不按 AMX 重排,改为用原始 CPU 参数建 xiaotu 引擎);
  - 覆盖 apply(路由用主线 select_experts,专家计算走 xiaotu 引擎)。

通用性:xiaotu 引擎支持 MXFP4 / FP8 / NVFP4 / BF16 / WNA16(见 xiaotu_moe.__init__);
本文件先落地 MXFP4(已实证与 checkpoint 逐字节一致),FP8/NVFP4/BF16 按同一模式扩展。
在 AMD(无 AMX)上,这是唯一能跑的 CPU MoE 内核;主线 CPUExperts* 要 Intel AMX。
"""
from __future__ import annotations

import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUExpertsMxfp4,
    select_experts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


def mixed_mode_enabled() -> bool:
    """True when experts are configured to live/compute on CPU on a GPU run."""
    return envs.VLLM_EXPERTS_LOAD_DEVICE == "cpu"


class XiaotuCPUExperts(CPUExpertsMxfp4):
    """MXFP4 W4A16 专家:权重在 CPU,计算用 xiaotu AVX512-VNNI 引擎(无需 AMX)。

    继承主线 CPUExpertsMxfp4 的 support 判定(_supports_quant_scheme / activation /
    routing),只替换"设备门控 / 权重后处理 / 计算"三处。
    """

    # 引擎类名(MXFP4)。其它格式在子类里改这个。
    _engine_attr = "MOE_MXFP4"

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._xiaotu_engine = None
        self._layer_ref: torch.nn.Module | None = None

    @staticmethod
    def _supports_current_device() -> bool:
        # 关键放宽:不再要求 is_cpu()/AMX。GPU 平台 + 混合模式 + x86 即可。
        from vllm.platforms import CpuArchEnum, current_platform

        return (
            mixed_mode_enabled()
            and current_platform.get_cpu_architecture() == CpuArchEnum.X86
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """不按 AMX 重排权重;记住 layer,直接用原始 CPU 参数构造 xiaotu 引擎。"""
        # 主线 CPU 版会 cpu_prepack_moe_weight 重排;引擎需要原始 [E,2I,H//2] 布局,
        # 所以这里只记住 layer,引擎在首次 apply 时懒建(直接从 layer 读原始权重/scale)。
        # 注意: w1_scale/w2_scale 是基类只读 property(取自 quant_config),不可赋值。
        self._layer_ref = layer

    def _ensure_engine(self, layer: torch.nn.Module):
        if self._xiaotu_engine is not None:
            return self._xiaotu_engine
        import xiaotu_moe

        ex_w13 = layer.w13_weight
        ex_w2 = layer.w2_weight
        s13 = layer.w13_weight_scale
        s2 = layer.w2_weight_scale
        cfg = xiaotu_moe.MOEConfigV2()
        cfg.num_processes = 1
        cfg.process_id = 0
        cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True
        cfg.expert_num = int(ex_w13.shape[0])
        cfg.top_k = self.moe_config.experts_per_token
        cfg.hidden_size = int(ex_w2.shape[1])          # w2: [E, H, I//2]
        cfg.intermediate_size = int(ex_w13.shape[1] // 2)  # w13: [E, 2I, H//2]
        cfg.max_batch_size = 8192
        cfg.max_num_seqs = 256
        cfg.stride = 32
        cfg.group_min_len = 10
        cfg.group_max_len = 4096 + 128
        cfg.activation_type = 0
        cfg.use_gpu_prefill = False
        cfg.groupN = 1
        cfg.groupK = 32
        engine_cls = getattr(xiaotu_moe, self._engine_attr)
        self._xiaotu_engine = engine_cls(
            cfg,
            ex_w13.data_ptr(), ex_w2.data_ptr(),
            s13.data_ptr(), s2.data_ptr(),
            0, 0,
        )
        return self._xiaotu_engine

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
        # 路由复用主线(与 CPUExpertsMxfp4 一致),只是"算"换成 xiaotu 引擎。
        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=num_expert_group is not None,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.moe_config.routing_method.name.startswith("Renormalize"),
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func="softmax",
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
            e_score_correction_bias=e_score_correction_bias,
        )
        # layer 在 process_weights_after_loading 时记住;引擎用 layer 参数(在 CPU)。
        layer = self._layer_ref
        if layer is None:
            raise RuntimeError(
                "XiaotuCPUExperts.apply called before process_weights_after_loading"
            )
        engine = self._ensure_engine(layer)
        qlen = hidden_states.size(0)
        hidden_size = int(w2.shape[1])
        out = torch.empty(qlen, hidden_size, dtype=torch.float32,
                          device=hidden_states.device)
        stream = torch.cuda.current_stream()
        # The binding's pointer extractor accepts numpy arrays or integer
        # data_ptr() values (NOT torch tensors -> nullptr), so pass data_ptr().
        h_bf16 = hidden_states.to(torch.bfloat16)
        ids_i32 = topk_ids.to(torch.int32)
        wts_f32 = topk_weights.to(torch.float32)
        engine.cpu_decode(
            stream.cuda_stream, qlen, self.moe_config.experts_per_token,
            h_bf16.data_ptr(), ids_i32.data_ptr(), wts_f32.data_ptr(),
            out.data_ptr(),
        )
        return out.to(hidden_states.dtype)


def register_mixed_cpu_backend() -> None:
    """混合模式下把 MXFP4 的 CPU 后端换成 XiaotuCPUExperts。

    mxfp4 oracle 对 CPU 后端是**惰性 import** `CPUExpertsMxfp4`;把该模块属性换成我们的类,
    即可让 oracle 选中 AVX512-VNNI 引擎(而非仅 Intel AMX 的主线内核)。
    注:oracle 里加一个正式的后端注册表是更干净的上游扩展点;这里先做 OOT 等价物。
    """
    if not mixed_mode_enabled():
        return
    import vllm.model_executor.layers.fused_moe.experts.cpu_moe as cpu_moe

    if getattr(cpu_moe, "CPUExpertsMxfp4", None) is not XiaotuCPUExperts:
        cpu_moe.CPUExpertsMxfp4 = XiaotuCPUExperts
        print(
            "[xiaotu] GPU/CPU Mixed: MXFP4 CPU backend -> XiaotuCPUExperts "
            "(AVX512-VNNI, no AMX)",
            flush=True,
        )

