"""vllm-xtu-moe: 主线 DS-V4 的 OOT hybrid 模型覆盖(把 MoE 层挪 CPU 走 xiaotu).

机制(已源码实证,见 docs/ARCHITECTURE.md):
  - vLLM 主线原生支持 DeepSeek-V4(vllm.models.deepseek_v4,nvidia 平台)。
  - ModelRegistry.register_model("DeepseekV4ForCausalLM", "<module>:<class>") 直接覆盖主线该
    arch(registry.py overwrite 明确);main 里 entry-points 把 register() 挂到 vllm.general_plugins。
  - 构造期 OOM 根因:主线把 138GB fp4 MoE 在 GPU 上 materialize(或 offload 只能搬"已构造"权重,
    阻止不了构造期分配)。=> 真解法 = 让每层 ffn 从构造起就持 CPU 权重。
  - 本模块替换 dv4_nvidia.DeepseekV4MoE → CpuXiaotuMoE:
      * gate(GPU,小)+ shared_experts(DeepseekV4MLP,GPU,小)复用主线;
      * routed experts = 自建 CPU 参数模块 CpuMegaExpertsParams,镜像主线 MegaMoE 的原始 fp4
        布局(w13_weight [E,2I,H//2] uint8 / w2_weight [E,H,I//2] uint8 / *_scale raw e8m0 uint8,
        groupK=32)—— 与该 checkpoint 的原始 fp4 及 xiaotu 引擎契约完全一致。
      * 让 self.use_mega_moe=True → 主线的 get_expert_mapping 走 mega mapping,主线的 load_weights
        直接把 checkpoint 原始 fp4 绑定进我们这组 CPU 参数(no GPU 138GB materialize)。
  - 权重加载完成后,process_weights_after_loading -> model.finalize_mega_moe_weights() ->
    layer.ffn.finalize_mega_moe_weights() 这一后置钩子里用这批 CPU 权重构造 xiaotu 引擎
    (数据指针直达,零拷贝)。
  - forward: 路由(GPU fused_topk_bias,含 hash/vision/biased 专家) + engine.cpu_decode(GPU 张量,
    内部 D2H→CPU forward_many→H2D) + 共享专家(GPU)。
"""
from __future__ import annotations

import os
import threading
import time

import torch
import torch.nn as nn

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.models.deepseek_v4.nvidia.model import extract_layer_index
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MLP
from vllm.models.deepseek_v4.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

# 主线 DS-V4 模型(不 fork、只 OOT 覆盖)
import vllm.models.deepseek_v4.nvidia.model as dv4_nvidia  # noqa: F401
from vllm_xiaotu_moe.gpu_prefill import gpu_prefill_min_tokens  # noqa: E402

_T: dict[str, float] = {"eng": 0.0, "calls": 0, "t0": time.perf_counter()}

MODEL_ARCH = "DeepseekV4ForCausalLM"

# Layer-index -> MoE module, so layer L can prefetch layer L+1's GPU weights.
_LAYERS: dict[int, "CpuXiaotuMoE"] = {}
_PREBUILD_STARTED = False


_PROF_STATE: dict = {"prof": None, "calls": 0}


def _maybe_profile() -> None:
    """Env-gated torch profiler around the first N GPU-prefill layers.

    Used to decompose the long-prefill TTFT into MoE vs attention/indexer
    kernels (the model runs in vLLM's worker process, so an external nsys
    session does not see its kernels). Set XIAOTU_TORCH_PROFILE=<path> to dump
    a chrome trace plus a per-kernel table; XIAOTU_TORCH_PROFILE_CALLS=N
    (default 86 = two passes) controls the window.
    """
    path = os.environ.get("XIAOTU_TORCH_PROFILE")
    if not path:
        return
    st = _PROF_STATE
    if st["prof"] is None:
        import torch.profiler as tp

        st["prof"] = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
        )
        st["prof"].__enter__()
    st["calls"] += 1
    limit = int(os.environ.get("XIAOTU_TORCH_PROFILE_CALLS", "86"))
    if st["calls"] >= limit:
        prof = st["prof"]
        st["prof"] = None
        prof.__exit__(None, None, None)
        print("[xiaotu-profile]\n" + prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=40), flush=True)
        prof.export_chrome_trace(path)
        print(f"[xiaotu-profile] chrome trace -> {path}", flush=True)


def _start_pinned_prebuild() -> None:
    """Build every layer's K-major pinned host cache in the background (once).

    Without this the first long prefill pays ~1 s per layer synchronously inside
    its forward (43 layers). The prebuild runs on a daemon thread while the first
    layers' kernels execute, so steady state is reached within the first few
    layers.
    """
    global _PREBUILD_STARTED
    if _PREBUILD_STARTED:
        return
    _PREBUILD_STARTED = True
    try:
        from vllm_xiaotu_moe.gpu_prefill import prebuild_pinned_kmajor

        quads = []
        for m in list(_LAYERS.values()):
            try:
                w13, s13, w2, s2, _, _, _ = m._gpu_shard()
                quads.append((w13, s13, w2, s2))
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(
            target=prebuild_pinned_kmajor, args=(quads,),
            kwargs={"workers": 6}, daemon=True,
        ).start()
    except Exception as e:  # noqa: BLE001
        print(f"[xiaotu] pinned prebuild not started: {type(e).__name__}: {e}",
              flush=True)


def _map_global_expert_id(expert_id: int) -> list[int]:
    # 单 rank、无 EPLB:全局专家 id == 本进程参数里的行号(我们持有全部 256 个)。
    return [expert_id]


class CpuMegaExpertsParams(nn.Module):
    """镜像主线 DeepseekV4MegaMoEExperts 的原始 fp4 参数布局,但分配在 CPU。

    形状/数据类型与该 checkpoint 的原始 fp4(及 xiaotu MOE_MXFP4 契约)完全一致:
      w13_weight        [E, 2I, H//2]  uint8
      w13_weight_scale  [E, 2I, H//32] uint8(raw e8m0,groupK=32)
      w2_weight         [E, H, I//2]   uint8
      w2_weight_scale   [E, H, I//32]  uint8
    weight_loader 语义与主线 mega 版一致(TP=1/EP=1:per-expert w1/w3 分半写入 w13)。
    关键:分配在 CPU => 模型构造不再在 GPU 上 materialize 138GB => 解决构造期 OOM。
    """

    def __init__(self, num_experts, hidden_size, intermediate_size, device="cpu"):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        weight_attrs = {"weight_loader": self.weight_loader}
        kw = dict(device=device, dtype=torch.uint8)
        self.w13_weight = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, hidden_size // 2, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size, hidden_size // 32, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"
        self.w2_weight = nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size // 2, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size // 32, **kw),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_ids = _map_global_expert_id(expert_id)
        loaded_any = False
        for local_expert_id in local_ids:
            expert_data = param.data[local_expert_id]
            if shard_id in ("w1", "w3"):
                if "w13_" not in weight_name:
                    continue
                shard_offset = 0 if shard_id == "w1" else self.intermediate_size
                expert_data = expert_data.narrow(
                    0, shard_offset, self.intermediate_size
                )
            elif shard_id == "w2":
                if "w2_" not in weight_name:
                    continue
            else:
                raise ValueError(f"Unsupported expert shard id: {shard_id}")
            if expert_data.shape != loaded_weight.shape:
                raise ValueError(
                    f"[xiaotu] expert weight shape mismatch {weight_name}: "
                    f"param {tuple(expert_data.shape)} vs ckpt {tuple(loaded_weight.shape)}"
                )
            expert_data.copy_(loaded_weight)
            loaded_any = True
        if return_success:
            return loaded_any
        return None

    def update_expert_map(self):
        # 无 EPLB,no-op(与主线 FusedMoEFactory 同名钩子对齐)。
        return None

    def finalize_weights(self, shared_experts=None):
        # 无 DeepGEMM 融合,no-op(引擎由 CpuXiaotuMoE.finalize_mega_moe_weights 构造)。
        return None


CpuMegaExpertsParams.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


class CpuXiaotuMoE(nn.Module):
    """把一层 DS-V4 的 routed MoE 挪到 CPU 用 xiaotu 引擎算;gate/共享专家留在 GPU。

    - 该模块即"该层 ffn"的 CPU 权重持有者:routed 专家参数在 CPU(不占 GPU,解决 138GB OOM)。
    - finalize_mega_moe_weights()(主线的模型后置加载钩子)里构造 xiaotu 引擎。
    - forward: 路由(GPU)+ engine.cpu_decode + 共享专家(GPU)。
    """

    def __init__(self, vllm_config, prefix: str = "", use_sequence_parallel: bool = False):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self._vllm_config = vllm_config
        self.prefix = prefix
        self.use_sequence_parallel = bool(use_sequence_parallel)
        # 关键:让主线的 get_expert_mapping() 走 mega mapping(绑定原始 fp4 进下面的 CPU 参数)。
        self.use_mega_moe = True
        self.use_fi_mega_moe = False
        self.tp_size = 1

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        self.n_shared_experts = config.n_shared_experts or 0
        self.n_local_experts = config.n_routed_experts
        self.n_logical_experts = config.n_routed_experts
        self.n_physical_experts = config.n_routed_experts  # 无 redundant
        self.n_local_physical_experts = config.n_routed_experts  # 单 rank 持全部
        self.n_redundant_experts = 0
        self.experts_start_idx = 0
        self.experts_end_idx = config.n_routed_experts
        self.hash_indices_dtype = torch.int64  # mega 语义
        self.image_sentinel_lo = (
            IMAGE_SENTINEL_BASE_ID
            if getattr(config, "vision_n_layers", 0) > 0
            else 0
        )

        # ---- gate(GPU,小)— 与主线一致(含 hash / biased 专家) ----
        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        self.gate.bias_vl = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        if is_hash_moe:
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        if getattr(config, "topk_method", None) == "noaux_tc" and not is_hash_moe:
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        # ---- 共享专家(GPU,小) — 复用主线 DeepseekV4MLP ----
        if self.n_shared_experts:
            intermediate_size = (
                config.moe_intermediate_size * self.n_shared_experts
            )
            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                is_sequence_parallel=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        # ---- routed experts:CPU 参数(主线的 load_weights 会把原始 fp4 绑进来) ----
        self.experts = CpuMegaExpertsParams(
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            device="cpu",
        )
        self.engine = None
        self._timing = os.environ.get("XIAOTU_TIMING") == "1"
        self._slot = None
        self._slot_device = None
        try:
            _LAYERS[extract_layer_index(prefix)] = self
        except Exception:
            pass

    # ---- long-prefill GPU path helpers ----
    def _gpu_shard(self):
        """(w13, s13, w2, s2, start, local_E, tp) CPU views for the GPU path."""
        ex = self.experts
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp > 1:
            # Expert-parallel: each rank streams only its 1/TP expert shard.
            rank = get_tensor_model_parallel_rank()
            L = self.n_routed_experts // tp
            st = rank * L
            return (
                ex.w13_weight.data[st:st + L], ex.w13_weight_scale.data[st:st + L],
                ex.w2_weight.data[st:st + L], ex.w2_weight_scale.data[st:st + L],
                st, L, tp,
            )
        return (
            ex.w13_weight.data, ex.w13_weight_scale.data,
            ex.w2_weight.data, ex.w2_weight_scale.data,
            0, self.n_routed_experts, tp,
        )

    def prefetch_gpu_weights(self, device) -> None:
        """Async H2D of this layer's GPU weights (issued by the previous layer)."""
        if self.engine is None:
            return
        from vllm_xiaotu_moe.gpu_prefill import _pinned_kmajor, prefetch_layer

        w13, s13, w2, s2, _, _, _ = self._gpu_shard()
        self._slot = prefetch_layer(
            _pinned_kmajor(w13), _pinned_kmajor(s13),
            _pinned_kmajor(w2), _pinned_kmajor(s2), device,
        )
        self._slot_device = device

    # ---- 引擎构造(后置加载钩子,权重已就位) ----
    def finalize_mega_moe_weights(self) -> None:
        if self.engine is not None:
            return
        import xiaotu_moe

        ex = self.experts
        w13 = ex.w13_weight.detach()
        w2 = ex.w2_weight.detach()
        s13 = ex.w13_weight_scale.detach()
        s2 = ex.w2_weight_scale.detach()

        cfg = xiaotu_moe.MOEConfigV2()
        cfg.num_processes = 1
        cfg.process_id = 0
        cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True
        cfg.expert_num = int(w13.shape[0])
        cfg.top_k = self.top_k
        cfg.hidden_size = self.hidden_size
        cfg.intermediate_size = self.moe_intermediate_size
        cfg.max_batch_size = 8192
        cfg.max_num_seqs = 256
        cfg.stride = 32
        cfg.group_min_len = 10
        # 引擎按 group_max_len 给 input_fp32_/gate_input_/… 定尺寸,而 forward_many_m
        # 用 token_id(<qlen)直接索引 → **必须 group_max_len >= qlen**(否则堆越界;
        # 曾经 qlen=8192/profile 而 group_max_len=4224,越界 ~2x)。canonical 公式:
        # min(4096, max_num_batched_tokens) + 128。
        _mnbt = int(
            getattr(
                getattr(self._vllm_config, "scheduler_config", None),
                "max_num_batched_tokens",
                4096,
            )
            or 4096
        )
        cfg.group_max_len = int(
            os.environ.get("XIAOTU_GROUP_MAX_LEN", str(min(4096, _mnbt) + 128))
        )
        if _mnbt > cfg.group_max_len:
            print(
                f"[xiaotu] WARNING: max_num_batched_tokens={_mnbt} > "
                f"group_max_len={cfg.group_max_len}; the engine's per-token "
                "buffers would overflow. Set --max-num-batched-tokens <= 4096.",
                flush=True,
            )
        cfg.activation_type = 0
        cfg.use_gpu_prefill = False
        cfg.groupN = 1
        cfg.groupK = 32  # raw e8m0 block=32(与 checkpoint/引擎契约一致)

        self.engine = xiaotu_moe.MOE_MXFP4(
            cfg,
            w13.data_ptr(), w2.data_ptr(),
            s13.data_ptr(), s2.data_ptr(),
            0, 0,
        )
        # 引擎持有这些参数的内存(达 data_ptr),必须防被替换/释放。
        self._w13, self._w2 = w13, w2
        self._s13, self._s2 = s13, s2
        del w13, w2, s13, s2
        print(f"[xiaotu] engine built {self.prefix} "
              f"E={self.n_routed_experts} topk={self.top_k}", flush=True)

    def build_engine(self):
        self.finalize_mega_moe_weights()

    # ---- forward ----
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.engine is None:
            # 结构前向退化(引擎未就绪):返回输入,保证模型可构造。
            return hidden_states
        if getattr(self.gate, "tid2eid", None) is not None and input_ids is None:
            raise ValueError("DS-V4 hash MoE routing requires input_ids.")

        gate_out = self.gate(hidden_states)
        router_logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out
        bias = getattr(self.gate, "e_score_correction_bias", None)
        bias_vl = getattr(self.gate, "bias_vl", None)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=bias.data if bias is not None else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=getattr(self.gate, "tid2eid", None),
            routed_scaling_factor=self.routed_scaling_factor,
            bias_vl=bias_vl.data if bias_vl is not None else None,
            image_sentinel_lo=self.image_sentinel_lo if bias_vl is not None else 0,
        )

        qlen = hidden_states.size(0)

        # ---- long-prefill GPU MoE path (env-gated) ----
        # When this layer's prefill batch is large enough, compute the MoE on
        # GPU by streaming this layer's raw MXFP4 weights H2D (one layer at a
        # time), instead of the CPU engine. Activations already live on the GPU
        # (attention path), so only weights stream. Short prefills stay on CPU.
        _gp_min = gpu_prefill_min_tokens()
        if (
            _gp_min > 0
            and qlen >= _gp_min
            and not torch.cuda.is_current_stream_capturing()
        ):
            from vllm.distributed import tensor_model_parallel_all_reduce
            from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

            # Overlap: kick off the NEXT layer's H2D before doing this layer's
            # kernels, so the DMA runs concurrently with the tensor cores.
            _ov = os.environ.get("XIAOTU_GPU_PREFETCH_AHEAD", "1") == "1"
            _start_pinned_prebuild()
            if os.environ.get("XIAOTU_DEBUG_QLEN") == "1":
                # Diagnostic: shows how the scheduler batches (one entry per MoE
                # forward). Used to explain why concurrent requests do not share
                # one weight stream (see docs/OPTIMIZATION_ROUND2.md §4).
                _dbg = _T.setdefault("qlen_log", [])
                if len(_dbg) < 3000:
                    _dbg.append(qlen)
                    if len(_dbg) % 43 == 0:
                        _tail = _dbg[-43:]
                        print(f"[qlen] pass#{len(_dbg) // 43} "
                              f"qlen={sorted(set(_tail))} layers={len(_tail)}",
                              flush=True)
            if _ov:
                try:
                    _nxt = _LAYERS.get(extract_layer_index(self.prefix) + 1)
                except Exception:
                    _nxt = None
                if _nxt is not None:
                    _nxt.prefetch_gpu_weights(hidden_states.device)

            _nvtx = os.environ.get("XIAOTU_NVTX") == "1"
            if _nvtx:
                torch.cuda.nvtx.range_push("xiaotu_moe")
            w13, s13, w2, s2, st, L, tp = self._gpu_shard()
            slot = self._slot if _ov else None
            self._slot = None
            if tp > 1:
                # Expert-parallel: remap to the local shard, then all-reduce the
                # partial MoE output across ranks.
                ids_l = topk_ids - st
                mask = (ids_l >= 0) & (ids_l < L)
                ids_l = torch.where(mask, ids_l, torch.full_like(ids_l, -1))
                wts_l = torch.where(mask, topk_weights, torch.zeros_like(topk_weights))
                gpu_out = gpu_moe_layer(
                    hidden_states, ids_l, wts_l, w13, s13, w2, s2,
                    H=self.hidden_size, I=self.moe_intermediate_size,
                    K=self.top_k, device=hidden_states.device, slot=slot,
                )
                gpu_out = tensor_model_parallel_all_reduce(gpu_out)
            else:
                gpu_out = gpu_moe_layer(
                    hidden_states, topk_ids, topk_weights,
                    w13, s13, w2, s2,
                    H=self.hidden_size, I=self.moe_intermediate_size,
                    K=self.top_k, device=hidden_states.device, slot=slot,
                )
            if _nvtx:
                torch.cuda.nvtx.range_pop()
            _maybe_profile()
            if self.shared_experts is not None:
                gpu_out = gpu_out + self.shared_experts(hidden_states)
            if os.environ.get("XIAOTU_DEBUG_L1") == "1":
                print(
                    f"[dbg] {self.prefix} GPU-prefill qlen={qlen} tp={tp} "
                    f"out_mean={gpu_out.abs().mean().item():.3e}",
                    flush=True,
                )
            return gpu_out

        # routed experts:xiaotu 引擎(CPU)。engine.cpu_decode 内部 D2H->CPU forward_many->H2D。
        out = torch.empty(
            qlen, self.hidden_size, dtype=torch.float32, device=hidden_states.device
        )
        stream = torch.cuda.current_stream()
        # 引擎 binding 的指针提取只认 numpy 数组或整数 data_ptr()(对 torch 张量返回
        # nullptr → cudaMemcpyAsync(nullptr) → invalid argument),故传 data_ptr()。
        h_bf16 = hidden_states.to(torch.bfloat16)
        ids_i32 = topk_ids.to(torch.int32)
        wts_f32 = topk_weights.to(torch.float32)
        _t0 = time.perf_counter() if self._timing else 0.0
        self.engine.cpu_decode(
            stream.cuda_stream, qlen, self.top_k,
            h_bf16.data_ptr(),
            ids_i32.data_ptr(),
            wts_f32.data_ptr(),
            out.data_ptr(),
        )
        if self._timing:
            _T["eng"] += time.perf_counter() - _t0
            _T["calls"] += 1
            if _T["calls"] == 1:
                _T["t0"] = time.perf_counter()
            if _T["calls"] % 43 == 0:
                _now = time.perf_counter()
                _step = _now - _T["t0"]
                print(
                    f"[xiaotu-timing] engine={_T['eng']:.3f}s step_wall={_step:.3f}s "
                    f"other={_step - _T['eng']:.3f}s qlen={qlen} "
                    f"engine_frac={_T['eng'] / max(_step, 1e-9):.2f}",
                    flush=True,
                )
                _T["eng"] = 0.0
                _T["t0"] = _now
        final_hidden_states = out.to(hidden_states.dtype)
        if os.environ.get("XIAOTU_DEBUG_L1") == "1":
            nz = (final_hidden_states != 0).sum().item()
            nn = final_hidden_states.numel()
            fn = (final_hidden_states != final_hidden_states).sum().item()  # NaN
            tids = topk_ids[:6].cpu().tolist()
            tw = topk_weights[:6].cpu().tolist()
            hin = hidden_states.abs().mean().item()
            ex = self.experts
            w13m = float(ex.w13_weight.abs().float().mean().item())
            w2m = float(ex.w2_weight.abs().float().mean().item())
            print(f"[dbg] {self.prefix} out-nz={nz}/{nn} NaN={fn} "
                  f"mean={final_hidden_states.abs().mean().item():.3e} "
                  f"hin={hin:.3e} w13m={w13m:.3e} w2m={w2m:.3e} "
                  f"topk={tids} wts={tw}", flush=True)

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(
                hidden_states
            )
        return final_hidden_states


class HybridDeepseekV4ForCausalLM(dv4_nvidia.DeepseekV4ForCausalLM):
    """OOT 覆盖 DeepseekV4ForCausalLM:decoder 层的 ffn 全部是 CpuXiaotuMoE(CPU MoE + GPU 注意力)。

    复用主线 model_cls=DeepseekV4Model(其层 ffn 因 register() 把 DeepseekV4MoE 换成
    CpuXiaotuMoE 而成为 CPU 版);引擎在 process_weights_after_loading ->
    model.finalize_mega_moe_weights() -> layer.ffn.finalize_mega_moe_weights() 里构造。
    """


def register():
    """vllm.general_plugins 入口:OOT 覆盖 DS-V4 arch(主线零改动)。"""
    if not getattr(dv4_nvidia, "_xiaotu_patched", False):
        dv4_nvidia.DeepseekV4MoE = CpuXiaotuMoE
        dv4_nvidia._xiaotu_patched = True  # 幂等

    ModelRegistry.register_model(
        MODEL_ARCH,
        f"{__name__}:HybridDeepseekV4ForCausalLM",
    )
    print(
        f"[vllm-xtu-moe] registered OOT override of {MODEL_ARCH} "
        f"(DeepseekV4MoE -> CpuXiaotuMoE)",
        flush=True,
    )
    return None


if __name__ == "__main__":
    register()
    print("registered OK")
