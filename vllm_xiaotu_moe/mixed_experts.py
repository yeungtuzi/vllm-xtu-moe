"""GPU/CPU Mixed Mode 的通用 CPU 计算后端(XiaotuCPUExperts*)。

背景:主线已引入 `VLLM_EXPERTS_LOAD_DEVICE=cpu`(见 vllm/envs.py + routed_experts.py),
让 routed-expert 权重从构造起就在 CPU(不 OOM)。但计算仍走 quant_method.apply(GPU 内核)
会设备不匹配。本模块提供 CPU 计算后端,并**按量化格式注册到主线的 CPU 后端槽位**,
因此任何使用 `FusedMoEFactory` 的 MoE 模型(DS-V4、GLM-5.x、Qwen、Mixtral…)只要
权重格式在支持列表里,就会自动选中 xiaotu 引擎,**不需要模型级 OOT 覆盖**。

主线侧需要配合的一点:oracle 在 GPU 平台上默认不会选 CPU 后端(它要求
`current_platform.is_cpu()`),所以混合模式下需要把 CPU 后端前置。MXFP4 的改动在
`patches/mainline_sm80_mixed_mode.patch`(PR1),FP8 的对称改动在同一补丁里。

支持格式(引擎 → 基线类):
  bf16   MOE_BF16   CPUUnquantizedExperts 无缩放        权重 bf16 [E,2I,H]/[E,H,I]
  mxfp4  MOE_MXFP4  CPUExpertsMxfp4  groupN=1    groupK=32   权重 u8 nibble + e8m0
  fp8    MOE_FP8    CPUExpertsFp8    groupN=128  groupK=128  权重 e4m3 + fp32 块缩放
  int4   MOE_WNA16  CPUExpertsInt4   groupN=32/128 groupK=32/128 (按 quant_config 取)

路由:主线把路由拆在 `fused_moe/router/*`(softmax / sigmoid / sqrtsoftplus /
grouped-topk / noaux_tc / custom_routing_function)。monolithic 后端拿到的只有
router_logits,所以本模块**复用主线的 router 对象**(`create_fused_moe_router`),
而不是像主线 cpu_moe 那样硬编码 softmax —— 后者对 GLM(sigmoid+noaux_tc)和
DS-V4(sqrtsoftplus)都会选错专家。

激活:引擎实现**packed 布局**的 gated 激活 `out = clamp(gate,max=L) * sigmoid(alpha*clamp(gate,max=L))
* (clamp(up,±L)+beta)`(与主线 `silu_and_mul_with_clamp` 同语义),因此支持主线的
`SILU` 与 `SWIGLUOAI_UNINTERLEAVE`(两者都是 packed);**不支持 `SWIGLUOAI`**
(gpt-oss 把 gate/up 交错存在 w13 里,我们按 packed 取数会静默算错)—— 由
`_supports_activation` 显式拒绝,让 oracle 去选别的后端。

在 AMD(无 AMX)上,这是唯一能跑的 CPU MoE 内核;主线 CPUExperts* 要 Intel AMX。
"""
from __future__ import annotations

import os

import torch

from vllm import envs
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
    CPUExpertsFp8,
    CPUExpertsInt4,
    CPUExpertsMxfp4,
    CPUUnquantizedExperts,
    select_experts,
)


_VERIFY_LAYER = os.environ.get("XIAOTU_VERIFY_LAYER", "") == "1"
_VERIFY_MAX = int(os.environ.get("XIAOTU_VERIFY_MAX", "1"))
_DUMP_LAYER = os.environ.get("XIAOTU_DUMP_LAYER", "")
_HID_LAYER = os.environ.get("XIAOTU_HID_LAYER", "")


def mixed_mode_enabled() -> bool:
    """True when experts are configured to live/compute on CPU on a GPU run.

    Delegates to `mainline_shims.mixed_mode_enabled`, which reads the env var
    directly so the plugin also works on a **stock** vLLM (whose `vllm.envs`
    has no `VLLM_EXPERTS_LOAD_DEVICE`).
    """
    from .mainline_shims import mixed_mode_enabled as _m

    return _m()


def _supports_mixed_device() -> bool:
    from vllm.platforms import CpuArchEnum, current_platform

    return (
        mixed_mode_enabled()
        and current_platform.get_cpu_architecture() == CpuArchEnum.X86
    )


# 主线 RoutedExperts 上保存的路由/激活配置(见 routed_experts.py __init__)。
_ROUTING_ATTRS = (
    "use_grouped_topk",
    "renormalize",
    "scoring_func",
    "custom_routing_function",
    "num_expert_group",
    "topk_group",
    "e_score_correction_bias",
    "routed_scaling_factor",
)
_SWIGLU_ATTRS = ("swiglu_limit", "swiglu_alpha", "swiglu_beta")
# 少数模型的 router 需要额外参数(本插件只在 layer 上存在时才透传)。
_ROUTER_EXTRA_ATTRS = (
    "hash_indices_table",
    "num_fused_shared_experts",
    "shared_expert_weight",
    "bias_vl",
    "image_sentinel_lo",
)


class _XiaotuExpertsMixin:
    """Shared behaviour: build the xiaotu engine from the layer's raw CPU weights.

    Subclasses set ``_engine_attr`` (engine class on the ``xiaotu_moe`` module) and
    ``_scale_attrs`` (candidate attribute names for w13/w2 scales, in order).
    """

    _engine_attr = ""
    _scale_attrs: tuple[str, str] = ("w13_weight_scale", "w2_weight_scale")
    # scale grouping passed to the engine (groupN, groupK)
    _group_n = 1
    _group_k = 32
    # weight element width in bytes (u8 for mxfp4/int4-packed, 1 for fp8)
    _w_bytes = 1
    # expected dtype of the layer's w13_weight (None = unchecked)
    _expect_dtype = None
    # dtype the engine reads block scales as; None = pass the layer's tensor
    # through unchanged (MXFP4 uses uint8 e8m0 bytes). Checkpoints disagree on
    # this (GLM-5.3 ships fp32 scale_inv, Qwen3 ships bf16), so we normalize to
    # fp32 for the formats whose engine kernel reads `const float*`.
    _scale_dtype = None

    def __init__(self, moe_config, quant_config):
        super().__init__(moe_config, quant_config)
        self._xiaotu_engine = None
        self._layer_ref: torch.nn.Module | None = None
        self._router = None
        # Mirror the router configuration that monolithic apply() must carry
        # itself (mainline captures the same fields in CPUUnquantizedExperts).
        self.use_grouped_topk = False
        self.renormalize = False
        self.scoring_func = "softmax"
        self.custom_routing_function = None
        self.num_expert_group = None
        self.topk_group = None
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0
        self.swiglu_limit = None
        self.swiglu_alpha = None
        self.swiglu_beta = None
        self._router_extra: dict = {}

    # ---- device / weight handling -------------------------------------
    @staticmethod
    def _supports_current_device() -> bool:
        return _supports_mixed_device()

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        # Routing is delegated to mainline's router objects (see _get_router),
        # so every routing method the layer can be built with is supported —
        # including DeepSeekV3 (sigmoid+noaux_tc), DeepseekV4 (sqrtsoftplus),
        # grouped top-k and custom routing functions.
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # The engine consumes the *packed* w13 layout (gate rows first, then up)
        # and applies an optional clamp; that is exactly mainline's SILU and
        # SWIGLUOAI_UNINTERLEAVE. SWIGLUOAI (gpt-oss) interleaves gate/up inside
        # w13, so accepting it here would produce silently wrong results.
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Do NOT AMX-prepack; remember the layer so the engine can use the raw
        CPU parameters (the checkpoint layout the engine consumes directly)."""
        self._layer_ref = layer
        for name in _ROUTING_ATTRS:
            if hasattr(layer, name):
                setattr(self, name, getattr(layer, name))
        for name in _SWIGLU_ATTRS:
            if hasattr(layer, name):
                setattr(self, name, getattr(layer, name))
        self._router_extra = {
            name: getattr(layer, name)
            for name in _ROUTER_EXTRA_ATTRS
            if getattr(layer, name, None) is not None
        }
        # Resolve block scales once (converted to the dtype the engine reads).
        # We keep our own tensors instead of mutating vLLM's parameters so the
        # GPU path (if used for the same layer elsewhere) is unaffected.
        self._scales: tuple = (None, None)
        if self._scale_dtype is not None:
            resolved = []
            for name in self._scale_attrs:
                cands = name if isinstance(name, (tuple, list)) else (name,)
                t = self._find_scale(layer, cands)
                if t is not None and t.dtype != self._scale_dtype:
                    t = t.to(self._scale_dtype).contiguous()
                resolved.append(t)
            self._scales = tuple(resolved)
        self._router = None

    # ---- routing -------------------------------------------------------
    def _get_router(self):
        if self._router is not None:
            return self._router
        from vllm.model_executor.layers.fused_moe.router.router_factory import (
            create_fused_moe_router,
        )

        kwargs = dict(
            top_k=self.moe_config.experts_per_token,
            global_num_experts=self.moe_config.num_experts,
            eplb_state=None,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )
        for name, value in self._router_extra.items():
            kwargs.setdefault(name, value)
        try:
            self._router = create_fused_moe_router(**kwargs)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[vllm-xtu-moe] router factory failed ({exc}); falling back to "
                f"cpu_moe.select_experts(scoring_func={self.scoring_func})",
                flush=True,
            )
            self._router = False  # sentinel: use the legacy fallback
        return self._router

    def _select_topk(self, hidden_states, router_logits):
        router = self._get_router()
        if router:
            return router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=torch.int32,
            )
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.moe_config.experts_per_token,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=(
                self.scoring_func if self.scoring_func in ("softmax", "sigmoid")
                else "softmax"
            ),
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

    def _local_expert_map(self, expert_map, device):
        """把 expert_map 缓存在 ids 所在的设备上(EP 下每层一份)。"""
        cached = getattr(self, "_expert_map_cached", None)
        if cached is not None and cached[0] == device:
            return cached[1]
        em = expert_map
        if em.device != device:
            em = em.to(device)
        if em.dtype not in (torch.int32, torch.int64):
            em = em.to(torch.long)
        self._expert_map_cached = (device, em)
        return em

    # ---- engine construction ------------------------------------------
    def _validate_weights(self, layer, ex_w13, ex_w2) -> None:
        """Per-format layout guard, run before the engine is built.

        Called with the tensors the engine will actually consume (i.e. after
        `_prepare_weights`). The engine reads the checkpoint layout directly (it
        never calls mainline's `prepare_*_for_cpu` AMX repack), so a format whose
        layout differs from the engine's must be converted or rejected here
        instead of silently computing garbage.
        """

    def _prepare_weights(self, layer) -> None:
        """Resolve the tensors/grouping the engine consumes.

        Default: the layer's own tensors — BF16/FP8/MXFP4/NVFP4 checkpoints
        already match the engine's layout. Formats that differ override this and
        set `self._engine_w13/_engine_w2`, `self._scales`,
        `self._group_n/_group_k` (INT4/WNA16 repacks the GPTQ int32 packing).
        """
        self._engine_w13 = None
        self._engine_w2 = None

    def _find_scale(self, layer, names):
        for n in names:
            if not n:  # unquantized backends have no scale tensors
                continue
            t = getattr(layer, n, None)
            if t is not None:
                return t
        return None

    def _ensure_engine(self, layer: torch.nn.Module):
        if self._xiaotu_engine is not None:
            return self._xiaotu_engine
        import xiaotu_moe

        self._prepare_weights(layer)
        ex_w13 = self._engine_w13 if self._engine_w13 is not None else layer.w13_weight
        ex_w2 = self._engine_w2 if self._engine_w2 is not None else layer.w2_weight
        self._validate_weights(layer, ex_w13, ex_w2)
        if self._expect_dtype is not None and ex_w13.dtype != self._expect_dtype:
            raise NotImplementedError(
                f"xiaotu {self._engine_attr} backend expects "
                f"{self._expect_dtype} weights, got {ex_w13.dtype}"
            )
        s13, s2 = self._scales
        if s13 is None and self._scale_dtype is None:
            # unquantized / e8m0 formats: use the layer tensors as-is
            s13 = self._find_scale(layer, (self._scale_attrs[0],))
            s2 = self._find_scale(layer, (self._scale_attrs[1],))
        num_local_experts = int(ex_w13.shape[0])
        if num_local_experts != int(self.moe_config.num_experts):
            # 专家并行:本 rank 只持有 local_num_experts 个专家,id 已在 apply 里
            # 按 expert_map 映射成局部 id。
            print(
                f"[vllm-xtu-moe] expert parallelism: local={num_local_experts} "
                f"global={self.moe_config.num_experts}",
                flush=True,
            )
        if not self._supports_activation(self.moe_config.activation):
            raise NotImplementedError(
                f"xiaotu CPU experts backend implements the packed gated "
                f"activation only (SILU / SWIGLUOAI_UNINTERLEAVE), got "
                f"{self.moe_config.activation}"
            )

        cfg = xiaotu_moe.MOEConfigV2()
        cfg.num_processes = 1
        cfg.process_id = 0
        cfg.gpu_id = torch.cuda.current_device()
        cfg.has_gate_proj = True
        cfg.expert_num = num_local_experts
        cfg.top_k = self.moe_config.experts_per_token
        cfg.hidden_size = int(ex_w2.shape[1])            # w2: [E, H, I//2] or [E, H, I]
        cfg.intermediate_size = int(ex_w13.shape[1] // 2)  # w13: [E, 2I, ...]
        # 每 token 缓冲在 V2 引擎里是按调用分配的,这两个值只影响预分配提示;
        # 仍按 vLLM 的调度配置设置,避免与真实 batch 规模脱节。
        sched = getattr(self.moe_config, "scheduler_config", None)
        cfg.max_batch_size = int(getattr(sched, "max_num_batched_tokens", 0) or 8192)
        cfg.max_num_seqs = int(getattr(sched, "max_num_seqs", 0) or 256)
        cfg.stride = int(self._group_k)
        cfg.group_min_len = 10
        cfg.group_max_len = int(getattr(sched, "max_num_batched_tokens", 0) or 4096) + 128
        # 激活:0=plain SiLU(gate*up),1=clamped SwiGLU(vLLM silu_and_mul_with_clamp
        # 同语义,GLM/DS-V4/MiniMax 的 swiglu_limit)。
        limit = float(self.swiglu_limit) if self.swiglu_limit else 0.0
        alpha = float(self.swiglu_alpha) if self.swiglu_alpha else 1.0
        beta = float(self.swiglu_beta) if self.swiglu_beta else 0.0
        clamped = limit > 0.0 or alpha != 1.0 or beta != 0.0
        cfg.activation_type = 1 if clamped else 0
        cfg.swiglu_limit = limit
        cfg.swiglu_alpha = alpha
        cfg.use_gpu_prefill = False
        cfg.groupN = int(self._group_n)
        cfg.groupK = int(self._group_k)
        engine_cls = getattr(xiaotu_moe, self._engine_attr)
        self._xiaotu_engine = engine_cls(
            cfg,
            ex_w13.data_ptr(), ex_w2.data_ptr(),
            s13.data_ptr() if s13 is not None else 0,
            s2.data_ptr() if s2 is not None else 0,
            0, 0,
        )
        print(
            f"[vllm-xtu-moe] xiaotu {self._engine_attr} engine: "
            f"E={cfg.expert_num} H={cfg.hidden_size} I={cfg.intermediate_size} "
            f"topk={cfg.top_k} group={self._group_n}x{self._group_k} "
            f"scales={'yes' if s13 is not None else 'no'} "
            f"routing={self.scoring_func}"
            f"{f'/grouped{self.num_expert_group}x{self.topk_group}' if self.use_grouped_topk else ''}"
            f"{f'/bias' if self.e_score_correction_bias is not None else ''} "
            f"swiglu={'clamp@' + str(limit) if clamped else 'plain'}",
            flush=True,
        )
        return self._xiaotu_engine

    # ---- optional in-process self-verification -------------------------
    def _verify_once(self, layer, hidden_states, topk_ids, topk_weights, out):
        """对比引擎输出与 torch 参考(同一批已加载权重),定位数值差异。

        XIAOTU_VERIFY_LAYER=1 打开;每层只跑一次(跳过路由 id 无效的 profile 调用)。
        参考实现(对 token 0 的全部 top-k 专家求和):
          gate/up = dequant(w13[e]) @ x ; act = silu / clamped swiglu
          out += w * (dequant(w2[e]) @ bf16(act))
        """
        try:
            w13 = self._engine_w13 if getattr(self, "_engine_w13", None) is not None \
                else layer.w13_weight
            w2w = self._engine_w2 if getattr(self, "_engine_w2", None) is not None \
                else layer.w2_weight
            s13, s2 = self._scales
            unquant = (self._scale_dtype is None) and (s13 is None)
            gn, gk = int(self._group_n), int(self._group_k)
            I = int(w13.shape[1] // 2)
            H = int(w2w.shape[1])
            int4 = self._engine_attr == "MOE_WNA16" and w13.dtype == torch.uint8

            def _unpack4(w):   # [N, K/2] u8 -> [N, K] f32 with value = nibble - 8
                b = w.to(torch.int16)
                lo = (b & 0x0F) - 8
                hi = ((b >> 4) & 0x0F) - 8
                return torch.stack((lo, hi), dim=-1).reshape(w.shape[0], -1).float()

            x = hidden_states[0].float().cpu()
            ref = torch.zeros(H, dtype=torch.float32)
            ids_row = [int(v) for v in topk_ids[0]]
            for r, e in enumerate(ids_row):
                w = float(topk_weights[0, r])
                if w == 0.0:
                    continue
                w13_e = _unpack4(w13[e]) if int4 else w13[e].float()
                if unquant:
                    deq13 = w13_e
                else:
                    s13_full = s13[e].float().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
                    deq13 = w13_e * s13_full[: 2 * I, :H]
                gate = deq13[:I] @ x
                up = deq13[I:] @ x
                if self.swiglu_limit or self.swiglu_alpha or self.swiglu_beta:
                    limit = float(self.swiglu_limit or 0.0)
                    alpha = float(self.swiglu_alpha or 1.0)
                    beta = float(self.swiglu_beta or 0.0)
                    if limit > 0:
                        gate = torch.clamp(gate, max=limit)
                        up = torch.clamp(up, -limit, limit)
                    act = (gate / (1 + torch.exp(-alpha * gate))) * (up + beta)
                else:
                    act = (gate / (1 + torch.exp(-gate))) * up
                act = act.to(torch.bfloat16).float()
                w2_e = _unpack4(w2w[e]) if int4 else w2w[e].float()
                if unquant:
                    down = w2_e @ act
                else:
                    s2_full = s2[e].float().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
                    down = (w2_e * s2_full[:H, :I]) @ act
                ref += w * down
            got = out[0].float().cpu()
            rms = float(torch.sqrt(torch.mean(ref * ref)))
            rel = float(torch.sqrt(torch.mean((got - ref) ** 2)) / (rms + 1e-12))
            print(
                f"[vllm-xtu-moe/verify] {self._engine_attr} "
                f"layer={getattr(layer, 'layer_name', '?')} M={hidden_states.shape[0]} ids0={ids_row} "
                f"w0={[round(float(v), 4) for v in topk_weights[0]]} "
                f"ref_rms={rms:.4f} rel_rms={rel:.4e} "
                f"max_abs={float((got - ref).abs().max()):.4e}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[vllm-xtu-moe/verify] failed: {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()

    # ---- optional one-shot dump/compare against the checkpoint -----------
    def _dump_layer(self, layer):
        """XIAOTU_DUMP_LAYER=<layer_name 子串> 时,打印已加载张量的形状/样例值;
        若 XIAOTU_CKPT_DIR 指向 checkpoint 目录,再与原始 per-expert 张量逐项比对。
        用途:判断"引擎与 torch 参考一致但模型输出不对"是否来自权重加载布局。"""
        import json
        import re

        import numpy as np

        name = getattr(layer, "layer_name", "")
        w13, w2 = layer.w13_weight, layer.w2_weight
        s13, s2 = self._scales
        print(
            f"[dump] layer={name} w13={tuple(w13.shape)}{w13.dtype} "
            f"w2={tuple(w2.shape)}{w2.dtype} "
            f"s13={None if s13 is None else tuple(s13.shape)} "
            f"s2={None if s2 is None else tuple(s2.shape)} "
            f"attrs_scale=({getattr(layer, 'w13_weight_scale_inv', None) is not None},"
            f"{getattr(layer, 'w2_weight_scale_inv', None) is not None})",
            flush=True,
        )
        ckpt = os.environ.get("XIAOTU_CKPT_DIR")
        m = re.search(r"layers\.(\d+)", name)
        if not ckpt or not m:
            return
        li = m.group(1)
        try:
            from safetensors import safe_open
        except ImportError:
            return
        idx_path = os.path.join(ckpt, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            return
        wm = json.load(open(idx_path))["weight_map"]

        def load(key, dtype_cast=None):
            with safe_open(os.path.join(ckpt, wm[key]), framework="pt") as f:
                t = f.get_tensor(key)
            return t

        import torch as _t

        pre = f"model.layers.{li}.mlp.experts.0."
        pairs = [
            ("w13[:I] vs gate", w13[0][: w13.shape[1] // 2],
             load(pre + "gate_proj.weight")),
            ("w13[I:] vs up", w13[0][w13.shape[1] // 2:],
             load(pre + "up_proj.weight")),
            ("w2 vs down", w2[0], load(pre + "down_proj.weight")),
        ]
        for label, got, want in pairs:
            g = got.view(_t.uint8).numpy() if got.dtype == _t.float8_e4m3fn else got.numpy()
            w = want.view(_t.uint8).numpy() if want.dtype == _t.float8_e4m3fn else want.numpy()
            same = g.shape == w.shape
            diff = float(np.abs(g.astype(np.int16) - w.astype(np.int16)).max()) if same else -1
            print(f"[dump]   {label}: shape got={g.shape} ckpt={w.shape} "
                  f"byte_max_diff={diff}", flush=True)
        if s13 is not None:
            gs = s13[0].float().numpy()
            gs_ck = load(pre + "gate_proj.weight_scale_inv").float().numpy()
            us_ck = load(pre + "up_proj.weight_scale_inv").float().numpy()
            d2 = s2[0].float().numpy()
            ds_ck = load(pre + "down_proj.weight_scale_inv").float().numpy()
            print(f"[dump]   s13 got={gs.shape} ckpt_cat={np.concatenate([gs_ck, us_ck], 0).shape} "
                  f"gate_diff={float(np.abs(gs[:gs_ck.shape[0]] - gs_ck).max()):.3e} "
                  f"up_diff={float(np.abs(gs[gs_ck.shape[0]:] - us_ck).max()):.3e} "
                  f"s2 got={d2.shape} ckpt={ds_ck.shape} "
                  f"diff={float(np.abs(d2 - ds_ck).max()):.3e}", flush=True)

    # ---- compute ------------------------------------------------------
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
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "xiaotu CPU experts backend does not support "
                "apply_router_weight_on_input"
            )
        # monolithic apply() 拿不到 input_ids,若模型的路由需要它(hash routing),
        # router 会在这里报错——比静默选错专家好。
        topk_weights, topk_ids = self._select_topk(hidden_states, router_logits)

        # 专家并行(EP):路由给出的是全局 expert id,需要按 expert_map 映射到本
        # rank 的本地 id;映射为 -1 表示该专家不在本 rank,把权重置 0。
        if expert_map is not None:
            em = self._local_expert_map(expert_map, topk_ids.device)
            local = em[topk_ids.to(torch.long)]
            miss = local < 0
            if bool(miss.any()):
                topk_weights = torch.where(
                    miss, torch.zeros_like(topk_weights), topk_weights
                )
                local = torch.where(miss, torch.zeros_like(local), local)
            topk_ids = local.to(torch.int32)

        layer = self._layer_ref
        if layer is None:
            raise RuntimeError(
                "XiaotuCPUExperts.apply called before process_weights_after_loading"
            )
        engine = self._ensure_engine(layer)
        qlen = hidden_states.size(0)
        # Use the tensor the ENGINE was built from: for formats whose checkpoint
        # layout differs (INT4/WNA16), the `w2` argument is the raw packed tensor
        # ([E, I/8, H]) and its dim 1 is NOT the hidden size.
        w2_engine = self._engine_w2 if self._engine_w2 is not None else w2
        hidden_size = int(w2_engine.shape[1])
        out = torch.empty(qlen, hidden_size, dtype=torch.float32,
                          device=hidden_states.device)
        stream = torch.cuda.current_stream()
        if os.environ.get("XIAOTU_STREAM_TRACE") and not getattr(self, "_st_traced", False):
            self._st_traced = True
            print(f"[stream] cur={stream.cuda_stream:#x} "
                  f"default={torch.cuda.default_stream().cuda_stream:#x} "
                  f"cur_device={stream.device} layer={getattr(layer,'layer_name','?')}",
                  flush=True)
        # The binding's pointer extractor accepts numpy arrays or integer
        # data_ptr() values (NOT torch tensors -> nullptr), so pass data_ptr().
        h_bf16 = hidden_states.to(torch.bfloat16)
        ids_i32 = topk_ids.to(torch.int32)
        wts_f32 = topk_weights.to(torch.float32)
        # 传裸指针给引擎做异步 D2H/H2D:必须保证这些张量在拷贝完成前不被释放或复用。
        # ① 保留上一代的强引用(同一层下一次调用在流序上晚于本次拷贝);
        # ② record_stream 告知分配器这些块仍被当前流使用。
        self._keepalive = (h_bf16, ids_i32, wts_f32, out)
        for t in (h_bf16, ids_i32, wts_f32, out):
            if t.is_cuda:
                t.record_stream(stream)
        # 引擎调用前后可选地与设备同步。跨请求状态污染已在 binding.cpp 中从根上修复
        # (每次调用使用不可变的参数块,回调不再读 per-engine 的可变字段),因此默认
        # 关闭;调试或怀疑时序问题时设 XIAOTU_SYNC_DECODE=1 可强制串行化。
        _sync = os.environ.get("XIAOTU_SYNC_DECODE", "0")
        _is_first = ".0." in getattr(layer, "layer_name", "")
        if _sync == "1" or (_sync == "pre") or (_sync == "first" and _is_first):
            torch.cuda.synchronize()
        engine.cpu_decode(
            stream.cuda_stream, qlen, self.moe_config.experts_per_token,
            h_bf16.data_ptr(), ids_i32.data_ptr(), wts_f32.data_ptr(),
            out.data_ptr(),
        )
        if _sync == "post":
            torch.cuda.synchronize()
        if _HID_LAYER and _HID_LAYER in getattr(layer, "layer_name", ""):
            hs = h_bf16
            rows = [0] if hs.shape[0] < 5 else [0, 4]
            bits = hs.view(torch.uint16)  # bf16 原始位模式
            fp = " ".join(
                f"r{r}:[" + ",".join(f"{int(v):04x}" for v in bits[r, :4].tolist()) + "]"
                for r in rows
            )
            print(f"[hid] layer={getattr(layer, 'layer_name', '?')} M={hs.shape[0]} {fp}",
                  flush=True)
        if _DUMP_LAYER and _DUMP_LAYER in getattr(layer, "layer_name", "") \
                and not getattr(self, "_dumped", False):
            self._dumped = True
            self._dump_layer(layer)
        if _VERIFY_LAYER and getattr(self, "_verified_n", 0) < _VERIFY_MAX:
            # 跳过 profile/warmup 等路由 id 无效的调用(此时 topk_ids 为 -1 哨兵)
            if bool((ids_i32 >= 0).all()):
                self._verified_n = getattr(self, "_verified_n", 0) + 1
                self._verify_once(layer, h_bf16, ids_i32, wts_f32, out)
        return out.to(hidden_states.dtype)


class XiaotuCPUExpertsBF16(_XiaotuExpertsMixin, CPUUnquantizedExperts):
    """无量化(BF16)专家:引擎 MOE_BF16。覆盖 Mixtral / Qwen2-MoE / Qwen3-MoE(bf16)等。"""

    _engine_attr = "MOE_BF16"
    _scale_attrs = ("", "")
    _group_n, _group_k = 1, 1
    _expect_dtype = torch.bfloat16


class XiaotuCPUExpertsMxfp4(_XiaotuExpertsMixin, CPUExpertsMxfp4):
    """MXFP4 W4A16 专家:权重在 CPU,计算用 xiaotu AVX512-VNNI 引擎(无需 AMX)。"""

    _engine_attr = "MOE_MXFP4"
    _scale_attrs = ("w13_weight_scale", "w2_weight_scale")
    _group_n, _group_k = 1, 32
    _expect_dtype = torch.uint8


class XiaotuCPUExpertsFp8(_XiaotuExpertsMixin, CPUExpertsFp8):
    """FP8 W8A16 块量化专家(GLM-5.x / DS-V4 等):引擎 MOE_FP8,块 128x128。"""

    _engine_attr = "MOE_FP8"
    _scale_attrs = ("w13_weight_scale_inv", "w2_weight_scale_inv")
    _group_n, _group_k = 128, 128
    _expect_dtype = torch.float8_e4m3fn
    _scale_dtype = torch.float32


class XiaotuCPUExpertsInt4(_XiaotuExpertsMixin, CPUExpertsInt4):
    """INT4 W4A16 组量化专家:引擎 MOE_WNA16。

    检查点布局与引擎布局不同,`_prepare_weights` 在**引擎构造时做一次重排**:

      GPTQ / compressed-tensors:`w13 [E, K/8, 2I] int32`(每 int32 沿 K 打包 8 个
      nibble,小端下 int32→u8 视图即为"低 nibble = 偶 k")⇒ `transpose(1,2).contiguous()
      .view(uint8)` 直接得到引擎的 `[E, 2I, K/2] u8`;`w2` 同理。缩放 `[E, G, N]`
      `transpose(1,2)` 得到引擎的 `[E, N, G]`(即 groupN=1、groupK=group_size)。

    零点:引擎的 int4 表是"中心 8"(`value = nibble - 8`),只等价于**对称**量化。
    GPTQ 检查点的 `qzeros` 存的是 `zp - 1`(主线 AMX 重排里 `+1` 还原),所以对称
    模型的 qzeros 全为 7 → zp=8;若出现非 8 的零点,`_require_symmetric` 会显式报错
    (逐组零点需要内核侧支持,见 docs/ROADMAP.md)。
    """

    _engine_attr = "MOE_WNA16"
    # Attribute names differ per quant method: auto_gptq / moe_wna16 use
    # w13_qweight / w13_scales / w13_qzeros, compressed-tensors uses
    # w13_weight_packed / w13_weight_scale / w13_weight_zero_point.
    _scale_attrs = (("w13_scales", "w13_weight_scale"),
                    ("w2_scales", "w2_weight_scale"))
    _w_names = (("w13_qweight", "w13_weight_packed", "w13_weight"),
                ("w2_qweight", "w2_weight_packed", "w2_weight"))
    _z_names = ("w13_qzeros", "w13_weight_zero_point",
                "w2_qzeros", "w2_weight_zero_point")
    _group_n, _group_k = 1, 128
    _scale_dtype = torch.float32

    def _prepare_weights(self, layer) -> None:
        w13 = self._find_scale(layer, self._w_names[0])
        w2 = self._find_scale(layer, self._w_names[1])
        if w13 is None or w2 is None:
            raise NotImplementedError(
                "xiaotu INT4 backend found no int4 weight tensors on the layer "
                f"(looked for {self._w_names[0]} / {self._w_names[1]})"
            )
        group = int(getattr(self.quant_config, "group_size", 0) or 0)
        if group <= 0:
            group = 128
        if bool(getattr(self.quant_config, "desc_act", False)):
            raise NotImplementedError(
                "xiaotu INT4 backend does not support GPTQ desc_act=True "
                "(g_idx reordering); the engine assumes sequential K order."
            )
        if w13.dtype == torch.int32:
            # GPTQ / compressed-tensors packing: nibbles along K, N last. On a
            # little-endian host the int32 -> uint8 view exposes two nibbles per
            # byte with the low nibble = lower K index, which is exactly the
            # engine's [E, 2I, K/2] layout.
            self._require_symmetric_zero_points(layer)
            w13 = torch.empty(w13.transpose(1, 2).shape, dtype=torch.int32).copy_(
                w13.transpose(1, 2)).view(torch.uint8)
            w2 = torch.empty(w2.transpose(1, 2).shape, dtype=torch.int32).copy_(
                w2.transpose(1, 2)).view(torch.uint8)
            s13, s2 = self._scales
            if s13 is not None and s2 is not None:
                # GPTQ scales are [E, K/group, N]; the engine wants [E, N, K/group].
                # Materialise into FRESH buffers with an explicit copy: chaining
                # .transpose().contiguous() can leave the result as a view whose
                # storage is owned elsewhere (observed as PROT_NONE pages once the
                # owner is released -> SIGSEGV inside the engine's weight copy).
                self._scales = (self._fresh_f32(s13.transpose(1, 2)),
                                self._fresh_f32(s2.transpose(1, 2)))
        elif w13.dtype != torch.uint8:
            raise NotImplementedError(
                f"xiaotu INT4 backend expects int32 (GPTQ/compressed-tensors) or "
                f"uint8 (engine layout) weights, got {w13.dtype}"
            )
        self._engine_w13, self._engine_w2 = w13, w2
        self._group_n, self._group_k = 1, group

    @staticmethod
    def _fresh_f32(t):
        out = torch.empty(t.shape, dtype=torch.float32)
        out.copy_(t)
        return out

    def _require_symmetric_zero_points(self, layer) -> None:
        """Reject asymmetric int4: the engine's table is centered at 8.

        GPTQ stores `zero_point - 1` (mainline's AMX repack adds 1 back), so a
        symmetric checkpoint shows nibbles of 7; compressed-tensors may store 8
        directly. Either way the effective zero point must be 8.
        """
        for name in self._z_names:
            qz = getattr(layer, name, None)
            if qz is None:
                continue
            b = qz.detach().to("cpu").contiguous()
            packed = b.dtype in (torch.int32, torch.int16) and b.shape[-1] * 8 == qz.shape[-1]
            if b.dtype != torch.uint8:
                b = b.to(torch.int32).view(torch.uint8)
            lo = (b & 0x0F).to(torch.int16)
            hi = (b >> 4).to(torch.int16)
            if packed:
                lo = lo + 1   # mainline's unpack adds 1 to GPTQ zeros
                hi = hi + 1
            vals = torch.unique(torch.cat((lo.reshape(-1), hi.reshape(-1))))
            vmin, vmax = int(vals.min()), int(vals.max())
            symmetric = bool(torch.all((vals == 8) | (vals == 7)))
            if not symmetric or vmin < 7:
                raise NotImplementedError(
                    f"xiaotu INT4 backend only implements symmetric quantization "
                    f"(effective zero point 8); {name} has zero points in "
                    f"[{vmin}, {vmax}]. Per-group zero points need kernel "
                    "support; see docs/ROADMAP.md (INT4 checkpoint adaptation)."
                )

    def _validate_weights(self, layer, ex_w13, ex_w2) -> None:
        # Engine layout: w13 [E, 2I, H/2] u8, w2 [E, H, I/2] u8 (nibbles along K).
        layout_ok = (
            ex_w13.dtype == torch.uint8
            and ex_w13.dim() == 3
            and ex_w2.dim() == 3
            and ex_w13.shape[2] * 2 == ex_w2.shape[1]
            and ex_w2.shape[2] * 2 == ex_w13.shape[1] // 2
        )
        if not layout_ok:
            raise NotImplementedError(
                "xiaotu INT4/WNA16 backend cannot consume this checkpoint: "
                f"w13 shape={tuple(ex_w13.shape)} dtype={ex_w13.dtype}, "
                f"w2 shape={tuple(ex_w2.shape)} dtype={ex_w2.dtype}. "
                "Expected the byte-packed [E, 2I, K/2] / [E, H, I/2] layout "
                "(GPTQ/compressed-tensors are repacked automatically; AWQ's "
                "N-packed layout is not supported yet). "
                "See docs/KNOWN_LIMITATIONS.md (INT4/WNA16)."
            )


# 格式 → 需要替换的主线 CPU 后端类
_BACKENDS = (
    ("CPUUnquantizedExperts", XiaotuCPUExpertsBF16, "BF16"),
    ("CPUExpertsMxfp4", XiaotuCPUExpertsMxfp4, "MXFP4"),
    ("CPUExpertsFp8", XiaotuCPUExpertsFp8, "FP8"),
    ("CPUExpertsInt4", XiaotuCPUExpertsInt4, "INT4"),
)


def register_mixed_cpu_backend() -> None:
    """混合模式下把主线的 CPU experts 后端换成 xiaotu 引擎实现。

    oracle 对这些后端是**惰性 import** 模块属性,所以替换 `cpu_moe` 里的类即可
    让 oracle 选中我们的实现(而不是仅 Intel AMX 的主线内核)。
    注:oracle 里加一个正式的后端注册表是更干净的上游扩展点;这里先做 OOT 等价物。
    """
    if not mixed_mode_enabled():
        return
    import vllm.model_executor.layers.fused_moe.experts.cpu_moe as cpu_moe

    patched = []
    for attr, cls, label in _BACKENDS:
        if getattr(cpu_moe, attr, None) is cls:
            continue
        if not hasattr(cpu_moe, attr):
            continue  # 该主线版本没有这个后端
        setattr(cpu_moe, attr, cls)
        patched.append(label)
    if patched:
        print(
            "[vllm-xtu-moe] GPU/CPU Mixed: CPU backends -> xiaotu engine "
            f"({', '.join(patched)}; AVX512, no AMX required)",
            flush=True,
        )


__all__ = [
    "mixed_mode_enabled",
    "register_mixed_cpu_backend",
    "XiaotuCPUExpertsBF16",
    "XiaotuCPUExpertsMxfp4",
    "XiaotuCPUExpertsFp8",
    "XiaotuCPUExpertsInt4",
    # backwards-compatible alias (older code imported XiaotuCPUExperts)
    "XiaotuCPUExpertsMxfp4",
]
XiaotuCPUExperts = XiaotuCPUExpertsMxfp4
