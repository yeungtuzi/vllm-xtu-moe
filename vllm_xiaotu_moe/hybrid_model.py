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
import sys
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


def gpu_resident_layers() -> set[int]:
    """常驻 GPU 的专家层(`XIAOTU_MOE_GPU_RESIDENT_LAYERS`,逗号+区间,如 "0-4,10")。

    这些层的专家权重**一次性**放进显存并常驻(不再每步走 CPU 引擎),因此它们
    贡献 0 往返、0 DRAM 权重流量;剩下的层仍在 CPU。对齐 lk-moe 的
    `LVLLM_GPU_RESIDENT_MOE_LAYERS`。默认空 = 全部走 CPU(与旧行为一致)。
    """
    spec = os.environ.get("XIAOTU_MOE_GPU_RESIDENT_LAYERS", "").strip()
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
        except ValueError:
            continue
    return out


def _resident_state() -> dict:
    """**进程级**共享状态(挂在 builtins 上)。

    不能只用模块级全局:实测(2026-09-11)插件的 hybrid_model 在同一进程里被加载成
    **两个模块对象**,各自的模块级全局互不相通 —— 结果同一个 prefix 的常驻层被建了
    两份(`[xiaotu] GPU-resident ...` 同 PID 同 prefix 出现两条),13 层吃掉 41 GiB
    ⇒ KV 8 GiB 分配失败 OOM。builtins 是解释器级的,模块复制也共享。
    """
    import builtins

    st = getattr(builtins, "_xiaotu_resident_state", None)
    if st is None:
        st = {"seen": {}, "used_bytes": 0}
        builtins._xiaotu_resident_state = st
    return st


def _instance_index(prefix: str) -> int:
    """同一 prefix 的第几次构造(0 = 目标模型,>=1 = draft/speculator 副本)。"""
    st = _resident_state()
    n = st["seen"].get(prefix, 0)
    st["seen"][prefix] = n + 1
    return n


def resident_already_built(prefix: str, instance: int = 0) -> bool:
    """这个 prefix 的常驻槽位是否已经建过(进程级)。

    插件模块被复制成两份时会各自调用一次 `finalize_mega_moe_weights`,同一个层就会
    建两份常驻显存(实测 13 层 ×2×1.59 GiB ≈ 41 GiB ⇒ KV OOM)。用进程级集合去重,
    比模块级 `self._resident_slot` 可靠。
    """
    st = _resident_state()
    keys = st.setdefault("built", set())
    # 必须带实例号:目标层与 draft 层的 prefix 相同(都叫 model.layers.0…),
    # 只按 prefix 去重会把 draft 的常驻槽位误判成"已建过"而跳过它。
    key = f"{prefix}#{instance}"
    if key in keys:
        return True
    keys.add(key)
    return False


def _spec_decode_requested() -> bool:
    """是否请求了投机解码(仅用于"预留 draft 常驻额度"这一件事)。

    在层构造期拿不到 `vllm_config`,而 `--speculative-config` 一定在 `sys.argv` 里,
    所以直接看命令行;也允许用 `XIAOTU_SPEC_DECODE=0/1` 显式覆盖。
    """
    flag = os.environ.get("XIAOTU_SPEC_DECODE")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return any("speculative" in a for a in sys.argv)


def _draft_layer_count() -> int:
    """DSpark 起草块的层数(本 checkpoint 是 `mtp.0/1/2` ⇒ 3)。"""
    try:
        return max(1, int(os.environ.get("XIAOTU_DRAFT_LAYERS", "3")))
    except ValueError:
        return 3


def reserve_draft_bytes(per_layer_bytes: int) -> None:
    """把 draft 层要占的显存**先从预算里预留出来**(只做一次)。

    构造顺序是"目标模型在前、draft 在后"(`LLMBaseProposer.load_model()`),若不预留,
    `XIAOTU_MOE_RESIDENT_BUDGET_GB` 会被 42 层目标层吃光,draft 只能落回 CPU ——
    而实测 draft 走 CPU 时投机是**净负**的(7.11 vs 不开投机 10.76 t/s,见 NOTES §245)。
    因此按用户要求:**draft 永远在 GPU**,预算先扣掉它的份额。
    """
    st = _resident_state()
    if st.get("draft_reserved"):
        return
    st["draft_reserved"] = True
    if not _spec_decode_requested():
        st["reserved_bytes"] = 0
        return
    st["reserved_bytes"] = per_layer_bytes * _draft_layer_count()
    print(
        f"[xiaotu] reserved {st['reserved_bytes'] / 2**30:.2f} GiB for "
        f"{_draft_layer_count()} GPU-resident draft layer(s)",
        flush=True,
    )


def resident_budget_ok(nbytes: int, mandatory: bool = False) -> bool:
    """本次常驻是否还在显存预算内(`XIAOTU_MOE_RESIDENT_BUDGET_GB`,0=不限)。

    `XIAOTU_MOE_RESIDENT_BUDGET_GB` 是控制**目标模型**常驻层数的硬旋钮(直接对应
    "塞进 40 GiB")。`mandatory=True` 表示这是 **draft 层**:按用户要求 draft 永远在
    GPU,所以不受预算限制(它很小:3 层 ≈ 4.8 GiB/rank),预算是留给目标层的。
    """
    gb = float(os.environ.get("XIAOTU_MOE_RESIDENT_BUDGET_GB", "0") or 0)
    st = _resident_state()
    if mandatory:
        st["used_bytes"] += nbytes
        return True
    if gb <= 0:
        return True
    reserved = int(st.get("reserved_bytes", 0))
    if st["used_bytes"] + reserved + nbytes > int(gb * (1 << 30)):
        return False
    st["used_bytes"] += nbytes
    return True


def _is_duplicate_model_instance(prefix: str) -> bool:
    """同一 MoE 层 prefix 是否**第二次**被构造(⇒ 是 draft/speculator 模型)。

    vLLM 的 DSpark 起草模型是目标模型的又一份完整实例,层名(prefix)相同;插件
    的模块构造顺序是"目标模型在前、draft 在后"(`LLMBaseProposer.load_model()`
    在目标模型加载完成后调用)。用它把 draft 排除在常驻层之外,省下的显存换更多
    目标模型常驻层(预填充的关键)。同一进程内每个模型实例只构造一次。
    """
    st = _resident_state()
    n = st["seen"].get(prefix, 0)
    st["seen"][prefix] = n + 1
    return n > 0


def ep_shm_enabled() -> bool:
    """EP 部分和改走 /dev/shm 归约(默认开;`XIAOTU_MOE_EP_SHM=0` 关回 NCCL)。

    两个 rank 在同一台机器上,跨 rank 求和用共享内存只需一次 memcpy + 两个自旋
    barrier(几十 µs),而 GPU 的 NCCL all-reduce 实测每层要 +4~6.5 ms。
    """
    return os.environ.get("XIAOTU_MOE_EP_SHM", "1") != "0"


_EP_SHM_FDS: list = []   # 保持 mmap/fd 存活


# 【第 193 轮·修 draft 计算 bug】shm 文件名必须区分**模型实例**。
# 背景:vLLM 的 DSpark 起草模型是目标模型的**又一份完整实例,层名(prefix)相同**
# ⇒ 只用 layer_idx 命名会让"目标第 L 层"和"draft 第 L 层"共用同一个 barrier 头与
# 同一块部分和区域 ⇒ 两套调用序列交织推进世代、互相覆写 ⇒ **draft 的 MoE 算错**
# ⇒ 接受率塌陷(实测 position-0: 26% vs 修好后的 72.5%,见 NOTES §236/§238)。
# 这里用一个**独立**的计数(不碰 _is_duplicate_model_instance 的 seen,避免互相干扰)。
_SHM_INSTANCE: dict = {"seen": {}}


def _shm_instance_tag(prefix: str) -> str:
    n = _SHM_INSTANCE["seen"].get(prefix, 0)
    _SHM_INSTANCE["seen"][prefix] = n + 1
    return "T" if n == 0 else f"D{n}"


def _ep_shm_attach(layer_idx: int, tokens: int, hidden: int, world: int,
                   instance_tag: str = "T"):
    """为某一层创建/打开跨 rank 共享的归约缓冲,返回 (mmap, stride_bytes)。

    布局:[Header 128B][rank0 部分和 stride B][rank1 部分和 stride B ...]
    Header 的前 64B 是两个 atomic(arrive/gen)与 read_done/gen2,由引擎侧解释;
    这里只需要保证两个 rank 打开**同名文件**且大小一致。
    """
    import mmap

    stride = (tokens * hidden * 4 + 63) // 64 * 64
    total = 128 + stride * world
    path = (f"/dev/shm/xiaotu_ep_L{layer_idx}_{instance_tag}"
            f"_{hidden}_{tokens}_{world}.bin")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, total)
        mm = mmap.mmap(fd, total, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        pass
    _EP_SHM_FDS.append((fd, mm, path))
    return mm, stride


def ep_enabled() -> bool:
    """Expert-parallel CPU decode for TP>1 (default on; `XIAOTU_MOE_EP=0` off).

    Without it every rank runs the *full* CPU expert set on the same tokens
    (redundant 2x work at TP=2), which is why TP=2 decode used to be slower than
    single-card. With it each rank owns `E/tp` experts, masks the routing pairs
    that belong to another rank (weight 0 -> the engine skips them, see
    `forward_many`'s `weights[ai] != 0.f` filter), and the partial MoE outputs are
    summed by one TP all-reduce.

    Set `XIAOTU_MOE_REDUNDANT=1` to force the old redundant behaviour.
    """
    if os.environ.get("XIAOTU_MOE_REDUNDANT", "0") == "1":
        return False
    return os.environ.get("XIAOTU_MOE_EP", "1") != "0"

# Layer-index -> MoE module, so layer L can prefetch layer L+1's GPU weights.
_LAYERS: dict[int, "CpuXiaotuMoE"] = {}
_PREBUILD_STARTED = False


_PROF_STATE: dict = {"prof": None, "calls": 0}


def _capture_guard() -> bool:
    """True while vLLM is capturing a CUDA graph.

    Every debug/profiling helper must bail out here: anything that synchronizes
    the stream (``torch.profiler.__enter__``, ``.item()``, ``.cpu()``) inside a
    capture region raises ``cudaErrorStreamCaptureUnsupported`` and invalidates
    the whole capture, which aborts engine startup. Costing a 13-minute reload to
    learn that is not worth it (TRIED_AND_REVERTED R101).
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


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
    if _capture_guard():
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
        prof.export_chrome_trace(f"{path}.r{os.getpid()}")
        print(f"[xiaotu-profile] chrome trace -> {path}.r{os.getpid()}", flush=True)


_PROF_DEC: dict = {"prof": None, "calls": 0}


def _maybe_profile_decode() -> None:
    """Env-gated torch profiler for the DECODE path.

    XIAOTU_TORCH_PROFILE_DECODE=<path> dump(默认抓 2×43 层)――decode 的
    step 由 43 次 D2H→host 回调→H2D + GPU 注意力组成,只有拿到进程内 trace
    才能看清"每层 6 ms 里有多少是内核、多少是等 CPU"。
    """
    path = os.environ.get("XIAOTU_TORCH_PROFILE_DECODE")
    if not path:
        return
    if _capture_guard():          # 捕获期启动 profiler 会作废捕获(见 R101)
        return
    st = _PROF_DEC
    if st["prof"] is None:
        import torch.profiler as tp

        st["prof"] = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]
        )
        st["prof"].__enter__()
    st["calls"] += 1
    limit = int(os.environ.get("XIAOTU_TORCH_PROFILE_DECODE_CALLS", "86"))
    if st["calls"] >= limit:
        prof = st["prof"]
        st["prof"] = None
        prof.__exit__(None, None, None)
        print("[xiaotu-dec-profile]\n" + prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30), flush=True)
        prof.export_chrome_trace(f"{path}.r{os.getpid()}")
        print(f"[xiaotu-dec-profile] chrome trace -> {path}.r{os.getpid()}", flush=True)


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

        def _runner():
            # 阶段 1(纯 CPU 转置)立刻开始:warmup 期间就把转置做完,否则服务路径
            # 会单线程逐层转置+锁页,启动拖成好几分钟。阶段 2(cudaHostRegister)
            # 由 prebuild_pinned_kmajor 自己等 CUDA graph 捕获静止后再做
            # (捕获期间调用会作废捕获甚至让驱动 segfault,见 TRIED_AND_REVERTED R4)。
            prebuild_pinned_kmajor(quads, workers=6)

        threading.Thread(target=_runner, daemon=True).start()
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

    def __init__(self, num_experts, hidden_size, intermediate_size, device="cpu",
                 ep_start=0, ep_local=None, ep_global=None):
        super().__init__()
        # 【NOTES §365】EP:只分配**本 rank 的专家分片**。
        # 主线原先按 config.n_routed_experts(=256) 全量分配 ⇒ 每 rank 138 GB,
        # 而 fork 用 `moe_config.num_local_experts`(=256/TP) ⇒ 只要一半。
        # 同机实测:fork+我们的引擎 105 GB/worker vs 主线+我们的插件 260 GB/worker。
        self.ep_start = int(ep_start)
        self.ep_local = int(ep_local if ep_local is not None else num_experts)
        self.ep_global = int(ep_global if ep_global is not None else num_experts)
        num_experts = self.ep_local          # 下面的 zeros 全部按本地数分配
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
        # global -> local(§365):不属于本 rank 的专家直接跳过(不写、不占内存)。
        try:
            _eid = int(expert_id)
        except Exception:  # noqa: BLE001 - 0-dim tensor / numpy 标量
            _eid = expert_id
        if isinstance(_eid, int):
            _lid = _eid - self.ep_start
            local_ids = [_lid] if 0 <= _lid < self.ep_local else []
        else:
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
        # ---- EP 存储分片(§365)----------------------------------------------
        # 默认跟 fork 一样:每个 rank **只分配/加载** n_routed_experts/TP 个专家。
        # XIAOTU_MOE_EP_SHARD_STORAGE=0 可回到旧的"每 rank 持全部"行为(排障用)。
        self._shard_storage = os.environ.get(
            "XIAOTU_MOE_EP_SHARD_STORAGE", "1") != "0"
        _tp, _rk = 1, 0
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            _tp = int(get_tensor_model_parallel_world_size())
            _rk = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            _tp, _rk = 1, 0
        if not self._shard_storage:
            _tp, _rk = 1, 0
        self._ep_tp, self._ep_rank = max(1, _tp), max(0, _rk)
        self._stor_local = config.n_routed_experts // self._ep_tp
        self._stor_start = self._ep_rank * self._stor_local
        self.n_local_experts = self._stor_local
        self.n_logical_experts = config.n_routed_experts
        self.n_physical_experts = config.n_routed_experts  # 无 redundant
        self.n_local_physical_experts = self._stor_local
        self.n_redundant_experts = 0
        self.experts_start_idx = self._stor_start
        self.experts_end_idx = self._stor_start + self._stor_local
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
        # reduce_results 必须跟主线一致(主线 mega 模式传 self.use_mega_moe=True):
        # RowParallelLinear 在 TP>1 时是**跨 rank 的中段分片**,reduce_results=False
        # 返回的是未归约的局部和 —— TP=1 看不出来,TP=2 就会静默算错(缺另一个 rank
        # 的那一半)。routed 部分由我们在 forward 里单独 all-reduce(EP),共享专家
        # 在这里自归约,与主线的组合方式一致。
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
                reduce_results=True,
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
            ep_start=self._stor_start,
            ep_local=self._stor_local,
            ep_global=config.n_routed_experts,
        )
        self.engine = None
        self._timing = os.environ.get("XIAOTU_TIMING") == "1"
        self._slot = None
        self._slot_device = None
        # ---- GPU 常驻层(见 gpu_resident_layers()) ----
        self._is_draft = False
        try:
            _li = extract_layer_index(prefix)
            # 每个 prefix 恰好登记一次(目标模型先构造,draft 后构造)。
            self._resident_instance = _instance_index(prefix)
            _dup = self._resident_instance > 0
            if _dup:
                # 同一 prefix 第二次出现 = **draft/speculator 模型**。
                # 【用户要求·第 210 轮】draft 层**永远在 GPU**:它只有几层,而且实测
                # draft 走 CPU 时投机是净负的(7.11 vs 10.76 t/s)。因此它的常驻
                # **不依赖** `XIAOTU_MOE_GPU_RESIDENT_LAYERS`,默认就开;只有显式
                # `XIAOTU_MOE_RESIDENT_DRAFT=0` 才关(关掉时会告警,因为那会让投机变负)。
                self._is_draft = True
                self._gpu_resident = (
                    os.environ.get("XIAOTU_MOE_RESIDENT_DRAFT", "1") != "0"
                )
                if not self._gpu_resident:
                    print(
                        f"[xiaotu] WARNING: draft layer {prefix} forced onto CPU by "
                        f"XIAOTU_MOE_RESIDENT_DRAFT=0. Speculative decoding is "
                        f"net-negative with a CPU draft (measured 7.11 vs 10.76 t/s); "
                        f"drop --speculative-config or unset that variable.",
                        flush=True,
                    )
            else:
                self._gpu_resident = _li in gpu_resident_layers()
        except Exception:
            self._gpu_resident = False
        self._resident_slot = None
        # ---- expert parallelism state (see ep_enabled()) ----
        self._ep = False
        self._ep_start = 0
        # 图安全的输出缓冲(见 _graph_out_buffers);cap 在 finalize 里定
        self._out_buf = None
        self._io_buf = None
        self._out_cap = 8
        self._ep_local = int(self.n_routed_experts)
        try:
            # 只登记**第一份**(目标模型)。draft 模型层名相同,若覆盖会让
            # prefetch-ahead(_LAYERS.get(L+1))与 pinned 预构建都作用在 draft 权重上,
            # 主模型的 H2D 重叠失效。
            _li = extract_layer_index(prefix)
            if _li not in _LAYERS:
                _LAYERS[_li] = self
        except Exception:
            pass

    # ---- long-prefill GPU path helpers ----
    def _graph_out_buffers(self, qlen: int, device):
        """CUDA 图安全的输出缓冲(地址**跨调用稳定**)。

        背景(第 18 轮,NOTES §347/§348):CPU 路径原先每次调用都
        `torch.empty(qlen,H,fp32)` 再 `out.to(bf16)`,**每次拿到新地址**。
        图捕获时内核把结果写到捕获期的地址,而 Python 侧返回的是新分配的张量
        ⇒ vLLM 读到的是未初始化数据 ⇒ 序列永远"没前进" ⇒ 每步重做整段预填充
        (`qlen=257 × 200/200`,1136.8 ms/token,实测 1.25 s/token)。
        修法:按**捕获范围内**的最大尺寸预分配两块持久缓冲,fp32 结果先 `copy_`
        成 bf16 再以**视图**返回(视图基址固定,replay 命中同一地址)。

        超过 `cap` 的形状(超长预填充)不会进图,退回临时分配即可,不浪费显存。
        """
        cap = self._out_cap
        if qlen > cap:
            return None
        buf = self._out_buf
        if buf is None or buf[0].device != device:
            f32 = torch.empty(cap, self.hidden_size, dtype=torch.float32, device=device)
            bf16 = torch.empty(cap, self.hidden_size, dtype=torch.bfloat16, device=device)
            buf = (f32, bf16)
            self._out_buf = buf
        # 【第 26 轮】ids/weights 也必须用持久缓冲。
        # fork 的 `_cpu_decode` 是**直接透传** `topk_ids.data_ptr()/topk_weights.data_ptr()`;
        # 我们这边为了做 EP 重映射,每层新建 4 个张量(减/夹/转 int32/where/转 f32)。
        # 在图重放下,图里记录的 D2H 源地址是**捕获期**的地址,而 Python 侧每层给的是新地址
        # ⇒ 引擎读到陈旧/未就绪数据,而且这种"读到的不是刚写的那份"会让异步握手的
        # 完成判定出问题,表现为**周期性等待**(实测主线 28 ms/层 vs fork 0.9 ms/层)。
        rb = self._io_buf
        if rb is None or rb[0].device != device:
            ids = torch.empty(cap, self.top_k, dtype=torch.int32, device=device)
            wts = torch.empty(cap, self.top_k, dtype=torch.float32, device=device)
            mask = torch.empty(cap, self.top_k, dtype=torch.bool, device=device)
            rb = (ids, wts, mask)
            self._io_buf = rb
        return buf[0][:qlen], buf[1][:qlen], rb[0][:qlen], rb[1][:qlen], rb[2][:qlen]

    def _gpu_shard(self):
        """(w13, s13, w2, s2, start, local_E, tp) CPU views for the GPU path."""
        ex = self.experts
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        if tp > 1:
            if getattr(self, "_shard_storage", False):
                # 【§365】参数已经是本 rank 分片,整份返回,不要再切。
                return (
                    ex.w13_weight.data, ex.w13_weight_scale.data,
                    ex.w2_weight.data, ex.w2_weight_scale.data,
                    self._stor_start, self._stor_local, tp,
                )
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
        if self.engine is None or self._gpu_resident:
            return   # 常驻层权重已在显存,不需要 ping-pong 预取
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
        if self._gpu_resident:
            # 【幂等 + 预算】常驻层不设 self.engine,这个钩子可能被调用两次(同一进程
            # 里插件模块被复制成两份时尤其明显)⇒ 必须自己防重复分配,否则显存翻倍。
            if self._resident_slot is not None or resident_already_built(
                self.prefix, getattr(self, "_resident_instance", 0)
            ):
                return
            _nb = self._resident_bytes()
            if _nb <= 0:   # 形状取不到时的兜底估计(打包 fp4 ≈ 12.6MB/专家 + 尺度)
                _nb = int(13.5e6 * max(1, int(self._ep_local)))
            if not self._is_draft:
                # 目标层第一个决预算之前,先把 draft 的份额扣掉(用户要求:
                # draft 永远在 GPU;预算只用来限制目标层)。
                reserve_draft_bytes(_nb)
            _gb = float(os.environ.get("XIAOTU_MOE_RESIDENT_BUDGET_GB", "0") or 0)
            _st = _resident_state()
            print(f"[xiaotu] resident try {self.prefix}: {_nb/2**30:.2f} GiB, "
                  f"used={_st['used_bytes']/2**30:.2f}, "
                  f"reserved={_st.get('reserved_bytes', 0)/2**30:.2f}, "
                  f"budget={_gb:.1f} GiB"
                  f"{' [draft, mandatory]' if self._is_draft else ''}", flush=True)
            if resident_budget_ok(_nb, mandatory=self._is_draft):
                self._build_resident_slot()
                return
            if self._is_draft:
                # draft 是 mandatory ⇒ 正常不会走到这里;真走到了说明显存彻底不够。
                print(
                    f"[xiaotu] ERROR: draft layer {self.prefix} could not be made "
                    f"GPU-resident. Speculative decoding is net-negative with a CPU "
                    f"draft (measured 7.11 vs 10.76 t/s). DISABLE it: drop "
                    f"--speculative-config, or lower XIAOTU_MOE_GPU_RESIDENT_LAYERS / "
                    f"raise XIAOTU_MOE_RESIDENT_BUDGET_GB.",
                    flush=True,
                )
            else:
                print(f"[xiaotu] resident budget exhausted -> {self.prefix} stays on CPU",
                      flush=True)
            # 预算用尽 ⇒ 本层当普通 CPU 层:继续往下把 CPU 引擎建起来,否则前向会
            # 因为 engine=None 而静默返回输入(算错)。
            self._gpu_resident = False
        # ---- 线程池自旋窗口(重要) -------------------------------------------
        # 引擎 worker 在两次调用之间自旋 spin_idle_us 等下一次任务,引擎默认 5 ms。
        # 但解码步里相邻两次调用只隔 ~2-4 ms ⇒ 192 个 worker 在整个解码期间几乎
        # 全在自旋,把整机烧满(实测:解码期间进程占 166-259 核,而真正的 MoE
        # 算力只需 ~21 核),并抢走驱动 GPU 的主线程 ⇒ 每层 `rest` 被推高。
        # 实测(scripts/sweep_spin.sh,负载 50-100 的同程序列,单位 ms/层):
        #   spin=5000: rest 3.30-3.47, period 6.6-7.1, C=1 7.53 tok/s(load 100)
        #   spin=1000: rest 1.27-1.42, period 4.3-4.9, C=1 10.74 tok/s(load 75)
        #   spin= 300: rest 0.94-0.97, period 3.7-3.9, C=1  9.61 tok/s(load 52)
        # 超时后 caller 会 cv_.notify_all() 唤醒(limited 与 non-limited 路径都有,
        # 见 numa_pool.hpp parallel_for 的 fallback),所以不会死锁。
        # 可用环境变量覆盖(例如 prefill 密集场景想要更大自旋)。
        os.environ.setdefault("XIAOTU_MOE_SPIN_IDLE_US", "300")
        # ---- 线程数:按"每 CCD 4-5 核"定(用户给定规则,2026-09-11)-----------
        # 规则:解码性能的最优解是**每个 CCD 只开 4-5 个核心**;更多核心拿不到更多
        # 带宽,反而抢 L3 与散热/power 预算,实测更慢。本机 24 CCD ⇒ 96-120 线程。
        # 佐证:本仓库 process_data/decomp/NUMA_BANDWIDTH_CCD.md(1 CCD ~30 GB/s、
        # 越过 6-8 CCD 后每 CCD 效率下降);LvLLM/lk_moe 官方 README(线程数 = 物理核
        # ÷ GPU 数;声明 L3 命中率 >50%、跨 node 通信 <3%)。
        # 实测(本仓库 NOTES §43/§44):192 线程 8.25-9.98 tok/s(C=1)、176→11.44、
        # 160→11.66、144→12.57、128→12.66 ⇒ 与"少开核"方向一致。
        # ⇒ 默认取 n_ccd × 5(上限 = 核数),用户仍可用 XIAOTU_MOE_THREADS 覆盖。
        if os.environ.get("XIAOTU_MOE_THREADS") is None:
            _ncpu = os.cpu_count() or 8
            _nccd = 0
            try:
                _l3 = set()
                for _c in range(_ncpu):
                    _p = f"/sys/devices/system/cpu/cpu{_c}/cache/index3/id"
                    if os.path.exists(_p):
                        with open(_p) as _f:
                            _l3.add(_f.read().strip())
                _nccd = len(_l3)
            except OSError:
                _nccd = 0
            if _nccd > 0:
                _want = _nccd * 5                     # 每 CCD 5 核
            else:
                _want = max(1, int(_ncpu * 5 / 8))    # 回退:核数 * 5/8
            os.environ["XIAOTU_MOE_THREADS"] = str(max(1, min(_ncpu, _want)))
        import xiaotu_moe

        ex = self.experts
        w13 = ex.w13_weight.detach()
        w2 = ex.w2_weight.detach()
        s13 = ex.w13_weight_scale.detach()
        s2 = ex.w2_weight_scale.detach()

        # ---- expert parallelism: this rank's engine holds only its shard ----
        # The CPU weights are still the FULL set on every rank (the mainline
        # loader is told tp_size=1 for this module), so slicing is a pure pointer
        # offset: the engine copies `expert_num` experts out of the given base
        # pointer, and the slice keeps dim-0 contiguity.
        tp = 1
        if ep_enabled():
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp = int(get_tensor_model_parallel_world_size())
            if tp > 1:
                rank = int(get_tensor_model_parallel_rank())
                if getattr(self, "_shard_storage", False):
                    # 【§365】参数**已经只含本 rank 分片**(分配期就切了)⇒ 绝不能再切一次,
                    # 否则切出空张量。直接沿用分配时记下的 start/local。
                    E = self.n_logical_experts
                    st, L = self._stor_start, self._stor_local
                else:
                    E = int(w13.shape[0])
                    L = E // tp
                    st = rank * L
                    w13 = w13[st:st + L].contiguous()
                    w2 = w2[st:st + L].contiguous()
                    s13 = s13[st:st + L].contiguous()
                    s2 = s2[st:st + L].contiguous()
                self._ep = True
                self._ep_start = st
                self._ep_local = L
                print(
                    f"[xiaotu] EP {self.prefix}: rank {rank}/{tp} owns experts "
                    f"[{st}, {st + L}) of {E}",
                    flush=True,
                )

        cfg = xiaotu_moe.MOEConfigV2()
        # 【v0.2·第 15 轮·NUMA 放置修复】把真实的 TP rank/world 告诉引擎,
        # **只用于 NUMA 权重分片与线程池核表切分**(归约仍走下面的 configure_ep
        # 共享内存,所以必须同时关掉引擎自建的 auto EP,否则重复归约)。
        #
        # 原先硬编码 `1 / 0` ⇒ `moe_v2.hpp` 里 `_world = num_processes = 1` ⇒
        #   `nshard_ = numa_node_count()/_world = 8`,
        #   `rank_node_base_ = process_id * (numnodes/_world) = 0`
        # ⇒ **两个 rank 都把各自那份权重分片绑到全部 8 个 NUMA node** ⇒ 每个 node
        # 承受两个 rank 的量。本机每个 node 只有 193 GB,而单 worker 需要 219–242 GB:
        #   oom-kill: constraint=CONSTRAINT_MEMORY_POLICY, nodemask=0,
        #             task=VLLM::Worker_TP, anon-rss:218866316kB
        # 进程**静默死亡**(无 Traceback)⇒ 表现为"日志只加载到一半就没了"。
        # 同一个 process_id 还决定 `shared_numa_pool(process_id, num_processes)` 的核表,
        # 硬编码 1 会让两个 rank 的线程池都横跨全部 24 CCD 互抢核(与 mode B 的
        # NOTES §334p "compute 双峰 0.41 vs 9.5 ms" 同源)。
        # 修法与 `mixed_experts.py`(mode B)完全一致,两处务必保持同步。
        os.environ.setdefault("XIAOTU_MOE_NO_AUTO_EP", "1")
        _tp, _rank = 1, 0
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            _tp = int(get_tensor_model_parallel_world_size())
            _rank = int(get_tensor_model_parallel_rank())
        except Exception:  # noqa: BLE001
            _tp, _rank = 1, 0
        cfg.num_processes = max(1, _tp)
        cfg.process_id = max(0, _rank)
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
        # V2 引擎(MOE_MXFP4/MOE_FP8/MOE_BF16)的每 token 缓冲是**按调用动态
        # 分配**的(qlen = 本次 forward 的 token 数),不再依赖 group_max_len;
        # group_max_len 只对旧 moe.cpp 引擎的静态缓冲有意义。这里仍然把它设成
        # max_num_batched_tokens,并在两者不一致时给出**信息性**提示(而不是
        # 让用户以为必须把 --max-num-batched-tokens 压到 4096)。
        cfg.group_max_len = int(
            os.environ.get("XIAOTU_GROUP_MAX_LEN", str(max(4096, _mnbt) + 128))
        )
        if _mnbt > cfg.group_max_len:
            print(
                f"[xiaotu] WARNING: max_num_batched_tokens={_mnbt} > "
                f"group_max_len={cfg.group_max_len}; the legacy per-token "
                "buffers would overflow (V2 engine allocates per call, so this "
                "is informational).",
                flush=True,
            )
        # ---- 图安全输出缓冲的上界(见 _graph_out_buffers)------------------
        # 只需要覆盖"会被 CUDA 图捕获"的形状:
        #   * 显式限了 --cudagraph-capture-sizes ⇒ max_cudagraph_capture_size 即上界;
        #   * 否则主线 PIECEWISE 会捕获到 MBT,取 MBT(MBT=256 时仅 6.3 MB/层)。
        # XIAOTU_MOE_GRAPH_OUT_MAX>0 可再兜一层显存(cap×H×(4+2)B×层数×rank)。
        _env_cap = int(os.environ.get("XIAOTU_MOE_GRAPH_OUT_MAX", "0") or 0)
        _cap = int(_mnbt)
        _cc = getattr(self._vllm_config, "compilation_config", None)
        _mcs = getattr(_cc, "max_cudagraph_capture_size", None) if _cc is not None else None
        if _mcs:
            _cap = int(_mcs)
        if _env_cap > 0:
            _cap = min(_cap, _env_cap)
        self._out_cap = max(8, _cap)
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
        # CUDA graph 安全:在**捕获之前**把 pinned 缓冲按最大捕获尺寸预分配好。
        # 捕获期间任何 cudaHostAlloc/cudaFreeHost 都会让 capture 失效
        # (实测:cudaErrorStreamCaptureInvalidated → 引擎初始化直接失败),
        # 所以这里一次性分配,稳态不再扩容;prefill 若需要更大缓冲会在
        # 非捕获路径上扩容(那时 sync+free 是合法的)。
        # ---- EP:部分和走共享内存(替代每层一次 NCCL all-reduce) ----
        self._ep_shm_tokens = 0
        if self._ep and ep_shm_enabled() and hasattr(self.engine, "configure_ep"):
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            _world = int(get_tensor_model_parallel_world_size())
            _rank = int(get_tensor_model_parallel_rank())
            # 覆盖 CPU 路径可能出现的最大 qlen:GPU prefill 阈值以上走 GPU 通路,
            # 所以这里取 max(预分配, 阈值) 即可(两者都可由环境变量调整)。
            _toks = int(os.environ.get("XIAOTU_MOE_EP_SHM_TOKENS", "1024"))
            if _world > 1:
                _mm, _stride = _ep_shm_attach(
                    extract_layer_index(self.prefix), _toks, self.hidden_size, _world,
                    _shm_instance_tag(self.prefix),
                )
                self.engine.configure_ep(_rank, _world, _mm, _stride)
                self._ep_shm_tokens = _toks
                self._ep_shm_mm = _mm
                print(
                    f"[xiaotu] EP-shm {self.prefix}: rank {_rank}/{_world} "
                    f"stride={_stride} tokens={_toks}",
                    flush=True,
                )
        _pre = int(os.environ.get("XIAOTU_CD_PREALLOC_TOKENS", "512"))
        if hasattr(self.engine, "prepare_decode_buffers"):
            try:
                self.engine.prepare_decode_buffers(max(1, _pre), self.top_k)
            except Exception as _e:  # 老 .so 没有该方法时静默跳过
                print(f"[xiaotu] prepare_decode_buffers skipped: {_e}", flush=True)
        # 【第 16 轮更正】原注释说"引擎持有这些参数的内存(达 data_ptr),必须防被
        # 替换/释放" —— 该前提已被 `scripts/test_engine_copies_weights.py` 证伪:
        # 引擎构造时就 memcpy 进自己的 NUMA 分片内存,原地改写调用方缓冲后输出逐位不变。
        # 这里保留强引用**另有原因**:`gpu_prefill._pin_key()` 会把源 storage 存进
        # `_PIN_CACHE`(见 NOTES §346c),GPU 预填充/常驻层路径需要这些 host 张量存活。
        self._w13, self._w2 = w13, w2
        self._s13, self._s2 = s13, s2
        del w13, w2, s13, s2
        print(f"[xiaotu] engine built {self.prefix} "
              f"E={self.n_routed_experts} topk={self.top_k}", flush=True)

    def _resident_bytes(self) -> int:
        """本层本 rank 常驻权重需要多少字节(只看形状,不做拷贝)。"""
        try:
            w13, s13, w2, s2, _st, _L, _tp = self._gpu_shard()
            return int(sum(t.numel() * t.element_size()
                           for t in (w13, s13, w2, s2) if t is not None))
        except Exception:  # noqa: BLE001
            return 0

    def _build_resident_slot(self) -> None:
        """把本层(本 rank 分片)的专家权重一次性放进显存并常驻。

        复用 gpu_prefill 的 K-major + PrefetchSlot 机制:`slot.bufs` 是设备缓冲,
        这里用一次性阻塞 H2D 填好并记录 ready 事件,之后每次 forward 只需
        `gpu_moe_layer(..., slot=self._resident_slot)`,不再有任何 H2D。
        """
        from vllm_xiaotu_moe.gpu_prefill import PrefetchSlot, _pinned_kmajor

        if self._resident_slot is not None:   # 幂等:见 finalize_mega_moe_weights()
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        w13, s13, w2, s2, st, L, tp = self._gpu_shard()
        src = (
            _pinned_kmajor(w13), _pinned_kmajor(s13),
            _pinned_kmajor(w2), _pinned_kmajor(s2),
        )
        slot = PrefetchSlot()
        slot.alloc(src, dev)
        for buf, s_ in zip(slot.bufs, src):
            buf.copy_(s_, non_blocking=False)
        slot.ready = torch.cuda.Event()
        slot.ready.record()
        slot.busy = None
        self._resident_slot = slot
        self._slot_device = dev
        gb = sum(b.numel() for b in slot.bufs) / 2**30
        print(
            f"[xiaotu] GPU-resident {self.prefix}: {gb:.2f} GiB on {dev} "
            f"(tp={tp}, experts={L})",
            flush=True,
        )

    def build_engine(self):
        self.finalize_mega_moe_weights()

    # ---- forward ----
    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self._gpu_resident and self._resident_slot is not None:
            # 常驻层:每步都在 GPU 上算,不需要阈值判断、不需要 CPU 引擎。
            self._resident_forward_kwargs = True
        if self.engine is None and not self._gpu_resident:
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
        _resident = self._gpu_resident and self._resident_slot is not None
        if _resident or (
            _gp_min > 0
            and qlen >= _gp_min
            and not torch.cuda.is_current_stream_capturing()
        ):
            from vllm.distributed import tensor_model_parallel_all_reduce
            from vllm_xiaotu_moe.gpu_prefill import gpu_moe_layer

            # Overlap: kick off the NEXT layer's H2D before doing this layer's
            # kernels, so the DMA runs concurrently with the tensor cores.
            _ov = (not _resident) and os.environ.get("XIAOTU_GPU_PREFETCH_AHEAD", "1") == "1"
            if not _resident:
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
            if _resident:
                slot = self._resident_slot
                w13, s13, w2, s2 = slot.bufs  # 已在显存(K-major),不需再传 host 张量
            else:
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
                if os.environ.get("XIAOTU_SKIP_AR", "0") != "1":
                    gpu_out = tensor_model_parallel_all_reduce(gpu_out)
                # XIAOTU_SKIP_AR=1 仅供**计时诊断**:跳过这条每层跨 rank 归约
                # (MoE 输出 (T,4096) bf16 = 33MB,A100-PCIE 无 NVLink;数值会不完整)。
                # 用来量"每层归约"在 GPU 预填充里占多少;默认关,交付路径不受影响。
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
            if os.environ.get("XIAOTU_DEBUG_L1") == "1" and not _capture_guard():
                print(
                    f"[dbg] {self.prefix} GPU-prefill qlen={qlen} tp={tp} "
                    f"out_mean={gpu_out.abs().mean().item():.3e}",
                    flush=True,
                )
            return gpu_out

        # routed experts:xiaotu 引擎(CPU)。engine.cpu_decode 内部 D2H->CPU forward_many->H2D。
        # 【图契约】优先用地址稳定的持久缓冲(见 _graph_out_buffers);超出捕获范围
        # 的形状(超长预填充)才临时分配。
        _gb = self._graph_out_buffers(qlen, hidden_states.device)
        if _gb is not None:
            out, _out_bf16, _ids_b, _wts_b, _mask_b = _gb
        else:
            out = torch.empty(
                qlen, self.hidden_size, dtype=torch.float32,
                device=hidden_states.device
            )
            _out_bf16 = None
            _ids_b = _wts_b = _mask_b = None
        stream = torch.cuda.current_stream()
        # 引擎 binding 的指针提取只认 numpy 数组或整数 data_ptr()(对 torch 张量返回
        # nullptr → cudaMemcpyAsync(nullptr) → invalid argument),故传 data_ptr()。
        h_bf16 = hidden_states.to(torch.bfloat16)
        if self._ep:
            # Expert-parallel: this rank only owns experts [st, st+L). Pairs that
            # belong to another rank get weight 0 — the engine's assignment scan
            # (`weights[ai] != 0.f`) drops them, so they cost nothing — and their
            # id is remapped into the local range so it can never index OOB.
            _st, _L = self._ep_start, self._ep_local
            if _ids_b is not None:
                # 就地写(地址稳定,图重放安全);见 _graph_out_buffers 的说明。
                _ids_b.copy_(topk_ids)          # int64 -> int32,copy_ 自带转换
                _ids_b.sub_(_st).clamp_(0, _L - 1)
                ids_i32 = _ids_b
                _wts_b.copy_(topk_weights)      # -> float32
                torch.logical_and(topk_ids >= _st, topk_ids < _st + _L, out=_mask_b)
                _wts_b.masked_fill_(~_mask_b, 0.0)
                wts_f32 = _wts_b
            else:
                ids_i32 = (topk_ids - _st).clamp_(0, _L - 1).to(torch.int32)
                _in = (topk_ids >= _st) & (topk_ids < _st + _L)
                wts_f32 = torch.where(
                    _in,
                    topk_weights,
                    torch.zeros((), dtype=topk_weights.dtype,
                                device=topk_weights.device),
                ).to(torch.float32)
        else:
            if _ids_b is not None:
                _ids_b.copy_(topk_ids)
                _wts_b.copy_(topk_weights)
                ids_i32, wts_f32 = _ids_b, _wts_b
            else:
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
        _maybe_profile_decode()
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
        if _out_bf16 is not None:
            # 就地 dtyped copy ⇒ 结果落在**固定地址**的持久缓冲上,返回其视图。
            _out_bf16.copy_(out)
            final_hidden_states = _out_bf16
        else:
            final_hidden_states = out.to(hidden_states.dtype)
        # EP 的跨 rank 求和:默认已在引擎的 host 回调里用 /dev/shm 完成
        # (`XIAOTU_MOE_EP_SHM=1`,qlen ≤ _ep_shm_tokens 时);只有超出该容量
        # (极长 prefill 走 CPU 路径)才回退到 NCCL all-reduce。
        _need_allreduce = self._ep and qlen > self._ep_shm_tokens
        if _need_allreduce:
            from vllm.distributed import tensor_model_parallel_all_reduce

            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states
            )
        if os.environ.get("XIAOTU_DEBUG_L1") == "1" and not _capture_guard():
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
            _sh = self.shared_experts(hidden_states)
            if _out_bf16 is not None:
                # 就地累加(缓冲区是持久视图,不能再产生新张量)
                final_hidden_states.add_(_sh)
            else:
                final_hidden_states = final_hidden_states + _sh
        return final_hidden_states


class HybridDeepseekV4ForCausalLM(dv4_nvidia.DeepseekV4ForCausalLM):
    """OOT 覆盖 DeepseekV4ForCausalLM:decoder 层的 ffn 全部是 CpuXiaotuMoE(CPU MoE + GPU 注意力)。

    复用主线 model_cls=DeepseekV4Model(其层 ffn 因 register() 把 DeepseekV4MoE 换成
    CpuXiaotuMoE 而成为 CPU 版);引擎在 process_weights_after_loading ->
    model.finalize_mega_moe_weights() -> layer.ffn.finalize_mega_moe_weights() 里构造。
    """


def register():
    """vllm.general_plugins 入口:OOT 覆盖 DS-V4 arch(主线零改动)。

    可用 `XIAOTU_OOT_OVERRIDE=0` 关掉本覆盖,改用 **lk/lvllm 同构**的路径:
    主线的 `RoutedExperts` 保持不动,由 `mixed_experts.register_mixed_cpu_backend()`
    把 `Mxfp4MoeBackend.CPU` 指向 `xiaotu_moe` 引擎(等价于上游 PR #56118 的
    `VLLM_EXPERTS_LOAD_DEVICE=cpu` + CPU 后端)。两条路的数值应一致,但 lk 那条
    能原生吃下主线的 cudagraph / prefix caching / chunked prefill / spec decode。
    """
    if os.environ.get("XIAOTU_OOT_OVERRIDE", "1") == "0":
        print(
            "[vllm-xtu-moe] OOT DS-V4 model override DISABLED "
            "(XIAOTU_OOT_OVERRIDE=0) -> mainline RoutedExperts + xiaotu CPU backend",
            flush=True,
        )
        return None

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
