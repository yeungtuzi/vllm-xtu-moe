"""让**原生(未打补丁)**的 vLLM 主线也能跑混合模式:4 处薄壳 monkey-patch。

背景:混合模式需要主线的 4 处配合(见 `docs/ARCHITECTURE.md` §5):
  1. routed-expert 权重必须建在 CPU(否则 100+ GB 专家在构造时就 OOM);
  2. oracle 必须优先选 CPU 后端(GPU 平台默认不选);
  3. CPU 后端的 AMX 重打包必须跳过(它会破坏原始权重布局,且依赖可能未编译的
     `torch.ops._C.convert_weight_packed`);
  4. 量化方法的 `process_weights_after_loading` 必须通知 experts 后端(fp8/wna16 主线不调);
  5. `oracle/mxfp4.convert_weight_to_mxfp4_moe_kernel_format` 必须为 CPU 后端原样返回
     原始权重(否则 `Mxfp4MoEMethod._setup_kernel` 直接
     `raise ValueError("Unsupported mxfp4_backend ... CPU")`)。
     第 5 处对应上游 PR #56118 的第三段;缺了它,**主线 RoutedExperts 路径根本起不来**
     (实测 `lkpath1`:权重加载完后在 `process_weights_after_loading` 处失败)。

这 4 处目前尚未合并进上游。为了让使用
**只装插件就能用**,这里用 monkey-patch 提供等价行为:

  shim 1  给每个 `FusedMoEMethodBase` 子类的 `create_weights` 套一层
          `with torch.device("cpu")`;对**之后才导入**的量化类用
          `__init_subclass__` 钩子自动补上(量化模块是惰性导入的)。
  shim 2  包住 `oracle/{fp8,mxfp4,int_wna16,unquantized}.py` 的
          `_get_priority_backends`,把 CPU 后端移到队首。
  shim 3  包住 `cpu_moe.prepare_{fp8,mxfp4,int4}_moe_layer_for_cpu`,混合模式下
          原样返回(不做 AMX 重打包)。
  shim 4  包住 `process_weights_after_loading`,调用 experts 后端的同名钩子
          (若源码里已经调过则跳过,避免重复)。

全部是**幂等**的:在已合并相应改动的 vLLM 上重复应用等于 no-op。
关闭:`XIAOTU_MAINLINE_SHIMS=0`。自检:`python -m vllm_xiaotu_moe.mainline_shims`。
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import os

import torch

_SHIM_FLAG = "XIAOTU_MAINLINE_SHIMS"


def mixed_mode_enabled() -> bool:
    """True when experts are configured to live/compute on CPU on a GPU run.

    Reads the env var directly (a stock vLLM has no `envs.VLLM_EXPERTS_LOAD_DEVICE`).
    """
    try:
        from vllm import envs

        if getattr(envs, "VLLM_EXPERTS_LOAD_DEVICE", None) == "cpu":
            return True
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("VLLM_EXPERTS_LOAD_DEVICE", "gpu").lower() == "cpu"


def _log(msg: str) -> None:
    print(f"[vllm-xtu-moe/shims] {msg}", flush=True)


# ---------------------------------------------------------------------------
# shim 1 + 4: quant method classes
# ---------------------------------------------------------------------------
_PWAL_DIAG = {"n": 0, "src": 0}


def _host_mem_mib() -> dict:
    """Read the disjoint host-memory counters from /proc/self/status."""
    out: dict = {}
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                key, _, rest = ln.partition(":")
                if key in ("RssAnon", "RssFile", "RssShmem", "VmRSS"):
                    out[key] = int(rest.split()[0]) >> 10  # MiB
    except Exception:  # noqa: BLE001
        pass
    return out


def _mem_diag_on() -> bool:
    """内存诊断探针是否开启(默认否)。

    这些探针会给**每个模型的每一层**加一次 `/proc/self/status` 读取与若干行打印。
    V4/V4.1 的默认路径必须零额外开销,所以按 env 门控,只有
    `XIAOTU_MEM_DIAG=1` 时才打开(排查内存时用)。
    """
    return os.environ.get("XIAOTU_MEM_DIAG") == "1"


_XTU_CPU_EXPERT_ATTR = "_xiaotu_cpu_expert"


def _install_device_loading_shim() -> list[str]:
    """让 `device_loading_context` **跳过**被标记的 CPU 专家参数。

    主线 `model_loader/utils.py:176-208` 会把这个模块里**所有 CPU 参数搬到
    `target_device`(GPU)**,跑完 `process_weights_after_loading` 再搬回 CPU 并
    **锁页**。对"专家常驻 CPU"的混合模式这是三重伤害(见 NOTES §390):

      * GPU 上要瞬时放下一整层专家(V4.1 = 6.33 GiB/层)⇒ **GPU OOM**
        (实测 `p.data = p.data.to(target_device)` 抛 OutOfMemoryError);
      * 搬回来走 `pin_memory=True` ⇒ 专家权重整体变成**不可回收**的 pinned;
      * 钩子时刻 `p.device` 是 cuda ⇒ 我们的 CPU 引擎拿到 **GPU 指针**去 memcpy ⇒ SIGSEGV。

    这里用一份等价实现替换它,只多一句"被标记则跳过"。未标记的参数行为**逐字不变**。
    """
    try:
        from vllm.model_executor.model_loader import utils as _u
    except Exception as exc:  # noqa: BLE001
        _log(f"skip device_loading_context shim: {type(exc).__name__}: {exc}")
        return []
    orig = getattr(_u, "device_loading_context", None)
    if orig is None or getattr(orig, "_xtu_shim", False):
        return []

    @contextlib.contextmanager
    def device_loading_context(module, target_device):
        if not mixed_mode_enabled() or target_device.type == "cpu":
            with orig(module, target_device) as m:
                yield m
            return
        from vllm import envs as _envs
        from vllm.utils.torch_utils import is_pin_memory_available

        moved: set[str] = set()
        for name, p in module.named_parameters():
            if getattr(p, _XTU_CPU_EXPERT_ATTR, False):
                continue                      # ← 我们的 CPU 专家:留在 CPU
            if p.device.type == "cpu":
                moved.add(name)
                p.data = p.data.to(target_device)
        try:
            yield module
        finally:
            use_pin_memory = (
                is_pin_memory_available()
                and not _envs.VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY
            )
            for name, p in module.named_parameters():
                if name in moved:
                    p.data = torch.empty_like(
                        p.data, device="cpu", pin_memory=use_pin_memory
                    ).copy_(p.data)

    device_loading_context._xtu_shim = True  # type: ignore[attr-defined]
    _u.device_loading_context = device_loading_context
    # 也要替换 loader 侧按名字导入的绑定
    for mod_name in ("vllm.model_executor.model_loader.utils",):
        try:
            importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            pass
    return ["device_loading_context"]


_ENGRAM_LAST_PENDING: list = []


def _engram_last_on() -> bool:
    """是否启用"Engram 最后加载"(IRON_RULES R11)。**默认关闭**。"""
    return os.environ.get("XIAOTU_ENGRAM_LAST") == "1"


def _stream_engram_from_ckpt(model, model_path, pending, chunk_rows: int = 2_000_000):
    """把 checkpoint 里的 Engram 大表**分块流式**读进已物化的 pinned 缓冲。

    为什么不用 `model.load_weights` 再跑一遍:那需要拿到 loader 的 weights iterator,
    而 `process_weights_after_loading` 的钩子上拿不到 loader。这里只涉及 **4 个大张量**
    (`layers.{1,14}.engram.embed.{weight,scale}`),命名映射简单,自己按名取更直接。

    **绝不能 `get_tensor(name)` 整块拿**:`embed.weight` 单层 98 GB,整块读会再吃一份
    内存,正好抵消掉 ENGRAM_LAST 的意义。用 `safe_open(...).get_slice(name)[a:b]`
    分块(默认 2e6 行 ≈ 512 MB/块)拷进目标,峰值只多一个 chunk。

    返回成功流式的张量数;失败只告警、不抛(让服务能起来,便于诊断)。
    """
    import glob
    import json
    import os
    import re

    from safetensors import safe_open

    if not model_path or not os.path.isdir(model_path):
        print(f"[xtu-engram-last] no checkpoint dir at {model_path!r}; cannot re-load",
              flush=True)
        return 0
    files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if not files:
        print(f"[xtu-engram-last] no safetensors under {model_path!r}", flush=True)
        return 0
    # key -> file (读 header 很便宜,shard 数 ~48)
    where = {}
    for f in files:
        try:
            with open(f, "rb") as fh:
                hl = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(hl))
        except Exception:  # noqa: BLE001
            continue
        for k in hdr:
            if "engram" in k and "embed" in k:
                where[k] = f

    name_of = {id(mm): n for n, mm in model.named_modules()}
    n_ok = 0
    for m in pending:
        q = name_of.get(id(m), "")
        mo = re.search(r"layers\.(\d+)\.", q)
        if not mo:
            print(f"[xtu-engram-last] cannot locate layer index for {q!r}; skipped",
                  flush=True)
            continue
        li = mo.group(1)
        for ck, attr in ((f"layers.{li}.engram.embed.weight", "weight"),
                         (f"layers.{li}.engram.embed.scale", "weight_scale_inv")):
            f = where.get(ck)
            dst = getattr(m, attr, None)
            if f is None or dst is None:
                print(f"[xtu-engram-last] missing {ck} (file={f}) — skipped", flush=True)
                continue
            try:
                with safe_open(f, "pt") as fh:
                    sl = fh.get_slice(ck)
                    n = int(dst.shape[0])
                    for a in range(0, n, chunk_rows):
                        b = min(n, a + chunk_rows)
                        dst[a:b].copy_(sl[a:b])
                n_ok += 1
                print(f"[xtu-engram-last] streamed {ck} -> {tuple(dst.shape)} "
                      f"({dst.numel() * dst.element_size() / 2**30:.1f} GiB)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[xtu-engram-last] FAILED to stream {ck}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    return n_ok


def _materialize_engram_tables(model=None, model_path=None) -> int:
    """把延后的 Engram pinned 大表真正建出来(专家阶段全部结束之后调用)。

    **真实权重支持**(NOTES §474/§476):分配好 pinned 缓冲后,若 checkpoint 里确实有
    Engram 张量,就**分块流式**读进这些缓冲(`_stream_engram_from_ckpt`);只有
    `--load-format dummy`(checkpoint 找不到表)才退回 dummy 填充。
    这样 `XIAOTU_ENGRAM_LAST=1` 在真实权重下可用:189 GiB pinned 表**不再与专家阶段叠加**。
    """
    n = 0
    _real = []
    for m in list(_ENGRAM_LAST_PENDING):
        try:
            w = torch.empty(
                m.part_num_embeddings, m.dim,
                dtype=torch.float8_e4m3fn, device="cpu", pin_memory=True,
            )
            s = torch.empty(
                m.part_num_embeddings, m.dim // m.block_size,
                dtype=torch.uint8, device="cpu", pin_memory=True,
            )
            # 真实权重优先:分块流式读真正的表(dummy 才 fill)
            w.fill_(1.0)      # 与主线 set_weight_attrs(dummy_weight_value=1.0) 一致
            s.fill_(127)      # ue8m0 的 1.0 = 指数 127
            _real.append((m, w, s))
            # ⚠️ 不能写 `m.weight.data = w`:占位是 **meta** 张量,而 `nn.Parameter`
            # 的 `.data=` 要求两侧 tensor type 相容(meta vs cpu 会抛
            # "incompatible tensor type")。所以**换掉整个 Parameter**,
            # 并把主线 `set_weight_attrs` 写在 `__dict__` 里的属性(如
            # `dummy_weight_value` / `weight_loader`)原样带过去。
            for attr, t in (("weight", w), ("weight_scale_inv", s)):
                old = getattr(m, attr)
                if isinstance(old, torch.nn.Parameter):
                    new = torch.nn.Parameter(t, requires_grad=old.requires_grad)
                    new.__dict__.update(getattr(old, "__dict__", {}))
                    setattr(m, attr, new)
                else:
                    setattr(m, attr, t)
            n += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[xtu-engram-last] materialize FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
    # 真实权重:把 checkpoint 的真表分块流式灌进刚物化的 pinned 缓冲。
    # 找不到表(如 --load-format dummy)就保留 dummy 填充(见 §474/§476)。
    if _real and model is not None:
        _m = [mm for (mm, _w, _s) in _real]
        _ok = _stream_engram_from_ckpt(model, model_path, _m)
        if _ok:
            print(f"[xtu-engram-last] re-loaded {_ok} real Engram tensor(s) from the "
                  f"checkpoint into the pinned tables (no extra peak)", flush=True)
    del _real
    _ENGRAM_LAST_PENDING.clear()
    if n:
        print(f"[xtu-engram-last] materialized {n} Engram table(s) AFTER the expert "
              f"phase (IRON_RULES R11)", flush=True)
    return n


def _install_engram_materialize_shim() -> list[str]:
    """把 Engram 表的**物化**挂到"专家阶段全部结束之后"。

    上游现成的模型级 post-load 钩子就在 `model_loader/utils.py` 的
    `process_weights_after_loading()` 末尾(`utils.py:167` 那句
    `model.process_weights_after_loading()` 之后)。我们在函数返回前物化,
    此时所有专家层都已完成 `process_weights_after_loading` + 释放。
    """
    if not _engram_last_on():
        return []
    try:
        from vllm.model_executor.model_loader import utils as _u
    except Exception as exc:  # noqa: BLE001
        _log(f"skip engram materialize shim: {type(exc).__name__}: {exc}")
        return []
    orig = getattr(_u, "process_weights_after_loading", None)
    if orig is None or getattr(orig, "_xtu_shim", False):
        return []

    @functools.wraps(orig)
    def process_weights_after_loading(model, model_config, target_device, *a, **kw):
        res = orig(model, model_config, target_device, *a, **kw)
        if _ENGRAM_LAST_PENDING:
            _materialize_engram_tables(
                model, getattr(model_config, "model", None)
            )
        return res

    process_weights_after_loading._xtu_shim = True  # type: ignore[attr-defined]
    _u.process_weights_after_loading = process_weights_after_loading
    # ⚠️ **必须同时替换按名字导入它的模块**。`base_loader.py:13-15` 用的是
    # `from ...utils import process_weights_after_loading`,它持有的是**另一份绑定**;
    # 只改 `utils` 上的属性,调用点仍然走旧函数 —— cellL 里
    # `materialized=0` 就是这么来的(Engram 表再也不会被物化)。
    # 与 shim 5(mxfp4 的两处绑定)是同一类坑。
    rebound = []
    for mod_name in (
        "vllm.model_executor.model_loader.base_loader",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.tensorizer_loader",
    ):
        try:
            m = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001
            continue
        if getattr(m, "process_weights_after_loading", None) is not None:
            setattr(m, "process_weights_after_loading", process_weights_after_loading)
            rebound.append(mod_name.rsplit(".", 1)[-1])
    return [f"process_weights_after_loading(+engram materialize; rebound={rebound})"]


_UPSTREAM_SEG = {"n": 0}


def _seg_sample(tag: str, before: dict) -> dict:
    now = _host_mem_mib()
    if _mem_diag_on():
        d_a = now.get("RssAnon", 0) - before.get("RssAnon", 0)
        d_s = now.get("RssShmem", 0) - before.get("RssShmem", 0)
        if abs(d_a) > 64 or abs(d_s) > 64:      # 只报 >64MiB 的分段,避免刷屏
            print(f"[xtu-seg] l#{_UPSTREAM_SEG['n']} {tag}: "
                  f"dAnon={d_a:+d}MiB dShmem={d_s:+d}MiB", flush=True)
    return now


def _install_upstream_seg_shims() -> list[str]:
    """把上游 `process_weights_after_loading` 再切细,定位每层 +9.18 GiB 匿名内存。

    `[xtu-diag-split]` 已证明这 9.18 GiB 完全落在**上游钩子**里(NOTES §407/§408),
    而该钩子体量很小:`Mxfp4MoEMethod.process_weights_after_loading` → `_setup_kernel`
    → `make_mxfp4_moe_kernel`。这里在三处各采一次样,把它落到具体一行。
    """
    applied: list[str] = []
    # (1) _setup_kernel:类属性,直接换不会踩"按名导入"的坑
    try:
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    except Exception as exc:  # noqa: BLE001
        _log(f"skip _setup_kernel seg: {type(exc).__name__}: {exc}")
        Mxfp4MoEMethod = None
    if Mxfp4MoEMethod is not None:
        orig = Mxfp4MoEMethod.__dict__.get("_setup_kernel")
        if orig is not None and not getattr(orig, "_xtu_shim", False):
            @functools.wraps(orig)
            def _setup_kernel(self, layer, *a, **kw):
                b = _host_mem_mib()
                res = orig(self, layer, *a, **kw)
                _seg_sample("_setup_kernel", b)
                return res

            _setup_kernel._xtu_shim = True  # type: ignore[attr-defined]
            Mxfp4MoEMethod._setup_kernel = _setup_kernel
            applied.append("Mxfp4MoEMethod._setup_kernel")

    # (2) 【已移除】`make_mxfp4_moe_kernel` 的包装:实测会让引擎初始化直接抛
    #     `TypeError: make_mxfp4_moe_kernel() takes from 4 to 5 positional arguments
    #      but 8 were given`,而绑定与签名检查都正常、机制未查明。
    #     它是"锦上添花"的二级细分,不值得为一个未查明的失败冒破坏可跑通路径的风险。
    #     `_setup_kernel` 这一级已足够二分(见 NOTES §409)。若日后要恢复,
    #     务必先用一次真实启动验证,而不是只做 import 级检查。
    return applied


def _install_build_kernel_probe() -> list[str]:
    """探 `Mxfp4MoEMethod._build_moe_kernel`(类属性,安全),定位剩余的匿名内存。"""
    if not _mem_diag_on():
        return []
    try:
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    except Exception:  # noqa: BLE001
        return []
    orig = Mxfp4MoEMethod.__dict__.get("_build_moe_kernel")
    if orig is None or getattr(orig, "_xtu_shim", False):
        return []

    @functools.wraps(orig)
    def _build_moe_kernel(self, layer):
        b = _host_mem_mib()
        res = orig(self, layer)
        _seg_sample("_build_moe_kernel", b)
        return res

    _build_moe_kernel._xtu_shim = True  # type: ignore[attr-defined]
    Mxfp4MoEMethod._build_moe_kernel = _build_moe_kernel
    return ["Mxfp4MoEMethod._build_moe_kernel"]


def _install_engram_last_shim() -> list[str]:
    """构造期只放 meta 占位,把 188.8 GiB pinned 表推迟到专家阶段之后(IRON_RULES R11)。

    分配点:`engram.py:266-296 ParallelEngramEmbedding._allocate_weights()` 的两个
    `torch.empty(..., pin_memory=True)`。它在**模块构造期**就执行,比 `load_weights`
    还早,于是内存最紧的专家阶段白扛 188.8 GiB 不可回收内存(cellK 实测,NOTES §394)。

    **默认关闭**(`XIAOTU_ENGRAM_LAST=1` 才开),保证其它模型行为逐字不变。
    """
    if not _engram_last_on():
        return []
    try:
        from vllm.models.deepseek_v41.nvidia import engram as _e
    except Exception as exc:  # noqa: BLE001
        _log(f"skip engram-last shim: {type(exc).__name__}: {exc}")
        return []
    cls = getattr(_e, "ParallelEngramEmbedding", None)
    if cls is None:
        return []
    orig = cls.__dict__.get("_allocate_weights")
    if orig is None or getattr(orig, "_xtu_shim", False):
        return []

    def _allocate_weights(self):
        # 只对"cpu_offload 但非 dp_shared_memory"这条会分配 188.8 GiB 的路径生效;
        # 其它路径(含 DP 共享)逐字走原实现。
        if (not _engram_last_on()) or (not getattr(self, "cpu_offload", False)) \
                or getattr(self, "dp_shared_memory", False):
            return orig(self)
        _ENGRAM_LAST_PENDING.append(self)
        return (
            torch.empty(self.part_num_embeddings, self.dim,
                        dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(self.part_num_embeddings, self.dim // self.block_size,
                        dtype=torch.uint8, device="meta"),
        )

    _allocate_weights._xtu_shim = True  # type: ignore[attr-defined]
    cls._allocate_weights = _allocate_weights
    return ["ParallelEngramEmbedding._allocate_weights"]


def _install_gpu_prefill_profile_guard() -> list[str]:
    """把 vLLM 的 `profile_run` 变成 GPU prefill 能看到的标志。

    为什么必须挡(NOTES §460):vLLM 用 profile run 那次 forward 的**峰值显存**给
    KV cache 定容。若流式 GPU prefill 在那期间生效,~14 GiB 的逐层 staging 会被算进
    峰值 ⇒ 实测 `Available KV cache memory: -1.57 GiB` ⇒ **服务起不来**。
    挡住之后 KV 按 CPU 路径正常定容,运行时再用预检决定是否真的走 GPU
    (腾不出显存就留在 CPU 并提示用户)。**默认不改变任何既有行为**(阈值默认 0)。
    """
    from vllm_xiaotu_moe.gpu_prefill import install_profile_guard

    return install_profile_guard()


def _release_on() -> bool:
    """释放开关是否打开(与 mixed_experts._release_source_enabled 同一判据)。

    ⚠️ 这里必须门控:下面的"钩子前快照"会**持有**权重张量的引用。若对所有模型
    (含 V4)都抓,就会把它们本可释放的存储多留一份 —— 那正是
    "适配新模型不能破坏已有模型"的红线。
    """
    try:
        from vllm_xiaotu_moe.mixed_experts import _release_source_enabled

        return _release_source_enabled()
    except Exception:  # noqa: BLE001
        return False


def _notify_experts(method, layer) -> None:
    kernel = getattr(method, "moe_kernel", None)
    experts = getattr(kernel, "fused_experts", None)
    fn = getattr(experts, "process_weights_after_loading", None)

    # Observability for the "shard one layer, release one layer" path: without
    # this, XIAOTU_RELEASE_SOURCE produces no output at all and there is no way
    # to tell whether the hook ran, which backend class is in use, or whether
    # the host expert tensors are anonymous or shmem-backed.
    _PWAL_DIAG["n"] += 1
    nbytes = 0
    for nm in ("w13_weight", "w2_weight"):
        t = getattr(layer, nm, None)
        if isinstance(t, torch.Tensor):
            nbytes += t.numel() * t.element_size()
    _PWAL_DIAG["src"] += nbytes
    if _mem_diag_on() and (_PWAL_DIAG["n"] == 1 or _PWAL_DIAG["n"] % 8 == 0):
        m = _host_mem_mib()
        print(
            f"[xtu-diag] pwal#{_PWAL_DIAG['n']} method={type(method).__name__} "
            f"experts={type(experts).__name__} hook={callable(fn)} "
            f"layer_src={nbytes / 2**30:.2f}GiB cum_src={_PWAL_DIAG['src'] / 2**30:.1f}GiB "
            f"RssAnon={m.get('RssAnon', -1)}MiB RssFile={m.get('RssFile', -1)}MiB "
            f"RssShmem={m.get('RssShmem', -1)}MiB",
            flush=True,
        )

    if callable(fn):
        fn(layer)


def _already_notifies(func) -> bool:
    try:
        return "process_weights_after_loading(layer)" in inspect.getsource(func)
    except Exception:  # noqa: BLE001
        return False


def _patch_quant_method_cls(cls) -> list[str]:
    out: list[str] = []
    cw = cls.__dict__.get("create_weights")
    if cw is not None and not getattr(cw, "_xtu_shim", False):
        @functools.wraps(cw)
        def create_weights(self, layer, *a, **kw):
            if mixed_mode_enabled():
                with torch.device("cpu"):
                    res = cw(self, layer, *a, **kw)
                # 给"因混合模式而被建在 CPU 上"的大参数打标记,供
                # `_install_device_loading_shim` 把它们排除在"搬上 GPU"之外。
                # 只标记大张量(≥64 MiB):专家权重是 GB 级,router/bias 之类是小的,
                # 不标记以免影响它们正常的 device 处理。
                for _pn, p in layer.named_parameters(recurse=True):
                    try:
                        _big = p.numel() * p.element_size() >= (64 << 20)
                        # 【NOTES §489】按**角色**判,不能只按**体积**判:
                        # DSpark draft 的 `w2_weight_scale` 只有 47,185,920 B(45 MiB),
                        # 低于下面的 64 MiB 阈值 ⇒ 曾被漏标 ⇒ `device_loading_context`
                        # 把它搬到了 **cuda** ⇒ 引擎按 cfg 算出的长度 memcpy 一个
                        # **设备指针** ⇒ SIGSEGV(无 Python 栈)。
                        # 专家的**权重与其块缩放是同一角色**,必须一起留在 CPU。
                        _is_scale = any(
                            _pn.endswith(_suf) for _suf in
                            ("_scale", "_scale_inv", "weight_scale", "scales")
                        )
                        if _big or _is_scale:
                            setattr(p, _XTU_CPU_EXPERT_ATTR, True)
                    except Exception:  # noqa: BLE001
                        pass
                return res
            return cw(self, layer, *a, **kw)

        create_weights._xtu_shim = True  # type: ignore[attr-defined]
        cls.create_weights = create_weights
        out.append(f"{cls.__name__}.create_weights")

    pwal = cls.__dict__.get("process_weights_after_loading")
    if (
        pwal is not None
        and not getattr(pwal, "_xtu_shim", False)
        and not _already_notifies(pwal)
    ):
        @functools.wraps(pwal)
        def process_weights_after_loading(self, layer, *a, **kw):
            # Split the per-layer memory growth between UPSTREAM's hook
            # (`_setup_kernel` -> make_mxfp4_moe_kernel -> experts/kernel ctor)
            # and OUR hook (Mixin.process_weights_after_loading -> _ensure_engine),
            # so the pinned/shmem growth is attributed to a concrete site.
            b = _host_mem_mib() if _mem_diag_on() else {}
            # 记录 upstream 钩子**前后**的权重指针。若指针变了(或被释放),就证明
            # "pwal 换掉了/搬走了原始存储",这正好解释 cellC 为什么在 pwal 返回后
            # 立刻建引擎会 SIGSEGV,而惰性路径(看到的是稳定后的张量)却没事。
            _pre = {}
            for _nm in (("w13_weight", "w2_weight") if _release_on() else ()):
                _t = getattr(layer, _nm, None)
                if isinstance(_t, torch.Tensor):
                    _pre[_nm] = (_t.data_ptr(), tuple(_t.shape), _t.storage_offset())
            if _pre:
                # 【保活】若 upstream 在 pwal 里换掉/搬走/释放了原始存储,那么 pwal
                # 返回后 `layer.w13_weight` 可能指向一块临时内存(cellC 的 SIGSEGV
                # 正是如此)。把**钩子前**的张量引用存下来:既让源保持可读,
                # 也给 `_ensure_engine` 一个安全的构建来源。
                layer._xiaotu_pre_pwal_src = {
                    _nm: getattr(layer, _nm) for _nm in _pre
                }
            res = pwal(self, layer, *a, **kw)
            m = _host_mem_mib() if _mem_diag_on() else {}
            if _pre and _PWAL_DIAG.get("n", 0) <= 2:
                for _nm, (_p0, _s0, _o0) in _pre.items():
                    _t = getattr(layer, _nm, None)
                    if not isinstance(_t, torch.Tensor):
                        print(f"[xtu-diag-ptr] {_nm}: {_p0:#x} {_s0} off={_o0} "
                              f"-> GONE ({type(_t).__name__})", flush=True)
                        continue
                    _p1, _s1, _o1 = _t.data_ptr(), tuple(_t.shape), _t.storage_offset()
                    if (_p0, _s0, _o0) != (_p1, _s1, _o1):
                        print(f"[xtu-diag-ptr] {_nm}: {_p0:#x} {_s0} off={_o0} "
                              f"-> {_p1:#x} {_s1} off={_o1}  ** REPLACED by pwal **",
                              flush=True)
                    else:
                        print(f"[xtu-diag-ptr] {_nm}: unchanged {_p1:#x} {_s1} off={_o1}",
                              flush=True)
            if mixed_mode_enabled():
                _notify_experts(self, layer)
            a2 = _host_mem_mib() if _mem_diag_on() else {}
            _PWAL_DIAG["split"] = _PWAL_DIAG.get("split", 0) + 1
            _UPSTREAM_SEG["n"] = _PWAL_DIAG["split"]
            if _mem_diag_on() and (
                _PWAL_DIAG["split"] == 1 or _PWAL_DIAG["split"] % 8 == 0
            ):
                print(
                    f"[xtu-diag-split] l#{_PWAL_DIAG['split']} "
                    f"upstream: dShmem={m.get('RssShmem', 0) - b.get('RssShmem', 0):+d}MiB "
                    f"dAnon={m.get('RssAnon', 0) - b.get('RssAnon', 0):+d}MiB | "
                    f"our_hook: dShmem={a2.get('RssShmem', 0) - m.get('RssShmem', 0):+d}MiB "
                    f"dAnon={a2.get('RssAnon', 0) - m.get('RssAnon', 0):+d}MiB",
                    flush=True,
                )
            return res

        process_weights_after_loading._xtu_shim = True  # type: ignore[attr-defined]
        cls.process_weights_after_loading = process_weights_after_loading
        out.append(f"{cls.__name__}.process_weights_after_loading")
    return out


def _walk_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_subclasses(sub)


def _install_quant_method_shims() -> list[str]:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    applied: list[str] = []
    for cls in _walk_subclasses(FusedMoEMethodBase):
        applied += _patch_quant_method_cls(cls)

    if not getattr(FusedMoEMethodBase, "_xtu_subclass_hook", False):
        def __init_subclass__(cls, **kwargs):  # noqa: N807
            try:
                object.__init_subclass__(**kwargs)
            except TypeError:
                pass
            _patch_quant_method_cls(cls)

        FusedMoEMethodBase.__init_subclass__ = classmethod(__init_subclass__)
        FusedMoEMethodBase._xtu_subclass_hook = True  # type: ignore[attr-defined]
        applied.append("FusedMoEMethodBase.__init_subclass__")
    return applied


# ---------------------------------------------------------------------------
# shim 2: oracle priority lists
# ---------------------------------------------------------------------------
_ORACLE_MODULES = (
    "vllm.model_executor.layers.fused_moe.oracle.fp8",
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.fused_moe.oracle.int_wna16",
    "vllm.model_executor.layers.fused_moe.oracle.unquantized",
)


def _cpu_first(backends: list):
    if not backends:
        return backends
    try:
        enum_cls = type(backends[0])
        cpu = next((m for m in enum_cls if m.name == "CPU"), None)
    except Exception:  # noqa: BLE001
        return backends
    if cpu is None:
        return backends
    rest = [b for b in backends if b is not cpu]
    return [cpu] + rest


def _install_oracle_shims() -> list[str]:
    applied: list[str] = []
    for mod_name in _ORACLE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            _log(f"skip {mod_name}: {type(exc).__name__}: {exc}")
            continue
        orig = getattr(mod, "_get_priority_backends", None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue

        @functools.wraps(orig)
        def wrapper(*a, _orig=orig, **kw):
            backends = list(_orig(*a, **kw))
            return _cpu_first(backends) if mixed_mode_enabled() else backends

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(mod, "_get_priority_backends", wrapper)
        applied.append(f"{mod_name.rsplit('.', 1)[-1]}._get_priority_backends")
    return applied


# ---------------------------------------------------------------------------
# shim 3: skip the AMX prepack
# ---------------------------------------------------------------------------
_PREPACK_PASSTHROUGH = {
    # name: indices of the tensors that are passed through unchanged
    "prepare_fp8_moe_layer_for_cpu": ("w13", "w2"),
    "prepare_mxfp4_moe_layer_for_cpu": ("w13", "w2", "w13_scale", "w2_scale"),
    "prepare_int4_moe_layer_for_cpu": (
        "w13_packed", "w2_packed", "w13_scale", "w2_scale", "w13_zeros", "w2_zeros",
    ),
}


def _install_prepack_shims() -> list[str]:
    try:
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe
    except Exception as exc:  # noqa: BLE001
        _log(f"skip prepack shims: {type(exc).__name__}: {exc}")
        return []

    applied: list[str] = []
    for name, params in _PREPACK_PASSTHROUGH.items():
        orig = getattr(cpu_moe, name, None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue
        try:
            sig = inspect.signature(orig)
        except (TypeError, ValueError):
            continue

        @functools.wraps(orig)
        def wrapper(*a, _orig=orig, _sig=sig, _params=params, **kw):
            if not mixed_mode_enabled():
                return _orig(*a, **kw)
            bound = _sig.bind_partial(*a, **kw)
            bound.apply_defaults()
            return tuple(bound.arguments[p] for p in _params)

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(cpu_moe, name, wrapper)
        applied.append(f"cpu_moe.{name}")
    return applied


# ---------------------------------------------------------------------------
# shim 5: CPU 后端跳过 mxfp4 kernel-format 转换
#
# 等价于上游 PR #56118 的第三处改动。上游在 `oracle/mxfp4.py` 里让
# `select_mxfp4_moe_backend()` 直接返回 CPU 后端,并在
# `convert_weight_to_mxfp4_moe_kernel_format()` 里为 CPU 后端原样返回原始权重:
#   "the CPU backend is an out-of-tree engine that consumes the raw
#    [E, 2I, H//2] / [E, H, I//2] uint8 weights and raw e8m0 scales directly,
#    so skip the AMX prepack."
# 主线(本仓 pin 的 commit)还没有这段,于是 `Mxfp4MoEMethod._setup_kernel`
# 会走到 `raise ValueError("Unsupported mxfp4_backend ... Mxfp4MoeBackend.CPU")`。
# 这里补上同样的语义。
#
# 注意:`quantization/mxfp4.py` 是 `from ...oracle.mxfp4 import (...)` **按名字**导入的,
# 所以两个模块的绑定都要替换,只改 oracle 不影响调用点。
_SHIM5_MODULES = (
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.quantization.mxfp4",
)


def _install_mxfp4_cpu_convert_shim() -> list[str]:
    applied: list[str] = []
    for mod_name in _SHIM5_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            _log(f"skip {mod_name}: {type(exc).__name__}: {exc}")
            continue
        orig = getattr(mod, "convert_weight_to_mxfp4_moe_kernel_format", None)
        if orig is None or getattr(orig, "_xtu_shim", False):
            continue

        @functools.wraps(orig)
        def wrapper(mxfp4_backend, layer, w13_weight, w2_weight,
                    w13_weight_scale, w2_weight_scale, w13_bias=None,
                    w2_bias=None, *a, _orig=orig, **kw):
            # CPU 后端 = OOT 引擎,直接吃 checkpoint 原始布局,不做任何重打包。
            _mb = _host_mem_mib() if _mem_diag_on() else {}
            if mixed_mode_enabled() and getattr(mxfp4_backend, "name", "") == "CPU":
                if _mem_diag_on():
                    _m = _host_mem_mib()
                    print(f"[xtu-seg] convert_weight(cpu passthrough): dAnon="
                          f"{_m.get('RssAnon',0)-_mb.get('RssAnon',0):+d}MiB", flush=True)
                return (w13_weight, w2_weight, w13_weight_scale,
                        w2_weight_scale, w13_bias, w2_bias)
            return _orig(
                mxfp4_backend, layer, w13_weight, w2_weight, w13_weight_scale,
                w2_weight_scale, w13_bias, w2_bias, *a, **kw,
            )

        wrapper._xtu_shim = True  # type: ignore[attr-defined]
        setattr(mod, "convert_weight_to_mxfp4_moe_kernel_format", wrapper)
        applied.append(f"{mod_name.rsplit('.', 1)[-1]}."
                       "convert_weight_to_mxfp4_moe_kernel_format")
    return applied


# ---------------------------------------------------------------------------
# shim 6: 把 monolithic 路径丢掉的 `input_ids` 存到 layer 上
#
# DeepSeek-V4 的前 3 层(`config.num_hash_layers=3`)是 **hash MoE**:
#   `DeepseekV4MoE.__init__`: `is_hash_moe = layer_idx < num_hash_layers`
#       ⇒ `gate.tid2eid` 是路由表,`gate.e_score_correction_bias = None`
#   `DeepseekV4MoE.forward` → `self.experts(x, router_logits=x, input_ids=input_ids)`
# 而 hash 路由必须查表:`fused_topk_bias(..., input_tokens=input_ids,
# hash_indices_table=...)`。链路是
#   RoutedExperts.forward_monolithic(x, router_logits, input_ids)
#     → quant_method.apply_monolithic(layer, x, router_logits, input_ids)   ← 收下了
#       → moe_kernel.apply_monolithic(...)                                  ← 没有 input_ids 这个参数
#         → fused_experts.apply(...)                                        ← 于是永远拿不到
# 即**两端都有、中间断了**。这里在断点上把 input_ids 暂存到 layer 上,
# 让 OOT 后端(我们的 experts)能取到,而不必改写整条调用链。
def _patch_apply_monolithic_input_ids(cls) -> list[str]:
    fn = cls.__dict__.get("apply_monolithic")
    if fn is None or getattr(fn, "_xtu_ids_shim", False):
        return []

    @functools.wraps(fn)
    def apply_monolithic(self, layer, x, router_logits, input_ids=None, *a, **kw):
        if input_ids is not None:
            try:
                layer._xiaotu_input_ids = input_ids
            except Exception:  # noqa: BLE001
                pass
        return fn(self, layer, x, router_logits, input_ids, *a, **kw)

    apply_monolithic._xtu_ids_shim = True  # type: ignore[attr-defined]
    cls.apply_monolithic = apply_monolithic
    return [f"{cls.__name__}.apply_monolithic"]


def _install_input_ids_shim() -> list[str]:
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )

    applied = _patch_apply_monolithic_input_ids(FusedMoEMethodBase)
    for cls in _walk_subclasses(FusedMoEMethodBase):
        applied += _patch_apply_monolithic_input_ids(cls)
    return applied


# ---------------------------------------------------------------------------
# shim 7: 把路由 extras(hash 表 / vision bias)也放到 `RoutedExperts` 上
#
# `FusedMoEFactory` 收下 `hash_indices_table` / `bias_vl` / `image_sentinel_lo`,
# 但它们**只**被送进 `create_fused_moe_router()` —— 也就是跑在 runner 上的那个
# router(modular 路径用)。`RoutedExperts.__init__` 只保存 `e_score_correction_bias`,
# **不保存这三样**。
# ⇒ monolithic 后端(我们的 CPU experts)自己在 `_get_router()` 里建 router 时,
#   既拿不到 hash 表也拿不到 vision bias,于是 factory 兜底选 `FusedTopKRouter`
#   ⇒ DeepSeek-V4 的 sqrtsoftplus 直接 `ValueError`。
# 实测诊断行:`router=FusedTopKRouter … bias=no hash=no vl=no`(lkpath4)。
# 这里在 factory 出口把这些 extras 补挂到 `RoutedExperts` 实例上,让
# `mixed_experts._ROUTER_EXTRA_ATTRS` 能取到 ⇒ 选中上游的 `FusedTopKBiasRouter`。
_ROUTER_EXTRAS = ("hash_indices_table", "bias_vl", "image_sentinel_lo")


def _install_router_extras_shim() -> list[str]:
    import sys

    from vllm.model_executor.layers.fused_moe import layer as _layer

    orig = getattr(_layer, "FusedMoEFactory", None)
    if orig is None or getattr(orig, "_xtu_extras_shim", False):
        return []

    @functools.wraps(orig)
    def FusedMoEFactory(*a, **kw):
        mod = orig(*a, **kw)
        if mixed_mode_enabled():
            extras = {n: kw.get(n) for n in _ROUTER_EXTRAS if kw.get(n) is not None}
            if extras:
                try:
                    for m in mod.modules():
                        if type(m).__name__ == "RoutedExperts":
                            for n, v in extras.items():
                                if getattr(m, n, None) is None:
                                    setattr(m, n, v)
                            break
                except Exception:  # noqa: BLE001
                    pass
        return mod

    FusedMoEFactory._xtu_extras_shim = True  # type: ignore[attr-defined]

    # 定义处 + 所有**按名字导入过**它的已加载模块(DS-V4 的 model.py 就是这种)。
    patched = ["fused_moe.layer.FusedMoEFactory"]
    for mod_name, m in list(sys.modules.items()):
        if m is None or m is _layer:
            continue
        try:
            if getattr(m, "FusedMoEFactory", None) is orig:
                setattr(m, "FusedMoEFactory", FusedMoEFactory)
                patched.append(f"{mod_name}.FusedMoEFactory")
        except Exception:  # noqa: BLE001
            continue
    setattr(_layer, "FusedMoEFactory", FusedMoEFactory)
    return patched


# ---------------------------------------------------------------------------
def apply_mainline_shims() -> list[str]:
    """Idempotently install all shims; returns the list of things applied."""
    if not mixed_mode_enabled():
        return []
    if os.environ.get(_SHIM_FLAG, "1") == "0":
        _log(f"disabled via {_SHIM_FLAG}=0")
        return []

    applied: list[str] = []
    for step in (
        _install_quant_method_shims,
        _install_device_loading_shim,
        _install_engram_last_shim,
        _install_upstream_seg_shims,
        _install_build_kernel_probe,
        _install_engram_materialize_shim,
        _install_oracle_shims,
        _install_prepack_shims,
        _install_mxfp4_cpu_convert_shim,
        _install_input_ids_shim,
        _install_router_extras_shim,
        _install_gpu_prefill_profile_guard,
    ):
        try:
            applied += step()
        except Exception as exc:  # noqa: BLE001
            _log(f"{step.__name__} failed: {type(exc).__name__}: {exc}")
    if applied:
        _log(f"mainline shims applied ({len(applied)}): " + ", ".join(applied))
    return applied


def self_check() -> int:
    """Print what the plugin will do on this machine/checkout."""
    print(f"mixed_mode_enabled = {mixed_mode_enabled()}")
    print(f"{_SHIM_FLAG}     = {os.environ.get(_SHIM_FLAG, '1')}")
    try:
        from vllm import envs

        print(f"vllm.envs.VLLM_EXPERTS_LOAD_DEVICE = "
              f"{getattr(envs, 'VLLM_EXPERTS_LOAD_DEVICE', '<absent: stock vLLM>')}")
    except Exception as exc:  # noqa: BLE001
        print(f"vllm.envs unavailable: {exc}")
    try:
        import xiaotu_moe

        print(f"xiaotu_moe variant = {getattr(xiaotu_moe, '__variant__', '?')}")
        print(f"xiaotu_moe version = {getattr(xiaotu_moe, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"xiaotu_moe unavailable: {exc}")
    applied = apply_mainline_shims()
    print(f"shims applied now  = {applied or '(none)'}")
    try:
        from vllm.model_executor.layers.fused_moe.experts import cpu_moe

        for attr in ("CPUUnquantizedExperts", "CPUExpertsMxfp4",
                     "CPUExpertsFp8", "CPUExpertsInt4"):
            cls = getattr(cpu_moe, attr, None)
            print(f"  {attr:26s} -> {cls.__name__ if cls else '<absent>'}")
    except Exception as exc:  # noqa: BLE001
        print(f"cpu_moe unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
