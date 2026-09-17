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
                    # 【§563 修】必须拷**本 rank 的 head shard**,不能总是从第 0 行开始!
                    # 依据上游 `_engram_head_shard_weight_loader`(deepseek_v41/common/engram.py:567):
                    #     shard = loaded_weight.narrow(0, param.engram_vocab_start, param.shape[0])
                    # 即偏移 = `engram_vocab_start`、长度 = 本 rank 的 part 行数。
                    # 表按 tp*dp 个 head shard 切(checkpoint 里是完整表 384,006,168 行,
                    # 每 rank 只有 192,009,666 行)⇒ 原先固定拷 [0:part] 会让**每个 rank 都拿前半张表**,
                    # rank≠0 的 Engram 查表全错 ⇒ 输出改变(greedy A/B 1/5,§562d)。
                    off = int(getattr(dst, "engram_vocab_start", 0) or 0)
                    n_full = int(sl.get_shape()[0])
                    assert off + n <= n_full, (
                        f"engram shard overrun: off={off} n={n} full={n_full}")
                    # 【§564】**内容级校验**(默认开;`XIAOTU_ENGRAM_VERIFY=0` 可关)。
                    # 为什么需要:§563 的 bug 就是"拷了 [0:part] 而不是 [off:off+part]"——
                    # 形状对、不报错、能起服务,只有查表结果错。所以光看日志"streamed ..."
                    # 不足以证明表内容对。这里**逐 chunk 抽验边界行**:把刚拷进 pinned 缓冲的
                    # 第 (a+r) 行与再用 safe_open 直接读 checkpoint 的同一绝对行逐字节比。
                    # 抽"最后一个行"能抓住 chunk 循环的 off-by-one;抽首批的"第一行"能抓住
                    # 全局偏移错。成本:每 chunk 1~2 次小随机读(共 ~100 次/tensor)。
                    verify = os.environ.get("XIAOTU_ENGRAM_VERIFY", "1") == "1"
                    n_checked = n_bad = 0
                    n_chunks = 0
                    for a in range(0, n, chunk_rows):
                        b = min(n, a + chunk_rows)
                        src = sl[off + a:off + b]
                        # ue8m0 尺度以 float8_e8m0fnu 到达,而 param 存 uint8 ⇒ 保持原始字节
                        if src.dtype == torch.float8_e8m0fnu:
                            src = src.view(torch.uint8)
                        dst[a:b].copy_(src)
                        if verify:
                            rows = [b - a - 1] if a else [0, b - a - 1]
                            for r in rows:
                                s2 = sl[off + a + r]
                                if s2.dtype == torch.float8_e8m0fnu:
                                    s2 = s2.view(torch.uint8)
                                n_checked += 1
                                if not torch.equal(dst[a + r], s2):
                                    n_bad += 1
                                    if n_bad <= 3:
                                        print(f"[xtu-engram-verify] MISMATCH {ck} "
                                              f"local_row={a + r} global_row={off + a + r}",
                                              flush=True)
                        n_chunks += 1
                    if verify:
                        verdict = "PASS" if n_bad == 0 else f"FAIL({n_bad})"
                        print(f"[xtu-engram-verify] {ck}: {verdict} "
                              f"checked={n_checked} chunks={n_chunks} "
                              f"vocab_start={off} rows={n}", flush=True)
                        if n_bad:
                            raise RuntimeError(
                                f"engram content mismatch: {n_bad}/{n_checked} rows of {ck}")
                n_ok += 1
                print(f"[xtu-engram-last] streamed {ck} -> {tuple(dst.shape)} "
                      f"(vocab_start={off} of {n_full} rows, "
                      f"{dst.numel() * dst.element_size() / 2**30:.1f} GiB)", flush=True)
            except Exception as exc:  # noqa: BLE001
                # 表内容校验失败**必须炸掉**(fail-closed):表错了服务还能起来,但输出是错的,
                # 静默继续比启动失败危险得多。
                if "engram content mismatch" in str(exc):
                    raise
                print(f"[xtu-engram-last] FAILED to stream {ck}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    return n_ok


def _load_format_is_dummy() -> bool:
    """当前是否 `--load-format dummy`(§565)。

    `dummy` 下**所有**权重都是 1.0/127 的占位,输出本来就无意义,再去 checkpoint
    读 189 GiB 真表纯属浪费启动时间与磁盘。所以此时保留 dummy 填充。
    """
    try:
        from vllm.config.vllm import get_current_vllm_config_or_none

        vc = get_current_vllm_config_or_none()
        lf = getattr(getattr(vc, "load_config", None), "load_format", None)
        return str(lf).lower() == "dummy"
    except Exception:  # noqa: BLE001
        return False


def _materialize_engram_tables(model=None, model_path=None) -> int:
    """把延后的 Engram pinned 大表真正建出来(专家阶段全部结束之后调用)。

    **真实权重支持**(NOTES §474/§476):分配好 pinned 缓冲后,若 checkpoint 里确实有
    Engram 张量,就**分块流式**读进这些缓冲(`_stream_engram_from_ckpt`);只有
    `--load-format dummy`(或 checkpoint 找不到表)才保留 dummy 填充。
    这样 `XIAOTU_ENGRAM_LAST=1` 在真实权重下可用:189 GiB pinned 表**不再与专家阶段叠加**。
    """
    n = 0
    _real = []
    dummy = _load_format_is_dummy()
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
    # 找不到表(或 --load-format dummy)就保留 dummy 填充(见 §474/§476/§565)。
    if _real and model is not None and not dummy:
        _m = [mm for (mm, _w, _s) in _real]
        _ok = _stream_engram_from_ckpt(model, model_path, _m)
        if _ok:
            print(f"[xtu-engram-last] re-loaded {_ok} real Engram tensor(s) from the "
                  f"checkpoint into the pinned tables (no extra peak)", flush=True)
    elif _real and dummy:
        print("[xtu-engram-last] --load-format dummy ⇒ 保留占位填充,"
              "不从 checkpoint 读真表(省 ~189 GiB 磁盘读;§565)", flush=True)
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
                # 【LvLLM(guqiong96/Lvllm)fork 适配,2026-09-16】该 fork 的
                # `mxfp4.Mxfp4MoEMethod.create_weights`(以及同族 fp8/int4)用
                #     device = cuda if current_platform.is_cuda_alike() else "cpu"
                #     if isinstance(layer, RoutedExperts) and not layer.is_gpu_resident_layer:
                #         device = "cpu"
                # 来选张量建在哪 —— **显式 device=**,所以下面那个
                # `with torch.device("cpu")` 对它无效。而它的
                # `is_lk_moe_gpu_resident_layer()` 在 `LVLLM_MOE_NUMA_ENABLED=0` 时
                # **恒返回 True**(`if not is_lk_moe_feature_enabled(): return True`)
                # ⇒ 我们的混合模式下(必须关掉 lk_moe)专家权重会被**建到 GPU 上**,
                # 43 层 × 1.59 GiB/rank ⇒ `torch.OutOfMemoryError`(实测 §500)。
                # 这里把它压成 False,让 fork 自己选 CPU。**只在属性存在时改**,
                # 主线(mainline)没有这个属性 ⇒ 行为逐字不变。
                try:
                    if hasattr(layer, "is_gpu_resident_layer"):
                        layer.is_gpu_resident_layer = False
                except Exception:  # noqa: BLE001
                    pass
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
def _install_lvllm_engine_substitution() -> list[str]:
    """**LvLLM fork(guqiong96/Lvllm)专用**:把 `RoutedExperts` 用的引擎换成我们的。

    为什么需要(§500 实测):LvLLM fork 把 **CPU MoE 的执行硬绑在 lk_moe 上** ——
        routed_experts.py:1841  _cpu_prefill -> self.lk_moe.cpu_prefill(...)
        routed_experts.py:~1770                self.lk_moe = lk_moe.MOE_MXFP4(config, w13,w2,s13,s2,0,0)
    而 `LVLLM_MOE_NUMA_ENABLED=0`(我们的混合模式**必须**关掉 lk_moe)时
    `self.lk_moe = None` ⇒ `AttributeError: 'NoneType' object has no attribute 'cpu_prefill'`。
    ❗主线(mainline)有"换后端类"的缝(`cpu_moe.CPUExpertsMxfp4` 等,见
    `register_mixed_cpu_backend`),**这个 fork 没有**。

    这个 shim 为什么这么小:我们的引擎本来就是 **lk_moe ABI 的等价实现** ——
      * `MOEConfigV2` 的字段名逐个相同(num_processes/process_id/gpu_id/has_gate_proj/
        expert_num/top_k/hidden_size/intermediate_size/max_batch_size/max_num_seqs/stride/
        group_min_len/group_max_len/groupN/groupK/activation_type/swiglu_*/use_gpu_prefill);
      * 构造签名相同 `MOE_MXFP4(cfg, w13, w2, s13, s2, gs13=0, gs2=0)`;
      * `cpu_prefill(qlen, top_k, eids, wts, x, out)` 与
        `gpu_prefill(x, out, ids, wts, qlen, k, stream)` 也相同
        (前者是我们引擎的原生 ABI,后者由 `xiaotu_moe/gpu_prefill_bridge.py` 提供)。
    ⇒ 只要把 fork 模块命名空间里的 `lk_moe` 这个名字指向我们的模块即可。

    安全性(逐条都可验证):
      * 只在 **LvLLM fork** 上生效:判据 = 该模块把 `is_lk_moe_feature_enabled`
        import 进了自己的命名空间(主线没有这个名字);
      * 只在 `mixed_mode_enabled()`(`VLLM_EXPERTS_LOAD_DEVICE=cpu`)时生效;
      * **真的 lk_moe 可用时绝不覆盖**(`self`/模块里的 `lk_moe` 非 None)⇒
        参考实现(lk_moe 自己跑)与"我们的 A/B arm A"都不受影响;
      * 发生任何异常都只记录、不影响启动。
    """
    try:
        import vllm.model_executor.layers.fused_moe.routed_experts as _re
    except Exception as exc:  # noqa: BLE001
        _log(f"skip lvllm engine substitution: {type(exc).__name__}: {exc}")
        return []
    if not hasattr(_re, "is_lk_moe_feature_enabled"):
        return []                      # 主线:没有这个 fork 的挂钩点,什么都不做
    if getattr(_re, "lk_moe", None) is not None:
        return []                      # 真 lk_moe 在场 ⇒ 不抢
    try:
        import xiaotu_moe as _xtu
    except Exception as exc:  # noqa: BLE001
        _log(f"skip lvllm engine substitution (no xiaotu_moe): {exc}")
        return []
    if not all(hasattr(_xtu, a) for a in ("MOEConfigV2", "MOE_MXFP4")):
        _log("skip lvllm engine substitution: xiaotu_moe lacks MOEConfigV2/MOE_MXFP4")
        return []
    _re.lk_moe = _xtu
    return ["routed_experts.lk_moe -> xiaotu_moe(LvLLM fork)"]


# ---------------------------------------------------------------------------
def _install_engram_ablate_shim() -> list[str]:
    """**诊断用**运行时消融:把 Engram 的注入变成"恒等"(§565)。

    为什么需要:要知道"那 196B 条件记忆到底买到了什么",唯一可信的办法是**消融后实测**
    (困惑度/生成质量),而不是断言。消融必须是"同一进程、只差一个开关",否则模型/编译/
    缓存差异都会混进来。

    做法:把 `Engram.forward` 换成"开关文件存在 ⇒ 直接返回输入(不注入)"。
    * **默认不安装**(`XIAOTU_ENGRAM_ABLATE_FILE` 未设时零开销、零行为变化);
    * 给了路径就安装,之后**运行时 `touch`/`rm` 该文件即可切换**,可在同一次服务里 A/B;
    * 每 pass 只有 2 次 `Engram.forward` 调用,`os.path.exists` 的开销可忽略。
    """
    path = os.environ.get("XIAOTU_ENGRAM_ABLATE_FILE")
    if not path:
        return []
    try:
        from vllm.models.deepseek_v41.common.engram import Engram
    except Exception as exc:  # noqa: BLE001
        _log(f"skip engram ablate shim: {type(exc).__name__}: {exc}")
        return []
    if getattr(Engram.forward, "_xtu_ablate", False):
        return []
    orig = Engram.forward

    @functools.wraps(orig)
    def forward(self, hidden_states, hash_ids, token_mask=None):
        if os.path.exists(path):
            # 恒等:等价于"该层不做任何 Engram 注入"。返回输入本身即可,
            # 调用方是 `residual = self.engram(previous_post, ...)`。
            return hidden_states
        return orig(self, hidden_states, hash_ids, token_mask)

    forward._xtu_ablate = True  # type: ignore[attr-defined]
    Engram.forward = forward
    return [f"Engram.forward(ablate via {path})"]


def _install_ced_diag_shim() -> list[str]:
    """③ 的诊断探针(**默认关**,`XIAOTU_CED_DIAG=1` 才装):把 fast-prefill 相关的
    "层名 / 类型 / compress_ratio / KV cache group" 一次性打出来(§575d)。

    为什么必须先问清:上游 `get_kv_sharing_fast_prefill_eligible_layers()` 用
    `get_layers_from_vllm_config(vllm_config, Attention)` 取层,而 V4.1 的注意力类是
    `DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC)` —— **是否属于 `Attention`
    必须实测**;而且 `init_attn_backend` 是用 **`kv_cache_group_spec.layer_names`** 去命中
    eligible 集合的,所以"该填哪个名字"也只有在真机上才看得准。
    **本探针只打印,不改任何行为。**
    """
    if os.environ.get("XIAOTU_CED_DIAG") != "1":
        return []
    try:
        from vllm.v1.worker.gpu import attn_utils as _au
    except Exception as exc:  # noqa: BLE001
        _log(f"skip ced diag shim: {type(exc).__name__}: {exc}")
        return []
    orig = getattr(_au, "init_attn_backend", None)
    if orig is None or getattr(orig, "_xtu_ced_diag", False):
        return []

    def _dump(kv_cache_config, vllm_config):
        import vllm.attention.layer as _al
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase as _ALB,
        )
        try:
            from vllm.config.vllm import get_layers_from_vllm_config as _gl
        except Exception:  # noqa: BLE001
            from vllm.config import get_layers_from_vllm_config as _gl  # type: ignore
        print(f"[ced-diag] kv_sharing_fast_prefill="
              f"{getattr(vllm_config.cache_config, 'kv_sharing_fast_prefill', '?')}",
              flush=True)
        for base_name, base in (("Attention", getattr(_al, "Attention", None)),
                                ("AttentionLayerBase", _ALB)):
            if base is None:
                continue
            try:
                d = _gl(vllm_config, base)
            except Exception as exc:  # noqa: BLE001
                print(f"[ced-diag] {base_name}: get_layers 失败 {exc}", flush=True)
                continue
            print(f"[ced-diag] {base_name}: {len(d)} 个模块", flush=True)
            shown = 0
            for k, m in d.items():
                cr = getattr(m, "compress_ratio", None)
                iks = getattr(m, "is_kv_source", None)
                kst = getattr(m, "kv_sharing_target_layer_name", None)
                if cr is None and iks is None and kst is None:
                    if shown < 3:
                        print(f"[ced-diag]    {k}  (无 CR/源/共享属性)", flush=True)
                        shown += 1
                    continue
                print(f"[ced-diag]    {k}  compress_ratio={cr} is_kv_source={iks} "
                      f"kv_sharing_target={kst!r}", flush=True)
            # 上游 eligible 判决的实际返回
            try:
                el = _au.get_kv_sharing_fast_prefill_eligible_layers(vllm_config)
                print(f"[ced-diag] 上游 eligible={len(el)} {sorted(el)[:6]}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[ced-diag] eligible 判决失败 {type(exc).__name__}: {exc}", flush=True)
        try:
            for g in kv_cache_config.kv_cache_groups[:3]:
                print(f"[ced-diag] kv_cache_group layer_names[:6]="
                      f"{list(g.layer_names)[:6]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ced-diag] groups 读取失败 {exc}", flush=True)

    @functools.wraps(orig)
    def init_attn_backend(kv_cache_config, vllm_config, device, *a, **kw):
        try:
            _dump(kv_cache_config, vllm_config)
        except Exception as exc:  # noqa: BLE001
            print(f"[ced-diag] dump 失败 {type(exc).__name__}: {exc}", flush=True)
        return orig(kv_cache_config, vllm_config, device, *a, **kw)

    init_attn_backend._xtu_ced_diag = True  # type: ignore[attr-defined]
    _au.init_attn_backend = init_attn_backend
    return ["init_attn_backend(ced diag)"]


# 【§604f】**真正要匹配的是"KV 缓存组的真实 layer_names"**,而这个集合只有
# `init_attn_backend(kv_cache_config, ...)` 拿得到。所以在这里接管它,把组名收集下来,
# 供 eligible 判据使用 —— 这样不依赖"名字空间猜测"(实测 SWA 那组就没被覆盖到)。
_CED_GROUP_NAMES: set = set()


def _install_ced_group_name_probe() -> list[str]:
    """接管 `init_attn_backend`,把 `kv_cache_config` 里每个组的真实组名记下来。

    ⚠️ `model_runner`/`speculator` 都是 **`from ... import init_attn_backend`**(直接导入),
    所以必须打在**它们各自的命名空间**上,改 `attn_utils.init_attn_backend` 是无效的。
    """
    if os.environ.get("XIAOTU_CED_FASTPREFILL") != "1":
        return []
    out: list[str] = []
    for modname in ("vllm.v1.worker.gpu.model_runner",
                    "vllm.v1.worker.gpu.spec_decode.speculator"):
        try:
            mod = __import__(modname, fromlist=["init_attn_backend"])
            orig = getattr(mod, "init_attn_backend", None)
            if orig is None or getattr(orig, "_xtu_ced_grp", False):
                continue

            @functools.wraps(orig)
            def init_attn_backend(kv_cache_config, vllm_config, device, *a, **kw):
                try:
                    names = set()
                    for g in kv_cache_config.kv_cache_groups:
                        names.update(getattr(g, "layer_names", ()) or ())
                    if names:
                        _CED_GROUP_NAMES.clear()
                        _CED_GROUP_NAMES.update(names)
                        if os.environ.get("XIAOTU_CED_DIAG") == "1":
                            _log(f"[ced-diag] KV 缓存组真实组名 {len(names)} 个,样例="
                                 f"{sorted(names)[:4]}")
                except Exception as exc:  # noqa: BLE001
                    _log(f"ced 组名收集失败: {type(exc).__name__}: {exc}")
                return orig(kv_cache_config, vllm_config, device, *a, **kw)

            init_attn_backend._xtu_ced_grp = True  # type: ignore
            setattr(mod, "init_attn_backend", init_attn_backend)
            out.append(f"{modname}.init_attn_backend(组名探针)")
        except Exception as exc:  # noqa: BLE001
            _log(f"ced 组名探针 {modname} 装不上: {type(exc).__name__}: {exc}")
    return out


# 【§604c fail-safe】metadata 侧与切片侧之间的**唯一握手**。
# metadata 侧每被调用一次就把 `win` 写成"本步实际改写成的窗口长度"(改写=窗口长,未改写=0);
# 切片侧**必须**看到 `win > 0` 才允许切。这样即使某一层没被换上 FastPrefill 后端
# (⇒ 元数据整步没改写),切片也会**自动停用** —— 从"崩"降级为"不生效"。
_CED_STATE: dict = {"win": 0, "toks": -1, "log": 0, "sliced_attn": None}


def _install_ced_fastprefill_shim() -> list[str]:
    """③ CED 预填充捷径(**默认关**,`XIAOTU_CED_FASTPREFILL=1` 才装,且需 `--kv-sharing-fast-prefill`)。

    复用上游已有的"eligible 层改写 attention metadata + 提前退出"框架,只补两处 V4.1 专属差异:

    A) **eligible 判据**:上游用 `get_layers_from_vllm_config(vllm_config, Attention)`,而 V4.1 的
       `DeepseekV4Attention` 继承的是 `AttentionLayerBase`(**不是** `Attention`)⇒ 上游对 V4.1
       结构上必然返回空集。这里改成用 `AttentionLayerBase` 遍历,筛
       `compress_ratio > 0 and not is_kv_source`(即"读共享压缩 KV、但自己仍有 SWA"的消费者层 = 21..39)。
       **不动 `get_kv_cache_spec`** ⇒ 各层 SWA 缓存照旧分配(§575a 的否决点绕开)。

    B) **保留哪些 query 位置**:上游只保留 `logits_indices`(每请求 1 个位置),而 V4.1 的 decoder 层
       每层有自己的 SWA(窗口 `w_win`),必须保留**最后 `2·w_win−1` 个位置**(§573b 的因果论证)。
       做法:把索引集合扩成"每个请求最后 W 个位置"再**委托给上游原函数** ⇒ 整套
       `query_start_loc`/`seq_lens`/block_table/slot_mapping 的重建逻辑全部免费复用。
    """
    if os.environ.get("XIAOTU_CED_FASTPREFILL") != "1":
        return []
    applied: list[str] = []
    try:
        import vllm.v1.worker.gpu.attn_utils as _au
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase as _ALB,
        )
        from vllm.config.vllm import get_layers_from_vllm_config as _gl
    except Exception as exc:  # noqa: BLE001
        _log(f"skip ced fastprefill shim: {type(exc).__name__}: {exc}")
        return []

    orig_elig = getattr(_au, "get_kv_sharing_fast_prefill_eligible_layers", None)
    if orig_elig is not None and not getattr(orig_elig, "_xtu_ced", False):
        @functools.wraps(orig_elig)
        def get_kv_sharing_fast_prefill_eligible_layers(vllm_config, *a, **kw):
            out = set(orig_elig(vllm_config, *a, **kw))    # 上游结果(对 V4.1 为空)
            try:
                d = _gl(vllm_config, _ALB)
                # 【§577/§578/§579 修】只取 **decoder 半区**(V4.1 = 层 21..39,共 19 层)。
                #
                # ⚠️ **必须按"名字里的层号"判定,不能用注意力模块名去比对**:`init_attn_backend`
                # 比对的键是 **`kv_cache_group_spec.layer_names`**(缓存子模块名,如
                # `...layers.21.attn.<cache>`),而 `get_layers_from_vllm_config(AttentionLayerBase)`
                # 给出的既有注意力模块名、也有各缓存子模块名 ⇒ 用模块名做 `in` 判定的结果
                # **命名空间不一致**、FastPrefill 后端套不上 ⇒ 元数据没被改写而层被切了 ⇒ 实测崩在
                # `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: slot_mapping must not exceed q row count`。
                # 按层号判定对两种命名都成立(多余的名字无害:循环只遍历缓存组里的名字)。
                import re as _re
                try:
                    n_layers = int(getattr(vllm_config.model_config.hf_text_config,
                                           "num_hidden_layers", 0) or 0)
                except Exception:  # noqa: BLE001
                    n_layers = 0
                mid = n_layers // 2 if n_layers else 0
                mine = set()
                for n in d:
                    mo = _re.search(r"layers\.(\d+)\.", n)
                    if not mo:
                        continue
                    idx = int(mo.group(1))
                    if mid and mid < idx < n_layers:
                        mine.add(n)
                # 【§604f】**再并上"KV 缓存组的真实组名"**(由 init_attn_backend 探针收集)。
                # 这是关键:元数据是按**缓存前缀名**(如 `...layers.21.attn.swa_cache`)取用的,
                # 而 `AttentionLayerBase` 那套名字**覆盖不到 SWA 那一组**(实测 SWA 元数据
                # 仍是全长 ⇒ 切片被正确地拦下、CED 不生效)。
                for n in list(_CED_GROUP_NAMES):
                    mo = _re.search(r"layers\.(\d+)\.", n)
                    if mo and mid and mid < int(mo.group(1)) < n_layers:
                        mine.add(n)
                if mine:
                    print(f"[ced-fastprefill] 追加 V4.1 eligible {len(mine)} 个名字"
                          f"(含 KV 缓存组真实组名): {sorted(mine)[:3]} … {sorted(mine)[-1:]}",
                          flush=True)
                out |= mine
            except Exception as exc:  # noqa: BLE001
                _log(f"ced eligible 追加失败: {type(exc).__name__}: {exc}")
            return out

        get_kv_sharing_fast_prefill_eligible_layers._xtu_ced = True  # type: ignore

        # 【§604d 诊断】**真正决定元数据会不会被改写的是"这个层有没有换上 FastPrefill 后端"**。
        # `attn_utils.py:225` 用**模块全局名**调用它 ⇒ 必须打在 `_au` 的命名空间上才生效。
        # 计数为 0 就说明"eligible 集合与 `kv_cache_group_spec.layer_names` 一个都没匹配上"。
        try:
            _orig_cfb = getattr(_au, "create_fast_prefill_custom_backend", None)
            if _orig_cfb is not None and not getattr(_orig_cfb, "_xtu_ced_cnt", False):
                _cfb_n = [0]

                @functools.wraps(_orig_cfb)
                def create_fast_prefill_custom_backend(prefix, backend, *a, **kw):
                    _cfb_n[0] += 1
                    if os.environ.get("XIAOTU_CED_DIAG") == "1" and _cfb_n[0] <= 25:
                        _log(f"[ced-diag] 换上 FastPrefill 后端 #{_cfb_n[0]} prefix={prefix} "
                             f"backend={getattr(backend, '__name__', backend)}")
                    return _orig_cfb(prefix, backend, *a, **kw)

                create_fast_prefill_custom_backend._xtu_ced_cnt = True  # type: ignore
                _au.create_fast_prefill_custom_backend = create_fast_prefill_custom_backend
        except Exception as exc:  # noqa: BLE001
            _log(f"ced cfb 计数探针装不上: {type(exc).__name__}: {exc}")
        _au.get_kv_sharing_fast_prefill_eligible_layers = (
            get_kv_sharing_fast_prefill_eligible_layers)
        applied.append("get_kv_sharing_fast_prefill_eligible_layers(V4.1 判据)")

    try:
        import vllm.v1.attention.backends.utils as _bu
        orig_mk = getattr(_bu, "make_kv_sharing_fast_prefill_common_attn_metadata", None)
    except Exception:  # noqa: BLE001
        orig_mk = None
    if orig_mk is not None and not getattr(orig_mk, "_xtu_ced", False):
        win = int(os.environ.get("XIAOTU_CED_WINDOW", "0") or 0)
        _md_n = [0]

        @functools.wraps(orig_mk)
        def make_kv_sharing_fast_prefill_common_attn_metadata(cam, *a, **kw):
            # 【§579 修】**武装判据必须与"层切片"完全一致**,否则会出现
            # "metadata 被改写成只剩 logits 位置、而层仍喂全长"⇒ 实测崩在
            # `slot_mapping must not exceed q row count`(1256 次)。
            # 故:未武装 ⇒ **原样返回**(不要委托上游 —— 上游会把 query 限制成 logits 位置)。
            try:
                w = win or 255
                idx = _ced_window_indices(cam, w)
            except Exception as exc:  # noqa: BLE001
                _log(f"ced 窗口扩展失败(本步不改写): {type(exc).__name__}: {exc}")
                return cam
            if idx is None:
                _CED_STATE["win"] = 0        # 【fail-safe】本步未改写 ⇒ 切片侧必须停用
                _CED_STATE["toks"] = -1
                # 【§604c 诊断】看清"为什么没武装":`_ced_window_indices` 需要
                # 单请求 + max_query_len > w;这两个属性在这版 vLLM 里叫什么必须实测。
                if os.environ.get("XIAOTU_CED_DIAG") == "1" and _md_n[0] < 10:
                    _md_n[0] += 1
                    _q = getattr(cam, "query_start_loc", None)
                    _log(f"[ced-diag] metadata **未武装** w={w} "
                         f"num_reqs={getattr(cam, 'num_reqs', 'NA')} "
                         f"max_query_len={getattr(cam, 'max_query_len', 'NA')} "
                         f"num_actual_tokens={getattr(cam, 'num_actual_tokens', 'NA')} "
                         f"qsl_shape={None if _q is None else tuple(_q.shape)} "
                         f"attrs={[k for k in dir(cam) if not k.startswith('_')][:24]}")
                return cam                      # 本步未武装:完全走上游"未改写"行为
            _CED_STATE["win"] = int(idx[1])   # 【fail-safe】本步确实改写成了这么长的窗口
            # 【§604d】还记录**本步的令牌数**:切片侧要求"当前层看到的 T"与它**完全相等**,
            # 这样上一步骤留下的陈旧标志绝不会误放行(实测陈旧标志正是崩溃的直接原因)。
            _CED_STATE["toks"] = int(getattr(cam, "num_actual_tokens", -1) or -1)
            if os.environ.get("XIAOTU_CED_DIAG") == "1" and _md_n[0] < 10:
                _md_n[0] += 1
                _log(f"[ced-diag] metadata **已改写** w={w} "
                     f"num_logits_indices={idx[1]} padded_shape={tuple(idx[0].shape)}")
            import dataclasses as _dc
            cam2 = _dc.replace(cam, logits_indices_padded=idx[0],
                               num_logits_indices=idx[1])
            return orig_mk(cam2, *a, **kw)

        make_kv_sharing_fast_prefill_common_attn_metadata._xtu_ced = True  # type: ignore
        _bu.make_kv_sharing_fast_prefill_common_attn_metadata = (
            make_kv_sharing_fast_prefill_common_attn_metadata)
        applied.append(f"make_kv_sharing_fast_prefill_common_attn_metadata(窗口={win or '默认'})")
    applied.extend(_install_ced_group_name_probe())
    return applied


def _ced_window_indices(cam, w: int):
    """把"每请求最后 w 个位置"展平成 padded 索引张量;返回 None = **本步不武装**。

    武装条件(必须与模型侧的层切片一致,§579):
    * **单请求**(多请求 flat batch 不能切"最后 N 行");
    * 本步是 prefill 且 query 长度 > w(`max_query_len > w`);
    ⇒ 只有满足时才改写 metadata;否则原样返回,保证"不切片 ⇒ 不改写"。
    """
    import torch
    qsl = getattr(cam, "query_start_loc", None)
    if qsl is None or qsl.numel() < 2:
        return None
    if int(getattr(cam, "num_reqs", 0) or 0) != 1:
        return None
    if int(getattr(cam, "max_query_len", 0) or 0) <= int(w):
        return None
    n_req = int(qsl.numel()) - 1
    idxs = []
    for r in range(n_req):
        s = int(qsl[r].item()); e = int(qsl[r + 1].item())
        if e <= s:
            continue
        idxs.append(list(range(max(s, e - w), e)))
    if not idxs:
        return None
    flat = [i for g in idxs for i in g]
    pad = getattr(cam, "logits_indices_padded", None)
    n = len(flat)
    if pad is not None and pad.numel() >= n:
        out = pad.clone()
        out[:n] = torch.tensor(flat, dtype=pad.dtype, device=pad.device)
        return out, n
    t = torch.tensor(flat, dtype=torch.int64, device=qsl.device)
    return t, n


def _install_ced_kvinsert_shim() -> list[str]:
    """【§604h】CED 切片后 **KV-insert 的 `slot_mapping` 仍是全长** ⇒ `slot_mapping must not exceed q row count`。

    根因(§604f):`attention.py:869` 的 fused insert 用的是
    `attn_metadata[self.swa_cache_layer.prefix].slot_mapping`,那是一个**独立的** metadata
    (实测它的 builder 不走 `FastPrefillAttentionBuilder.build()`,所以我们改写的
    `logits_indices_padded` 到不了它)。而层已经被切到 `w` 行 ⇒ 长度不匹配。

    做法(**不改 vLLM 文件**):在 `DeepseekV4Attention._fused_qnorm_rope_kv_insert` 外面包一层,
    当"本步确实做过 CED 改写"(`_CED_STATE["win"] > 0`)且 `slot_mapping` 比 `q` 行数长时,
    **把 slot_mapping 切到最后 `q.shape[0]` 项**(浅拷贝 metadata,不动共享对象)。

    语义:`positions`/`q`/`kv` 都已被层切片切成"最后 w 个位置",slot_mapping 的后 w 项正是
    这些位置对应的 cache 槽位 ✓;窗口外的 SWA 条目本就不需要(decoder 层只依赖最后
    `2·w_win−1` 个位置,§573b)。
    """
    if os.environ.get("XIAOTU_CED_FASTPREFILL") != "1":
        return []
    try:
        from vllm.models.deepseek_v41.attention import DeepseekV4Attention as _A
    except Exception as exc:  # noqa: BLE001
        _log(f"skip ced kvinsert shim: {type(exc).__name__}: {exc}")
        return []
    orig = getattr(_A, "_fused_qnorm_rope_kv_insert", None)
    if orig is None or getattr(orig, "_xtu_ced_kv", False):
        return []

    @functools.wraps(orig)
    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        try:
            # 【§604k】**必须是"步级"条件,不能用"逐层"**:实测(§604j)把门控收紧成
            # "只对本层被切片的 attn 生效"后,**层 22..39 也崩了** —— 因为它们收到的是
            # 已被切短的 t(=w)、自己没被切片,但它们的 SWA `slot_mapping` **仍是全长**
            # (那一组 metadata 不走 FastPrefill 改写),所以**同样需要对它们做对齐**。
            # ⇒ 只要"本步做过 CED 改写"(`win>0`),凡 slot_mapping 比 q 行数长的都要对齐。
            if (int(_CED_STATE.get("win", 0)) > 0
                    and isinstance(attn_metadata, dict)):
                swa = getattr(self, "swa_cache_layer", None)
                pfx = getattr(swa, "prefix", None)
                md = attn_metadata.get(pfx) if pfx is not None else None
                sm = getattr(md, "slot_mapping", None)
                qn = int(q.shape[0])
                # 【§604i】**四者行数必须一致**(kernel:1005 `q/kv/position_ids row counts
                # must match`)。层切片只切了"模型的逐 token 入参",而这几个是**在
                # attention 内部**由入参派生/由调用方单独传入的 ⇒ 可能仍是全长。
                # 语义与层切片相同(保留最后 q 行),所以在这里统一按 q 行数对齐。
                if os.environ.get("XIAOTU_CED_DIAG") == "1" and _CED_STATE["log"] < 30:
                    _CED_STATE["log"] += 1
                    _log(f"[ced-diag] 行数 q={qn} kv={int(kv.shape[0])} "
                         f"pos={int(positions.shape[0]) if hasattr(positions, 'shape') else 'NA'} "
                         f"sm={int(sm.numel()) if sm is not None else 'NA'}")
                if hasattr(positions, "shape") and int(positions.shape[0]) > qn > 0:
                    positions = positions[-qn:].contiguous()
                if hasattr(kv, "shape") and int(kv.shape[0]) > qn > 0:
                    kv = kv[-qn:].contiguous()
                if sm is not None and int(sm.numel()) > qn > 0:
                    import copy as _cp
                    nmd = _cp.copy(md)
                    try:
                        object.__setattr__(nmd, "slot_mapping", sm[-qn:].contiguous())
                    except Exception:  # noqa: BLE001
                        setattr(nmd, "slot_mapping", sm[-qn:].contiguous())
                    _nd = dict(attn_metadata)
                    _nd[pfx] = nmd
                    attn_metadata = _nd
                    if os.environ.get("XIAOTU_CED_DIAG") == "1" and _CED_STATE["log"] < 24:
                        _CED_STATE["log"] += 1
                        _log(f"[ced-diag] KV-insert:slot_mapping {int(sm.numel())} → {qn}"
                             f"(按 q 行数对齐)")
        except Exception as exc:  # noqa: BLE001
            _log(f"ced kv-insert 对齐失败(本层走原路径): {type(exc).__name__}: {exc}")
        return orig(self, q, kv, positions, attn_metadata)

    _fused_qnorm_rope_kv_insert._xtu_ced_kv = True  # type: ignore
    _A._fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert
    return ["DeepseekV4Attention._fused_qnorm_rope_kv_insert(CED slot_mapping 对齐)"]


def _install_ced_slice_shim() -> list[str]:
    """③ CED **层循环级切片**(§578):让 decoder 半区的层(21..39)只处理**尾部 token**。

    为什么必须做在"层"上而不是只改 attention metadata(§577c):本机预填充由 **CPU MoE 主导**,
    只限制注意力 query 的话 MLP/MoE 仍会对全部 T 个 token 跑 ⇒ 几乎拿不到收益。

    做法(纯插件,不改主线文件):
    * 层循环 `model.py:706-721` 把每层返回值**串下去**,所以某一层返回尾部尺寸后,
      后续层自然只处理尾部 ✓;
    * 本 shim 包住 `DeepseekV4DecoderLayer.forward`:**该层的 attn 属于 eligible 集合**时,
      把逐 token 入参切到**最后 `window` 行**(`window = XIAOTU_CED_WINDOW`,默认 255 = §573b 的**精确**档);
    * 包住 `DeepseekV4Model.forward`:返回前把隐藏状态**零填充回全长 T**
      (runner 用 `hidden_states[logits_indices]` 取采样位,返回短张量会索引错位)。

    门控(§578b,全部 fail-open:任一不满足就完全走原路径):
    * `XIAOTU_CED_FASTPREFILL=1` 且 非投机解码(aux/DSpark 预热会缺上下文);
    * 本批是**单个连续** prefill(`positions` 连续)—— 多请求 flat batch 不能切"最后 N 行";
    * `T > window`;非 CUDA graph 捕获期;非 profile run。
    """
    if os.environ.get("XIAOTU_CED_FASTPREFILL") != "1":
        return []
    try:
        from vllm.models.deepseek_v41.nvidia import model as _m
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase as _ALB,
        )
        from vllm.config.vllm import get_layers_from_vllm_config as _gl
    except Exception as exc:  # noqa: BLE001
        _log(f"skip ced slice shim: {type(exc).__name__}: {exc}")
        return []
    Layer = getattr(_m, "DeepseekV4DecoderLayer", None)
    Model = getattr(_m, "DeepseekV4Model", None)
    if Layer is None or Model is None:
        return []
    if getattr(Layer.forward, "_xtu_ced_slice", False):
        return []
    try:
        from vllm_xiaotu_moe.gpu_prefill import in_profile_run as _in_prof
    except Exception:  # noqa: BLE001
        def _in_prof():
            return False

    st = {"win": int(os.environ.get("XIAOTU_CED_WINDOW", "0") or 0) or 255,
          "armed": False, "full_t": 0, "hits": 0, "idx": {}, "n": 0}
    orig_layer = Layer.forward
    orig_model = Model.forward
    # 逐 token 的入参名(见 model.py:315 的签名)
    TOK = ("x", "positions", "input_ids", "pre_mix", "post_mix", "res_mix",
           "residual", "engram_hashes", "engram_mask")

    def _eligible(layer) -> bool:
        # 【§604 修】**第一判据 = 层号**,必须与 metadata 侧 §579 的
        # `mid < idx < n_layers` 逐字一致(两侧不一致正是长 prompt 崩的根因)。
        _ix = st.get("idx") or {}
        _i = _ix.get(id(layer))
        if _i is not None and st.get("n"):
            _n = int(st["n"])
            ok = (_n // 2) < _i < _n
            if os.environ.get("XIAOTU_CED_DIAG") == "1" and not ok:
                _log(f"[ced-diag] slice-side layer_idx={_i} eligible=False(层号判据)")
            return ok
        # 兜底(拿不到层号时):旧的属性判据。**注意它实测恒 False**
        # (`layer.attn.kv_source_layer_id` 为 None)⇒ 只在极端情况下才会走到这里,
        # 一旦走到就**必须告警**,因为它会让两侧判据再次不一致。
        attn = getattr(layer, "attn", None)
        if attn is None:
            return False
        src = getattr(attn, "kv_source_layer_id", None)
        srcs = getattr(attn, "kv_source_layers", ()) or ()
        ok = (src is not None and bool(srcs)
              and not getattr(attn, "is_kv_source", False)
              and int(src) == int(max(srcs)))
        if ok is False:
            _log("[ced-diag] ⚠️ 层号判据不可用且属性判据为假 ⇒ 本层不切片;"
                 "若 metadata 侧已改写就会崩(应尽快修)")
        return ok

    @functools.wraps(orig_layer)
    def forward(self, *a, **kw):
        if not st["armed"] or not _eligible(self):
            return orig_layer(self, *a, **kw)
        import inspect
        ba = inspect.signature(orig_layer).bind(self, *a, **kw)
        x = ba.arguments.get("x")
        t = int(x.shape[0]) if hasattr(x, "shape") else 0
        w = st["win"]
        if t <= w:
            return orig_layer(self, *a, **kw)
        # 【§604e】**真正决定切片长度的不是"我们改写的那个 metadata",而是
        # `swa_metadata.slot_mapping`** —— `attention.py:869` 的 fused KV-insert 用的是
        # `attn_metadata[self.swa_cache_layer.prefix].slot_mapping`,它与 query 侧是
        # **两个不同的 metadata 对象**。所以这里直接从 forward context 里把 SWA 侧的
        # slot_mapping 长度读出来,并以它为准;读不到或与窗口不一致就**不切**(安全)。
        _swa_len = -1
        try:
            from vllm.forward_context import get_forward_context as _gfc
            _md = _gfc().attn_metadata
            _attn = getattr(self, "attn", None)
            _swa = getattr(_attn, "swa_cache_layer", None)
            _pfx = getattr(_swa, "prefix", None)
            if os.environ.get("XIAOTU_CED_DIAG") == "1" and _CED_STATE["log"] < 3:
                _CED_STATE["log"] += 1
                _log(f"[ced-diag] SWA 前缀={_pfx!r} | metadata 键({len(_md) if isinstance(_md, dict) else 'NA'})="
                     f"{list(_md.keys())[:6] if isinstance(_md, dict) else _md}")
            if isinstance(_md, dict) and _pfx is not None and _pfx in _md:
                _sm = getattr(_md[_pfx], "slot_mapping", None)
                if _sm is not None:
                    _swa_len = int(_sm.numel())
        except Exception:  # noqa: BLE001
            _swa_len = -1
        # 【§604h】**不再因为 SWA 长度不等就放弃切片**:KV-insert 那一侧由
        # `_install_ced_kvinsert_shim` 把 `slot_mapping` 对齐到 q 行数(浅拷贝,
        # 语义见该函数 docstring)。这里只在诊断下记录长度差,便于审计。
        if _swa_len > 0 and _swa_len != int(w) and os.environ.get("XIAOTU_CED_DIAG") == "1" \
                and _CED_STATE["log"] < 8:
            _CED_STATE["log"] += 1
            _log(f"[ced-diag] 切片照做;SWA slot_mapping 长度={_swa_len} ≠ 窗口 {w} "
                 f"⇒ 由 KV-insert 对齐 shim 处理(t={t})")
        if not (int(_CED_STATE.get("win", 0)) > 0
                and int(_CED_STATE.get("toks", -1)) == int(t)):
            # 【§604c fail-safe】metadata 侧本步**没有**改写 ⇒ 绝不能切,否则
            # slot_mapping(全长)> q 行(窗口长)⇒ `slot_mapping must not exceed q row count`。
            if os.environ.get("XIAOTU_CED_DIAG") == "1" and _CED_STATE["log"] < 5:
                _CED_STATE["log"] += 1
                _log(f"[ced-diag] 切片被 fail-safe 拦住(metadata 本步未改写或令牌数不匹配:"
                     f"win={_CED_STATE.get('win')} toks={_CED_STATE.get('toks')} t={t})⇒ 本步不切片")
            return orig_layer(self, *a, **kw)
        for name in TOK:
            v = ba.arguments.get(name)
            if hasattr(v, "shape") and v is not None and v.shape and int(v.shape[0]) == t:
                ba.arguments[name] = v[-w:]
        if os.environ.get("XIAOTU_CED_DIAG") == "1" and st["hits"] < 30:
            _log(f"[ced-diag] 层**切片** layer_idx={st.get('idx', {}).get(id(self))} "
                 f"t={t} -> {w}")
        st["hits"] += 1
        # 【§604j】把"本层被切片"这件事**只对本层的 attn 模块**生效:
        # 之前用的是**步级** `_CED_STATE["win"]>0`,于是对没被切片的层(22..39,t 已是 w)
        # 也会触发 KV-insert 的对齐 ⇒ 索引到不该动的 SWA 槽位 ⇒ CUDA 非法访存。
        _CED_STATE["sliced_attn"] = id(getattr(self, "attn", None))
        try:
            return orig_layer(*ba.args, **ba.kwargs)
        finally:
            _CED_STATE["sliced_attn"] = None

    forward._xtu_ced_slice = True  # type: ignore[attr-defined]
    Layer.forward = forward
    Model.forward = _wrap_model_forward(orig_model, st)
    return [f"DeepseekV4DecoderLayer.forward(CED 切片, window={st['win']})",
            "DeepseekV4Model.forward(CED 零填充)"]


def _wrap_model_forward(orig_model, st):
    import functools as _ft
    import torch

    @_ft.wraps(orig_model)
    def forward(self, input_ids=None, *a, **kw):
        T = int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
        positions = a[0] if a else kw.get("positions")
        armed = False
        if T > st["win"] and 0 < st["win"]:
            try:
                p = positions
                if p is not None and int(p.shape[0]) == T:
                    armed = (int(p[-1].item()) - int(p[0].item()) == T - 1)
            except Exception:  # noqa: BLE001
                armed = False
        if not armed:
            return orig_model(self, input_ids, *a, **kw)
        st["armed"], st["full_t"] = True, T
        # 【§604 修】**两侧判据必须同源**:metadata 侧(§579)用**层号** `mid<idx<n_layers`,
        # 而切片侧的旧 `_eligible` 读 `layer.attn.kv_source_layer_id` —— 实测该属性是 `None`
        # ⇒ 切片侧**从未切片过任何层**,而 metadata 已被改写 ⇒
        # `slot_mapping must not exceed q row count`(短 prompt 不触发窗口所以看不出来)。
        # 这里在武装时按模型自己的层列表建 `id(layer) -> 层号`,切片侧据此判断。
        try:
            _ls = getattr(self, "layers", None)
            if _ls is not None:
                st["idx"] = {id(l): i for i, l in enumerate(_ls)}
                st["n"] = len(_ls)
        except Exception:  # noqa: BLE001
            st["idx"] = {}
        try:
            out = orig_model(self, input_ids, *a, **kw)
        finally:
            st["armed"] = False

        def _pad(h):
            n = int(h.shape[0])
            if n >= T:
                return h
            return torch.nn.functional.pad(h, (0, 0) * (h.dim() - 1) + (T - n, 0))

        if isinstance(out, tuple):
            return tuple(_pad(o) if hasattr(o, "shape") else o for o in out)
        return _pad(out) if hasattr(out, "shape") else out

    return forward


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
        _install_engram_ablate_shim,
        _install_ced_diag_shim,
        _install_ced_fastprefill_shim,
        _install_ced_slice_shim,
        _install_ced_kvinsert_shim,
        _install_oracle_shims,
        _install_prepack_shims,
        _install_mxfp4_cpu_convert_shim,
        _install_input_ids_shim,
        _install_router_extras_shim,
        _install_gpu_prefill_profile_guard,
        _install_lvllm_engine_substitution,
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
