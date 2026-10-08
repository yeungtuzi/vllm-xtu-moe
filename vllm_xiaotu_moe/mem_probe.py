"""引擎内内存探针 —— 让 **A39 关心的那个量(谷底余量)** 变成可测的量。

═══════════════════════════════════════════════════════════════════════════
 为什么需要它(2026-10-08,B349 / B356 的实测结论)
═══════════════════════════════════════════════════════════════════════════
A39 的崩溃判据是**崩溃【谷底】free**(`aten::new_empty` 要 512 MiB 而只剩 223.5 MiB),
而现有两个手段**都测不到它**:

* ✗ **`nvidia-smi memory.free`**:长 prefill **全程 0 跌幅**(实测 64 s、65 个采样,恒 9,154 MiB)
  —— 因为工作缓冲是从**分配器【已预留】的池**里拿的,设备级 free 根本不动 ✓
* ✗ **引擎的 GPU 预填闸门**:读 `torch.cuda.mem_get_info()`,但**每设备每进程只判一次**
  (42 行重复打印 = 同一个瞬时值)⇒ 之后内存退化它**不会再拒绝** ✗

⇒ ⭐ 所以要测的是**三件一起看**:
    `free`(设备还能给多少) + `reserved − allocated`(**分配器池里还剩多少可用**)
    ⇒ 两者之和 = "再来一次大分配,能拿到多少" ⇒ 与要分配的字节比,才是谷底余量 ✓

═══════════════════════════════════════════════════════════════════════════
 用法
═══════════════════════════════════════════════════════════════════════════
  XIAOTU_MEM_PROBE=1          打开
  XIAOTU_MEM_PROBE_SEC=2.0    常规打印节流(秒);**刷新最低余量时总是打印** ✓

⭐ 本模块**绝不能影响主流程** —— 所有调用点都包 try/except,失败即静默 ✓
"""
from __future__ import annotations

import os
import time

_STATE: dict = {
    "on": None,
    "t_last": 0.0,
    "n": 0,
    "min_free": None,       # GiB,设备级
    "min_avail": None,      # GiB,free + (reserved - allocated)
    "min_at": None,         # (qlen, layer)
    "start_free": None,
}


def enabled() -> bool:
    v = _STATE.get("on")
    if v is None:
        v = os.environ.get("XIAOTU_MEM_PROBE", "0") == "1"
        _STATE["on"] = v
    return bool(v)


def _sec() -> float:
    try:
        return float(os.environ.get("XIAOTU_MEM_PROBE_SEC", "2.0"))
    except Exception:  # noqa: BLE001
        return 2.0


def _read(dev):
    """返回 (free_GiB, reserved_GiB, allocated_GiB, avail_GiB)。"""
    import torch

    free_b, total_b = torch.cuda.mem_get_info(dev)
    reserved = torch.cuda.memory_reserved(dev)
    allocated = torch.cuda.memory_allocated(dev)
    g = 2 ** 30
    free = free_b / g
    resv = reserved / g
    alloc = allocated / g
    # ⭐ 再来一次大分配最多能拿到:设备级 free + 分配器池里未用的那部分
    avail = free + max(0.0, resv - alloc)
    return free, resv, alloc, avail


def sample(tag: str, qlen: int | None = None, layer=None, dev=None) -> None:
    """采一次样。⚠️ 绝不抛异常影响主流程 ✓"""
    try:
        if not enabled():
            return
        import torch

        if dev is None:
            dev = torch.cuda.current_device()
        free, resv, alloc, avail = _read(dev)
        s = _STATE
        if s["start_free"] is None:
            s["start_free"] = free
        s["n"] = int(s["n"]) + 1
        new_min = s["min_avail"] is None or avail < float(s["min_avail"])
        if new_min:
            s["min_avail"] = avail
            s["min_free"] = free
            s["min_at"] = (qlen, layer)
        now = time.perf_counter()
        if new_min or (now - float(s["t_last"])) >= _sec():
            s["t_last"] = now
            flag = " ⭐新低" if new_min else ""
            print(
                "[xtu-mem] %s %-10s qlen=%s layer=%s | free=%.2f reserved=%.2f "
                "alloc=%.2f ⇒ **可用(可再分配)=%.2f GiB**%s"
                % (time.strftime("%H:%M:%S"), tag,
                   "-" if qlen is None else qlen,
                   getattr(layer, "layer_name", layer) if layer is not None else "-",
                   free, resv, alloc, avail, flag),
                flush=True)
    except Exception:  # noqa: BLE001
        pass


def summary(tag: str = "END") -> None:
    """打一行谷底汇总。⚠️ 同样不抛异常。"""
    try:
        if not enabled() or _STATE["min_avail"] is None:
            return
        s = _STATE
        print(
            "[xtu-mem] ===== %s 谷底汇总:采样 %d 次 · 起始 free=%.2f · "
            "**最低可用=%.2f GiB**(free=%.2f)@qlen/layer=%s"
            % (tag, int(s["n"]),
               float(s["start_free"] or 0.0), float(s["min_avail"]),
               float(s["min_free"] or 0.0), s["min_at"]),
            flush=True)
    except Exception:  # noqa: BLE001
        pass
