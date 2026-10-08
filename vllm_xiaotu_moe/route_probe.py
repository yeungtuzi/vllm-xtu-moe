"""路由探针(**轻量版**)—— 判"每层驻留 C 个热专家"能省掉多少搬运。

═══════════════════════════════════════════════════════════════════════════
 为什么重写(B374 的教训)
═══════════════════════════════════════════════════════════════════════════
旧版 `observe()` 里做 `ids.detach().to("cpu").reshape(-1).tolist()` ⇒
**每层一次【同步 D2H】** ✗(40 层 × 9 chunk = 360 次)⇒ 实测把预填拖慢 **~11 s(+17%)** ✗✗
⇒ ⭐ 探针本身污染了被测对象 —— 这正是量性能时最忌讳的 ✗

**本版做法(零 D2H)**:
* `observe()`:在 **GPU 端**对该层 top-k ids 做 `bincount`(E 个整数,极便宜 ✓),
  累加进**常驻 GPU 的**每层计数器 ✓ —— **完全不跨 PCIe、不破坏流水** ✓
* `report()`:**只在被调用时**拷一次小直方图(每层 E 个数,几十 KB ✓),
  算 ⭐ **"top-C 覆盖率"** = 命中率最高的 C 个专家覆盖了多少比例的**被路由槽位** ✓
  ⇒ 这是**静态驻留缓存命中率的【上界】** ✓(真实缓存只会更低 ✓)

用法:
  XIAOTU_ROUTE_PROBE=1            打开
  XIAOTU_ROUTE_PROBE_C=64         每层驻留槽数
  XIAOTU_ROUTE_PROBE_SEC=30       上报节流(秒)—— **只有上报才 D2H** ✓
⚠️ 绝不抛异常影响主流程 ✓
"""
from __future__ import annotations

import os
import time

_S: dict = {"on": None, "c": None, "sec": None, "t": 0.0}
_COUNTS: dict = {}      # layer_key -> GPU int64[E] 计数(常驻 GPU ✓)
_NAMES: dict = {}
_STEPS: dict = {}
_FIRST: dict = {}   # ⭐ B380:每层【第一次观察】的计数(模拟"首个 chunk 学到的热集")✓


def enabled() -> bool:
    if _S["on"] is None:
        _S["on"] = os.environ.get("XIAOTU_ROUTE_PROBE", "0") == "1"
        try:
            _S["c"] = int(os.environ.get("XIAOTU_ROUTE_PROBE_C", "64"))
            _S["sec"] = float(os.environ.get("XIAOTU_ROUTE_PROBE_SEC", "30"))
        except Exception:  # noqa: BLE001
            _S["c"], _S["sec"] = 64, 30.0
        if _S["on"]:
            print(f"[xtu-route] probe armed(轻量版:C={_S['c']},节流={_S['sec']}s,零 D2H)✓", flush=True)
    return bool(_S["on"])


def _ids_of(routed):
    if isinstance(routed, (tuple, list)) and len(routed) >= 2:
        return routed[1]
    return getattr(routed, "topk_ids", None)


def observe(self, routed, layer=None) -> None:
    """⭐ 只做 GPU 端累加,**不跨 PCIe** ✓"""
    try:
        if not enabled():
            return
        import torch

        # ⭐⭐ B378:**图捕获期间绝不分配、绝不发内核** ✗
        #    实测:探针位置修对后真跑进捕获期 ⇒ `bincount`/`zeros`/`cat` 让捕获作废 ⇒
        #    `cudaStreamCaptureStatusInvalidated INTERNAL ASSERT FAILED` ⇒ **实例起不来** ✗
        #    (旧版从没崩,只是因为它**从没被调用过** —— 它挂在死分支里 ✗)
        if torch.cuda.is_current_stream_capturing():
            return

        ids = _ids_of(routed)
        if ids is None or not isinstance(ids, torch.Tensor) or not ids.numel():
            return
        key = getattr(self, "layer_name", None) or ("mod%d" % id(self))
        flat = ids.reshape(-1).to(torch.int64)
        mx = int(flat.max()) + 1
        cnt = _COUNTS.get(key)
        if cnt is None:
            cnt = torch.zeros(mx, dtype=torch.int64, device=ids.device)
            _COUNTS[key] = cnt
            _NAMES[key] = str(key)
        elif mx > cnt.numel():                      # 扩容(仍在 GPU 端 ✓)
            cnt = torch.cat([cnt, torch.zeros(mx - cnt.numel(), dtype=torch.int64,
                                              device=ids.device)])
            _COUNTS[key] = cnt
            # ⭐⭐ B382:**`_FIRST` 必须同步扩容** ✗ 否则 `report()` 里 `cnt - f` 形状不匹配
            #     ⇒ 抛异常 ⇒ 被 `try/except` 吞掉 ⇒ `tot_r` 恒 0 ⇒ 恒报"待后续chunk"
            #     (实测真机 43 层全如此 ⇒ **这就是根因** ✓;语义上新 id 在首 chunk 计数为 0 ✓)
            _f0 = _FIRST.get(key)
            if _f0 is not None and _f0.numel() < mx:
                _FIRST[key] = torch.cat([_f0, torch.zeros(mx - _f0.numel(), dtype=torch.int64,
                                                          device=ids.device)])
        cnt += torch.bincount(flat, minlength=cnt.numel())[:cnt.numel()]
        # ⭐ B380:把【第一次】观察单独留一份 ⇒ 用来模拟"首个 chunk 学到的热集,之后常驻" ✓
        if key not in _FIRST:
            _FIRST[key] = torch.bincount(flat, minlength=cnt.numel())[:cnt.numel()].clone()
        _STEPS[key] = int(_STEPS.get(key, 0)) + 1

        now = time.perf_counter()
        if now - float(_S["t"]) >= float(_S["sec"] or 30):
            _S["t"] = now
            report()
    except Exception:  # noqa: BLE001
        pass


def report() -> None:
    """⭐ 只有这里才 D2H(一次,几十 KB),并算 top-C 覆盖率 ✓"""
    try:
        if not _COUNTS:
            return
        import torch

        cap = int(_S["c"] or 64)
        caps = sorted({4, 8, 16, 32, cap})
        # ⭐ B378:一次给出【C 扫描曲线】—— 因为显存只养得起几个/层,而探针只报一个 C 没用 ✓
        acc = {c: 0.0 for c in caps}      # 每层 top-c 命中的槽位总和
        tot_all = 0.0
        nlayer = 0
        for k, cnt in _COUNTS.items():
            v = cnt.float()
            tot = float(v.sum())
            if tot <= 0:
                continue
            nlayer += 1
            tot_all += tot
            sv, _ = torch.sort(v, descending=True)
            cs = torch.cumsum(sv, 0)
            for c in caps:
                acc[c] += float(cs[min(c, cs.numel()) - 1]) if cs.numel() else 0.0
        if tot_all <= 0:
            return
        print(f"[xtu-route] ==== ⭐ top-C 覆盖率({nlayer} 层)====", flush=True)
        for c in caps:
            gib = nlayer * c * 8.8 / 1024
            # ⭐ oracle 上界:用【累积】计数选热集、覆盖累积 ✓
            g = 100.0 * acc[c] / tot_all
            # ⭐⭐ B380 实际值:用【第 1 个 chunk】学到的热集,覆盖【后续】槽位 ✓
            #    (= 可实现方案:首个 chunk 学习 → 之后常驻)
            hit_r = 0.0
            tot_r = 0.0
            for k, cnt in _COUNTS.items():
                f = _FIRST.get(k)
                if f is None:
                    continue
                later = (cnt - f).clamp(min=0).float()
                tr = float(later.sum())
                if tr <= 0:
                    continue
                tot_r += tr
                # 取 f 的 top-c 那批专家的**下标**,再看它们在 later 里占多少 ✓
                idx = torch.topk(f.float(), min(c, f.numel())).indices
                hit_r += float(later[idx].sum())
            # ✅ B381 更正:此列**没有坏** —— 先前判"未修好"是**误判**(见 B381)。
            #    第一次 report(节流可能在【第 1 个 chunk】就触发)时还没有"后续"⇒ later=0
            #    ⇒ 正确地报 '-' ✓;第二次起就有值 ✓(合成稳定场景实测:实际 == oracle ✓)
            gr = (100.0 * hit_r / tot_r) if tot_r > 0 else None
            print(f"[xtu-route]   C={c:>3}/层 ⇒ oracle(累积)**{g:5.1f}%**"
                  f" · ⭐实际(首chunk学→后续)**{('待后续chunk' if gr is None else f'{gr:5.1f}%')}**"
                  f"   (装这些需 ~{gib:5.1f} GiB)", flush=True)
        print("[xtu-route]   ⇒ ⭐ 若「实际」≈「oracle」⇒ 热集【跨 chunk 稳定】,方案可行 ✓;"
              "若差很多 ⇒ 需在线自适应 ✓", flush=True)
    except Exception as e:  # noqa: BLE001
        # ⭐ B382:`observe` 早就做了"失败可见",**`report` 漏了** ✗
        #    根因正是这里静默吞掉了 `cnt - f` 的形状错误 ⇒ 我多绕了 3 轮 ✗
        if not _S.get("warned"):
            _S["warned"] = True
            import traceback
            print(f"[xtu-route] ⚠️ report() 异常(只报一次): {e!r}", flush=True)
            traceback.print_exc()
