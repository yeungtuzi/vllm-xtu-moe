"""按专家暂存权重到显存(ping/pong) —— 默认关,文件门控。

目的(见 dev-docs/PLAN_DEVICE_STAGING.md):
  · 现在权重经 UVA 直接读宿主,内核在 token 块循环里【重复读 N 遍】(N=ceil(token/专家 ÷ BM))
    且每遍是 32B 跨行跳读 ✗
  · 改为:把该 (专家, 节点) 的权重块【一次连续搬到显存】,之后 N 遍重读都在显存内 ✓
    ⇒ PCIe 流量 = V(1 遍)✓ 且搬运连续 ✓
  · 单块大小(本模型):w13 每节点 2*crows13*kh13 = 737,280 B ≈ 0.70 MB
                          w2  每节点   crows2   *kh2  = 737,280 B ≈ 0.70 MB
    ⇒ ping/pong 两槽 ≈ 1.5 MB ⇒ 即使浪费也只有几 MB ✓

门控:存在 /tmp/xiaotu_stage ⇒ 启用(默认不启用 ✓)
"""

from __future__ import annotations

import os
import torch

ENABLED = os.path.exists("/tmp/xiaotu_stage")
SLOTS = int(os.environ.get("XIAOTU_STAGE_SLOTS", "2") or 2)


def plan_from_geometry(g: dict) -> dict:
    """由 shard_views 报告的几何推出【每 (专家,节点) 的字节数】—— 零模型常数。

    g 需含:crows13, crows2, kh13, kh2, ns, E
    """
    c13, c2 = int(g["crows13"]), int(g["crows2"])
    k13, k2 = int(g["kh13"]), int(g["kh2"])
    ns, E = int(g["ns"]), int(g["E"])
    w13_blk = 2 * c13 * k13          # 一个专家 × 一个节点的 w13 字节
    w2_blk = c2 * k2                 # 一个专家 × 一个节点的 w2 字节
    return {
        "w13_blk": w13_blk, "w2_blk": w2_blk, "ns": ns, "E": E,
        "slot_bytes": max(w13_blk, w2_blk),
        "pool_bytes": max(w13_blk, w2_blk) * SLOTS,
        "note": "ping/pong: 槽数=%d ⇒ 池 %.2f MB" % (SLOTS, max(w13_blk, w2_blk) * SLOTS / 1e6),
    }


class SlotPool:
    """ping/pong 槽池:每个槽一块设备缓冲 + 一个 ready 事件(供计算流等待)。"""

    def __init__(self, slot_bytes: int, device, slots: int = SLOTS):
        self.slot_bytes = int(slot_bytes)
        self.bufs = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=device)
                     for _ in range(slots)]
        self.ready = [torch.cuda.Event() for _ in range(slots)]
        self.side = torch.cuda.Stream(device=device)     # 拷贝专用侧流 ✓
        self.i = 0

    def next_slot(self):
        self.i = (self.i + 1) % len(self.bufs)
        return self.i

    def submit(self, src_u8: torch.Tensor, nbytes: int):
        """把 src 的前 nbytes 字节拷进下一个槽(侧流,异步 ✓),返回槽号。"""
        k = self.next_slot()
        with torch.cuda.stream(self.side):
            self.bufs[k][:nbytes].copy_(src_u8[:nbytes], non_blocking=True)
            self.ready[k].record(self.side)
        return k

    def wait(self, k: int):
        """计算流等待槽 k 就绪 ✓"""
        torch.cuda.current_stream().wait_event(self.ready[k])
        return self.bufs[k]


# ⚠️ 接线前必须确认的两点(见 PLAN_DEVICE_STAGING.md §4 与台账 §241):
#   ① 调用点(likely gpu_moe_layer_uva / 其 node 循环)里 w_ptr / s_ptr 与 w_off / s_off
#      的【精确计算方式】—— 因为槽内布局要与内核的寻址公式一致(否则算错)
#   ② 宿主侧 w13v[i][pid_e] / w2v[i][pid_e] 是否是【可直接 .copy_ 的张量】
#      (_view_from_ptr 的返回类型)⇒ 若不是,需要先转成 uint8 视图


_POOLS: dict = {}


def stage_tensor(host_view):
    """把一个宿主侧权重视图(某个节点的 w13/w2)按 ping/pong 搬进显存,返回【同形状的设备张量】。

    关键:内核是直接收【张量】的(sh["w13"][n])=> 换成设备张量即可,
         槽内保持与宿主相同的行主序布局 => 内核寻址公式【一字不改】✓
    未启用时原样返回(零行为变化)✓
    """
    if not ENABLED:
        return host_view
    key = (tuple(host_view.shape), host_view.dtype)
    p = _POOLS.get(key)
    if p is None:
        p = {"bufs": [torch.empty(host_view.shape, dtype=host_view.dtype, device="cuda")
                      for _ in range(SLOTS)],
             "ev": [torch.cuda.Event() for _ in range(SLOTS)],
             "side": torch.cuda.Stream(), "i": -1}
        _POOLS[key] = p
    p["i"] = (p["i"] + 1) % len(p["bufs"])
    k = p["i"]
    if p.get("n", 0) < 4:      # ⭐ 回执:证明暂存真的在执行(PRECHECK #12)✓
        p["n"] = p.get("n", 0) + 1
        print("[stage] slot=%d shape=%s dtype=%s bytes=%.2f MB"
              % (k, tuple(host_view.shape), host_view.dtype,
                 host_view.numel() * host_view.element_size() / 1e6), flush=True)
    with torch.cuda.stream(p["side"]):
        p["bufs"][k].copy_(host_view, non_blocking=True)
        p["ev"][k].record(p["side"])
    torch.cuda.current_stream().wait_event(p["ev"][k])
    return p["bufs"][k]
