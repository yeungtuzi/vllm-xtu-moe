#!/usr/bin/env python3
"""量「预填与解码能不能真并行」—— **同一长 prompt,有/无并发解码时的预填墙钟**。

═══════════════════════════════════════════════════════════════════════════
 动机(2026-10-08 用户提出)
═══════════════════════════════════════════════════════════════════════════
已知(本仓台账):
  * **B329**: GPU 预填期间 `GPU1/2 92–100%`,而 **CPU 仅 2–5/192 核忙** ⇒ **CPU/DRAM 总体闲置**
  * **B284/B285**: 预填**不是算力受限** —— 真瓶颈是设备侧 **strided 转置**(84 vs 1361 GB/s),
    `asm` 占 **250 ms/层 = 85%**,而该层 MoE 计算只要 **5–50 ms** ⇒ **SM 在空转等 DMA**
  ⇒ ⇒ 两侧都有余量:预填吃 **PCIe/装配带宽**,解码吃 **CPU 核 + DRAM** ⇒ **理论上可不冲突**

⚠️ 但 **B273 现成的 C=1(33.6 s) vs C=4(92 s)** **不能**回答这个问题 ——
   那个差里混了**准入/排队**,不是"预填被解码拖慢"✗

═══════════════════════════════════════════════════════════════════════════
 做法(两条臂,同长度、不同内容 ⇒ 避开前缀缓存)
═══════════════════════════════════════════════════════════════════════════
  臂 0(**基线 T0**): 只发一条长 prompt(seed 97)⇒ 量它自己的 **TTFT**
  臂 1(**并发 T1**): 先发 A(短 prompt + 长输出,流式)⇒ 等 `--delay` 秒后
                      注入同长度的长 prompt(seed 99)⇒ 量 **B 自己的 TTFT**;同时量 A 的 ITL

  ⇒ ⭐ 判据:`T1 / T0`
      ≈ 1.0  ⇒ **余量真实** ⇒ "解码 forward 与预填并行"值得立项 ✓
      ≫ 1.0  ⇒ GPU 侧已被预填占满 ⇒ "chunk 间插入解码步"已是合理近似 ✓
  ⚠️ 本探针**只发 HTTP 请求**:不启停进程、不碰 GPU ⇒ **不在 R38 复审门范围内** ✓

用法:
  PORT=8070 LEN=32768 DELAY=3 python scripts/probe_prefill_overlap.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import requests


def stream_one(url, model, prompt, max_tokens, sink, tag):
    """流式发一个请求;记录 TTFT 与逐 token 到达时刻(推理模型的 delta 在 reasoning 字段)。"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    marks, last, first = [], t0, None
    try:
        with requests.post(url, json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload.strip() == b"[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                ch = (delta.get("content") or delta.get("reasoning")
                      or delta.get("reasoning_content"))
                if ch:
                    now = time.time()
                    if first is None:
                        first = now - t0
                    marks.append((now - t0, (now - last) * 1000.0))
                    last = now
    except Exception as e:  # noqa: BLE001
        sink[tag] = {"error": repr(e), "ttft": None, "marks": []}
        return
    sink[tag] = {"ttft": first, "marks": marks, "wall": time.time() - t0}


_NONCE = os.environ.get("NONCE") or f"[run{int(time.time())}]"


def filler(n: int, seed: int) -> str:
    """定长"无意义文本" ⇒ 长度可控、内容随 seed 变。

    ⭐⭐ 2026-10-08 修:只靠 seed 区分**不够** —— 同一 seed 若在别的探针里用过,
    前缀缓存会命中,整组测量作废 ✗ ⇒ 必须带**唯一 nonce**(可用 NONCE= 覆盖以便复现)✓
    """
    return ("请逐字复述下面这段无意义文本,不要总结、不要解释:" + _NONCE +
            "".join(chr(0x4E00 + ((i * 7 + seed) % 2000)) for i in range(n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8070)))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--len-b", type=int, default=int(os.environ.get("LEN", 32768)),
                    help="长 prompt 字数(两条臂必须相同)")
    ap.add_argument("--len-a", type=int, default=64)
    ap.add_argument("--tok-a", type=int, default=256, help="A 的输出长度(要够长以覆盖 B 的预填窗口)")
    ap.add_argument("--tok-b", type=int, default=8)
    ap.add_argument("--delay", type=float, default=float(os.environ.get("DELAY", 3.0)))
    ap.add_argument("--decoders", type=int, default=int(os.environ.get("DECODERS", 1)),
                    help="⭐ 并发解码路数 N(目标判据:预填墙钟是否随 N 恶化)")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    # ── 臂 0:长 prompt 单独跑(基线 T0)────────────────────────────────
    print(f"=== 臂 0(基线 T0):长 prompt 单独 · len={args.len_b} · seed=97 ===", flush=True)
    s0: dict = {}
    stream_one(url, args.model, filler(args.len_b, 97), args.tok_b, s0, "B0")
    print(f"  (prompt nonce={_NONCE!r} —— 保证两臂都是【冷】预填)")
    r0 = s0.get("B0", {})
    if r0.get("error") or r0.get("ttft") is None:
        print(f"  ✗ 臂 0 失败: {r0.get('error') or '没有产出 token'}")
        return
    t0 = r0["ttft"]
    print(f"  T0(预填墙钟 / TTFT) = {t0:.2f} s", flush=True)

    time.sleep(2.0)   # 让服务回到空闲(且 A/B 内容不同 ⇒ 无前缀缓存命中)

    # ── 臂 1:长 prompt + 一路并发解码(并发 T1)──────────────────────
    print(f"=== 臂 1(并发 T1):A(len={args.len_a}/tok={args.tok_a})先跑,"
          f"延迟 {args.delay}s 后注入 B(len={args.len_b} · seed=99)===", flush=True)
    s1: dict = {}
    tAs = []
    for j in range(args.decoders):
        t = threading.Thread(target=stream_one,
                             args=(url, args.model, filler(args.len_a, 1 + j), args.tok_a, s1, f"A{j}"),
                             daemon=True)
        t.start(); tAs.append(t)
    time.sleep(args.delay)
    tB = threading.Thread(target=stream_one,
                          args=(url, args.model, filler(args.len_b, 99), args.tok_b, s1, "B"),
                          daemon=True)
    tB.start()
    tB.join()
    for t in tAs:
        t.join()

    ra, rb = s1.get("A0", {}), s1.get("B", {})
    if rb.get("error") or rb.get("ttft") is None:
        print(f"  ✗ 臂 1 的 B 失败: {rb.get('error') or '没有产出 token'}")
        return
    t1 = rb["ttft"]

    print(f"  T1(B 的预填墙钟 / TTFT) = {t1:.2f} s", flush=True)
    for j in range(args.decoders):
        rj = s1.get(f"A{j}", {})
        if rj.get("marks"):
            itls = [d for _, d in rj["marks"]]
            print(f"  A{j}(并发解码)ITL:median={statistics.median(itls):.0f} ms "
                  f"max={max(itls):.0f} ms tokens={len(itls)}", flush=True)

    print()
    print("=" * 74)
    print(f"  ⭐ N={args.decoders} 路并发:T_N / T0 = {t1 / t0:.2f}×  "
          f"(T0={t0:.2f}s 是【无解码】基线)")
    if t1 / t0 < 1.15:
        print("  ⇒ **余量真实**:并发解码几乎不拖慢预填 ⇒ 「解码 forward 与预填并行」值得立项 ✓")
    elif t1 / t0 < 1.5:
        print("  ⇒ **部分余量**:有明显争用但不大 ⇒ 可考虑更省的并行方案,需再量 ✓")
    else:
        print("  ⇒ **已占满**:并发解码显著拖慢预填 ⇒ 「chunk 间插入解码步」已是合理近似 ✓")
    print("  ⚠️ 单次配对,不是统计结论;要下结论请重复 3 轮取中位 ✓")
    print("=" * 74)


if __name__ == "__main__":
    main()
