#!/usr/bin/env python3
"""量【长 prefill 全程的谷底 free VRAM】—— A/B 的**基线臂**,零重启。

═══════════════════════════════════════════════════════════════════════════
 为什么必须测这个(第 5 轮 R38 复审的判定)
═══════════════════════════════════════════════════════════════════════════
* A39 崩溃的判据是 **崩溃【谷底】free**(`aten::new_empty` 要 512 MiB 而只剩 **223.5 MiB**)
* 而 `gpu_prefill` 闸门读的是 **预检【瞬时】free**(且 **每设备每进程只判一次**)
* ⭐ **A41 明文:这两种口径不可互换** ✗
⇒ ⇒ 本脚本测的就是那个缺失的量:**长 prompt 全过程中 free 的最低点** ✓
   基线臂 = **当前在跑的实例(MBT=6144)**;之后若把 MBT 改 8192,再跑同一脚本比谷底 ✓

用法:
  PORT=8070 LEN=32768 python3 scripts/probe_trough_free.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time

import requests


def free_mib():
    """所有 GPU 的 free MiB。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        d = {}
        for line in out.strip().splitlines():
            i, f = [x.strip() for x in line.split(",")]
            d[int(i)] = int(f)
        return d
    except Exception:
        return {}


def sampler(stop: threading.Event, samples: list, interval: float):
    while not stop.is_set():
        samples.append((time.time(), free_mib()))
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8070)))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--len", type=int, default=int(os.environ.get("LEN", 32768)))
    ap.add_argument("--tok", type=int, default=8)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/v1/completions"

    before = free_mib()
    print("=== 请求前 free(MiB)===")
    for k, v in sorted(before.items()):
        print(f"  GPU{k}: {v:,} MiB = {v/1024:.2f} GiB")

    samples: list = []
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop, samples, args.interval), daemon=True)
    th.start()

    # ⭐ nonce 保证冷预填(不命中前缀缓存)
    nonce = os.environ.get("NONCE") or f"[run{int(time.time())}]"
    prompt = ("请逐字复述下面这段无意义文本,不要总结、不要解释:" + nonce +
              "".join(chr(0x4E00 + ((i * 7 + 99) % 2000)) for i in range(args.len)))
    t0 = time.time()
    body = {"model": args.model, "prompt": prompt, "max_tokens": args.tok,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    ttft = None
    try:
        with requests.post(url, json=body, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                    if ttft is None:
                        ttft = time.time() - t0
                    break
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 请求异常: {e!r}")
    wall = time.time() - t0
    time.sleep(3.0)
    stop.set(); th.join(timeout=5)

    print(f"\n=== 请求(nonce={nonce!r})===")
    print(f"  TTFT = {ttft:.2f} s · 总墙钟 = {wall:.2f} s" if ttft else f"  总墙钟 = {wall:.2f} s")

    per_gpu: dict = {}
    for _, snap in samples:
        for k, v in snap.items():
            per_gpu.setdefault(k, []).append(v)
    print("\n=== ⭐ 全程 free(MiB):min / 中位 / max ===")
    worst = {}
    for k in sorted(per_gpu):
        vs = per_gpu[k]
        worst[k] = min(vs)
        print(f"  GPU{k}: min {min(vs):>7,}  中位 {int(statistics.median(vs)):>7,}  max {max(vs):>7,}"
              f"   (采样 {len(vs)} 次)")
    before_theoretical = {k: v for k, v in before.items()}
    print("\n=== ⭐⭐ 谷底 vs 门槛 ===")
    for k in sorted(worst):
        b = before_theoretical.get(k, 0)
        drop = b - worst[k]
        print(f"  GPU{k}: 请求前 {b:,} → **谷底 {worst[k]:,} MiB** ({worst[k]/1024:.2f} GiB)"
              f"  最大跌幅 {drop:,} MiB")
    print()
    print("⇒ ⭐ A39 的判据是**谷底**(要能容纳 512 MiB 的 q_out 分配);")
    print("   与『预检瞬时 free』**不可互换**(A41)✗")
    print("⇒ 记下这个谷底作为 **MBT=6144 基线**;改 MBT=8192 后跑同一脚本比谷底 ✓")


if __name__ == "__main__":
    main()
