#!/usr/bin/env python3
"""临时实例 `exp8192`(MBT=8192 + 两个探针)的**一次到位**采集。

一次运行同时拿三个答案:
  (a) ⭐ **显存谷底**  ← 日志里的 `[xtu-mem]`(验 A39 的 `q_out`=512 MiB 风险)
  (b) ⭐ **端到端预填墙钟** ← 冷长 prompt 的 TTFT,与 MBT=6144 基线 64.76 s 对比
  (c) ⭐ **专家复用率** ← 日志里的 `[xtu-route]`(= 杠杆②能省掉的搬运比例)

用法:
  PORT=8071 LEN=32768 python scripts/measure_mbt8192_trial.py
"""
from __future__ import annotations

import os
import re
import subprocess
import time

import requests

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", 8071))
LEN = int(os.environ.get("LEN", 32768))
LOG = os.environ.get("LOG", "dev-docs/report/tuning/logs/exp8192.log")
MODEL = "DeepSeek-V4.1-Flash"


def port_up() -> bool:
    try:
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=10).stdout
        return f":{PORT} " in out
    except Exception:  # noqa: BLE001
        return False


def wait_ready(timeout_s: int = 2400) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if port_up():
            try:
                r = requests.get(f"http://{HOST}:{PORT}/v1/models", timeout=10)
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
        time.sleep(15)
    return False


def probe_lines(pat: str, n: int = 40) -> list[str]:
    try:
        out = subprocess.run(["grep", "-h", pat, LOG], capture_output=True, text=True, timeout=30).stdout
        return [ln for ln in out.splitlines() if ln.strip()][-n:]
    except Exception:  # noqa: BLE001
        return []


def main() -> int:
    print(f"等待 :{PORT} 就绪(最多 40 分钟)…", flush=True)
    if not wait_ready():
        print("✗ 等不到就绪"); return 2
    print(f"✅ :{PORT} 就绪 @ {time.strftime('%H:%M:%S')}", flush=True)

    nonce = os.environ.get("NONCE") or f"[t{int(time.time())}]"
    prompt = ("请逐字复述下面这段无意义文本,不要总结、不要解释:" + nonce +
              "".join(chr(0x4E00 + ((i * 7 + 99) % 2000)) for i in range(LEN)))
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 8,
            "temperature": 0.0, "ignore_eos": True, "stream": True}
    print(f"冷长 prompt(nonce={nonce!r},LEN={LEN} 字符)…", flush=True)
    t0 = time.time(); ttft = None
    try:
        with requests.post(f"http://{HOST}:{PORT}/v1/completions", json=body,
                           stream=True, timeout=7200) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                    ttft = time.time() - t0
                    break
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 请求异常: {e!r}")
    print(f"\n=== (b) 端到端预填墙钟 ===\n  TTFT = {ttft:.2f} s" if ttft else "  请求失败")
    print(f"  MBT=6144 基线(生产口径) = 64.76 s ⇒ 本次 {'快' if ttft and ttft < 64.76 else '慢'} "
          f"{abs(100*(ttft/64.76-1)):.1f}%" if ttft else "")

    time.sleep(5)
    print("\n=== (a) 显存谷底 `[xtu-mem]`(**验 A39 风险**)===")
    for ln in probe_lines("xtu-mem", 12):
        print("  " + ln)
    print("\n=== (c) 专家复用率 `[xtu-route]`(= 杠杆②的上限)===")
    for ln in probe_lines("xtu-route", 30):
        print("  " + ln)
    print("\n⚠️ 若 `[xtu-mem]` 的**最低可用** < 512 MiB + 余量 ⇒ MBT=8192 有 A39 风险 ✗")
    print("⚠️ 若 `[xtu-route]` 的全局命中率 < ~10% ⇒ 热专家驻留(②)不值得做 ✗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
