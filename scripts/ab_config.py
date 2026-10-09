#!/usr/bin/env python
"""交错 A/B harness —— 修掉 B375 暴露的两个方法学硬伤。

═══════════════════════════════════════════════════════════════════════════
 为什么需要它(B375 的教训)
═══════════════════════════════════════════════════════════════════════════
① ⛔ **就绪判定不可靠**:用 `ss | grep ':8071'` 判"就绪" ⇒ **上一个实例的 socket 还没释放时它也为真** ✗
   ⇒ 实测:第二轮"约 0 分就绪",量到的根本不是新实例(16.49 s 假数据)✗
   ⇒ ⭐ 本脚本改为:**等新 PID 出现,且端口【归属该 PID】**(`ss -ltnp` 的 users 字段)✓
② ⛔ **顺序测会引入漂移**:先测完 A 再测完 B ⇒ **运行中的逐步退化全记到 B 头上** ✗
   ⇒ 实测:MBT=6144 三连测 **66.30 / 68.00 / 84.81 s**(离散 28%)✗
   ⇒ ⭐ 本脚本改为:**交错(interleaved)** `A1 B1 A2 B2 A3 B3` ✓

并**强制**报**中位 + 离散度**(min/max 与相对离散)⇒ 没有离散度的 "改善 X%" 不得当结论 ✓

用法:
  python scripts/ab_config.py --reps 3 --mbt 6144,8192 --len 32768
⚠️ 每次换配置都要**重启**(MBT 是启动参数)⇒ 本脚本自动 stop/start ✓
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(REPO, "dev-docs/report/tuning/logs")
PORT = int(os.environ.get("PORT", 8071))


def sh(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)


def pid_file(name: str) -> str:
    return os.path.join(LOGDIR, f"{name}.pid")


def read_pid(name: str) -> int | None:
    try:
        return int(open(pid_file(name)).read().strip())
    except Exception:  # noqa: BLE001
        return None


def port_owner(port: int) -> int | None:
    """⭐ 端口真正属于哪个 PID(不是"端口在听"就算就绪)✓"""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if f":{port} " in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def wait_ready(name: str, timeout_s: int = 900) -> bool:
    """⭐ B375①:必须【新 PID 活着】且【端口归属该 PID】✓"""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        p = read_pid(name)
        own = port_owner(PORT)
        if p and alive(p) and own == p:
            return True
        # 崩溃早退
        try:
            lg = open(os.path.join(LOGDIR, f"{name}.log"), errors="replace").read()[-200000:]
            if "illegal memory access" in lg or "Engine core initialization failed" in lg:
                print(f"    ⛔ {name} 启动失败", flush=True)
                return False
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    return False


def hard_stop(name: str) -> None:
    """按 PID 与端口归属精确停;绝不按名字匹配 ✓"""
    p = read_pid(name)
    pids = set()
    if p:
        pids.add(p)
        # 展开子进程(纯 PID)
        frontier = [p]
        for _ in range(4):
            nxt = []
            for q in frontier:
                try:
                    out = subprocess.run(["pgrep", "-P", str(q)], capture_output=True, text=True).stdout
                    nxt += [int(x) for x in out.split() if x.isdigit()]
                except Exception:  # noqa: BLE001
                    pass
            pids.update(nxt)
            frontier = nxt
    own = port_owner(PORT)
    if own:
        pids.add(own)
    for q in sorted(pids, reverse=True):
        try:
            os.kill(q, 15)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(6)
    for q in pids:
        if alive(q):
            try:
                os.kill(q, 9)
            except Exception:  # noqa: BLE001
                pass
    # 兜底:GPU 上的算进程(逐个 PID,不按名字)
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout
        for x in out.split():
            if x.isdigit():
                os.kill(int(x), 9)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


def start(mbt: int, name: str) -> None:
    env = dict(
        PORT=str(PORT), TAG=name, GPUS="1,2", TP="2", MAXLEN="524288", MBT=str(mbt),
        MAXSEQS="2", KV_DTYPE="fp8_ds_mla", KV_CACHE_BYTES="2684354560", GPU_UTIL="0.90",
        COMPILE="1", EAGER="0", SPEC="1", MM_IMAGES="4", WARMUP="1",
        VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS="2560",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
        SERVED="DeepSeek-V4.1-Flash",
        EXTRA_ENV="XIAOTU_GP_ACT_RESERVE_GIB=1.5 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128",
    )
    cmd = ["bash", "scripts/proc.sh", "spawn", name, "env"] + [f"{k}={v}" for k, v in env.items()] \
          + ["bash", "scripts/serve_v41.sh"]
    sh(cmd, timeout=120)


def measure(name: str, length: int) -> float | None:
    """一次冷长 prompt ⇒ TTFT(秒)"""
    py = f"""
import os,time,requests
n = str(int(time.time()*1000))
p = "请逐字复述下面这段无意义文本,不要总结、不要解释:" + n + "".join(chr(0x4E00+((i*7+99)%2000)) for i in range({length}))
b = {{"model":"DeepSeek-V4.1-Flash","prompt":p,"max_tokens":8,"temperature":0.0,"ignore_eos":True,"stream":True}}
t=time.time()
with requests.post("http://127.0.0.1:{PORT}/v1/completions", json=b, stream=True, timeout=7200) as r:
    r.raise_for_status()
    for l in r.iter_lines():
        if l and l.startswith(b"data: ") and l[6:].strip() != b"[DONE]":
            print(f"{{time.time()-t:.2f}}")
            break
"""
    r = subprocess.run([sys.executable, "-c", py], capture_output=True, text=True, timeout=7200, cwd=REPO)
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        print(f"    ⚠️ 测量失败: {r.stderr[-200:]}", flush=True)
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbt", default="6144,8192", help="要对比的 MBT,逗号分隔")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--len", type=int, default=32768)
    ap.add_argument("--name-prefix", default="ab")
    args = ap.parse_args()
    cfgs = [int(x) for x in args.mbt.split(",")]

    res: dict[int, list[float]] = {c: [] for c in cfgs}
    print(f"交错 A/B:{cfgs} × {args.reps} 轮(LEN={args.len} 字符)", flush=True)
    for rep in range(1, args.reps + 1):
        for c in cfgs:
            name = f"{args.name_prefix}{c}r{rep}"
            print(f"  ── 第 {rep} 轮 · MBT={c} · {name} ──", flush=True)
            hard_stop(name)
            start(c, name)
            if not wait_ready(name):
                print("    ⛔ 未就绪,跳过", flush=True)
                hard_stop(name)
                continue
            t = measure(name, args.len)
            if t:
                res[c].append(t)
                print(f"    TTFT = {t:.2f} s", flush=True)
            hard_stop(name)

    print("\n═══ 汇总(中位 ± 离散)═══", flush=True)
    med = {}
    for c in cfgs:
        v = res[c]
        if not v:
            print(f"  MBT={c}: 无有效样本 ✗", flush=True)
            continue
        m = statistics.median(v)
        med[c] = m
        spread = (max(v) - min(v)) / m * 100 if m else 0
        print(f"  MBT={c}: 中位 {m:.2f} s · 样本 {[round(x,2) for x in v]} · 离散 {spread:.1f}%", flush=True)
    if len(med) == 2:
        a, b = cfgs
        lo, hi = med[a], med[b]
        print(f"\n  ⇒ MBT {a} → {b}:{hi-lo:+.2f} s({(hi-lo)/lo*100:+.1f}%)"
              f"{'  ⭐ 更快' if hi < lo else '  ⚠️ 更慢/无改善'}", flush=True)
        sp = max((max(res[a])-min(res[a]))/lo*100, (max(res[b])-min(res[b]))/hi*100)
        print(f"  ⚠️ 最大离散 {sp:.1f}% ⇒ {'⭐ 结论可用' if sp < 5 else '⛔ 离散过大,结论仍不可用,需加轮次或查漂移'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
