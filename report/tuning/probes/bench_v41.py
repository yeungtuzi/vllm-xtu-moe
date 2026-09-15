#!/usr/bin/env python
"""DeepSeek-V4.1-Flash 性能基准(纯 stdlib)。

用法:
    python bench_v41.py <port> <label> [--maxtok M] [--prompt P]
                        [--repeat R] [--conc C] [--reqs N]

设计要点(来自 report/tuning/BENCH_REFERENCE.md 的教训):
  * **必须看方差,不只看均值** —— lk-moe 的数据里"上限没提高但方差被消掉"才是关键信号,
    所以这里对重复测量给出 P50 / P90 / P99 与 min/max;
  * 所有请求 temperature=0;
  * 测吞吐必须用**会持续生成**的 prompt(短 prompt 会 3 个 token 就 EOS)。
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8108"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


MAXTOK = arg("--maxtok", 64)
# --synth N:生成 N 个 token 量级的长 prompt(用于长上下文 prefill 计时)。
# 用重复但可预测的文本,避免 tokenizer 把它压得过短。
_SYNTH = arg("--synth", 0)
if _SYNTH:
    _unit = "The quick brown fox jumps over the lazy dog near the river bank. "
    _need = max(1, _SYNTH // 10)          # 每 unit 约 11 个 token
    PROMPT = _unit * _need
else:
    PROMPT = arg("--prompt", "Count from 1 to 1000, separated by commas: 1, 2, 3,")
REPEAT = arg("--repeat", 1)
CONC = arg("--conc", 1)
REQS = arg("--reqs", CONC)
URL = f"http://127.0.0.1:{PORT}/v1/completions"


def one(prompt, maxtok):
    body = json.dumps(
        {"model": "dsv41", "prompt": prompt, "max_tokens": maxtok, "temperature": 0}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    return (dt, d["usage"]["completion_tokens"], d["choices"][0]["text"],
            d["usage"].get("prompt_tokens", 0))


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))
    return xs[i]


def report(name, dts, cts):
    tps = [c / d for d, c in zip(dts, cts) if d > 0]
    if not tps:
        return
    print(
        f"{name:22s} n={len(tps):3d}  lat P50={statistics.median(dts):6.2f}s "
        f"P90={pct(dts, 90):6.2f}s  max={max(dts):6.2f}s | "
        f"tok/s P50={statistics.median(tps):6.2f} P90={pct(tps, 90):6.2f} "
        f"min={min(tps):6.2f} max={max(tps):6.2f}"
    )


print(f"=== bench label={LABEL} port={PORT} maxtok={MAXTOK} repeat={REPEAT} conc={CONC} ===")
dt, ct, txt, pt0 = one(PROMPT, 4)
print(f"warmup                : {dt:6.2f}s  prompt_tok={pt0:6d}  text={txt[:30]!r}")
if pt0:
    print(f"  prefill: {pt0 / dt:8.1f} tok/s  ({dt:.2f}s for {pt0} prompt tokens)")

# 串行重复:看抖动(均值之外的稳定性)
dts, cts = [], []
for _ in range(REPEAT):
    d, c, t = one(PROMPT, MAXTOK)
    dts.append(d)
    cts.append(c)
report(f"serial x{REPEAT}", dts, cts)
if REPEAT == 1:
    print(f"                       text={t[:60]!r}")

# 并发:每请求各自计时,再算聚合
if CONC > 1:
    res = []
    lock = threading.Lock()

    def worker():
        d, c, _ = one(PROMPT, MAXTOK)
        with lock:
            res.append((d, c))

    ts = [threading.Thread(target=worker) for _ in range(REQS)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0
    tot = sum(c for _, c in res)
    report(f"conc={CONC}", [d for d, _ in res], [c for _, c in res])
    print(
        f"{'':22s} wall={wall:6.2f}s  total={tot:4d} tok  "
        f"**aggregate={tot / wall:7.2f} tok/s**  (ideal if perfect scaling = "
        f"{statistics.median([c / d for d, c in res]) * CONC:7.2f})"
    )
