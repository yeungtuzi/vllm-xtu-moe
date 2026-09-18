#!/usr/bin/env python3
"""一次加载内跑完整个测量组 —— 消灭"每改一个旋钮重启一次 13 分钟"的浪费。

每个 13 分钟的模型加载只换来**一个**数据点的做法是本轮最大的时间浪费(见 NOTES §171)。
本脚本在一次服务生命周期内依次测完:

  1. 预填充阶梯(nat8192 热身 + nat8192 + nat32768),每档后探活
     —— 探活同时是"32K 后引擎是否存活"的稳定性判据(R91/R120 争议点);
  2. 解码 C=1 / C=2 / C=3(无投机),给**单流 tok/s、聚合 tok/s、TPOT**;
  3. 可选:投机解码(SPEC_K>0)重复 2 那一步。

用法:
  python3 scripts/bench_battery.py [--port 8070] [--spec-k 5] [--decode-tokens 128]
"""
import argparse, json, threading, time, urllib.request, urllib.error

DS = "dev-docs/report/tuning/datasets"


def _post(port, prompt, max_tokens):
    body = json.dumps({
        "model": "DeepSeek-V4-Flash-xiaotu", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": True, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft, ntok = None, 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line:
                continue
            if ttft is None:
                ttft = time.time() - t0
            ntok += 1
    return ttft, time.time() - t0, ntok


def ds_prompt(n):
    with open(f"{DS}/nat{n}.jsonl") as f:
        return json.loads(f.readline())["prompt"]


def liveness(port):
    """短 prompt 走一次完整 decode,引擎若已死会抛错。"""
    try:
        _post(port, "hello world", 4)
        return True
    except Exception as e:                                    # noqa: BLE001
        print(f"    !! 探活失败: {type(e).__name__}: {e}", flush=True)
        return False


def prefill_ladder(port, lengths=(8192, 8192, 32768)):
    print("== 1. 预填充阶梯(客户端 TTFT 为准)==", flush=True)
    for n in lengths:
        p = ds_prompt(n)
        try:
            ttft, wall, _ = _post(port, p, 1)
            print(f"  len={n:6d} TTFT={ttft * 1000:8.0f}ms  rate={n / ttft:7.0f} t/s"
                  f"  wall={wall * 1000:.0f}ms", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"  len={n} **崩溃** {type(e).__name__}: {e}", flush=True)
            return False
        if not liveness(port):
            return False
    return True


def decode_conc(port, conc, max_tokens, prompt_len=32):
    """C 路并发、每路 max_tokens 个输出 token;返回 (单流 t/s, 聚合 t/s, TPOT ms)。"""
    p = ds_prompt(prompt_len)
    res = [None] * conc

    def worker(i):
        try:
            res[i] = _post(port, p, max_tokens)
        except Exception as e:                                # noqa: BLE001
            res[i] = e

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    ok = [r for r in res if isinstance(r, tuple)]
    if len(ok) != conc:
        print(f"  C={conc} **有请求失败** {[type(r).__name__ for r in res if not isinstance(r, tuple)]}",
              flush=True)
        return None
    per_stream = max(r[1] for r in ok)          # 最慢那路的墙钟
    tot_tok = sum(r[2] for r in ok)
    ttft = max(r[0] for r in ok)
    ntok = min(r[2] for r in ok)
    tpot = (per_stream - ttft) / max(1, ntok - 1) * 1000
    return ntok / per_stream, tot_tok / wall, tpot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8070)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--decode-conc", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--skip-prefill", action="store_true")
    a = ap.parse_args()

    if not a.skip_prefill and not prefill_ladder(a.port):
        print("== 预填充/稳定性失败,停止 ==", flush=True)
        return 2
    print("== 2. 解码(无投机)==", flush=True)
    for c in a.decode_conc:
        r = decode_conc(a.port, c, a.decode_tokens)
        if r is None:
            return 3
        print(f"  C={c}  单流={r[0]:6.2f} t/s  聚合={r[1]:6.2f} t/s  TPOT={r[2]:6.2f} ms",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
