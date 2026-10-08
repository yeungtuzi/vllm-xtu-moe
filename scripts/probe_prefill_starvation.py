#!/usr/bin/env python3
"""测「长预填是否饿死正在解码的请求」—— 错峰注入法。

背景(B273/B276):C>1 时长 prompt 会把 p99 ITL 打到 ~8 秒,而 `mean/median ITL` 只涨
1.4–2.2×。官方 `vllm bench serve` 是**同时**发 N 个请求,**看不出"谁饿死了谁"** ⇒ 本探针
用**错峰注入**把因果隔离开:

  1. 先发 A(短 prompt + 长输出,流式)⇒ 它开始逐 token 生成;
  2. 等 `--delay` 秒后,注入 B(**长 prompt**,流式);
  3. 记录 **A 的相邻 token 间隔(ITL)** 与时间轴 ⇒ B 的预填期间 A 是否停摆、停多久。

判据:若 A 在 B 的预填窗口内出现**远大于基线**的 ITL 尖峰,则"预填独占 step"成立 ✓;
若 A 的 ITL 平稳,则 B273 的 8 秒 p99 另有原因 ✗

用法:
  PORT=8090 DELAY=2 LEN_B=32768 TOK_A=256 python scripts/probe_prefill_starvation.py
"""
import argparse, json, os, statistics, threading, time

import requests


def stream_one(url, model, prompt, max_tokens, sink, tag):
    """流式发一个请求,记录每个 token 的到达时刻(相对本请求开始)。"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
    }
    t0 = time.time()
    marks, last = [], t0
    first = None
    try:
        with requests.post(url, json=body, stream=True, timeout=1800) as r:
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
                # ⚠️ 本模型是**推理模型**:流式 delta 里是 `reasoning`(实测 `content` 为 None)⇒
                #    三个字段都要算 token,否则会得到"没有产出 token"的假象 ✗
                ch = (delta.get("content") or delta.get("reasoning")
                      or delta.get("reasoning_content"))
                if ch:
                    now = time.time()
                    if first is None:
                        first = now - t0
                    marks.append((now - t0, (now - last) * 1000.0))
                    last = now
    except Exception as e:  # noqa: BLE001
        sink[tag] = {"error": repr(e), "t0": t0, "marks": marks, "first": first}
        return
    sink[tag] = {"t0": t0, "marks": marks, "first": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8090)))
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--delay", type=float, default=float(os.environ.get("DELAY", 2.0)),
                    help="A 开始后多久注入 B(秒)")
    ap.add_argument("--len-a", type=int, default=int(os.environ.get("LEN_A", 64)))
    ap.add_argument("--len-b", type=int, default=int(os.environ.get("LEN_B", 32768)))
    ap.add_argument("--tok-a", type=int, default=int(os.environ.get("TOK_A", 256)))
    ap.add_argument("--tok-b", type=int, default=int(os.environ.get("TOK_B", 8)))
    ap.add_argument("--tag", default=os.environ.get("TAG", "starve"))
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    # A、B 用**不同内容**的 prompt(避免前缀缓存让 B 变便宜),且都是"可判定的无意义长文本"
    # ⭐⭐ 2026-10-08 修:只靠 seed 区分**不够** —— 同一 seed 在别的探针里用过 ⇒ 前缀缓存命中 ⇒
    #    B 的"预填窗口"掉到 1.1 s(实测),整组数据作废 ✗ ⇒ 必须加**唯一 nonce** ✓
    _nonce = os.environ.get("NONCE") or f"[run{int(time.time())}]"
    filler = lambda n, seed: ("请逐字复述下面这段无意义文本,不要总结、不要解释:" + _nonce +
                              "".join(chr(0x4E00 + ((i * 7 + seed) % 2000)) for i in range(n)))
    print(f"  (prompt nonce={_nonce!r} —— 保证两臂都是【冷】预填)")
    pa, pb = filler(args.len_a, 1), filler(args.len_b, 99)

    sink = {}
    tA = threading.Thread(target=stream_one, args=(url, args.model, pa, args.tok_a, sink, "A"),
                          daemon=True)
    tA.start()
    time.sleep(args.delay)
    inj = time.time()
    tB = threading.Thread(target=stream_one, args=(url, args.model, pb, args.tok_b, sink, "B"),
                          daemon=True)
    tB.start()
    tB.join()
    tA.join()

    ra, rb = sink.get("A", {}), sink.get("B", {})
    print(f"=== 探针 tag={args.tag}  delay={args.delay}s  A: len={args.len_a}/tok={args.tok_a}"
          f"  B: len={args.len_b}/tok={args.tok_b} ===")
    if "error" in ra:
        print(f"  A 失败: {ra['error']}")
    if "error" in rb:
        print(f"  B 失败: {rb['error']}")
    if not ra.get("marks"):
        print("  A 没有产出 token,无法判定")
        return
    a0 = ra["t0"]
    itls = [d for _, d in ra["marks"]]
    # B 的注入时刻(相对 A 开始)
    b_start = rb["t0"] - a0 if rb.get("t0") else None
    b_end = (rb["t0"] + (rb["marks"][-1][0] if rb.get("marks") else 0)) - a0 if rb.get("t0") else None
    print(f"  A: tokens={len(itls)}  TTFT={ra['first']*1000:.0f} ms  "
          f"ITL median={statistics.median(itls):.1f} ms  max={max(itls):.1f} ms  p90={sorted(itls)[int(.9*len(itls))-1]:.1f} ms")
    if b_start is not None:
        print(f"  B: 注入于 A 开始后 {b_start:.1f}s" +
              (f",预填结束(首个 token)于 {b_end:.1f}s ⇒ 预填窗口 {b_end-b_start:.1f}s" if b_end else ""))
    # 找 A 的最长间隔,并标出它落在 B 的预填窗口内与否
    worst = max(range(len(itls)), key=lambda i: itls[i])
    wt = ra["marks"][worst][0]
    inside = (b_start is not None and b_end is not None and b_start <= wt <= b_end)
    print(f"  A 的最大间隔 = {itls[worst]:.1f} ms,发生在 A 开始后 {wt:.1f}s "
          f"⇒ {'**落在 B 的预填窗口内** ✓ 预填独占成立' if inside else '不在 B 的预填窗口内'}")
    # 打印时间轴(粗粒度)
    print("  A 时间轴(相对 A 开始,ms;★=该 token 的 ITL > 3× 中位):")
    med = statistics.median(itls)
    for (t, d) in ra["marks"]:
        if d > 3 * med or len(itls) <= 24:
            print(f"    t={t*1000:8.0f}  ITL={d:8.1f} {'★' if d > 3*med else ''}")


if __name__ == "__main__":
    main()
