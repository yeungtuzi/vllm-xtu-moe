#!/usr/bin/env python3
# 对**已在运行**的 vLLM OpenAI 服务抓一段 torch profiler trace:一次长 prefill + 一次批量 decode。
#
# 为什么需要它(§564,目标 item ⑤/编排归因):服务级每层 0.80→1.43 ms 的差距到底落在
# "GPU kernel"还是"kernel 之间的空档/编排"上,只有 kernel 级 trace 能回答。探针里的
# CD_TIMING 只给 period/compute/rest 三个数,再往下必须看 trace。
#
# 前置:服务必须带 VLLM_TORCH_PROFILER_DIR 启动(见 xtu_own_v41_mem.sh 的 PROFILE_DIR)。
#
# 用法:
#   python3 report/tuning/probes/prof_capture.py --port 8312 --out-dir /tmp/engprof
#   可选:--prefill-tokens 7000 --decode-seqs 8 --decode-tokens 128
#
# 产物:服务端写 <out_dir>/*.pt.trace.json(.gz);本脚本最后打印新出现的文件路径。
# License: Apache-2.0
import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request


def _post(port: int, path: str, payload: dict | None = None, timeout: float = 30.0):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="ignore")[:400]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="ignore")[:400]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="dsv41-xtu")
    ap.add_argument("--out-dir", default="/tmp/engprof")
    ap.add_argument("--prefill-tokens", type=int, default=7000)
    ap.add_argument("--decode-seqs", type=int, default=8)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--skip-prefill", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.out_dir, "*trace*")))

    code, body = _post(args.port, "/start_profile")
    print(f"[prof] start_profile -> {code} {body}")
    if code != 200:
        return 2

    if not args.skip_prefill:
        # ~4.5 token/词 的粗估足够:长 prompt 只为把 prefill 路径打满
        filler = "The quick brown fox jumps over the lazy dog. "
        n_rep = max(1, args.prefill_tokens // 9)
        prompt = filler * n_rep
        t0 = time.time()
        code, body = _post(
            args.port,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            timeout=600.0,
        )
        print(
            f"[prof] prefill({len(prompt)} chars, ~{args.prefill_tokens} tok) "
            f"-> {code} in {time.time() - t0:.1f}s {body[:200]}"
        )

    # 批量 decode:并发发 decode_seqs 个短请求,每个 decode_tokens 步
    import concurrent.futures

    def _one(i: int):
        t0 = time.time()
        code, body = _post(
            args.port,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": f"Question {i}: explain tensor parallelism briefly.",
                "max_tokens": args.decode_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            timeout=900.0,
        )
        return code, time.time() - t0

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.decode_seqs) as ex:
        res = list(ex.map(_one, range(args.decode_seqs)))
    dt = time.time() - t0
    ok = sum(1 for c, _ in res if c == 200)
    tpot = (dt * 1000.0 / args.decode_tokens) if args.decode_tokens else 0.0
    print(
        f"[prof] decode x{args.decode_seqs} x{args.decode_tokens} tok: {ok} ok in {dt:.2f}s "
        f"-> 每步(整批) {tpot:.2f} ms"
    )

    code, body = _post(args.port, "/stop_profile")
    print(f"[prof] stop_profile -> {code} {body}")

    for _ in range(30):
        new = sorted(set(glob.glob(os.path.join(args.out_dir, "*trace*"))) - before)
        if new:
            for p in new:
                print(f"[prof] TRACE {p} ({os.path.getsize(p) / 1e6:.1f} MB)")
            return 0
        time.sleep(1)
    print("[prof] 没等到新 trace 文件(检查 VLLM_TORCH_PROFILER_DIR 与日志)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
